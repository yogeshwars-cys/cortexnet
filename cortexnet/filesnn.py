"""A spiking file manager: leaky integrate-and-fire neurons decide which files the model will want next.

One LIF neuron per file CortexNET knows (the repository's indexed source files, plus any file the model reads, edits
or writes). Four input channels per neuron; a channel's weight is shared by every file:
  touch    a nearest-spike trace: set to 1 each time the model uses the file, x `leak` per model action. It encodes the
           time since the last use exactly (an accumulating trace let a file used three times a while ago outrank the
           file just used). leak 0.82 brings it down to the rheobase in 12 actions, the gap inside which 90% of a
           session's returns to a file happen (bench/results/working_set.md)
  path     one spike when the model's words name the file (its path, or a path ending in it)
  symbol   one spike when the words name a symbol defined in this file and at most one other known file
  habit    the same touch spikes into a slow trace (x `slow` per action), through a saturating synapse h / (1 + h): how
           much the session works on the file (unsaturated, a file used 20 times outweighed every other input)
After each model turn the inputs drive the membranes for one window of `ticks` clock steps (dt = window / ticks,
membrane time constant tau = 1): v <- exp(-dt) v + I dt, a spike and a reset by subtraction when v crosses the
threshold. The threshold is set by the rheobase: the weakest input that still fires inside the window. An inhibitory
neuron counts first spikes and silences the layer once `k` neurons have fired (k-winners-take-all), so the run stops
early. The first k neurons to fire, in spike order (latency coding; ties inside one tick go to the higher membrane
potential), are the prediction: those files are warmed ahead of the model. With constant input over the window,
first-spike order is the order of input current; what the spiking adds is the threshold (a file whose input stays
below the rheobase does not fire, so a quiet session warms nothing) and the early stop.

Two populations share the neurons' inputs: an innate one with fixed weights `w0`, and a plastic one that learns per
model. A gate tracks each population's recent hit@3 (an exponential average) and lets the better one drive the
prediction, so learning is used only for a model it helps (measured: it helps Llama, which opens the file the question
names, and hurts Claude, which returns to the file it just used; bench/results/files_replay.md).
Plastic learning: when the model's real next file arrives, a three-factor rule moves the weights: presynaptic activity
(each neuron's channel inputs) x postsynaptic error (1 at the true file, minus each neuron's chance of winning a noisy
first-spike race: a softmax of the input currents at temperature `temp`) x a rate, normalised by the layer's
presynaptic power; synaptic scaling then restores the weights' total strength, so learning shifts the balance between
channels and never the layer's excitability. Two rules measured worse: error-only updates (a perceptron on the spike
order) learned from a biased sample, the misses; without scaling, depressing the neurons that fired ahead of a silent
true file shrank every weight until most predictions were empty. The weights are shared by all files, so a population
learns a model's habit, not which file follows which question.
State per model: <state>/files/<model>.npz. Telemetry: agreement (hit@1..k), hits on files the session had not
touched yet (the only ones warming speeds up), spikes and silent neurons per run, which population drives.
"""
import os
import re
import threading

import numpy as np

CHANNELS = ("touch", "path", "symbol", "habit")
PATH_RX = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,5}\b")
IDENT_RX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
COMMON = {"self", "None", "True", "False", "return", "import", "from", "class", "def", "with", "this", "that",
          "file", "line", "lines", "code", "test", "tests", "data", "main", "init", "path", "name", "value", "type"}


def _slug(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model or "model")[:80]


class SpikingFileManager:
    def __init__(self, model="model", state_dir=None, k=3, leak=0.82, slow=0.95, ticks=128, window=4.0,
                 rheobase=0.1, eta=0.05, temp=0.2, gate=0.98, learn=True, w0=(1.0, 1.0, 0.5, 0.2)):
        self.model, self.k, self.leak, self.slow, self.eta, self.temp = model, k, leak, slow, eta, temp
        self.learn, self.gate = learn, gate
        self.ticks, self.dt = ticks, window / ticks
        self.decay = float(np.exp(-self.dt))
        self.theta = rheobase * (1 - np.exp(-window))           # input `rheobase` reaches it at the window's end
        self.w0 = np.array(w0, float)                            # innate population
        self.w = self.w0.copy()                                  # plastic population
        self.strength = float(np.abs(self.w0).sum())             # synaptic scaling keeps this total
        self.score = np.zeros(2)                                 # recent hit@3 of (innate, plastic)
        self.files, self.names, self.defs = [], {}, {}          # neuron order; file -> symbol names; name -> files
        self.idx = {}
        self.trace, self.habit = {}, {}                          # synaptic traces of the touch spikes
        self.touched = set()                                     # files the model used this session
        self.pending = None
        self.hits, self.steps, self.fresh_steps, self.fresh_hits = [0] * k, 0, 0, 0
        self.spikes = self.runs = self.neurons = self.silent = 0
        self.lock = threading.RLock()
        self.path = os.path.join(state_dir, "files", _slug(model) + ".npz") if state_dir else None
        if self.path and os.path.exists(self.path):
            st = np.load(self.path)
            if st["w"].shape == self.w.shape:
                self.w, self.score = st["w"].astype(float), st["score"].astype(float)
                self.hits, self.steps = list(st["hits"]), int(st["steps"])

    # ------------------------------------------------------------------ neurons
    def know(self, path, names=()):
        """Give a file a neuron, or refresh the symbols it defines (they feed its symbol channel)."""
        with self.lock:
            if path not in self.idx:
                self.idx[path] = len(self.files)
                self.files.append(path)
            for n in self.names.get(path, ()):
                self.defs.get(n, set()).discard(path)
            names = tuple(n for n in names if len(n) >= 4 and n not in COMMON)
            self.names[path] = names
            for n in names:
                self.defs.setdefault(n, set()).add(path)

    def new_session(self):
        """Forget the session's traces (the weights and the neurons stay)."""
        with self.lock:
            self.trace, self.habit, self.touched, self.pending = {}, {}, set(), None

    def inputs(self, words):
        """[files, channels]: each neuron's presynaptic activity for the model's latest words."""
        X = np.zeros((len(self.files), len(CHANNELS)))
        for p, v in self.trace.items():
            X[self.idx[p], 0] = v
        for p, v in self.habit.items():
            X[self.idx[p], 3] = v / (1 + v)                     # saturating synapse
        low = (words or "").lower()
        for m in PATH_RX.findall(low):
            m = m.replace("\\", "/").strip("./")
            for p in self.files:
                q = p.lower()
                if q == m or q.endswith("/" + m):
                    X[self.idx[p], 1] = 1.0
        for name in set(IDENT_RX.findall(words or "")):
            fs = self.defs.get(name, ())
            if 0 < len(fs) <= 2:
                for p in fs:
                    X[self.idx[p], 2] = 1.0
        return X

    def run(self, I):
        """One window of LIF dynamics under k-WTA. Returns (first-spiking neurons in spike order, telemetry)."""
        v = np.zeros(len(I))
        fired, spikes, live = [], 0, I > 0                       # neurons with no input get no events
        for _ in range(self.ticks):
            v[live] = self.decay * v[live] + I[live] * self.dt
            over = np.flatnonzero(live & (v >= self.theta))
            if len(over):
                spikes += len(over)
                fired += [int(i) for i in over[np.argsort(-v[over], kind="stable")] if i not in fired]
                v[over] -= self.theta
                if len(fired) >= self.k:                         # the inhibitory neuron silences the layer
                    break
        return fired[:self.k], dict(neurons=len(I), active=int(live.sum()), spikes=spikes,
                                    silent=len(I) - len(set(fired)))

    def plastic_drives(self):
        return self.learn and self.score[1] > self.score[0]

    def predict(self, words):
        """The model just said `words`: the files it will want next, most likely first ([] if nothing fires)."""
        with self.lock:
            if not self.files:
                self.pending = None
                return []
            X = self.inputs(words)
            runs = [self.run(X @ self.w0), self.run(X @ self.w) if self.learn else None]
            drive = int(self.plastic_drives())
            fired, rep = runs[drive]
            self.runs += 1
            self.spikes += rep["spikes"]
            self.neurons += rep["neurons"]
            self.silent += rep["silent"]
            self.last_report = rep
            pred = [self.files[i] for i in fired]
            self.pending = dict(X=X, fired=[r[0] if r else None for r in runs], drive=drive, pred=pred)
            return pred

    def used(self, path):
        """The model used `path` (read, edit, write): score the pending prediction, learn, then spike the file's
        touch synapses. Returns the file's rank in the prediction (None if missed or nothing was predicted)."""
        with self.lock:
            pend, self.pending = self.pending, None
            rank = None
            if pend is not None and path in self.idx and self.idx[path] < len(pend["X"]):
                t = self.idx[path]
                fired = pend["fired"][pend["drive"]]
                rank = fired.index(t) if t in fired else None
                self.steps += 1
                for j in range(self.k):
                    self.hits[j] += int(rank is not None and rank <= j)
                if path not in self.touched:
                    self.fresh_steps += 1
                    self.fresh_hits += int(rank is not None)
                for i, f in enumerate(pend["fired"]):            # the gate: each population's recent hit@k
                    if f is not None:
                        self.score[i] = self.gate * self.score[i] + (1 - self.gate) * float(t in f)
                if self.learn:
                    self._teach(pend["X"], t)
            if path not in self.idx:
                self.know(path)
            self.trace = {p: x * self.leak for p, x in self.trace.items() if x * self.leak > 1e-3}
            self.habit = {p: x * self.slow for p, x in self.habit.items() if x * self.slow > 1e-3}
            self.trace[path] = 1.0                               # nearest-spike trace
            self.habit[path] = self.habit.get(path, 0.0) + 1.0
            self.touched.add(path)
            return rank

    def _teach(self, X, t):
        race = np.flatnonzero(X.any(1) | (np.arange(len(X)) == t))   # every neuron with input, and the truth
        Xr = X[race]
        s = Xr @ self.w / self.temp
        p = np.exp(s - s.max())
        err = -p / p.sum()
        err[int(np.flatnonzero(race == t)[0])] += 1.0                # postsynaptic error: truth - race share
        w = self.w + self.eta * (err @ Xr) / (float((Xr * Xr).sum()) / len(Xr) + 1e-6)
        self.w = w * self.strength / max(float(np.abs(w).sum()), 1e-9)   # synaptic scaling

    # ------------------------------------------------------------------ telemetry
    def agreement(self):
        n = max(self.steps, 1)
        return dict(steps=self.steps, fresh=self.fresh_steps,
                    fresh_hits=self.fresh_hits / max(self.fresh_steps, 1),
                    spikes=self.spikes / max(self.runs, 1), silent=self.silent / max(self.neurons, 1),
                    plastic=bool(self.plastic_drives()),
                    **{f"hit@{j + 1}": self.hits[j] / n for j in range(self.k)})

    def habits(self):
        """The plastic population's learned weights (the innate ones are w0)."""
        return dict(zip(CHANNELS, (round(float(x), 2) for x in self.w)))

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            np.savez(self.path, w=self.w, score=self.score, hits=np.array(self.hits), steps=self.steps)
