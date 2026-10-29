"""The spiking file manager (cortexnet/filesnn.py).

1. Dynamics: the file just used fires first; a file idle past the touch trace's horizon falls below the rheobase and
   stays silent; k-winners-take-all stops the run at k files.
2. A file the model has not opened yet fires when the words name it (path or a distinctive symbol).
3. Learning: an agent that always opens the file it names is mirrored once the plastic population learns the path
   channel, and the gate hands it the prediction; synaptic scaling keeps the weights' total strength.
4. Per-model state is saved and reloaded; an unknown file is not scored.
5. The proxy teaches a manager per model from a forwarded conversation (CORTEXNET_SNN=1).
6. The CLI agent: the repository index gets neurons, a file the model names is warmed before it opens it, /files and
   the status line report the manager.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from cortexnet.filesnn import SpikingFileManager

FILES = [f"pkg/mod_{c}.py" for c in "abcdefgh"]


def manager(**kw):
    fm = SpikingFileManager(**kw)
    for i, f in enumerate(FILES):
        fm.know(f, [f"helper_{i}", "main"])
    return fm


class DynamicsTest(unittest.TestCase):
    def test_last_used_fires_first_and_idle_files_go_silent(self):
        fm = manager(w0=(1.0, 0.0, 0.0, 0.0))
        for f in FILES[:3]:
            fm.used(f)
        self.assertEqual(fm.predict("next step"), [FILES[2], FILES[1], FILES[0]])
        for _ in range(13):                                         # past the 12-action horizon
            fm.used(FILES[3])
        self.assertEqual(fm.predict("next step"), [FILES[3]])

    def test_k_winners_take_all(self):
        fm = manager(k=2)
        for f in FILES:
            fm.used(f)
        pred = fm.predict("next step")
        self.assertEqual(len(pred), 2)
        self.assertGreaterEqual(fm.last_report["silent"], len(FILES) - 2)

    def test_named_file_fires_before_it_is_opened(self):
        fm = manager()
        self.assertEqual(fm.predict("let me open pkg/mod_e.py"), ["pkg/mod_e.py"])
        self.assertEqual(fm.predict("where is helper_6 defined"), ["pkg/mod_g.py"])
        self.assertEqual(fm.predict("nothing in particular"), [])   # no input reaches the rheobase
        fm.predict("open mod_b.py")
        self.assertEqual(fm.used("pkg/mod_b.py"), 0)
        self.assertEqual(fm.agreement()["fresh_hits"], 1.0)


class LearningTest(unittest.TestCase):
    def habit(self, fm, n=200, seed=0):
        """The agent names a file, then opens it; it rarely returns to the last one."""
        rng = np.random.default_rng(seed)
        ranks = []
        for _ in range(n):
            f = FILES[int(rng.integers(len(FILES)))]
            fm.predict(f"next I will look at {os.path.basename(f)}")
            ranks.append(fm.used(f))
        return ranks

    def test_plastic_population_learns_a_habit_and_takes_over(self):
        blind = (1.0, 0.0, 0.0, 0.5)                                # innate weights ignore names
        learn, frozen = manager(w0=blind, eta=0.1), manager(w0=blind, learn=False)
        r_learn, r_frozen = self.habit(learn), self.habit(frozen)
        top1 = lambda r: np.mean([k == 0 for k in r[100:]])
        self.assertGreater(learn.habits()["path"], 0.3)
        self.assertTrue(learn.agreement()["plastic"])
        self.assertGreater(top1(r_learn), top1(r_frozen) + 0.3)
        self.assertAlmostEqual(float(np.abs(learn.w).sum()), 1.5)  # synaptic scaling

    def test_state_per_model(self):
        with tempfile.TemporaryDirectory() as d:
            fm = manager(model="llama:local", state_dir=d, eta=0.1, w0=(1.0, 0.0, 0.0, 0.5))
            self.habit(fm, 40)
            fm.save()
            again = SpikingFileManager(model="llama:local", state_dir=d)
            np.testing.assert_allclose(again.w, fm.w)
            self.assertEqual(again.steps, fm.steps)
            self.assertEqual(SpikingFileManager(model="muse-spark", state_dir=d).steps, 0)

    def test_unknown_file_is_not_scored(self):
        fm = SpikingFileManager()
        self.assertEqual(fm.predict("anything"), [])                # no neurons yet
        fm.know("a.py")
        fm.predict("a.py")
        self.assertIsNone(fm.used("never_seen.py"))
        self.assertEqual(fm.steps, 0)
        self.assertIn("never_seen.py", fm.idx)                      # it has a neuron now


class ProxyTest(unittest.TestCase):
    def test_proxy_teaches_a_manager_per_model(self):
        from cortexnet import Reader
        from cortexnet.proxy import Rewriter
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "mod.py")
            with open(src, "w") as f:
                f.write("def helper():\n    return 1\n")
            os.environ["CORTEXNET_SNN"] = "1"
            try:
                rw = Rewriter(os.path.join(d, "proxy"), reader=Reader(os.path.join(d, "r")), rewrite=False)
            finally:
                del os.environ["CORTEXNET_SNN"]
            msgs = [{"role": "user", "content": "fix the helper"}]
            for i in range(3):
                msgs.append({"role": "assistant", "content": [
                    {"type": "text", "text": "reading mod.py"},
                    {"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {"file_path": src}}]})
                msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                                                          "content": "1\tdef helper():"}]})
                _, stats = rw.rewrite(json.dumps({"model": "claude-x", "messages": msgs}).encode())
            self.assertIn("claude-x", rw.managers)
            self.assertEqual(rw.managers["claude-x"].steps, 2)      # the first read had no neuron to predict yet
            self.assertEqual(stats["files"]["steps"], 2)


class AgentTest(unittest.TestCase):
    def test_cli_agent_warms_the_named_file_and_reports(self):
        from cortexnet.cli import Agent, Workspace, slash
        from cortexnet.cli_ui import PlainUI
        from test_cli import Scripted
        with tempfile.TemporaryDirectory() as d:
            for name in ("calc.py", "util.py"):
                with open(os.path.join(d, name), "w") as f:
                    f.write(f"def {name[:-3]}_entry(a):\n    return a\n")
            log = []
            ag = Agent(Scripted([("", [("read_file", {"path": "calc.py"})]),
                                 ("calc calls into util.py, reading it next", [("read_file", {"path": "calc.py"})]),
                                 ("", [("read_file", {"path": "util.py"})]),
                                 ("done", [])]),
                       Workspace(d), ui=PlainUI(out=log.append))
            try:
                ag.files = SpikingFileManager(model="test", state_dir=os.path.join(d, ".cortexnet"))
                ag.prefetch.ready.wait(10)
                ag.turn("look at calc")
                self.assertIn("util.py", ag.files.idx)                   # indexed before the model opened it
                self.assertEqual(ag.files.steps, 3)
                self.assertGreaterEqual(ag.files.agreement()["fresh_hits"], 0.5)   # util.py was predicted fresh
                self.assertIsNotNone(ag.status()["files"])
                slash(ag, ag.ui, "/files")
                self.assertIn("spiking file manager for test", log[-1])
            finally:
                ag.close()
            self.assertTrue(os.path.exists(os.path.join(d, ".cortexnet", "files", "test.npz")))


if __name__ == "__main__":
    unittest.main()
