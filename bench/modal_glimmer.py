"""Muse Glimmer 30B on Modal: frozen encoder for the cache engine, and a live coding agent on SWE-bench Lite.

Volumes: glimmer-weights (the model), cortexnet-subnn (private session cases at /data, live cases at /data/live),
swe-data (public SWE-bench repos and per-instance snapshots).

  download     CPU: weights into the Volume.
  embed        H100: every text of a case table through the text tower (cut after the deepest layer used);
               masked-mean hidden states of LAYERS, centred and L2-normalised -> <data>/glimmer_L{l}.pt.
  train        L4: bench/subnn_train.py on one encoder / data dir; returns the report.
  swe_prepare  CPU: pick SWE-bench Lite instances (stratified over repos), clone each repo once, tar the tree at
               base_commit.
  swe_run      H100: Glimmer runs the instances as an agent (bench/swe_agent.py) until the spend cap; decision states
               of every read_file at LAYERS (layer hooks, forward stopped after the deepest); live cases; then the
               live text table embedded as in `embed`.

Usage: modal run bench/modal_glimmer.py                  (replay: embed, then train per encoder)
       modal run bench/modal_glimmer.py::swe --n 2       (live smoke; --n 40 --cap 10 for the batch, --train)
"""
import os

import modal

REPO = "meta-models/Muse-Glimmer-30B"
DIR = "/w/Muse-Glimmer-30B"
LAYERS = (13, 26, 39)           # of 52; mid-depth states carry the most general semantics
USAGE_LAYER = 28                # full-attention (NoPE) layer that sees the whole context: attention usage of reads
FADE_AGE = 2                    # fade arm: a read fades to its outline after 2 turns without touching its file
MAX_TOK = 256
H100_PER_S = 0.001097           # modal.com/pricing, 2026-10-27

app = modal.App("cortexnet-glimmer")
weights = modal.Volume.from_name("glimmer-weights", create_if_missing=True)
data = modal.Volume.from_name("cortexnet-subnn", create_if_missing=True)
swe_vol = modal.Volume.from_name("swe-data", create_if_missing=True)
here = os.path.dirname(os.path.abspath(__file__))
dl_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.30", "hf_xet")
prep_image = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("datasets")
image = (modal.Image.debian_slim(python_version="3.13")      # must match subnn_data.py: ast parses differ by version
         .pip_install("torch==2.14.1", "transformers==5.21.0", "accelerate", "numpy", "scikit-learn")
         .env({"SUBNN_DATA": "/data"})
         .add_local_python_source("cortexnet")
         .add_local_file(os.path.join(here, "subnn_train.py"), "/root/bench/subnn_train.py")
         .add_local_file(os.path.join(here, "swe_agent.py"), "/root/bench/swe_agent.py"))


@app.function(image=dl_image, volumes={"/w": weights}, cpu=4, memory=8192, timeout=2 * 3600)
def download():
    import json
    import time

    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(REPO)
    t = time.time()
    snapshot_download(REPO, local_dir=DIR, max_workers=16,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "*.tiktoken"])
    weights.commit()
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(DIR) for f in fs)
    cfg = json.load(open(os.path.join(DIR, "config.json")))
    return dict(size_gb=round(size / 1e9, 1), secs=round(time.time() - t), files=files, config=cfg)


def load_prep(data_dir="/data"):
    import sys
    sys.path.insert(0, "/root/bench")
    import subnn_train as S
    S.DATA = data_dir
    cases, texts = S.load()
    S.PREP = S.Prep(cases, texts)
    return S


def pool_texts(lm, tok, texts, t0, log_every=4000):
    """Masked-mean hidden states of LAYERS for every text (uncentred), batched by length."""
    import time

    import torch
    ids = tok(texts, truncation=True, max_length=MAX_TOK)["input_ids"]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = {l: torch.zeros(len(texts), lm.config.hidden_size, dtype=torch.float32) for l in LAYERS}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    i, n_tok = 0, 0
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(ids[order[j]]) <= 16384:   # ~16k padded tokens per batch
                j += 1
            j = max(j, i + 1)
            b = order[i:j]
            L = max(len(ids[k]) for k in b)
            x = torch.full((len(b), L), pad, dtype=torch.long)
            msk = torch.zeros(len(b), L, dtype=torch.long)
            for r, k in enumerate(b):
                x[r, :len(ids[k])] = torch.tensor(ids[k])
                msk[r, :len(ids[k])] = 1
            x, msk = x.cuda(), msk.cuda()
            hs = lm(input_ids=x, attention_mask=msk, output_hidden_states=True).hidden_states
            m = msk.unsqueeze(-1).float()
            for l in LAYERS:
                out[l][torch.as_tensor(b)] = ((hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu()
            n_tok += int(msk.sum())
            if i // log_every != j // log_every:
                print(f"  {j}/{len(order)} texts, {n_tok / 1e6:.1f}M tokens ({time.time() - t0:.0f}s)", flush=True)
            i = j
    return out, n_tok


def save_pooled(S, out, data_dir):
    import torch
    import torch.nn.functional as Fn
    sha = S.texts_sha()
    for l in LAYERS:
        e = out[l] - out[l].mean(0, keepdim=True)     # centre: raw LM states share a large common direction
        torch.save(dict(emb=Fn.normalize(e, dim=-1).half(), texts_sha=sha, name=f"Muse Glimmer 30B layer {l}"),
                   f"{data_dir}/glimmer_L{l}.pt")


@app.function(image=image, gpu="H100", volumes={"/w": weights, "/data": data}, memory=65536, timeout=2 * 3600)
def embed(data_dir="/data"):
    import time

    import torch
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    S = load_prep(data_dir)
    texts = S.PREP.tab.texts
    tok = AutoTokenizer.from_pretrained(DIR)
    model = AutoModel.from_pretrained(DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.language_model
    lm.layers = lm.layers[:max(LAYERS)]              # nothing deeper is used
    lm.config.num_hidden_layers = max(LAYERS)
    print(f"{len(texts)} texts, model loaded ({time.time() - t0:.0f}s)", flush=True)
    out, n_tok = pool_texts(lm, tok, texts, t0)
    save_pooled(S, out, data_dir)
    data.commit()
    return dict(texts=len(texts), tokens=n_tok, secs=round(time.time() - t0))


@app.function(image=image, gpu="L4", volumes={"/data": data}, memory=32768, timeout=3 * 3600)
def train(source, data_dir="/data", extra=(), tag=None):
    import sys
    os.environ["SUBNN_DATA"] = data_dir
    sys.path.insert(0, "/root/bench")
    import subnn_train as S
    S.DATA = data_dir
    tag = tag or os.path.basename(source).replace(".pt", "")
    out = f"{data_dir}/subnn_{tag}.md"
    sys.argv = ["subnn_train.py", "--emb", source, "--device", "cuda", "--out", out, *extra]
    S.main()
    data.commit()
    return open(out, encoding="utf8").read(), open(out + ".rr.npz", "rb").read()


# ---------------------------------------------------------------- live agent on SWE-bench Lite

def pick(ds, n):
    """Round-robin over repos (sorted ids) of instances whose gold patch changes 1-2 .py files, <= 40 lines."""
    import re
    by = {}
    for r in ds:
        files = re.findall(r"^\+\+\+ b/(\S+)", r["patch"], re.M)
        changed = sum(1 for l in r["patch"].split("\n") if l[:1] in "+-" and not l.startswith(("+++", "---")))
        if 1 <= len(files) <= 2 and all(f.endswith(".py") for f in files) and changed <= 40:
            by.setdefault(r["repo"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["instance_id"])
    out, k = [], 0
    while len(out) < n and any(len(v) > k for v in by.values()):
        for repo in sorted(by):
            if len(by[repo]) > k and len(out) < n:
                out.append(by[repo][k])
        k += 1
    return out


@app.function(image=prep_image, volumes={"/swe": swe_vol}, cpu=4, memory=8192, timeout=3600)
def swe_prepare(n):
    import json
    import subprocess

    from datasets import load_dataset
    keep = ("instance_id", "repo", "base_commit", "patch", "problem_statement")
    insts = [{k: r[k] for k in keep} for r in pick(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"), n)]
    os.makedirs("/swe/repos", exist_ok=True)
    os.makedirs("/swe/snap", exist_ok=True)
    for r in insts:
        repo = f"/swe/repos/{r['repo'].replace('/', '__')}"
        snap = f"/swe/snap/{r['instance_id']}.tar"
        if os.path.exists(snap):
            continue
        if not os.path.exists(repo):
            subprocess.run(["git", "clone", "--quiet", f"https://github.com/{r['repo']}.git", repo], check=True)
        subprocess.run(["git", "-C", repo, "archive", "--format=tar", "-o", snap, r["base_commit"]], check=True)
        print(f"  {r['instance_id']}", flush=True)
    json.dump(insts, open("/swe/instances.json", "w"))
    swe_vol.commit()
    return [r["instance_id"] for r in insts]


@app.function(image=image, gpu="H100", volumes={"/w": weights, "/data": data, "/swe": swe_vol}, memory=65536,
              timeout=3 * 3600)
def swe_run(n, cap_usd, arm=None):
    """arm=None: the live-case run (cortex Reader, hidden states, live dataset). arm='cortex'|'whole': the token A/B,
    episodes under /data/ab/<arm>, per-episode metrics returned, no hidden states or dataset."""
    import json
    import sys
    import tarfile
    import tempfile
    import time

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    t0 = time.time()
    sys.path.insert(0, "/root/bench")
    import swe_agent as A
    live, epdir = "/data/live", (f"/data/ab/{arm}" if arm else "/data/live/episodes")
    os.makedirs(epdir, exist_ok=True)
    insts = json.load(open("/swe/instances.json"))[:n]
    tok = AutoTokenizer.from_pretrained(DIR)
    cls = getattr(transformers, "MuseGlimmerForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
    model = cls.from_pretrained(DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.model.language_model
    stops = [i for i in (tok.convert_tokens_to_ids(t) for t in ("<|eom|>", "<|eot|>", "<|end|>"))
             if isinstance(i, int) and i != tok.unk_token_id]
    cfg = lm.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    per_layer = 2 * getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * hd * 2   # K and V, bf16
    types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
    n_full = sum(t == "full_attention" for t in types)
    window = getattr(cfg, "sliding_window", None) or 10 ** 9
    kv_at = lambda T: per_layer * (n_full * T + (len(types) - n_full) * min(T, window))  # noqa: E731
    kv_bytes = per_layer * len(types)
    print(f"model loaded ({time.time() - t0:.0f}s), stop ids {stops}, KV {per_layer} B/token/layer, {n_full} full "
          f"layers, window {window}", flush=True)
    tools, system = (A.TOOLS_WHOLE, A.SYSTEM_WHOLE) if arm == "whole" else (A.TOOLS, A.SYSTEM)
    W_U = model.get_output_embeddings().weight          # (vocab, hidden), for the logit lens
    final_norm = getattr(lm, "norm", None) or getattr(lm, "final_layernorm")
    usage_attn = lm.layers[USAGE_LAYER - 1].self_attn
    assert types[USAGE_LAYER - 1] == "full_attention", "usage layer must see the whole context"
    metrics = []
    encode = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    n_gen = [0, 0.0]

    def gen(ids):
        t = time.time()
        x = torch.tensor([ids], device="cuda")
        for attempt in range(2):
            try:
                with torch.no_grad():
                    y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=1024, eos_token_id=stops,
                                       pad_token_id=stops[-1])
                break
            except torch.OutOfMemoryError:        # the context no longer fits the GPU: the episode ends, recorded
                del x
                torch.cuda.empty_cache()
                print(f"  out of GPU memory at {len(ids):,} context tokens", flush=True)
                return None
            except RuntimeError as e:             # cuDNN attention backend failure: retry once without it
                if attempt or "cuDNN" not in str(e):
                    raise
                torch.cuda.empty_cache()
                torch.backends.cuda.enable_cudnn_sdp(False)
                print(f"  cuDNN attention failed at {len(ids):,} tokens; retrying without cuDNN SDPA", flush=True)
        new = y[0, len(ids):].tolist()
        n_gen[0] += len(new)
        n_gen[1] += time.time() - t
        return new

    class Stop(Exception):
        pass

    def states(ids, pos):
        """Hidden states at the decision positions for LAYERS; the forward stops after the deepest layer."""
        got = {}

        def hook(l):
            def f(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                got[l] = o[0, pos].float().cpu().numpy()
                if l == max(LAYERS):
                    raise Stop
            return f
        hs = [lm.layers[l - 1].register_forward_hook(hook(l)) for l in LAYERS]
        try:
            with torch.no_grad():
                lm(input_ids=torch.tensor([ids], device="cuda"))
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        return got

    def probe(ep, ws):
        """One forward over the finished episode (stopped after the deepest LAYERS layer):
        hid    hidden states at every read_file decision position, per layer in LAYERS
        lens   logit lens at those states: log-prob of each leaf symbol name of the file being read (first token,
               with and without a leading space), and the top tokens, per layer
        usage  attention of the full-attention layer USAGE_LAYER from generated tokens after each read_file result
               back to it, averaged over heads and queries, summed per leaf symbol of the file (body lines only)"""
        import bisect
        import re as _re
        from cortexnet.coproc import leaves
        ids, pos = ep["ids"], [e["pos"] for e in ep["events"]]
        got, hin = {}, {}

        def hook(l):
            def f(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                got[l] = o[0, pos].float()
                if l == max(LAYERS):
                    raise Stop
            return f

        def pre(mod, args, kwargs):
            hin["h"] = kwargs.get("hidden_states", args[0] if args else None)
        hs = [lm.layers[l - 1].register_forward_hook(hook(l)) for l in LAYERS]
        hs.append(usage_attn.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            with torch.no_grad():
                lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()

        def syms_of(path):
            try:
                with open(ws.path(path), encoding="utf8", errors="replace") as f:
                    text = f.read()
                return leaves(text, ws.rel(ws.path(path)))
            except (OSError, ValueError):
                return []

        lens = []
        with torch.no_grad():
            for i, ev in enumerate(ep["events"]):
                names = sorted({s.name for s in syms_of(ev["path"])})
                first = {nm: {t[0] for t in (encode(nm), encode(" " + nm)) if t} for nm in names}
                row = {}
                for l in LAYERS:
                    h = got[l][i]
                    logp = torch.log_softmax(final_norm(h.to(W_U.dtype)).float() @ W_U.float().T, -1)
                    top = torch.topk(logp, 20).indices.tolist()
                    row[l] = dict(sym={nm: max(float(logp[t]) for t in ts) for nm, ts in first.items() if ts},
                                  top=[decode([t]) for t in top])
                lens.append(dict(call=ev["call"], path=ev["path"], layers=row))

        usage = []
        reads = ep["reads"]
        if reads and "h" in hin:
            with torch.no_grad():
                h = hin["h"]
                L = h.shape[1]
                gpos = [p for t in ep["turns"] for p in range(t["ctx"], t["ctx"] + t["n"])]
                first_end = min(r["at"] + r["n"] for r in reads)
                q = [p for p in gpos if first_end <= p < L]
                if len(q) > 1024:
                    q = [q[j] for j in np.linspace(0, len(q) - 1, 1024).astype(int)]
                if q:
                    a = usage_attn
                    K = a.qk_norm(a.k_proj(h).view(1, L, -1, hd).transpose(1, 2))[0]
                    K = K.repeat_interleave(a.num_key_value_groups, 0).float()            # (heads, L, hd)
                    qp = torch.tensor(q, device="cuda")
                    Q = a.qk_norm(a.q_proj(h[:, qp]).view(1, len(q), -1, hd).transpose(1, 2))[0] * a.qk_scale_factor
                    Q = Q.float()
                    acc = [torch.zeros(r["n"], device="cuda") for r in reads]
                    cnt = [0] * len(reads)
                    ar = torch.arange(L, device="cuda")
                    for c0 in range(0, len(q), 32):
                        qc = qp[c0:c0 + 32]
                        s = torch.einsum("hqd,hkd->hqk", Q[:, c0:c0 + 32], K) * a.scaling
                        s = s.masked_fill(ar[None, None, :] > qc[None, :, None], float("-inf"))
                        w = torch.softmax(s, -1).mean(0)                                 # (c, L)
                        for j, r in enumerate(reads):
                            sel = qc >= r["at"] + r["n"]
                            if sel.any():
                                acc[j] += w[sel][:, r["at"]:r["at"] + r["n"]].sum(0)
                                cnt[j] += int(sel.sum())
                    for j, r in enumerate(reads):
                        if not cnt[j]:
                            continue
                        mass = (acc[j] / cnt[j]).cpu().numpy()
                        blk = A.tool_block("read_file", r["out"])
                        enc = tok(blk, add_special_tokens=False, return_offsets_mapping=True)
                        o0 = blk.index(r["out"])
                        starts = [o0] + [o0 + m.end() for m in _re.finditer("\n", r["out"])]
                        lines = r["out"].split("\n")
                        if len(enc["input_ids"]) == r["n"]:
                            cs = [o for o, _ in enc["offset_mapping"]]
                        else:                                # tokenization drift: spread tokens over characters
                            cs = list(np.linspace(0, len(blk) - 1, r["n"]).astype(int))
                        fl = []
                        for ln in lines:
                            m = _re.match(r"^\s*(\d+)\t", ln)
                            fl.append(int(m.group(1)) if m else None)
                        sy = syms_of(r["path"])
                        sym, sym_tok, outl = {}, {}, 0.0
                        for t, c in enumerate(cs):
                            li = bisect.bisect_right(starts, c) - 1
                            if li < 0 or li >= len(lines):
                                continue
                            if fl[li] is None:
                                outl += float(mass[t])
                                continue
                            for s in sy:
                                if s.start <= fl[li] <= s.end:
                                    sym[s.name] = sym.get(s.name, 0.0) + float(mass[t])
                                    sym_tok[s.name] = sym_tok.get(s.name, 0) + 1
                                    break
                        usage.append(dict(call=r["call"], path=r["path"], n=r["n"], queries=cnt[j],
                                          block=float(mass.sum()), outline=outl, sym=sym, sym_tok=sym_tok))
        hid = {l: v.cpu().numpy() for l, v in got.items()}
        return hid, lens, usage

    done = {f[:-5] for f in os.listdir(epdir) if f.endswith(".json")}
    per_ep = []
    for k, inst in enumerate(insts):
        spent = (time.time() - t0) * H100_PER_S
        est = np.median(per_ep) if per_ep else 0.4
        if inst["instance_id"] in done:
            continue
        if spent + est + (0.0 if arm else 0.4) > cap_usd:           # keep ~$0.40 for embedding the live text table
            print(f"spend cap: ${spent:.2f} spent, next episode ~${est:.2f}; stopping", flush=True)
            break
        te = time.time()
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "repo")
            with tarfile.open(f"/swe/snap/{inst['instance_id']}.tar") as tf:
                tf.extractall(root, filter="data")
            ws = A.Workspace(root, os.path.join(d, "state"), arm="whole" if arm == "whole" else "cortex")
            msgs = [dict(role="system", content=system),
                    dict(role="user", content=f"Repository: {inst['repo']}\n\nIssue:\n{inst['problem_statement']}")]
            start = encode(tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False))
            ep = A.run_episode(gen, encode, decode, start, ws, fade_age=FADE_AGE if arm == "fade" else 0)
            lens, usage = [], []
            if not arm:
                hid = states(ep["ids"], [e["pos"] for e in ep["events"]]) if ep["events"] else {}
            elif arm != "fade" and not ep.get("oom") and (ep["events"] or ep["reads"]):     # fading rewrites positions: no probe there
                tp = time.time()
                try:
                    hid, lens, usage = probe(ep, ws)
                except Exception as e:                   # never lose the episode over the probe
                    torch.cuda.empty_cache()
                    hid, lens, usage = {}, [], []
                    print(f"  probe failed, skipped: {type(e).__name__}: {e}", flush=True)
                print(f"  probe {time.time() - tp:.0f}s", flush=True)
            else:
                hid = {}
            cases, texts = A.cases_from_episode(inst, ep, ws, order=k, qv0=0)
            score = A.score_submission(inst, ws)
        ctx = [t["ctx"] for t in ep["turns"]]
        m = dict(instance_id=inst["instance_id"], arm=arm, start=len(start), peak=len(ep["ids"]),
                 prefill=sum(ctx), gen=sum(t["n"] for t in ep["turns"]),
                 gen_secs=round(sum(t["secs"] for t in ep["turns"]), 1), turns=len(ep["turns"]), calls=ep["calls"],
                 reads=len(ep["events"]), ranges=len(ep["ranges"]), tool_tok=ep["tool_tok"], faded=ep["faded"],
                 hit_limit=ep["hit_limit"], oom=ep.get("oom", False), submitted=bool(ep["submitted"]), kv_bytes=kv_bytes,
                 peak_kv_mb=round(kv_at(max(ctx + [len(ep["ids"])])) / 2 ** 20, 1),
                 kv_mb_turns=round(sum(kv_at(c) for c in ctx) / 2 ** 20, 1), **score)
        rec = dict(instance_id=inst["instance_id"], calls=ep["calls"], events=ep["events"], turns=ep["turns"],
                   submitted=ep["submitted"], tokens=len(ep["ids"]), cases=cases, texts=texts,
                   transcript=decode(ep["ids"][len(start):]), secs=round(time.time() - te), metrics=m,
                   lens=lens, usage=usage, gold=sorted(A.gold_lines(inst["patch"])),
                   read_paths=[dict(call=r["call"], path=r["path"], n=r["n"], faded=r["faded"]) for r in ep["reads"]])
        m["secs"] = rec["secs"]
        metrics.append(m)
        json.dump(rec, open(f"{epdir}/{inst['instance_id']}.json", "w"))
        for l, v in hid.items():
            np.save(f"{epdir}/{inst['instance_id']}_L{l}.npy", v.astype(np.float16))
        per_ep.append((time.time() - te) * H100_PER_S)
        data.commit()
        sub = ep["submitted"][1] if ep["submitted"] else "-"
        print(f"[{k + 1}/{len(insts)}] {inst['instance_id']}: {ep['calls']} calls, {len(ep['events'])} reads, "
              f"{len(cases)} cases, submitted {sub}, {rec['secs']} s; gen {n_gen[0] / max(n_gen[1], 1e-9):.1f} tok/s; "
              f"spent ${(time.time() - t0) * H100_PER_S:.2f}", flush=True)

    if arm:                                         # A/B run: metrics of every finished episode of this arm
        out = []
        for f in sorted(x for x in os.listdir(epdir) if x.endswith(".json")):
            r = json.load(open(f"{epdir}/{f}"))
            if "metrics" in r:
                out.append(dict(r["metrics"], secs=r["secs"]))
        return dict(arm=arm, episodes=out, spent=round((time.time() - t0) * H100_PER_S, 2),
                    gen_tok_s=round(n_gen[0] / max(n_gen[1], 1e-9), 1))

    # assemble every episode so far into one live dataset (subnn_data.py format + qv rows)
    cases, texts, hrem, hidden = [], {}, {}, {l: [] for l in LAYERS}
    for f in sorted(x for x in os.listdir(epdir) if x.endswith(".json")):
        rec = json.load(open(f"{epdir}/{f}"))
        off = sum(len(h) for h in hidden[LAYERS[0]])          # first qv row of this episode
        for c in rec["cases"]:
            c["qv"] += off
            cases.append(c)
        texts.update(rec["texts"])
        hrem[rec["instance_id"]] = {str(e["call"]): rec["calls"] - e["call"] for e in rec["events"]}
        for l in LAYERS:
            p = f"{epdir}/{rec['instance_id']}_L{l}.npy"
            hidden[l].append(np.load(p) if os.path.exists(p) else np.zeros((0, lm.config.hidden_size), np.float16))
    with open(f"{live}/cases.jsonl", "w", encoding="utf8") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")
    json.dump(texts, open(f"{live}/texts.json", "w", encoding="utf8"))
    json.dump(hrem, open(f"{live}/hrem.json", "w"))
    for l in LAYERS:
        np.save(f"{live}/hidden_L{l}.npy", np.concatenate(hidden[l]).astype(np.float32))
    S = load_prep(live)
    lm.layers = lm.layers[:max(LAYERS)]
    out, n_tok = pool_texts(lm, tok, S.PREP.tab.texts, t0, log_every=10 ** 9)
    save_pooled(S, out, live)
    data.commit()
    return dict(episodes=sum(f.endswith('.json') for f in os.listdir(epdir)), cases=len(cases),
                spent=round((time.time() - t0) * H100_PER_S, 2), gen_tok_s=round(n_gen[0] / max(n_gen[1], 1e-9), 1))


@app.function(image=image, gpu="H100", volumes={"/w": weights, "/data": data}, memory=65536, timeout=1800)
def lens_pieces(arms=("whole", "cortex")):
    """Logit lens again from the saved decision states, scoring every word piece of a symbol name (identifier split
    on _ and case, with and without a leading space) instead of only its first token: score = max log-prob."""
    import json
    import re

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DIR)
    cls = getattr(transformers, "MuseGlimmerForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
    model = cls.from_pretrained(DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.model.language_model
    norm = getattr(lm, "norm", None) or getattr(lm, "final_layernorm")
    W = model.get_output_embeddings().weight
    word = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

    def cands(name):
        out = set()
        for w in {name, *word.findall(name)}:
            for v in {w, w.lower()}:
                for t in (tok.encode(v, add_special_tokens=False), tok.encode(" " + v, add_special_tokens=False)):
                    if t and re.search(r"[A-Za-z0-9]", tok.decode(t[:1])):
                        out.add(t[0])
        return sorted(out)
    res = {}
    for arm in arms:
        d = f"/data/ab/{arm}"
        for f in sorted(x for x in os.listdir(d) if x.endswith(".json")):
            r = json.load(open(f"{d}/{f}"))
            iid, rows = r["instance_id"], []
            H = {l: np.load(f"{d}/{iid}_L{l}.npy") for l in LAYERS if os.path.exists(f"{d}/{iid}_L{l}.npy")}
            if not H or not r.get("lens"):
                continue
            for i, ln in enumerate(r["lens"]):
                names = list(next(iter(ln["layers"].values()))["sym"])
                cs = {n: cands(n) for n in names}
                row = {}
                with torch.no_grad():
                    for l, h in H.items():
                        if i >= len(h):
                            continue
                        x = torch.as_tensor(h[i], device="cuda").to(W.dtype)
                        logp = torch.log_softmax(norm(x).float() @ W.float().T, -1).cpu().numpy()
                        row[str(l)] = {n: float(max(logp[t] for t in c)) for n, c in cs.items() if c}
                rows.append(dict(call=ln["call"], path=ln["path"], layers=row))
            res.setdefault(arm, {})[iid] = rows
    return res


@app.local_entrypoint()
def lens():
    """Recompute the logit lens with word pieces for the A/B episodes; writes D:/dev-cache/cortexnet/ab/lens_pieces.json."""
    import json
    out = "D:/dev-cache/cortexnet/ab/lens_pieces.json"
    res = lens_pieces.remote()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    print({a: len(v) for a, v in res.items()}, "->", out)


def save_report(name, report, rr):
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    path = os.path.join(here, "results", f"{name}.md")
    open(path, "w", encoding="utf8").write(report)
    open(path + ".rr.npz", "wb").write(rr)
    return path


@app.local_entrypoint()
def main():
    print(embed.remote())
    sources = ["minilm"] + [f"/data/glimmer_L{l}.pt" for l in LAYERS]
    for src, (report, rr) in zip(sources, train.map(sources)):
        path = save_report(f"subnn_{os.path.basename(src).replace('.pt', '')}", report, rr)
        print(f"\n===== {src} -> {path}\n{report[:2500]}")


@app.local_entrypoint()
def swe(n: int = 2, cap: float = 10.0, train_live: bool = False):
    print(swe_prepare.remote(n))
    print(swe_run.remote(n, cap))
    if train_live:
        live = "/data/live"
        arms = [("minilm", (), "live_minilm"),
                (f"{live}/glimmer_L26.pt", (), "live_glimmer_L26_text")]
        arms += [(f"{live}/glimmer_L{l}.pt", ("--qvec", f"{live}/hidden_L{l}.npy"), f"live_glimmer_L{l}_state")
                 for l in LAYERS]
        common = ("--configs", "full", "--seeds", "1")
        for (src, extra, tag), (report, rr) in zip(arms, train.starmap(
                [(src, live, (*common, *extra), tag) for src, extra, tag in arms])):
            path = save_report(f"subnn_{tag}", report, rr)
            print(f"\n===== {tag} -> {path}\n{report[:2000]}")


@app.local_entrypoint()
def ab(n: int = 40, cap: float = 6.0):
    """Token A/B with Glimmer as the worker: the same SWE-bench Lite instances; arms whole-file reads, the CortexNET
    Reader, and the Reader with fading (old reads drop to their outline); one H100 per arm, spend cap per arm. Whole
    and cortex episodes also carry the probe (decision states, logit lens, attention usage of every read). Writes results/swe_ab.json (per-episode metrics, public data only)."""
    import json
    print(swe_prepare.remote(n))
    res = list(swe_run.starmap([(n, cap, a) for a in ("whole", "cortex", "fade")]))
    path = os.path.join(here, "results", "swe_ab.json")
    json.dump(res, open(path, "w"), indent=1)
    for r in res:
        print(f"{r['arm']}: {len(r['episodes'])} episodes, ${r['spent']}, {r['gen_tok_s']} tok/s")
    print(path)


@app.local_entrypoint()
def ab_arm(arm: str, n: int = 40, cap: float = 2.0):
    """Resume one A/B arm (finished instances are skipped) and rewrite results/swe_ab_<arm>.json."""
    import json
    r = swe_run.remote(n, cap, arm)
    path = os.path.join(here, "results", f"swe_ab_{arm}.json")
    json.dump(r, open(path, "w"), indent=1)
    print(f"{arm}: {len(r['episodes'])} episodes, ${r['spent']} this run -> {path}")


@app.local_entrypoint()
def dump():
    """Full model only, out-of-fold per-symbol scores into the Volume (/data/oof_*.npz) for bench/subnn_hedge.py."""
    arms = [("minilm", "oof_minilm"), ("/data/glimmer_L26.pt", "oof_glimmer_L26")]
    for (src, tag), (report, _) in zip(arms, train.starmap(
            [(src, "/data", ("--configs", "full", "--seeds", "1", "--dump", f"/data/{tag}.npz"), tag)
             for src, tag in arms])):
        print(f"\n===== {tag}\n" + "\n".join(l for l in report.split("\n") if "full Cortex-SubNN" in l or "zero-shot" in l))
