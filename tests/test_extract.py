"""User extract (cortexnet/extract.py) and the bridge into the agent turn (Agent.bridge)."""
import hashlib
import os
import tempfile
import unittest

import numpy as np

from cortexnet.cli import Agent, Workspace
from cortexnet.cli_ui import PlainUI
from cortexnet.coproc import Coprocessor
from cortexnet.extract import extract
from cortexnet.prefetch import Prefetcher

from test_cli import BUGGY, Scripted

PARSER = "\n".join(f"def parse_{i}(s):\n    return s[{i}:]\n" for i in range(40)) + \
    "\ndef tokenize_header(s):\n    return s.split(':')\n"


class FakeEncoder:
    """Deterministic 8-d vectors from a text hash; counts what it was asked to encode."""

    def __init__(self):
        self.seen = []

    def __call__(self, texts):
        self.seen += list(texts)
        out = []
        for t in texts:
            v = np.frombuffer(hashlib.sha256(t.encode()).digest()[:32], np.uint8).astype(np.float32)[:8] - 128
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def fake_coproc():
    cp = Coprocessor.__new__(Coprocessor)                 # no trained head needed for the query cache
    cp.dim, cp.gamma, cp.encoder, cp.store = 8, 0.5, FakeEncoder(), {}
    return cp


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "pkg"))
        os.makedirs(os.path.join(self.root, "bench"))
        for rel, text in (("calc.py", BUGGY), ("pkg/parser.py", PARSER), ("notes.txt", "todo\n"),
                          ("bench/proxy_ab.py", "def command(arm):\n    return [arm]\n"),
                          ("pkg/prefetch.py", "class Prefetcher:\n    pass\n")):
            with open(os.path.join(self.root, rel), "w") as f:
                f.write(text)
        self.log = []

    def agent(self, steps, **kw):
        ws = Workspace(self.root)
        ag = Agent(Scripted(steps), ws, ui=PlainUI(out=self.log.append), **kw)
        self.addCleanup(ag.close)
        ag.prefetch.ready.wait(10)
        return ag, ws

    def test_extract_finds_files_symbols_quotes_and_kind(self):
        ws = Workspace(self.root)
        pf = Prefetcher(ws)
        self.addCleanup(pf.close)
        pf.ready.wait(10)
        x = extract("tokenize_header fails on `a:b:c` in pkg/parser.py, see notes.txt and total()", pf, ws)
        self.assertEqual(x.kind, "fix")
        self.assertEqual(x.paths, ["pkg/parser.py", "notes.txt"])
        self.assertIn(("tokenize_header", "pkg/parser.py", PARSER.split("\n").index("def tokenize_header(s):") + 1),
                      x.symbols)
        self.assertIn(("total", "calc.py", 1), x.symbols)
        self.assertEqual(x.quoted, ["a:b:c"])
        self.assertEqual(x.files[:2], ["pkg/parser.py", "notes.txt"])
        self.assertIn("tokenize_header -> pkg/parser.py:", x.note())
        self.assertIn("nothing has been read yet", x.note())
        self.assertEqual(extract("hello there", pf, ws).note(), "")
        self.assertEqual(extract("how does it work?").kind, "explain")

    def test_plain_words_are_not_symbols_but_file_stems_are_files(self):
        ws = Workspace(self.root)
        pf = Prefetcher(ws)
        self.addCleanup(pf.close)
        pf.ready.wait(10)
        x = extract("Can you explain prefetch command", pf, ws)        # the session that went to bench/proxy_ab.py
        self.assertEqual((x.paths, x.symbols, x.files), (["pkg/prefetch.py"], [], ["pkg/prefetch.py"]))
        for marked in ("explain command()", "explain `command`", "explain the function command"):
            self.assertEqual(extract(marked, pf, ws).symbols, [("command", "bench/proxy_ab.py", 1)], marked)
        self.assertEqual(extract("explain Prefetcher", pf, ws).symbols, [("Prefetcher", "pkg/prefetch.py", 1)])

    def test_turn_carries_the_note_and_the_named_file_is_warmed_before_the_first_read(self):
        ag, ws = self.agent([("", [("read_file", {"path": "pkg/parser.py", "query": "header"})]), ("ok", [])])
        ag.turn("why does tokenize_header split on every colon?")
        first = ag.backend.seen[0][-1]["content"]
        self.assertTrue(first.startswith("why does tokenize_header split on every colon?\n[CortexNET: your request"))
        self.assertEqual(ws.task, "why does tokenize_header split on every colon?")   # reads query the raw request
        self.assertEqual(ag.prefetch.hits, 1)                       # the first read was the predicted file
        self.assertTrue(any("warming pkg/parser.py" in line for line in self.log))
        self.assertIn("pkg/parser.py", ag.extracted.files)

    def test_no_hints_keeps_the_message_as_typed(self):
        ag, ws = self.agent([("ok", [])], hints=False)
        ag.turn("look at total in calc.py")
        self.assertEqual(ag.backend.seen[0][-1]["content"], "look at total in calc.py")
        self.assertEqual(ag.prefetch.predicted, ["calc.py"])        # prefetch still runs

    def test_query_cache_gives_the_same_views_and_skips_re_encoding(self):
        cold, warm = fake_coproc(), fake_coproc()
        self.assertGreater(warm.prime("fix tokenize_header"), 0)
        n = len(warm.encoder.seen)
        a, b = cold.views(prompt="fix tokenize_header"), warm.views(prompt="fix tokenize_header")
        for k in ("eq", "eh", "eqh"):
            np.testing.assert_allclose(a[k], b[k])
        self.assertEqual(warm.encoder.seen[n:], [])                  # every view text was primed
        warm.views(prompt="fix tokenize_header", trail=[(1.0, "def f(): pass")])
        warm.views(prompt="fix tokenize_header", trail=[(1.0, "def f(): pass")])
        self.assertEqual(warm.encoder.seen[n:], ["def f(): pass"])   # the trail text is encoded once

    def test_bridge_primes_the_coprocessor_in_parallel_with_the_turn(self):
        ag, ws = self.agent([("ok", [])])
        ws.reader.coproc = fake_coproc()
        ag.turn("explain tokenize_header")
        for t in ag.bridge_threads:
            t.join(10)
        self.assertGreater(ag.primed, 0)
        self.assertIn("explain tokenize_header", ws.reader.coproc.qcache)
        self.assertEqual(ag.status()["primed"], ag.primed)

    def test_scouts_do_not_extract(self):
        ag, ws = self.agent([("", [("scout", {"question": "where is total?"})]), ("total is in calc.py", []),
                             ("done", [])])
        ag.turn("find total")
        self.assertEqual(ag.backend.seen[1][-1]["content"], "where is total?")


if __name__ == "__main__":
    unittest.main()
