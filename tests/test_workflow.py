import json
import os
import tempfile
import unittest

from cortexnet.card import TaskCard
from cortexnet.cli import Agent, Workspace
from cortexnet.cli_ui import PlainUI
from cortexnet.fade import MARK, watermark_fade
from cortexnet.prefetch import Prefetcher
from cortexnet.reader import Reader

from test_cli import BUGGY, Scripted

PARSER = "\n".join(f"def parse_{i}(s):\n    t = s.strip()\n    return t[{i}:]\n" for i in range(60)) + \
    "\ndef tokenize_header(s):\n    return s.split(':')\n"


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "pkg"))
        for rel, text in (("calc.py", BUGGY), ("pkg/parser.py", PARSER)):
            with open(os.path.join(self.root, rel), "w") as f:
                f.write(text)
        self.log = []

    def agent(self, steps, **kw):
        ws = Workspace(self.root, Reader(os.path.join(self.root, ".cortexnet", "agent"), split_tokens=600))
        ag = Agent(Scripted(steps), ws, ui=PlainUI(out=self.log.append), **kw)
        self.addCleanup(ag.close)
        return ag, ws

    def test_card_keeps_plan_findings_and_the_latest_want(self):
        c = TaskCard("fix total")
        c.apply(plan=["find total", "fix it"], phase="plan")
        out = c.apply(check=[1], finding="calc.py:5 subtracts", want="calc.py total", phase="nonsense")
        self.assertIn("[task card v2] goal: fix total", out)
        self.assertIn("1. [x] find total", out)
        self.assertIn("2. [ ] fix it", out)
        self.assertIn("- calc.py:5 subtracts", out)
        self.assertIn("next: calc.py total", out)
        self.assertEqual(c.phase, "plan")                       # an unknown phase is ignored

    def test_note_returns_the_card_sets_the_latent_request_and_is_logged(self):
        ag, ws = self.agent([
            ("", [("note", {"plan": ["read total", "fix"], "want": "the total function in calc.py"})]),
            ("", [("read_file", {"path": "calc.py"})]),
            ("done", []),
        ])
        ag.turn("total() is wrong")
        tools = [m["content"] for m in ag.msgs if m["role"] == "tool"]
        self.assertTrue(tools[0].startswith("[task card v1] goal: total() is wrong"))
        self.assertEqual(ws.intent, "the total function in calc.py")
        with open(os.path.join(self.root, ".cortexnet", "agent", "cases.jsonl")) as f:
            ev = [json.loads(line) for line in f]
        read = next(e for e in ev if e["ev"] == "read")
        self.assertEqual(read["intent"], "the total function in calc.py")   # the read's query carried the want
        self.assertEqual(ag.status()["card"], 1)

    def test_only_the_newest_card_survives_a_fade(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        for i, (name, content) in enumerate([("note", "[task card v1]" + "x" * 900), ("read_file", "a" * 900),
                                             ("note", "[task card v2]" + "y" * 900), ("read_file", "b" * 900)]):
            args = {"path": f"f{i}.py"} if name == "read_file" else {}
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": content})
        watermark_fade(msgs, 1000, 1000, lambda p: None, keep=1, low=0.01)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertTrue(tools[0].startswith(MARK + "an older task card"))
        self.assertTrue(tools[2].startswith("[task card v2]"))   # never faded, even at the lowest mark

    def test_scout_answers_in_its_own_context_with_read_only_tools(self):
        ag, ws = self.agent([
            ("", [("scout", {"question": "where is total defined?"})]),
            ("", [("edit_file", {"path": "calc.py", "old": "s -= x", "new": "s += x"})]),   # the scout tries
            ("", [("read_file", {"path": "calc.py", "query": "total"})]),
            ("total is at calc.py:1", []),                                                 # the scout's answer
            ("It is in calc.py:1.", []),
        ])
        out = ag.turn("where is total?")
        self.assertEqual(out, "It is in calc.py:1.")
        tools = [m["content"] for m in ag.msgs if m["role"] == "tool"]
        self.assertEqual(tools, ["scout answer:\ntotal is at calc.py:1"])        # no raw read in the parent
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("s -= x", f.read())                                    # the scout could not edit
        self.assertEqual(ws.task, "where is total?")                             # restored after the scout
        self.assertEqual(ag.status()["scouts"], 1)

    def test_prefetch_predicts_from_paths_and_symbols_and_counts_hits(self):
        ag, ws = self.agent([
            ("I will look at tokenize_header next.", [("list_dir", {"path": "."})]),
            ("", [("read_file", {"path": "pkg/parser.py", "query": "tokenize_header"})]),
            ("done", []),
        ])
        pf = ag.prefetch
        self.assertTrue(pf.ready.wait(10))
        self.assertEqual(pf.predict("open parser.py"), ["pkg/parser.py"])
        self.assertEqual(pf.predict("tokenize_header splits"), ["pkg/parser.py"])
        self.assertEqual(pf.predict("the self path value"), [])                # common words predict nothing
        ag.turn("how are headers split?")
        pf.wait()
        self.assertIn("pkg/parser.py", pf.warm)
        self.assertEqual((pf.hits, pf.misses), (1, 0))
        self.assertGreaterEqual(ag.status()["warmed"], 1)

    def test_warm_parses_once_and_a_later_read_reuses_it(self):
        r = Reader(os.path.join(self.root, ".cortexnet", "w"), split_tokens=600)
        p = os.path.join(self.root, "pkg", "parser.py")
        self.assertTrue(r.warm(p))
        self.assertFalse(r.warm(p))                                              # already parsed
        n = len(r.parsed)
        r.read(p, "tokenize_header")
        self.assertEqual(len(r.parsed), n)                                       # the read hit the warm parse

    def test_close_stops_background_work(self):
        ws = Workspace(self.root)
        pf = Prefetcher(ws)
        pf.close()
        self.assertTrue(all(not t.is_alive() for t in pf.threads))


if __name__ == "__main__":
    unittest.main()


class UiAndCommandsTest(WorkflowTest):
    def test_claude_code_style_steps_card_scout_and_diff(self):
        from rich.console import Console
        from cortexnet.cli_ui import RichUI
        con = Console(record=True, width=100, force_terminal=True, color_system="truecolor")
        ui = RichUI(console=con, inp=lambda p: "y")
        ui.step("read_file", {"path": "calc.py", "query": "total"}, "calc.py: 9 lines, ~900 tokens. Outline (...):\nL1")
        ui.step("note", {"plan": ["find", "fix"]}, "[task card v2] goal: g\nphase: edit\nplan:\n  1. [x] find\n"
                "  2. [ ] fix\nfindings:\n  - calc.py:5 subtracts\nnext: fix calc.py")
        with ui.nested("where is total?"):
            ui.step("grep", {"pattern": "def total"}, "calc.py:1: def total(xs):")
        ui.step("scout", {"question": "where is total?"}, "scout answer:\ncalc.py:1")
        ui.step("edit_file", {"path": "calc.py"}, "edited calc.py\n--- a\n+++ b\n-    s -= x\n+    s += x")
        self.assertTrue(ui.confirm("apply this edit?", "--- a/calc.py\n+++ b/calc.py\n@@ -4,3 +4,3 @@\n     for x in xs:"
                                   "\n-        s -= x\n+        s += x\n     return s"))
        ui.answer("Fixed `total`.")
        t = con.export_text()
        for want in ("● Read(calc.py", "⎿  outline", "● Card(plan · edit)", "☒ find", "☐ fix", "· calc.py:5 subtracts",
                     "next: fix calc.py", "● Scout(where is total?)", "    ● Search(/def total/", "⎿  calc.py:1",
                     "Updated calc.py with 1 addition and 1 removal", "Update(calc.py)", "    5 -         s -= x",
                     "    5 +         s += x", "● Fixed total"):
            self.assertIn(want, t)

    def test_slash_commands_run_without_the_model(self):
        from cortexnet.cli import slash
        ag, ws = self.agent([("", [("note", {"plan": ["read calc"], "want": "calc.py"})]),
                             ("", [("read_file", {"path": "calc.py"})]),
                             ("", [("read_lines", {"path": "calc.py", "start": 1, "end": 6})]),
                             ("done", [])])
        ag.turn("fix total")
        said = []
        ag.ui.say = lambda text, style="": said.append(text)
        ag.ui.table = lambda title, header, rows: said.append(title + "\n" + "\n".join(
            " | ".join(map(str, r)) for r in rows))
        n_calls = len(ag.backend.seen)
        self.assertFalse(slash(ag, ag.ui, "fix it"))
        for c in ("/help", "/card", "/context", "/reads", "/learn", "/compact", "/nope"):
            self.assertTrue(slash(ag, ag.ui, c))
        self.assertEqual(slash(ag, ag.ui, "/exit"), "exit")
        self.assertEqual(len(ag.backend.seen), n_calls)                          # no model call
        out = "\n".join(said)
        for want in ("/compact", "[task card v1]", "of 8,192 tokens", "calc.py | ", "calc.py:1-6 | ", "| live |",
                     "1 reads logged, 1 ranges", "unknown command /nope"):
            self.assertIn(want, out)
        self.assertTrue(slash(ag, ag.ui, "/clear"))
        self.assertEqual(len(ag.msgs), 1)
        self.assertTrue(ag.card.empty())

    def test_learn_turns_read_then_range_into_labelled_cases(self):
        from cortexnet.cases import build, main as learn, write
        at = PARSER.splitlines().index("def tokenize_header(s):") + 1
        ag, ws = self.agent([("", [("read_file", {"path": "pkg/parser.py", "query": "headers"})]),
                             ("", [("read_lines", {"path": "pkg/parser.py", "start": at, "end": at + 1})]),
                             ("", [("read_file", {"path": "calc.py"})]),            # no range after: unlabelled
                             ("done", [])])
        ag.turn("how are headers tokenized?")
        with open(os.path.join(ws.cases_dir, "cases.jsonl"), "a") as f:
            f.write('{"ev": "read", "id"')                                       # a run killed mid-write
        cases, texts, st = build(ws.cases_dir)
        self.assertEqual((st["reads"], st["ranges"], st["labelled"]), (2, 1, 1))
        c = cases[0]
        from cortexnet.coproc import leaves
        syms = leaves(texts[c["text"]], c["path"])
        self.assertEqual([syms[k].name for k in c["y"]], ["tokenize_header"])
        self.assertIn("how are headers tokenized?", c["q"])
        self.assertEqual(c["source"], "sessions")
        out = os.path.join(self.root, "ds")
        self.assertEqual(write(out, cases, texts), 1)
        base = os.path.join(self.root, "ds2")
        self.assertEqual(write(base, cases, texts, base=out), 2)                # merged with an existing dataset
        self.assertEqual(learn(["--repo", self.root, "--out", os.path.join(self.root, "ds3")]), 0)
