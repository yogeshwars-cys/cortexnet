"""The Cortex CLI's local commands and prompt shortcuts (docs/cli-ui.md): none of them calls the model."""
import json
import os
import tempfile
import unittest

from cortexnet.cli import (TOOLS, MUTATING, Agent, AnthropicMessages, Sessions, Workspace, preset, slash, typed,
                           why_lines)
from cortexnet.cli_ui import PlainUI
from cortexnet.fade import MARK

from test_cli import BUGGY, Scripted


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        for rel, text in (("calc.py", BUGGY), ("small.py", "def one():\n    return 1\n")):
            with open(os.path.join(self.root, rel), "w") as f:
                f.write(text)
        self.log, self.said = [], []

    def agent(self, steps=(), confirm=lambda q, d: True, **kw):
        ws = Workspace(self.root, confirm=confirm)
        ag = Agent(Scripted(list(steps)), ws, ui=PlainUI(out=self.log.append), **kw)
        ag.ui.say = lambda text, style="": self.said.append(text)
        ag.ui.table = lambda title, header, rows: self.said.append(
            title + "\n" + "\n".join(" | ".join(map(str, r)) for r in rows))
        self.addCleanup(ag.close)
        return ag, ws

    def out(self):
        return "\n".join(self.said)

    # -------------------------------------------------------------- read N, #id
    def test_read_by_number_pulls_outline_symbols_without_a_model_call(self):
        ag, ws = self.agent()
        self.assertTrue(slash(ag, ag.ui, "/outline calc.py"))
        rows = ws.symbol_list("calc.py")
        self.assertIn(f"  2. {rows[1][2]}", self.out())
        self.assertTrue(typed(ag, ag.ui, "read 1,3"))                     # bare: the file /outline just showed
        self.assertEqual(ag.backend.seen, [])
        tools = [m for m in ag.msgs if m["role"] == "tool"]
        self.assertEqual([t["name"] for t in tools], ["read_lines", "read_lines"])
        self.assertIn(f"{rows[0][0]}", tools[0]["content"])
        self.assertTrue(typed(ag, ag.ui, "read 999 calc.py"))
        self.assertIn("no 999", self.out())
        self.assertFalse(typed(ag, ag.ui, "read the file and fix it"))   # an ordinary request goes to the model

    def test_a_cortex_shell_command_at_the_prompt_is_answered_locally(self):
        ag, ws = self.agent()
        self.assertTrue(typed(ag, ag.ui, r"cortex --coproc D:\dev-cache\head.npz"))
        self.assertIn("shell command", self.out())
        self.assertEqual((ag.backend.seen, len(ag.msgs)), ([], 1))
        self.assertFalse(typed(ag, ag.ui, "cortexnet looks slow, why?"))

    def test_a_turn_after_typed_calls_sees_them_and_roles_alternate(self):
        ag, ws = self.agent([("the bug is on line 5", [])])
        typed(ag, ag.ui, "read 1 calc.py")
        typed(ag, ag.ui, "#3 #9")                                          # no menu yet: unknown ids, still a result
        ag.turn("what is wrong?")
        sent = ag.backend.seen[0]
        roles = [m["role"] for m in sent]
        self.assertEqual(roles[1:], ["user", "assistant", "tool", "user", "assistant", "tool", "user"])
        self.assertIn("unknown menu id #3", sent[6]["content"])
        _, conv = AnthropicMessages.convert(sent)
        self.assertTrue(all(a["role"] != b["role"] for a, b in zip(conv, conv[1:])))

    # -------------------------------------------------------------- /why, /map, /reads, /context
    def test_why_explains_the_bodies_a_read_showed(self):
        ag, ws = self.agent()
        ws.t_read_file("calc.py", "total sums the list")
        text = why_lines(ws, "calc.py")
        self.assertIn("policy budget", text)
        self.assertIn("closest left out:", text)
        ws.t_read_file("small.py")
        self.assertIn("shown whole", why_lines(ws, "small.py"))
        self.assertIn("has not been read", why_lines(ws, "nope.py"))

    def test_map_lists_top_level_symbols(self):
        ag, ws = self.agent()
        slash(ag, ag.ui, "/map")
        self.assertIn("calc.py (81): total, helper_0", self.out())
        self.assertIn("small.py (1): one", self.out())

    def test_pins_keep_reads_from_fading_and_context_previews_without_changing(self):
        ag, ws = self.agent([
            ("", [("read_file", {"path": "calc.py", "query": "total"})]),
            ("", [("read_lines", {"path": "small.py", "start": 1, "end": 2})]),
            ("", [("read_lines", {"path": "calc.py", "start": 1, "end": 6})]),
            ("", [("list_dir", {"path": "."})]), ("", [("list_dir", {"path": "."})]),
            ("", [("grep", {"pattern": "def one"})]), ("", [("grep", {"pattern": "helper_1"})]),
            ("done", [])], ctx=1500)
        ag.mark = 9.0                                                       # no automatic fade during the turn
        ag.turn("look around")
        before = json.dumps(ag.msgs)
        slash(ag, ag.ui, "/context")
        self.assertIn("next fade at", self.out())
        self.assertEqual(json.dumps(ag.msgs), before)                       # preview only
        slash(ag, ag.ui, "/pin calc.py")
        self.assertEqual(ag.pinned, {"calc.py"})
        ag.compact()
        calc = [m for m, n in ((m, m.get("name")) for m in ag.msgs) if m["role"] == "tool"
                and n in ("read_file", "read_lines") and "calc.py" in json.dumps(
                    [c for a in ag.msgs for c in a.get("tool_calls") or [] if c["id"] == m["tool_call_id"]])]
        self.assertTrue(calc and not any(m["content"].startswith(MARK) for m in calc))
        slash(ag, ag.ui, "/reads")
        self.assertIn("calc.py (pinned) | ", self.out())
        slash(ag, ag.ui, "/unpin calc.py")
        self.assertEqual(ag.pinned, set())

    # -------------------------------------------------------------- edits: undo, diff, revise, CRLF
    def test_undo_restores_the_file_and_tells_the_model_next_turn(self):
        ag, ws = self.agent([("", [("edit_file", {"path": "calc.py", "old": "s -= x", "new": "s += x"})]),
                             ("fixed", []), ("ok", [])])
        ag.turn("fix total")
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("s += x", f.read())
        slash(ag, ag.ui, "/diff")
        self.assertIn("+        s -= x", self.out())
        slash(ag, ag.ui, "/undo")
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("s -= x", f.read())
        ag.turn("thanks")
        self.assertIn("reverted a change with /undo", ag.backend.seen[-1][-1]["content"])
        slash(ag, ag.ui, "/undo")
        self.assertIn("nothing to undo", self.out())

    def test_undo_of_a_created_file_removes_it(self):
        ag, ws = self.agent()
        ws.t_write_file("new.py", "x = 1\n")
        self.assertTrue(os.path.exists(os.path.join(self.root, "new.py")))
        self.assertIn("removed new.py", ws.undo())
        self.assertFalse(os.path.exists(os.path.join(self.root, "new.py")))

    def test_e_revises_the_change_before_it_is_applied(self):
        answers, prompts = iter(["e", "y"]), []
        ui = PlainUI(out=self.log.append, inp=lambda prompt: prompts.append(prompt) or next(answers))
        ui.editor = lambda text: text.replace("s += x", "s = s + x")
        ws = Workspace(self.root, confirm=ui.confirm)
        out = ws.t_edit_file("calc.py", "s -= x", "s += x")
        self.assertIn("revised your change", out)
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("s = s + x", f.read())
        self.assertEqual(len(prompts), 2)                                    # asked again after the revision
        self.assertIn("[y/N/e]", prompts[0])

    def test_an_edit_keeps_a_crlf_file_crlf(self):
        p = os.path.join(self.root, "win.py")
        with open(p, "w", newline="") as f:
            f.write("def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n")
        ws = Workspace(self.root, confirm=lambda q, d: True)
        self.assertIn("edited win.py", ws.t_edit_file("win.py", "def f():\n    return 1", "def f():\n    return 3"))
        with open(p, "rb") as f:
            raw = f.read()
        self.assertIn(b"return 3\r\n", raw)
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))

    # -------------------------------------------------------------- /test, /lint, plan mode
    def test_test_and_lint_presets(self):
        ag, ws = self.agent()
        self.assertEqual(preset(ag, "test"), "")
        with open(os.path.join(self.root, "pyproject.toml"), "w") as f:
            f.write("[project]\nname='x'\n")
        self.assertEqual(preset(ag, "test"), "python -m pytest -q")
        self.assertEqual(preset(ag, "lint"), "python -m ruff check .")
        typed_cmd = 'python -c "print(42)"'
        slash(ag, ag.ui, f"/test {typed_cmd}")
        last = ag.msgs[-1]
        self.assertEqual((last["role"], last["name"]), ("tool", "run"))
        self.assertIn("42", last["content"])
        ag.msgs[-2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "python -m pytest -x tests"})
        self.assertEqual(preset(ag, "test"), "python -m pytest -x tests")      # the last test command run

    def test_plan_mode_has_no_mutating_tools(self):
        tools = [t for t in TOOLS if t["function"]["name"] not in MUTATING]
        ag, ws = self.agent([("", [("edit_file", {"path": "calc.py", "old": "s -= x", "new": "s += x"})]),
                             ("plan: change line 5", [])], tools=tools)
        ag.turn("fix total")
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("s -= x", f.read())
        self.assertIn("not available here", [m for m in ag.msgs if m["role"] == "tool"][0]["content"])

    # -------------------------------------------------------------- sessions
    def test_resume_restores_the_conversation_and_fork_saves_elsewhere(self):
        ag, ws = self.agent([("", [("read_file", {"path": "calc.py", "query": "total"})]),
                             ("", [("note", {"finding": "calc.py:5 subtracts", "want": "calc.py total"})]),
                             ("found it", [])])
        ag.turn("why is total wrong?")
        ag.pinned.add("calc.py")
        s = Sessions(self.root)
        s.start(None)
        s.save(ag)
        self.assertTrue(os.path.exists(s.latest))

        ag2, ws2 = self.agent([("", [("read_file", {"path": "calc.py", "query": "total"})]), ("ok", [])])
        s2 = Sessions(self.root)
        resumed = s2.load("last", ag2)
        s2.start(resumed, fork=True)
        self.assertNotEqual(s2.path, s.latest)
        self.assertEqual([m["role"] for m in ag2.msgs], [m["role"] for m in ag.msgs])
        self.assertEqual(ag2.card.findings, ["calc.py:5 subtracts"])
        self.assertEqual(ws2.intent, "calc.py total")
        self.assertEqual(ag2.pinned, {"calc.py"})
        self.assertTrue(all(any(r["msg"] is m for m in ag2.msgs) for r in ag2.recs))
        ag2.turn("and now?")                                                 # the restored read is still live
        self.assertRegex([m for m in ag2.msgs if m["role"] == "tool"][-1]["content"],
                         r"same result as your earlier read_file|unchanged since your read above")
        s2.save(ag2)
        with open(s.latest) as f:
            self.assertEqual(len(json.load(f)["msgs"]), len(ag.msgs))       # the fork left the original alone

        s3 = Sessions(self.root)
        s3.start(None)                                                       # a fresh run archives the last session
        self.assertFalse(os.path.exists(s3.latest))
        self.assertTrue([f for f in os.listdir(s3.archive) if not f.startswith("fork-")])


if __name__ == "__main__":
    unittest.main()
