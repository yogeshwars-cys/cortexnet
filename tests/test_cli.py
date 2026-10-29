import json
import os
import tempfile
import textwrap
import unittest

from cortexnet.cli import Agent, AnthropicMessages, Workspace
from cortexnet.cli_ui import PlainUI, RichUI, brief, make_ui, summarize

BUGGY = textwrap.dedent('''\
    def total(xs):
        """Sum of a list."""
        s = 0
        for x in xs:
            s -= x
        return s
    ''') + "\n".join(f"def helper_{i}(a):\n    b = a * {i}\n    return b + {i}\n" for i in range(80))


class Scripted:
    """A fake model: each call returns the next scripted (text, calls); a RuntimeError entry is raised instead."""

    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def chat(self, msgs, tools):
        self.seen.append(json.loads(json.dumps(msgs)))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        text, calls = step
        return text, [dict(id=f"c{len(self.seen)}_{i}", name=n, args=a) for i, (n, a) in enumerate(calls)], 0


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "calc.py"), "w") as f:
            f.write(BUGGY)
        self.log = []
        self.addCleanup(self.tmp.cleanup)          # cleanups run last-in first-out: agents close before this

    def agent(self, steps, confirm=lambda q, d: True, ctx=8192):
        ws = Workspace(self.root, confirm=confirm)
        ag = Agent(Scripted(steps), ws, ctx=ctx, ui=PlainUI(out=self.log.append))
        self.addCleanup(ag.close)
        return ag, ws

    def test_read_edit_run_answer(self):
        ag, ws = self.agent([
            ("", [("read_file", {"path": "calc.py", "query": "total sums the list"})]),
            ("", [("read_lines", {"path": "calc.py", "start": 1, "end": 6})]),
            ("the sign is wrong", [("edit_file", {"path": "calc.py", "old": "s -= x", "new": "s += x"})]),
            ("", [("run", {"command": "python -c \"import calc; print(calc.total([1, 2, 3]))\""})]),
            ("Fixed total(): it subtracted instead of adding.", []),
        ])
        out = ag.turn("total() returns the wrong result, fix it")
        self.assertIn("Fixed total()", out)
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("s += x", f.read())
        tools = [m for m in ag.msgs if m["role"] == "tool"]
        self.assertIn("Outline", tools[0]["content"])              # an outline-first read, not the whole file
        self.assertGreater(ws.saved, 0)
        self.assertIn("exit code 0", tools[3]["content"])
        self.assertIn("6", tools[3]["content"])                     # the fixed function ran

    def test_declined_edit_and_command_change_nothing(self):
        ag, _ = self.agent([
            ("", [("edit_file", {"path": "calc.py", "old": "s -= x", "new": "s += x"}),
                  ("run", {"command": "python -c \"open('x.txt', 'w').write('1')\""})]),
            ("ok", []),
        ], confirm=lambda q, d: False)
        ag.turn("fix it")
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("s -= x", f.read())
        self.assertFalse(os.path.exists(os.path.join(self.root, "x.txt")))
        self.assertEqual([m["content"] for m in ag.msgs if m["role"] == "tool"],
                         ["the user declined this edit", "the user declined this command"])

    def test_paths_stay_in_the_repository_and_edits_need_one_exact_match(self):
        ag, _ = self.agent([
            ("", [("read_file", {"path": "../outside.py"}), ("edit_file", {"path": "calc.py", "old": "return",
                                                                            "new": "yield"})]),
            ("ok", []),
        ])
        ag.turn("x")
        out = [m["content"] for m in ag.msgs if m["role"] == "tool"]
        self.assertIn("outside the repository", out[0])
        self.assertIn("must occur exactly once", out[1])

    def test_fade_runs_near_the_context_limit(self):
        for i in range(6):
            with open(os.path.join(self.root, f"m{i}.py"), "w") as f:
                f.write(BUGGY)
        reads = [("", [("read_file", {"path": f"m{i}.py", "query": f"q{i}"})]) for i in range(6)]
        ag, _ = self.agent(reads + [("done", [])], ctx=1500)
        ag.turn("read a lot")
        self.assertGreater(ag.faded, 0)
        self.assertTrue(any("faded" in line for line in self.log))

    def test_repeated_calls_point_back_and_rereads_show_only_new_parts(self):
        ag, ws = self.agent([
            ("", [("read_file", {"path": "calc.py", "query": "total"})]),
            ("", [("read_file", {"path": "calc.py", "query": "total"})]),          # identical: a pointer
            ("", [("read_file", {"path": "calc.py", "query": "helper_7"})]),       # same file, other query: delta
            ("", [("list_dir", {"path": "."})]),
            ("", [("list_dir", {"path": "."})]),                                    # identical: a pointer
            ("", [("edit_file", {"path": "calc.py", "old": "s -= x", "new": "s += x"})]),
            ("", [("list_dir", {"path": "."})]),                                    # after an edit: fresh
            ("", [("read_file", {"path": "calc.py", "query": "total"})]),          # file changed: fresh
            ("done", []),
        ])
        ag.turn("look around")
        out = [m["content"] for m in ag.msgs if m["role"] == "tool"]
        self.assertIn("Outline", out[0])
        self.assertTrue(out[1].startswith("[same result as your earlier read_file call above"))
        self.assertTrue(out[2].startswith("[calc.py is unchanged since your read above"))
        self.assertNotIn("Outline", out[2])                                        # the outline is not repeated
        self.assertTrue(out[4].startswith("[same result as your earlier list_dir call above"))
        self.assertFalse(out[6].startswith("[same result"))
        self.assertIn("Outline", out[7])
        self.assertEqual(ag.reused, 3)

    def test_a_faded_earlier_read_is_read_again_in_full(self):
        ag, _ = self.agent([("", [("read_file", {"path": "calc.py", "query": "total"})]), ("ok", [])])
        ag.turn("x")
        first = next(m for m in ag.msgs if m["role"] == "tool")
        first["content"] = "[faded to save context: calc.py; read it again if you need it]"   # as fade leaves it
        ag.backend.steps += [("", [("read_file", {"path": "calc.py", "query": "total"})]), ("ok", [])]
        ag.turn("again")
        last = [m for m in ag.msgs if m["role"] == "tool"][-1]["content"]
        self.assertIn("Outline", last)                                             # not a pointer to faded text

    def test_an_unparsable_tool_call_is_retried_not_fatal(self):
        bad = RuntimeError('HTTP 500: {"error":{"message":"The model produced output that does not match the '
                           'expected peg-native format"}}')
        ag, _ = self.agent([bad, ("answer", [])])
        self.assertEqual(ag.turn("hi"), "answer")
        self.assertTrue(any("not a valid tool call" in str(m.get("content")) for m in ag.msgs))
        ag2, _ = self.agent([RuntimeError("HTTP 401: Invalid API Key")])
        with self.assertRaises(RuntimeError):                       # other errors still surface
            ag2.turn("hi")

    def test_prompt_cache_breakpoints_and_usage_totals(self):
        conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]}]
        tools = [{"name": "a"}, {"name": "b"}]
        sysb, conv2, tools2 = AnthropicMessages.cacheable("sys", conv, tools)
        self.assertEqual(sysb[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(tools2[-1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", tools2[0])
        self.assertEqual(conv2[-1]["content"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", conv[-1]["content"][-1])               # the stored conversation is untouched
        self.assertNotIn("cache_control", str(conv2[:-1]))                       # one breakpoint in the messages
        bk = Scripted([("a", []), ("b", [])])
        ag = Agent(bk, Workspace(self.root), ui=PlainUI(out=self.log.append))
        self.addCleanup(ag.close)
        bk.last = dict(sent=1000, cached=800, out=20)
        ag.turn("one")
        ag.turn("two")
        st = ag.status()
        self.assertEqual((st["sent"], st["cached"], st["out"]), (2000, 1600, 40))

    def test_anthropic_conversion_groups_results_of_one_turn(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}},
                    {"id": "b", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "a", "content": "r1"},
                {"role": "tool", "tool_call_id": "b", "content": "r2"}]
        system, conv = AnthropicMessages.convert(msgs)
        self.assertEqual(system, "sys")
        self.assertEqual([b["input"] for b in conv[1]["content"]], [{"pattern": "x"}, {}])
        self.assertEqual([b["tool_use_id"] for b in conv[2]["content"]], ["a", "b"])   # one user message

    def test_outline_only_reads_carry_a_read_before_you_claim_note(self):
        ws = Workspace(self.root)
        only = ws.t_read_file("calc.py", "unrelated words")
        self.assertIn("only the outline is shown", only)
        with_body = ws.t_read_file("calc.py", "total sum of a list")
        if "--- lines" in with_body:
            self.assertNotIn("only the outline is shown", with_body)

    def test_context_window_from_llama_server_or_defaults(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from cortexnet.cli import server_ctx
        replies = [{"default_generation_settings": {"n_ctx": 16384}}, {"n_ctx": 4096}]

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps(replies.pop(0)).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_port}/v1"
            self.assertEqual(server_ctx(base, "openai"), 16384)
            self.assertEqual(server_ctx(base, "openai"), 4096)
        finally:
            srv.shutdown()
        self.assertEqual(server_ctx(f"http://127.0.0.1:{srv.server_port}/v1", "openai", timeout=1), 8192)  # gone
        self.assertEqual(server_ctx("https://api.meta.ai", "anthropic"), 32768)

    def test_ctrl_c_leaves_every_tool_call_with_a_result(self):
        ag, _ = self.agent([])
        ag.msgs += [{"role": "user", "content": "x"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
                        {"id": "b", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "a", "name": "grep", "content": "r"}]
        ag.interrupted()
        ids = [m["tool_call_id"] for m in ag.msgs if m["role"] == "tool"]
        self.assertEqual(sorted(ids), ["a", "b"])
        self.assertEqual(ag.msgs[-1]["content"], "(stopped by the user)")


class UITest(unittest.TestCase):
    def test_step_summaries(self):
        outline_read = ("x.py: 300 lines, ~5,083 tokens. Outline (ask for line ranges to see bodies):\nL1-4 def f()\n"
                        "--- lines 1-4 ---\n 1\tdef f():")
        self.assertEqual(summarize("read_file", {}, outline_read)[:18], "outline + 1 body, ")
        self.assertTrue(summarize("run", {}, "exit code 1\nboom").startswith("exit code 1, "))
        self.assertEqual(summarize("edit_file", {}, "edited a.py:\n--- a.py\n+++ a.py\n-x\n+y\n+z"), "+2 -1")
        self.assertEqual(brief("read_lines", {"path": "a.py", "start": 3, "end": 9}), "a.py 3-9")

    def test_rich_ui_renders_without_a_terminal(self):
        from rich.console import Console
        con = Console(record=True, width=100, force_terminal=False, color_system=None)
        answers = iter(["y", "n"])
        ui = RichUI(console=con, inp=lambda prompt: next(answers))
        ui.header(dict(repo="/r", backend="openai local @ http://x", ctx=8192, mode="auto-approve"))
        ui.step("read_file", {"path": "a.py", "query": "q"}, "a.py: 9 lines, ~900 tokens. Outline (...):\nL1 x")
        ui.step("run", {"command": "pytest"}, "exit code 1\nfail")
        self.assertTrue(ui.confirm("apply this edit to a.py?", "--- a.py\n+++ a.py\n-x\n+y"))
        self.assertFalse(ui.confirm("run: rm -rf build", ""))
        ui.answer("**Fixed** the sign.")
        ui.status(dict(context=6000, ctx=8192, faded=2, saved=1234))
        text = con.export_text()
        for want in ("Cortex CLI", "Read(a.py", "Bash(pytest)", "exit code 1", "+ y", "Update(a.py)", "Fixed the sign",
                     "6,000/8,192", "faded 2"):
            self.assertIn(want, text)

    def test_spinner_survives_a_cp1252_console(self):
        import io
        from rich.console import Console

        class Cp1252(io.StringIO):
            encoding = "cp1252"

            def write(self, s):
                s.encode("cp1252")                       # raises on characters the code page lacks
                return super().write(s)
        con = Console(file=Cp1252(), force_terminal=True, legacy_windows=False, width=80)
        ui = RichUI(console=con, inp=lambda p: "y")
        with ui.thinking():
            pass
        # every glyph the UI draws has an ASCII stand-in on a legacy code page
        ui.step("read_file", {"path": "a.py"}, "a.py: 9 lines, ~900 tokens. Outline (...):\nL1 x")
        ui.step("note", {"plan": ["a"]}, "[task card v1] goal: g\nphase: plan\nplan:\n  1. [x] a\n  2. [ ] b")
        with ui.nested("where?"):
            ui.step("grep", {"pattern": "x"}, "a.py:1: x")
        ui.step("scout", {"question": "where?"}, "scout answer:\na.py:1")
        ui.confirm("apply?", "--- a.py\n+++ a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n z")
        ui.answer("done")
        ui.status(dict(context=10, ctx=100, faded=0, saved=5, sent=100, cached=50, prefetch_hits=1, prefetch_misses=1))
        ui.toolbar()

    def test_logo_banner_by_terminal_width_and_encoding(self):
        import io
        from rich.console import Console
        from cortexnet.cli_ui import BRAIN_ART, gradient

        info = dict(repo="/r", backend="openai m @ http://x", ctx=8192, mode="")
        self.assertIsNotNone(BRAIN_ART())                                    # the packaged asset is there
        self.assertEqual(gradient(0), "#2e9345")
        self.assertEqual(gradient(1), "#eced62")

        def shown(width, encoding="utf-8"):
            class F(io.StringIO):
                pass
            F.encoding = encoding
            con = Console(file=F(), force_terminal=True, color_system="truecolor", width=width, legacy_windows=False,
                          record=True)
            RichUI(console=con, inp=lambda p: "").header(info)
            return con.export_text()
        wide, narrow, tiny, legacy = shown(120), shown(70), shown(40), shown(120, "cp1252")
        self.assertIn("▀", wide)                                             # brain + wordmark
        self.assertIn("██████╗", wide)
        self.assertNotIn("▀", narrow)                                        # wordmark only
        self.assertIn("██████╗", narrow)
        for out in (tiny, legacy):                                           # the plain panel header
            self.assertNotIn("█", out)
            self.assertIn("Cortex CLI", out)

    def test_plain_ui_when_not_a_terminal(self):
        self.assertIsInstance(make_ui(plain=True), PlainUI)
        self.assertNotIsInstance(make_ui(plain=True), RichUI)


if __name__ == "__main__":
    unittest.main()
