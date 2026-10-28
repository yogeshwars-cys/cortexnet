import json
import os
import tempfile
import unittest

from cortexnet.proxy import Rewriter, shell_command
from cortexnet.shellout import outline_output

PYTEST = "\n".join(["============ test session starts ============", "platform linux, python 3.12",
                    "collected 400 items", ""]
                   + [f"tests/test_mod.py::test_case_{i} PASSED" for i in range(300)]
                   + ["tests/test_mod.py::test_parse_header FAILED", "", "=========== FAILURES ===========",
                      "____ test_parse_header ____", "    def test_parse_header():", ">       assert parse(b'x') == 1",
                      "E       AssertionError: assert 0 == 1", "tests/test_mod.py:88: AssertionError"]
                   + [f"tests/test_other.py::test_more_{i} PASSED" for i in range(100)]
                   + ["======= 1 failed, 399 passed in 12.3s ======="])


def bash_body(cmd, output, name="Bash"):
    return {"model": "m", "messages": [
        {"role": "user", "content": "run the tests"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "b1", "name": name, "input": {"command": cmd}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b1",
                                      "content": [{"type": "text", "text": output}]}]}]}


class ShellOutlineTest(unittest.TestCase):
    def test_keeps_start_end_failures_and_collapses_the_rest(self):
        out = outline_output(PYTEST)
        self.assertIsNotNone(out)
        self.assertIn("test session starts", out)                       # head
        self.assertIn("1 failed, 399 passed", out)                      # tail
        self.assertIn("AssertionError: assert 0 == 1", out)             # the failure block
        self.assertIn("test_parse_header FAILED", out)
        self.assertIn("lines omitted", out)
        self.assertNotIn("test_case_150 PASSED", out)                   # the passing middle is gone
        self.assertLess(len(out), len(PYTEST) / 3)

    def test_rare_intent_words_keep_their_lines(self):
        out = outline_output(PYTEST, words=["test_case_150"])
        self.assertIn("test_case_150 PASSED", out)

    def test_short_output_is_left_alone(self):
        self.assertIsNone(outline_output("ok\n" * 20))


class ProxyShellTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def send(self, rw, body):
        raw, stats = rw.rewrite(json.dumps(body).encode())
        return json.loads(raw)["messages"][2]["content"][0]["content"][0]["text"], stats

    def test_off_by_default_on_with_shell_and_replayed_byte_for_byte(self):
        body = bash_body("pytest -q", PYTEST)
        off, _ = self.send(Rewriter(os.path.join(self.tmp.name, "a")), body)
        self.assertEqual(off, PYTEST)
        rw = Rewriter(os.path.join(self.tmp.name, "b"), shell=True)
        on, stats = self.send(rw, body)
        self.assertTrue(on.startswith("[cortexnet] long output"))
        self.assertEqual(stats.get("shell_rewrites"), 1)
        again, _ = self.send(rw, body)                                   # same call id: the memo, byte for byte
        self.assertEqual(again, on)
        reloaded, _ = self.send(Rewriter(os.path.join(self.tmp.name, "b"), shell=True), body)   # after a restart
        self.assertEqual(reloaded, on)

    def test_content_commands_are_never_outlined(self):
        from cortexnet.shellout import is_content_command
        for cmd in ("grep -rn foo src", "git diff HEAD~1", "rg pattern | head -50", "ls -la", "python -c 'print(x)'",
                    "cd repo && git log -p -3"):
            self.assertTrue(is_content_command(cmd), cmd)
        for cmd in ("pytest -q", "npm test", "cargo build --release", "pip install -e ."):
            self.assertFalse(is_content_command(cmd), cmd)
        rw = Rewriter(os.path.join(self.tmp.name, "d"), shell=True)
        text, _ = self.send(rw, bash_body("grep -rn PASSED tests", PYTEST))
        self.assertEqual(text, PYTEST)                                   # search hits are the content: untouched

    def test_editor_commands_and_file_reads_are_not_shell_calls(self):
        self.assertEqual(shell_command(("str_replace_based_edit_tool", {"command": "create"})), "")
        self.assertEqual(shell_command(("Bash", {"command": "pytest -q"})), "pytest -q")
        rw = Rewriter(os.path.join(self.tmp.name, "c"), shell=True)
        text, _ = self.send(rw, bash_body("str_replace", PYTEST, name="str_replace_editor"))
        self.assertEqual(text, PYTEST)


if __name__ == "__main__":
    unittest.main()
