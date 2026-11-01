"""Offline tests for cortexnet.local: no network, no GPU.

1. asset(cpu) archive names per OS/arch (platform.system/machine patched).
2. source_files() in a temporary git repo: tracked + untracked .py listed,
   .gitignore- and .git/info/exclude-matched files not listed.
3. Session.path(): a bare file name resolves to the matching source file,
   a path outside the repository raises ValueError (no server contact).
"""
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from cortexnet.local import BUILD, Session, asset, source_files


class AssetTest(unittest.TestCase):
    def _asset(self, system, machine, cpu):
        with mock.patch("platform.system", return_value=system), \
                mock.patch("platform.machine", return_value=machine):
            return asset(cpu)

    def test_windows_x64_vulkan(self):
        self.assertEqual(self._asset("Windows", "AMD64", False),
                         f"llama-{BUILD}-bin-win-vulkan-x64.zip")

    def test_windows_x64_cpu(self):
        self.assertEqual(self._asset("Windows", "AMD64", True),
                         f"llama-{BUILD}-bin-win-cpu-x64.zip")

    def test_linux_x64_vulkan(self):
        self.assertEqual(self._asset("Linux", "x86_64", False),
                         f"llama-{BUILD}-bin-ubuntu-vulkan-x64.tar.gz")

    def test_linux_arm64_cpu(self):
        self.assertEqual(self._asset("Linux", "arm64", True),
                         f"llama-{BUILD}-bin-ubuntu-arm64.tar.gz")

    def test_macos_arm64(self):
        self.assertEqual(self._asset("Darwin", "arm64", False),
                         f"llama-{BUILD}-bin-macos-arm64.tar.gz")


class SourceFilesGitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        subprocess.run(["git", "init", "-q", self.root], check=True, timeout=30)
        self._write("tracked.py", "x = 1\n")
        self._write("untracked.py", "y = 2\n")
        self._write("ignored.py", "z = 3\n")
        self._write("excluded.py", "w = 4\n")
        self._write(".gitignore", "ignored.py\n")
        with open(os.path.join(self.root, ".git", "info", "exclude"), "a") as fh:
            fh.write("excluded.py\n")
        subprocess.run(["git", "-C", self.root, "add", "tracked.py"], check=True, timeout=30)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, content):
        with open(os.path.join(self.root, name), "w") as fh:
            fh.write(content)

    def test_tracked_and_untracked_listed_ignored_and_excluded_not(self):
        found = source_files(self.root)
        self.assertIn(os.path.join(self.root, "tracked.py"), found)
        self.assertIn(os.path.join(self.root, "untracked.py"), found)
        self.assertNotIn(os.path.join(self.root, "ignored.py"), found)
        self.assertNotIn(os.path.join(self.root, "excluded.py"), found)


class SessionPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "sub"))
        with open(os.path.join(self.root, "sub", "target.py"), "w") as fh:
            fh.write("x = 1\n")
        self.state = tempfile.TemporaryDirectory()
        self.sess = Session(self.root, "http://unused", 8192, self.state.name)

    def tearDown(self):
        self.tmp.cleanup()
        self.state.cleanup()

    def test_bare_name_resolves_to_matching_source_file(self):
        self.assertEqual(self.sess.path("target.py"),
                         os.path.join(self.root, "sub", "target.py"))

    def test_path_outside_repository_rejected(self):
        with self.assertRaises(ValueError):
            self.sess.path(os.path.join("..", "outside.py"))

    def test_overflow_teaches_the_ratio_and_the_question_is_retried(self):
        for i in range(30):                                       # a long session of earlier questions
            self.sess.msgs += [{"role": "user", "content": f"q{i} " + "w" * 900},
                               {"role": "assistant", "content": "a" * 900}]
        calls = []

        def post(url, body):
            calls.append(len(body["messages"]))
            if len(calls) == 1:                                   # the estimate was low: the server says 9,500
                raise RuntimeError('HTTP 400: {"error":{"n_prompt_tokens":9500,"n_ctx":8192}}')
            return {"choices": [{"message": {"content": "ANSWER: 42"}}], "usage": {"prompt_tokens": 3000}}
        with mock.patch("cortexnet.local.post", post):
            answer, _, _ = self.sess.ask("what is it?")
        self.assertEqual(answer, "ANSWER: 42")
        self.assertLess(calls[1], calls[0])                      # the retry went out after old exchanges were dropped


if __name__ == "__main__":
    unittest.main()
