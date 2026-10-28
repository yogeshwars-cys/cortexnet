import os
import tempfile
import textwrap
import unittest

from cortexnet import Reader, symbols
from cortexnet.outline import skeleton

SRC = textwrap.dedent('''\
    class Wrapper:
        def __init__(self, width=70,
                     indent="",
                     placeholder=" [...]"):
            self.width = width
            self.indent = indent
            self.placeholder = placeholder

    def chain(seq, autojunk=True):
        """Index the sequence."""
        total = 0
        for i, x in enumerate(seq):
            total += i
            total *= 2
            total -= 1
            total += x
        if autojunk and len(seq) >= 200:
            seq = seq[:200]
        out = []
        for x in seq:
            out.append(x + 1)
            out.append(x + 2)
            out.append(x + 3)
        return out
    ''') + "\n".join(f"def filler_{i}(a, b):\n    c = a + b\n    d = c * 2\n    return d - {i}\n" for i in range(40))


class SkeletonTest(unittest.TestCase):
    def setUp(self):
        self.lines = SRC.split("\n")
        self.syms = {s.name: s for s in symbols(SRC, "m.py")}

    def test_multiline_signature_is_kept_to_its_closing_line(self):
        sk = skeleton(self.syms["__init__"], self.lines, max_lines=12)
        self.assertEqual(sk[:3], [2, 3, 4])                 # the default placeholder sits on the 3rd signature line

    def test_rare_words_bring_their_lines_and_flow_fills_the_rest(self):
        s = self.syms["chain"]
        sk = skeleton(s, self.lines, {"autojunk"}, max_lines=12, window=1)
        self.assertIn(17, sk)                               # `if autojunk and len(seq) >= 200:`
        self.assertIn(18, sk)                               # its window
        self.assertIn(24, sk)                               # `return out`, from the control-flow spine
        self.assertLessEqual(len(sk), 12)
        self.assertEqual(sk, sorted(sk))


class XrayReaderTest(unittest.TestCase):
    def test_reader_xrays_the_symbol_with_the_rare_query_word_and_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w") as f:
                f.write(SRC)
            q = "above what length does autojunk apply?"
            plain = Reader(os.path.join(d, "a"), full_below=50).read(p, q)
            self.assertNotIn("x-ray of", plain)
            x = Reader(os.path.join(d, "b"), full_below=50, xray_top=1).read(p, q)
            self.assertIn("x-ray of chain", x)
            self.assertIn(">= 200", x)
            tight = Reader(os.path.join(d, "c"), full_below=50, xray_top=3, xray_budget=5).read(p, q)
            self.assertNotIn("x-ray of", tight)             # nothing fits a 5-token budget


if __name__ == "__main__":
    unittest.main()
