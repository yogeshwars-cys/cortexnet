import io
import json
import os
import tempfile
import textwrap
import unittest

from cortexnet import Reader, outline, symbols
from cortexnet.mcp_server import main as serve

PY = textwrap.dedent('''\
    import os
    from re import compile

    LIMIT = 10


    class Cache:
        """Holds things."""

        def get(self, key):
            return self.d.get(key)

        @property
        def size(self):
            return len(self.d)


    def helper(x, y=2):
        return Cache().get(x) + y
    ''')


def big_py(n=60):
    out = ["import json", ""]
    for i in range(n):
        out += [f"def func_{i}(a, b):", f'    """Doc {i}."""', f"    v = a * {i} + b", "    w = json.dumps(v)",
                f"    return w + 'item_{i}'", ""]
    return "\n".join(out)


def _w(path, text):
    with open(path, "w", encoding="utf8") as f:
        f.write(text)


class OutlineTest(unittest.TestCase):
    def test_python_spans(self):
        s = {x.name: x for x in symbols(PY, "m.py")}
        self.assertEqual((s["Cache"].start, s["Cache"].end), (7, 15))
        self.assertEqual((s["get"].start, s["get"].end, s["get"].depth), (10, 11, 1))
        self.assertEqual(s["size"].start, 13)          # decorator line starts the span
        self.assertEqual((s["helper"].start, s["helper"].end), (18, 19))
        self.assertEqual(s["Cache"].doc, "Holds things.")
        self.assertIn("imports: os, re", outline(PY, "m.py"))

    def test_generic_and_markdown(self):
        js = "const a = 1;\nfunction foo(x) {\n  return x;\n}\n\nexport async function bar() {\n  return 2;\n}\n"
        names = [s.head for s in symbols(js, "a.js")]
        self.assertTrue(any("function foo" in h for h in names) and any("function bar" in h for h in names))
        md = "# Title\ntext\n## Part A\nmore\n## Part B\nend\n"
        s = symbols(md, "r.md")
        self.assertEqual([(x.start, x.end) for x in s if x.kind == "heading"], [(1, 6), (3, 4), (5, 6)])

    def test_split_big_symbols_into_blocks(self):
        body = "\n".join(f"    x{i} = compute(x{i - 1}, 'padding text to make the function big {i}')" for i in range(1, 30))
        src = f"def big(a):\n{body}\n    if a:\n        return x1\n    for k in range(3):\n        a += k\n    return a\n"
        plain = symbols(src, "b.py")
        split = symbols(src, "b.py", split_tokens=100)
        blocks = [s for s in split if s.kind == "block"]
        self.assertEqual(len(plain), 1)
        self.assertGreaterEqual(len(blocks), 2)
        self.assertTrue(next(s for s in split if s.name == "big").children)
        self.assertTrue(all(s.depth == 1 and 1 <= s.start <= s.end <= 36 for s in blocks))

    def test_unparseable_python_falls_back(self):
        s = symbols("def broken(:\n    pass\n" * 30, "b.py")
        self.assertTrue(len(s) > 0)


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMP"))
        self.dir = self.tmp.name
        self.small = os.path.join(self.dir, "small.py")
        self.big = os.path.join(self.dir, "big.py")
        _w(self.small, PY)
        _w(self.big, big_py())
        self.r = Reader(os.path.join(self.dir, "state"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_small_file_whole(self):
        out = self.r.read(self.small)
        self.assertIn("def helper", out)
        self.assertNotIn("Outline", out)

    def test_big_file_outline_and_query_include(self):
        out = self.r.read(self.big, query="func_42 returns the wrong item")
        self.assertIn("Outline", out)
        self.assertIn("item_42", out)                  # body of the named symbol was included
        self.assertNotIn("item_7'", out)
        self.assertLess(len(out), len(big_py()) / 2)

    def test_range_miss_trains_model(self):
        self.r.read(self.big)
        s = {x.name: x for x in symbols(big_py(), self.big)}["func_10"]
        out = self.r.range(self.big, s.start, s.end)
        self.assertIn("item_10", out)
        for _ in range(12):                            # push past the horizon so the read event is labelled
            self.r.range(self.small, 1, 2)
        self.assertGreaterEqual(self.r.model.updates, 1)
        ev = [json.loads(l) for l in open(self.r.log_path, encoding="utf8").read().splitlines()]
        self.assertTrue(any(e["op"] == "range" and e["miss"] for e in ev))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "state", "include.json")))

    def test_changed_lines_are_detected(self):
        self.r.read(self.big)
        src = big_py().replace("v = a * 30 + b", "v = a * 30 - b")
        _w(self.big, src)
        out = self.r.read(self.big)
        self.assertIn("a * 30 - b", out)               # the edited symbol is shown again


class ServerTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMP")) as d:
            f = os.path.join(d, "m.py")
            _w(f, PY)
            os.environ["CORTEXNET_STATE"] = os.path.join(d, "state")
            reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "cortex_outline", "arguments": {"path": f}}},
                    {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "cortex_range", "arguments": {"path": f, "start": 18, "end": 19}}},
                    {"jsonrpc": "2.0", "id": 5, "method": "nope"}]
            out = io.StringIO()
            serve(io.StringIO("\n".join(json.dumps(r) for r in reqs) + "\n"), out)
            resp = [json.loads(l) for l in out.getvalue().splitlines()]
            self.assertEqual([r["id"] for r in resp], [1, 2, 3, 4, 5])
            self.assertEqual(len(resp[1]["result"]["tools"]), 5)
            self.assertIn("L18-19 def helper", resp[2]["result"]["content"][0]["text"])
            self.assertIn("return Cache().get(x) + y", resp[3]["result"]["content"][0]["text"])
            self.assertEqual(resp[4]["error"]["code"], -32601)
            del os.environ["CORTEXNET_STATE"]


class GraphMenuTest(unittest.TestCase):
    def test_connected_symbol_menu_and_approve(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMP")) as d:
            a, b = os.path.join(d, "orders.py"), os.path.join(d, "rates.py")
            _w(a, big_py(40) + "\n\ndef compute_total(order):\n    return order.amount * (1 + tax_rate(order.region))\n")
            _w(b, big_py(40).replace("func_", "rate_") + "\n\ndef tax_rate(region):\n    return {'eu': 0.2}.get(region, 0.1)\n")
            r = Reader(os.path.join(d, "state"), menu=True, tau_menu=0.05)
            r.read(a)
            s = {x.name: x for x in symbols(open(a, encoding="utf8").read(), a)}["compute_total"]
            r.range(a, s.start, s.end)
            self.assertIn("tax", r.trail)                       # the trail remembers what was just viewed
            out = r.read(a)
            self.assertIn("rates.py:tax_rate", out)             # connected symbol offered from the neighbour file
            mid = next(k for k, m in r.menu.items() if m["name"] == "tax_rate")
            body = r.approve([mid])
            self.assertIn("{'eu': 0.2}", body)

    def test_feature_mask_disables_graph(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMP")) as d:
            a = os.path.join(d, "m.py")
            _w(a, big_py(40))
            r = Reader(os.path.join(d, "state"), features=["query_name", "query_overlap"])
            r.read(a, "func_3")
            self.assertEqual(r._menu(os.path.abspath(a), ""), [])


class OptionsTest(unittest.TestCase):
    def test_env_switches_map_to_reader_options(self):
        from cortexnet.mcp_server import options
        with tempfile.TemporaryDirectory() as d:
            kw = options({"CORTEXNET_MENU": "1", "CORTEXNET_GRAPH": "1", "CORTEXNET_STATE": d})
            self.assertTrue(kw["menu"])
            self.assertIsNotNone(kw["graph"])
            r = Reader(d, **kw)                              # the full system's switches build a working reader
            self.assertTrue(r.menu_on)
        self.assertNotIn("menu", options({}))                # off by default
        kw = options({"CORTEXNET_COPROC": os.path.join(tempfile.gettempdir(), "no-such-head.npz")})
        self.assertNotIn("coproc", kw)                       # a missing head turns the coprocessor off, no crash


if __name__ == "__main__":
    unittest.main()
