import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import swe_agent as S  # noqa: E402

SRC = textwrap.dedent('''\
    def alpha(x):
        return x + 1


    def beta(x):
        return x * 2


    def gamma(x):
        y = x - 3
        return y
    ''') * 20

PATCH = textwrap.dedent('''\
    diff --git a/pkg/mod.py b/pkg/mod.py
    --- a/pkg/mod.py
    +++ b/pkg/mod.py
    @@ -9,3 +9,3 @@ def beta(x):
     def gamma(x):
    -    y = x - 3
    +    y = x - 4
         return y
    ''')


def call(name, **args):
    params = "".join(f'<atem:parameter name="{k}">{v}</atem:parameter>' for k, v in args.items())
    return (f' to={name}<|message|><atem:function_calls>\n<atem:invoke name="{name}">{params}</atem:invoke>\n'
            f'</atem:function_calls><|eom|>')


class SweAgentTest(unittest.TestCase):
    def test_gold_lines(self):
        self.assertEqual(S.gold_lines(PATCH), {"pkg/mod.py": {10}})

    def test_episode_and_cases(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "repo", "pkg"))
            with open(os.path.join(d, "repo", "pkg", "mod.py"), "w") as f:
                f.write(SRC)
            script = [" to=self<|message|>The bug is in gamma, I should read mod.py.<|eom|>",
                      call("read_file", path="pkg/mod.py", query="gamma subtraction"),
                      call("read_lines", path="pkg/mod.py", start=5, end=6),
                      call("read_file", path="../outside.py"),
                      call("submit", file="pkg/mod.py", symbol="gamma", patch="-3 -> -4")]
            replies = iter(script)
            encode = lambda t: [ord(c) for c in t]  # noqa: E731
            decode = lambda ids: "".join(chr(i) for i in ids)  # noqa: E731
            ws = S.Workspace(os.path.join(d, "repo"), os.path.join(d, "state"))
            ep = S.run_episode(lambda ids: encode(next(replies)), encode, decode, encode("<|start|>assistant"), ws)
            self.assertEqual(ep["calls"], 4)
            self.assertEqual(ep["submitted"][1], "gamma")
            self.assertEqual(len(ep["events"]), 2)
            ev = ep["events"][0]
            self.assertTrue(decode(ep["ids"][ev["pos"] + 1:]).startswith("<atem:function_calls>"))
            self.assertIn("gamma", ev["snap"])
            self.assertIn("ValueError", decode(ep["ids"]))       # the outside path was refused
            inst = dict(instance_id="x__y-1", repo="x/y", patch=PATCH, problem_statement="gamma is off by one")
            cases, texts = S.cases_from_episode(inst, ep, ws, order=0, qv0=10)
            self.assertEqual(len(cases), 1)
            c = cases[0]
            syms = S.leaves(SRC, "pkg/mod.py")
            names = {syms[j].name for j in c["y"]}
            self.assertEqual(names, {"gamma", "beta"})           # gold patch + the later read_lines
            self.assertEqual(c["qv"], 10)
            self.assertIn(c["text"], texts)


if __name__ == "__main__":
    unittest.main()
