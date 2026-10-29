"""Neural-dynamics options.

1. Homeostatic threshold: misses lower the threshold, a quiet session raises it, off by default.
2. Refractory bodies: a re-read does not resend an unchanged body; an edit or forget() (the fade) sends it again.
3. Winner-take-all menu: a peaked field keeps its winner, a flat one offers nothing.
4. NLMS include model: per-feature normalised steps, state saved and reloaded.
5. Charge-ordered fade: the file the agent keeps touching outlives an older-touched one; on_fade is called.
"""
import json
import os
import tempfile
import textwrap
import unittest

import numpy as np

from cortexnet import Reader
from cortexnet.fade import MARK, watermark_fade
from cortexnet.learn import FEATURES, IncludeModel

BIG = textwrap.dedent('''\
    def alpha(x):
        """First."""
        return x + 1
''') + "\n\n".join(f"def helper_{i}(value):\n    total = value * {i}\n" + "    total += 1\n" * 40 + "    return total"
                   for i in range(12))


class ReaderDynamicsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "big.py")
        with open(self.path, "w") as f:
            f.write(BIG)

    def tearDown(self):
        self.tmp.cleanup()

    def test_homeostasis_moves_the_threshold(self):
        r = Reader(os.path.join(self.tmp.name, "s"))
        self.assertEqual(r.tau_now(), r.tau)                       # off by default
        h = Reader(os.path.join(self.tmp.name, "h"), homeostasis=2.0, miss_target=0.3)
        h.h_miss = 0.9
        self.assertLess(h.tau_now(), h.tau)
        h.h_miss = 0.0
        self.assertGreater(h.tau_now(), h.tau)

    def test_refractory_body_is_not_resent_until_edited_or_forgotten(self):
        r = Reader(os.path.join(self.tmp.name, "s"), refractory=True, full_below=10)
        first = r.read(self.path, "alpha")
        self.assertIn("return x + 1", first)
        again = r.read(self.path, "alpha")
        self.assertNotIn("return x + 1", again)
        self.assertIn("unchanged since read #1", again)
        r.forget(self.path)                                         # the fade shrank that read
        self.assertIn("return x + 1", r.read(self.path, "alpha"))
        with open(self.path, "w") as f:
            f.write(BIG.replace("return x + 1", "return x + 2"))
        self.assertIn("return x + 2", r.read(self.path, "alpha"))  # changed body: sent again

    def test_wta_menu_competition(self):
        r = Reader(os.path.join(self.tmp.name, "s"), menu_rule="wta", menu_floor=0.25)
        peaked = [("a", 10.0), ("b", 0.5), ("c", 0.4)]
        self.assertEqual([c[0] for c in r._compete(peaked, lambda c: c[1])], ["a"])
        flat = [(str(i), 1.0) for i in range(8)]
        self.assertEqual(r._compete(flat, lambda c: c[1]), [])
        top = Reader(os.path.join(self.tmp.name, "t"))
        self.assertEqual(len(top._compete(flat, lambda c: c[1])), 3)   # default: the first menu_size


class NlmsTest(unittest.TestCase):
    def test_normalised_steps_and_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            m = IncludeModel(os.path.join(d, "inc.json"), rule="nlms")
            X = np.zeros((4, len(FEATURES)))
            X[:, FEATURES.index("log_tokens")] = 8.0                   # a large-scale feature
            X[:, FEATURES.index("query_name")] = 1.0
            w0 = m.w.copy()
            m.update(X, [1, 1, 0, 0])
            self.assertGreater(m.power[FEATURES.index("log_tokens")], 1.0)
            self.assertFalse(np.allclose(m.w, w0))
            m.save()
            self.assertIn("power", json.load(open(os.path.join(d, "inc.json"))))
            np.testing.assert_allclose(IncludeModel(os.path.join(d, "inc.json"), rule="nlms").power, m.power)


class ChargeFadeTest(unittest.TestCase):
    def _session(self):
        msgs = []

        def call(name, path, content):
            i = f"c{len(msgs)}"
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": i, "function": {"name": name, "arguments": json.dumps({"path": path})}}]})
            msgs.append({"role": "tool", "tool_call_id": i, "content": content})
        call("read_file", "old.py", "x" * 4000)          # read first, then touched again and again
        call("read_file", "once.py", "y" * 4000)         # read later, never touched again
        for _ in range(4):
            call("read_lines", "old.py", "z")
        for k in range(4):
            call("read_lines", f"new{k}.py", "w")
        return msgs

    def test_charge_order_keeps_the_file_in_use(self):
        age, charge = self._session(), self._session()
        watermark_fade(age, 2400, 1000, lambda p: None, keep=1, low=1.5)        # one fade is enough
        faded = []
        watermark_fade(charge, 2400, 1000, lambda p: None, keep=1, low=1.5, order="charge", on_fade=faded.append)
        content = lambda msgs, path: next(m["content"] for m in msgs if m.get("role") == "tool" and
                                          m["tool_call_id"] == next(c["id"] for a in msgs if a.get("tool_calls")
                                                                    for c in a["tool_calls"]
                                                                    if json.loads(c["function"]["arguments"])["path"] == path))
        self.assertTrue(content(age, "old.py").startswith(MARK))       # oldest first
        self.assertFalse(content(age, "once.py").startswith(MARK))
        self.assertFalse(content(charge, "old.py").startswith(MARK))   # still in use: kept
        self.assertTrue(content(charge, "once.py").startswith(MARK))
        self.assertIn("once.py", faded)


if __name__ == "__main__":
    unittest.main()
