import json
import unittest

from cortexnet.fade import MARK, watermark_fade


def history(paths):
    """system, user, then one read per path (assistant call + tool result of 1,000 chars each)."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    for i, p in enumerate(paths):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": p})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read_file", "content": p + ":" + "x" * 1000})
    return msgs


class TestFade(unittest.TestCase):
    def test_below_the_mark_nothing_changes(self):
        msgs = history(["a.py", "b.py"])
        before = json.dumps(msgs)
        self.assertEqual(watermark_fade(msgs, 800, 1000, lambda p: "outline " + p), 0)
        self.assertEqual(json.dumps(msgs), before)              # append-only: the cached prefix stays valid

    def test_one_batch_fades_every_stale_read_and_keeps_recent_ones(self):
        msgs = history(["a.py", "b.py", "c.py", "a.py"])
        n = watermark_fade(msgs, 900, 1000, lambda p: "outline " + p, keep=2)
        self.assertEqual(n, 2)        # the first a.py (a later copy exists) and the stale b.py; c.py was touched
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertTrue(tools[0].startswith(MARK + "a.py; superseded") and tools[1].startswith(MARK + "b.py"))
        self.assertTrue(tools[2].startswith("c.py:") and tools[3].startswith("a.py:"))
        # the caller re-estimates after a batch (now ~430 < 850): nothing more changes, the prefix stays stable
        self.assertEqual(watermark_fade(msgs, 430, 1000, lambda p: "outline " + p, keep=2), 0)

    def test_reused_call_ids_still_pair_each_result_with_its_own_call(self):
        msgs = history(["a.py", "b.py", "c.py"])
        for m in msgs:                                          # every turn says id "c1", as some servers do
            for c in m.get("tool_calls") or []:
                c["id"] = "c1"
            if m["role"] == "tool":
                m["tool_call_id"] = "c1"
        self.assertEqual(watermark_fade(msgs, 900, 1000, lambda p: "outline " + p, keep=1), 2)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertTrue(tools[0].startswith(MARK + "a.py") and tools[1].startswith(MARK + "b.py"))
        self.assertTrue(tools[2].startswith("c.py:"))

    def test_superseded_reads_go_first_and_fading_stops_at_the_low_mark(self):
        msgs = history(["a.py", "b.py", "a.py", "c.py", "d.py"])
        # 1 fade brings 900 down by ~235: the low mark (0.7 x 1000) is reached after the superseded a.py alone
        n = watermark_fade(msgs, 900, 1000, lambda p: "outline " + p, low=0.7, keep=1)
        self.assertEqual(n, 1)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertTrue(tools[0].startswith(MARK + "a.py; superseded"))
        self.assertTrue(all(t.endswith("x") for t in tools[1:]))

    def test_older_copies_of_a_recent_file_are_still_superseded(self):
        msgs = history(["a.py", "a.py", "b.py", "a.py"])        # a.py is recent, its first two copies are not needed
        n = watermark_fade(msgs, 950, 1000, lambda p: "outline " + p, low=0.3, keep=2, keep_frac=1.0)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertEqual(n, 2)
        self.assertTrue(tools[0].startswith(MARK + "a.py; superseded") and tools[1].startswith(MARK + "a.py; supers"))
        self.assertTrue(tools[2].startswith("b.py:") and tools[3].startswith("a.py:"))

    def test_protection_is_capped_by_tokens_not_only_by_count(self):
        msgs = history(["a.py", "b.py", "c.py"])                  # each result ~250 tokens
        # keep=4 would protect all three; a 30% cap of 1000 tokens protects only the newest (c.py) plus nothing else
        n = watermark_fade(msgs, 900, 1000, lambda p: "outline " + p, keep=4, keep_frac=0.3, low=0.1)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertEqual(n, 2)
        self.assertTrue(tools[0].startswith(MARK) and tools[1].startswith(MARK) and tools[2].startswith("c.py:"))
        msgs = history(["a.py"])                                  # the newest result is kept even above the cap
        self.assertEqual(watermark_fade(msgs, 900, 100, lambda p: "o", keep=4, keep_frac=0.01), 0)

    def test_without_a_short_outline_a_stale_read_becomes_a_pointer(self):
        msgs = history(["a.py", "b.py", "c.py"])
        n = watermark_fade(msgs, 900, 1000, lambda p: None if p == "a.py" else "o" * 5000, keep=1)
        self.assertEqual(n, 2)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertTrue(all(t.startswith(MARK) and t.endswith("read it again if you need it]") for t in tools[:2]))

    def test_outlines_from_an_earlier_batch_shrink_to_pointers_when_still_above_the_mark(self):
        msgs = history(["a.py", "b.py", "c.py"])
        watermark_fade(msgs, 900, 1000, lambda p: "o" * 300, keep=1, low=0.8)   # a.py -> outline only
        n = watermark_fade(msgs, 900, 1000, lambda p: "o" * 300, keep=1, low=0.1)
        tools = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertGreaterEqual(n, 2)
        self.assertTrue(tools[0].endswith("read it again if you need it]"))


if __name__ == "__main__":
    unittest.main()
