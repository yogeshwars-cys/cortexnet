import os
import tempfile
import textwrap
import unittest
import zlib

import numpy as np

from cortexnet import Reader
from cortexnet.codec import BitplaneCodec
from cortexnet.coproc import Coprocessor
from cortexnet.graph import CoNeedGraph, symbol_cells
from cortexnet.learn import FEATURES, idents

D = 64


def stub_encoder(texts):
    """Deterministic bag-of-identifiers vectors (stands in for MiniLM)."""
    out = np.zeros((len(texts), D), np.float32)
    for i, t in enumerate(texts):
        for w in idents(t):
            out[i, zlib.crc32(w.encode()) % D] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


def export(path, codec=None):
    """An identity-initialised head (the zero-shot ranker), Platt a = 20, b = -10."""
    F = len(FEATURES)
    out = dict(g=np.zeros((3, D), np.float32), dg=np.zeros(D, np.float32), A=np.zeros((4, D), np.float32),
               B=np.zeros((4, D), np.float32), Wg_w=np.zeros((3, 3 * D + 4), np.float32), Wg_b=np.zeros(3, np.float32),
               ws_w=np.zeros((1, F), np.float32), ws_b=np.zeros(1, np.float32), phi_mu=np.zeros(F, np.float32),
               phi_sd=np.ones(F, np.float32), platt=np.array([20.0, -10.0], np.float32),
               flags=np.array([True, True, True, True, True, True]), gamma=np.float32(0.6),
               features=np.array(FEATURES), encoder=np.array("stub"), val_mrr=np.float32(0))
    if codec is not None:
        out.update(codec_V=codec.V, codec_bits=codec.bits, codec_levels=codec.levels)
    np.savez(path, **out)


SRC = textwrap.dedent('''\
    def parse_config(path):
        """Read the config file."""
        text = open(path).read()
        return dict(line.split("=") for line in text.splitlines())


    def render_template(name, values):
        """Fill a template with values."""
        tpl = TEMPLATES[name]
        for key, value in values.items():
            tpl = tpl.replace("{" + key + "}", str(value))
        return tpl


    def send_email(address, body):
        """Deliver a message over smtp."""
        server = smtplib.SMTP("localhost")
        server.sendmail("noreply", address, body)
        server.quit()
    ''') * 3


class GraphTest(unittest.TestCase):
    def test_frames_neighbours_and_reload(self):
        g = CoNeedGraph(frame_calls=2, decay=0.5, horizon=1)
        g.need([1, 2])
        g.tick()
        g.need([3])          # within the horizon: pairs with 1 and 2
        g.tick()             # closes frame 0
        g.need([1, 4])
        g.tick(tag="task-b")  # new tag closes frame 1
        self.assertEqual([f[0] for f in g.frames()], [0, 1])
        self.assertEqual({tuple(e[:2]) for e in g.frames()[0][3]}, {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(g.neighbours([1])[0][0] in (2, 3, 4), True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.npz")
            g.save(p)
            h = CoNeedGraph.load(p)
        self.assertEqual(len(h.frames()), 2)
        np.testing.assert_array_equal(h.frames()[1][3], g.frames()[1][3])
        self.assertEqual(h.frames()[1][1], "")
        self.assertAlmostEqual(sum(h.live.values()), sum(g.live.values()))

    def test_symbol_cell_is_majority(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 32)).astype(np.float32)
        c = BitplaneCodec.fit(X, code_bytes=8)
        codes = c.encode(X[:3])
        cell = symbol_cells(c, codes, m=10)
        bits = (c.plane0(codes)[:, :10].sum(0) * 2 >= 3).astype(int)
        self.assertEqual(cell, int("".join(map(str, bits)), 2))


class CoprocTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "mod.py")
        with open(self.path, "w") as f:
            f.write(SRC)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_head_is_zero_shot_cosine(self):
        ex = os.path.join(self.tmp.name, "head.npz")
        export(ex)
        cp = Coprocessor(ex, encoder=stub_encoder)
        from cortexnet.coproc import leaves
        lines = SRC.split("\n")
        syms = leaves(SRC, self.path)
        v = cp.views(prompt="smtp email delivery")
        s = cp.scores(v, cp.symbol_vectors(self.path, lines, syms), np.zeros((len(syms), len(FEATURES))))
        E = cp.symbol_vectors(self.path, lines, syms)
        np.testing.assert_allclose(s, (E @ v["eqh"]).max(1), atol=1e-6)
        self.assertEqual(syms[int(np.argmax(s))].name, "send_email")

    def test_reader_semantic_includes_rule_ab_and_graph_menu(self):
        X = stub_encoder(sorted(idents(SRC) | {"smtp", "template", "config"}) * 20)
        codec = BitplaneCodec.fit(X + 1e-3 * np.random.default_rng(0).normal(size=X.shape), code_bytes=16)
        ex = os.path.join(self.tmp.name, "head.npz")
        export(ex, codec)
        cp = Coprocessor(ex, encoder=stub_encoder)
        other = os.path.join(self.tmp.name, "mail.py")
        with open(other, "w") as f:
            f.write(SRC.replace("send_email", "queue_email"))
        r = Reader(os.path.join(self.tmp.name, "st"), full_below=50, coproc=cp, policy="rule_ab",
                   graph=CoNeedGraph(frame_calls=5), ctx_base=40000)
        out = r.read(self.path, query="deliver email over smtp")
        self.assertIn("smtplib", out)                      # the semantically matching body is shown up front
        self.assertTrue(r.cell_of)
        r.read(other, query="deliver email over smtp")
        for _ in range(3):                                 # co-needs: ranges in both files
            r.range(self.path, 1, 4)
            r.range(other, 15, 19)
        self.assertTrue(r.graph.live)
        r.ctx_base, r.served = 0, 0                        # cheap round trips: outline (+ menu), not a bypass
        out3 = r.read(self.path, query="config")
        self.assertIn("[connected]", out3)
        self.assertIn("mail.py:queue_email (lines 34-38", out3)   # lines 15-19 were shown already
        r2 = Reader(os.path.join(self.tmp.name, "st2"), full_below=50, coproc=cp, policy="rule_ab", ctx_base=0,
                    h_rem=1000)                              # context re-read 1000 more times: round trips are cheap
        out2 = r2.read(self.path, query="unrelated words")
        self.assertIn("Outline", out2)
        self.assertNotIn("smtplib.SMTP", out2)
        r3 = Reader(os.path.join(self.tmp.name, "st3"), full_below=50, coproc=cp, policy="rule_ab", ctx_base=0,
                    h_rem=0)                                 # no compounding: a 400-token round trip > this file
        self.assertNotIn("Outline", r3.read(self.path, query="unrelated words"))


if __name__ == "__main__":
    unittest.main()
