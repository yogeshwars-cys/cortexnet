import os
import tempfile
import unittest

import numpy as np

from cortexnet.codec import BitplaneCodec


def data(n=4000, d=96, seed=0):
    """Anisotropic, non-centred unit vectors (like sentence embeddings)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * np.exp(-np.arange(d) / 20) + 0.3
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


class CodecTest(unittest.TestCase):
    def setUp(self):
        self.X = data()
        self.c = BitplaneCodec.fit(self.X[:3000], code_bytes=32)

    def test_size_and_bits(self):
        codes = self.c.encode(self.X)
        self.assertEqual(codes.shape, (len(self.X), 32))
        self.assertEqual(self.c.code_bits, 256)
        self.assertTrue(np.all(np.diff(self.c.bits.astype(int)) <= 0))

    def test_round_trip_and_unpack(self):
        Xt = self.X[3000:]
        idx = self.c.quantize(self.c.project(Xt))
        codes = self.c.encode(Xt)
        np.testing.assert_array_equal(self.c.unpack(codes), idx)
        cos = (self.c.decode(codes) * Xt).sum(1)
        self.assertGreater(cos.mean(), 0.97)

    def test_lut_score_is_cosine_of_decoded(self):
        Xt, q = self.X[3000:3500], self.X[3999]
        codes = self.c.encode(Xt)
        qp = self.c.V @ self.c.project(q)          # q restricted to the codec's subspace
        want = self.c.decode(codes) @ (qp / np.linalg.norm(qp))
        np.testing.assert_allclose(self.c.score(codes, q), want, atol=1e-4)

    def test_prefilter_recall(self):
        Xt = self.X[:3000]
        codes = self.c.encode(Xt)
        hit = []
        for q in self.X[3000:3050]:
            true = set(np.argsort(-(Xt @ q))[:10])
            hit.append(len(true & set(self.c.prefilter(codes, q, 300))) / 10)
        self.assertGreater(np.mean(hit), 0.9)
        ids, s = self.c.search(codes, self.X[3001], k=5)
        self.assertEqual(len(ids), 5)
        self.assertTrue(np.all(np.diff(s) <= 1e-6))

    def test_save_load_and_cells(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "codec.npz")
            self.c.save(p)
            c2 = BitplaneCodec.load(p)
        codes = self.c.encode(self.X[:10])
        np.testing.assert_array_equal(c2.encode(self.X[:10]), codes)
        cells = self.c.cells(codes, m=12)
        self.assertTrue(np.all((cells >= 0) & (cells < 2 ** 12)))
        self.assertEqual(cells[0], self.c.cells(codes[:1], m=12)[0])


if __name__ == "__main__":
    unittest.main()
