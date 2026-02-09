import unittest

import numpy as np

from compound_processing import metrics


class TestSearchMetrics(unittest.TestCase):
    def test_tanimoto_packed_matches_dense(self) -> None:
        q_bits = np.array([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=np.uint8)
        x_bits = np.array([[1, 0, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)

        q_packed = np.packbits(q_bits, axis=1)
        x_packed = np.packbits(x_bits, axis=1)

        packed_meta = {
            "packed_bits": True,
            "packed_dim": q_packed.shape[1],
            "dim": q_bits.shape[1],
            "dtype": "uint8",
        }
        dense_meta = {
            "packed_bits": False,
            "dim": q_bits.shape[1],
            "dtype": "uint8",
        }

        ti_packed = metrics.score_block(
            "tanimoto",
            q_packed,
            x_packed,
            q_meta=packed_meta,
            x_meta=packed_meta,
            device="cpu",
        )
        ti_dense = metrics.score_block(
            "tanimoto",
            q_bits,
            x_bits,
            q_meta=dense_meta,
            x_meta=dense_meta,
            device="cpu",
        )

        self.assertTrue(np.allclose(ti_packed, ti_dense, atol=1e-6))

    def test_cosine_cpu(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        x = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        meta = {"packed_bits": False, "dim": 2, "dtype": "float32"}

        sim = metrics.score_block(
            "cosine",
            q,
            x,
            q_meta=meta,
            x_meta=meta,
            device="cpu",
            assume_normalized=False,
        )

        expected = np.array([[1.0, 1.0 / np.sqrt(2.0)], [0.0, 1.0 / np.sqrt(2.0)]], dtype=np.float32)
        self.assertTrue(np.allclose(sim, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
