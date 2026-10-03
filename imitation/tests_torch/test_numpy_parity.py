"""The NumPy serving forward (main_v7._forward) must compute the trained torch function.

This is the compute-parity gate in miniature, on SYNTHETIC rows and a tiny random
model: the torch side is built by train_ss.collate_fast exactly as training builds it,
the NumPy side is fed the same rows through the gate's own row-to-token converter.
"""
import unittest

import numpy as np

import synthetic as syn
import main_v7
from gate_compute_parity import _row_to_tok

TRAINED = {"mask_pad_state": True, "mask_pad_options": True, "relations": True}
CHAMPION = {"mask_pad_state": False, "mask_pad_options": False, "relations": False}


class NumpyParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = syn.make_model()
        cls.rows = syn.make_rows(16)
        cls.weights = syn.numpy_weights(cls.model)

    def numpy_logits(self, mode):
        out = []
        for row in self.rows:
            tok = _row_to_tok(row, syn.NS, syn.NO)
            n = sum(1 for t in row["option_type"] if t >= 0)
            out.append(np.asarray(main_v7._forward(
                tok, self.weights, layers=syn.LAYERS, heads=syn.HEADS, mode=mode),
                dtype=np.float64)[:n])
        return out

    def max_logit_gap(self, serve_mode, train_mode):
        syn.set_training_semantics(**{"mask_state": train_mode["mask_pad_state"],
                                      "mask_options": train_mode["mask_pad_options"],
                                      "relations": train_mode["relations"]})
        ref = syn.torch_logits(self.model, self.rows)
        got = self.numpy_logits(serve_mode)
        gaps = [float(np.abs(a - b).max()) for a, b in zip(ref, got)]
        flips = sum(int(a.argmax() != b.argmax()) for a, b in zip(ref, got))
        return max(gaps), flips

    def test_serving_the_trained_mode_matches_torch(self):
        gap, flips = self.max_logit_gap(TRAINED, TRAINED)
        self.assertLess(gap, 1e-4)
        self.assertEqual(flips, 0)

    def test_champion_mode_matches_a_model_trained_without_masks(self):
        gap, flips = self.max_logit_gap(CHAMPION, CHAMPION)
        self.assertLess(gap, 1e-4)
        self.assertEqual(flips, 0)

    def test_an_option_that_does_not_exist_has_no_logit(self):
        # "Illegal actions are never constructed, so they have no logit": with option
        # masking on, every padded slot must come back at the floor, not a score.
        for row in self.rows:
            tok = _row_to_tok(row, syn.NS, syn.NO)
            n = sum(1 for t in row["option_type"] if t >= 0)
            out = np.asarray(main_v7._forward(tok, self.weights, layers=syn.LAYERS,
                                              heads=syn.HEADS, mode=TRAINED))
            self.assertTrue((out[n:] < -1e29).all())
            self.assertTrue((out[:n] > -1e29).all())

    def test_serving_the_wrong_mode_is_detected(self):
        # The I-1 defect: weights trained with masks and relations, served without.
        # A gate that cannot see this would read green while the agent played a
        # different function from the one it was fit to.
        gap, _ = self.max_logit_gap(CHAMPION, TRAINED)
        self.assertGreater(gap, 1e-2)


if __name__ == "__main__":
    unittest.main()
