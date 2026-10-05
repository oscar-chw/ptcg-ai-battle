"""model_ss: construction guards, the parameter count the README quotes, and padding."""
import unittest

import numpy as np

import synthetic as syn
import model_ss


class Construction(unittest.TestCase):
    def test_default_config_parameter_count(self):
        # imitation/README.md quotes this number; the d512 checkpoints (45.4M) are a
        # different configuration, see ppo/docs/BASELINE.md.
        model = model_ss.SixthSenseNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 23_782_511)

    def test_v2_refuses_fewer_than_three_pool_seeds(self):
        with self.assertRaises(ValueError):
            model_ss.SixthSenseNet(d=32, layers=1, heads=4, hidden=48,
                                   arch_v2=True, pool_seeds=2)

    def test_unknown_num_scale_cannot_be_stamped(self):
        with self.assertRaises(ValueError):
            model_ss._scale_id("log1P")

    def test_card_bow_row_count_is_checked(self):
        with self.assertRaises(ValueError):
            model_ss.SixthSenseNet(d=32, layers=1, heads=4, hidden=48,
                                   card_bow=np.zeros((10, 5), dtype=np.float32))

    def test_num_stats_reject_scalars_nan_and_negative_sigma(self):
        model = model_ss.SixthSenseNet(d=32, layers=1, heads=4, hidden=48,
                                       num_w=6, opt_num_w=6, arch_v2=True)
        ok = np.ones(6, dtype=np.float32)
        for bad_mu, bad_sigma in ((np.float32(1.0), ok),                 # would broadcast
                                  (ok, np.array([1, 1, 1, 1, 1, np.nan], dtype=np.float32)),
                                  (ok, np.array([1, 1, 1, 1, 1, -1], dtype=np.float32))):
            with self.assertRaises(ValueError):
                model.set_num_stats(bad_mu, bad_sigma)
        model.set_num_stats(ok, ok)  # a well-formed file is accepted


class Padding(unittest.TestCase):
    """Zero padding is not neutral unless it is masked (docs/architecture.pdf, section 2)."""

    @staticmethod
    def widened(rows, extra_state, extra_opt):
        out = []
        for r in rows:
            r = dict(r)
            r["family"] = r["family"] + [0] * extra_state
            r["owner"] = r["owner"] + [0] * extra_state
            r["card"] = r["card"] + [0] * extra_state
            r["card_type"] = r["card_type"] + [0] * extra_state
            r["energy_type"] = r["energy_type"] + [0] * extra_state
            r["numeric"] = r["numeric"] + [[0.0] * syn.NUM_W] * extra_state
            r["option_type"] = r["option_type"] + [-1] * extra_opt
            r["option_numeric"] = r["option_numeric"] + [[0.0] * syn.OPT_NUM_W] * extra_opt
            r["option_ptr"] = r["option_ptr"] + [[-1, -1]] * extra_opt
            r["option_attack"] = r["option_attack"] + [0] * extra_opt
            out.append(r)
        return out

    def logit_gap(self, mask_state, mask_options):
        model = syn.make_model()
        rows = syn.make_rows(12)
        syn.set_training_semantics(mask_state, mask_options, relations=True)
        narrow = syn.torch_logits(model, rows)
        wide = syn.torch_logits(model, self.widened(rows, 16, 4))
        return max(float(np.abs(a - b).max()) for a, b in zip(narrow, wide))

    def test_masked_padding_leaves_real_logits_unchanged(self):
        self.assertLess(self.logit_gap(True, True), 1e-5)

    def test_unmasked_padding_changes_real_logits(self):
        # The probe must be able to observe the defect, or the test above is vacuous.
        self.assertGreater(self.logit_gap(False, False), 1e-3)


if __name__ == "__main__":
    unittest.main()
