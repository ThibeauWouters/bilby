"""NFDist with the neural_priors_gym preprocessing record, and old-format models.

New format: a small glasflow model trained with the standard scaler,
ln(lambda_1) and ln(1 - q). Old format: a glasflow model with only a pickled
MinMaxScaler, compared bit for bit with a frozen copy of the old NFDist.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, norm

try:
    import joblib
    import torch
    from neural_priors_gym.config.schema import TrainingConfig
    from neural_priors_gym.config.schemas.flow import GlasflowConfig
    from neural_priors_gym.flows.glasflow import GlasflowNSF
    from neural_priors_gym.preprocessing import (
        flow_to_physical,
        load_preprocessing,
        physical_log_prob,
    )
    from neural_priors_gym.training.trainer import FlowTrainer
    from sklearn.preprocessing import MinMaxScaler

    _GYM = True
except ImportError:
    _GYM = False

import bilby.core.prior as bcp

NAMES = ["chirp_mass_source", "mass_ratio", "lambda_1"]
MEAN = np.array([1.19, np.log(0.2), 6.0])
COV = np.array([[1e-4, 1.5e-3, -2.4e-3], [1.5e-3, 0.09, -0.036], [-2.4e-3, -0.036, 0.16]])


def _draw(n, seed):
    v = np.random.default_rng(seed).multivariate_normal(MEAN, COV, size=n)
    return np.column_stack([v[:, 0], 1.0 - np.exp(v[:, 1]), np.exp(v[:, 2])])


def _train_new_format(directory):
    torch.manual_seed(0)
    x = _draw(5000, seed=0)
    config = TrainingConfig(
        output_dir=str(directory),
        flow={"backend": "glasflow", "n_transforms": 3, "n_neurons": 32,
              "n_blocks_per_transform": 1, "num_bins": 8},
        training={"parameter_names": NAMES, "data_path": str(directory / "unused.npz"),
                  "log_lambda": True, "log_one_minus_q": True, "scaler": "standard",
                  "num_epochs": 20, "learning_rate": 3e-3, "batch_size": 256,
                  "max_patience": 20, "validation_split": 0.2, "scale_input": True},
    )
    data = {name: x[:, i] for i, name in enumerate(NAMES)}
    flow, *_ = FlowTrainer(config).train(data, NAMES, output_dir=directory)
    flow.save(directory)
    return flow


@unittest.skipUnless(_GYM, "neural_priors_gym not installed")
class TestNFDistPreprocessingRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls._tmp.name) / "model"
        cls.flow = _train_new_format(cls.model_dir)
        cls.meta = load_preprocessing(cls.model_dir)
        cls.dist = bcp.NFDist(names=NAMES, flow_dir=str(cls.model_dir))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_record_used_and_scaler_ignored(self):
        self.assertTrue((self.model_dir / "scaler.gz").exists())
        self.assertIsNotNone(self.dist.preprocessing)
        self.assertIsNone(self.dist.scaler)

    def test_ln_prob_equals_gym_helper(self):
        """Same float32 flow call on the same inputs: equal to 1e-10 (in practice exact)."""
        x = _draw(1000, seed=1)
        expected = physical_log_prob(self.flow.log_prob, x, self.meta)
        np.testing.assert_allclose(self.dist.ln_prob(x), expected, rtol=0, atol=1e-10)

    def test_ln_prob_equals_hand_written_change_of_variables(self):
        """Independent computation from the JSON; the flow runs in float32, so atol 1e-4."""
        kw = json.loads((self.model_dir / "model_kwargs.json").read_text())["preprocessing"]
        scale = np.array(kw["scaler"]["scale"])
        shift = np.array(kw["scaler"]["min"])
        x = _draw(500, seed=2)
        u = np.column_stack([x[:, 0], np.log(1 - x[:, 1]), np.log(x[:, 2])])
        y = u * scale + shift
        # |dy/dx| = prod(scale) / ((1 - q) * Lambda)
        expected = (self.flow.log_prob(y) + np.sum(np.log(scale))
                    - np.log(1 - x[:, 1]) - np.log(x[:, 2]))
        np.testing.assert_allclose(self.dist.ln_prob(x), expected, rtol=0, atol=1e-4)

    def test_prior_dict_ln_prob_matches_joint(self):
        priors = bcp.NFPriorDict({n: bcp.NFPrior(dist=self.dist, name=n) for n in NAMES})
        x = _draw(200, seed=3)
        lp = priors.ln_prob({n: x[:, i] for i, n in enumerate(NAMES)}, axis=0)
        np.testing.assert_allclose(lp, self.dist.ln_prob(x), rtol=0, atol=1e-10)

    def test_rescale_is_flow_pushforward_and_matches_flow_samples(self):
        u = np.random.default_rng(4).uniform(size=(5000, 3))
        resc = self.dist.rescale(u)
        expected = flow_to_physical(self.flow.latent_to_data(norm.ppf(u)), self.meta)
        np.testing.assert_allclose(resc, expected, rtol=1e-12)
        torch.manual_seed(5)
        own = flow_to_physical(self.flow.sample(5000), self.meta)
        for i, name in enumerate(NAMES):
            p = ks_2samp(resc[:, i], own[:, i]).pvalue
            self.assertGreater(p, 1e-3, f"{name}: KS p = {p}")

    def test_rescale_stays_in_support_and_ln_prob_finite_there(self):
        u = np.random.default_rng(6).uniform(size=(2000, 3))
        u[:10] = 1e-12
        u[10:20] = 1 - 1e-12
        resc = self.dist.rescale(u)
        self.assertTrue(np.all(resc[:, 1] < 1))
        self.assertTrue(np.all(resc[:, 2] > 0))
        self.assertTrue(np.all(np.isfinite(self.dist.ln_prob(resc))))

    def test_minus_inf_outside_support(self):
        x = np.array([[1.19, 0.8, 300.0], [1.19, 1.0, 300.0], [1.19, 1.2, 300.0],
                      [1.19, 0.8, 0.0], [1.19, 0.8, -5.0]])
        lp = self.dist.ln_prob(x)
        self.assertTrue(np.isfinite(lp[0]))
        self.assertTrue(np.all(lp[1:] == -np.inf))

    def test_bounds_applied_to_every_parameter(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir),
                          bounds=[(1.0, 1.5), (0.0, 1.0), (0.0, 5000.0)])
        x = np.array([[1.19, 0.8, 300.0], [1.19, 0.8, 6000.0], [1.6, 0.8, 300.0]])
        lp = dist.ln_prob(x)
        self.assertTrue(np.isfinite(lp[0]))
        self.assertTrue(np.all(lp[1:] == -np.inf))

    def test_warns_when_bounds_cut_flow_mass(self):
        with self.assertLogs("bilby", level="WARNING") as cm:
            bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir),
                       bounds=[(None, None), (0.8, 1.0), (None, None)])
        self.assertTrue(any("outside the declared bounds" in m for m in cm.output))

    def test_name_mismatch_raises(self):
        with self.assertRaises(ValueError):
            bcp.NFDist(names=["mass_ratio", "chirp_mass_source", "lambda_1"],
                       flow_dir=str(self.model_dir))


@unittest.skipUnless(_GYM, "neural_priors_gym not installed")
class TestNFDistOldFormatUnchanged(unittest.TestCase):
    """Old-format models: bit-identical to the NFDist at cdbd6ef4."""

    @classmethod
    def setUpClass(cls):
        from ._nfdist_reference import ReferenceNFDist

        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls._tmp.name) / "model"
        torch.manual_seed(1)
        flow = GlasflowNSF.from_config(
            GlasflowConfig(n_transforms=2, n_neurons=16, n_blocks_per_transform=1, num_bins=4), 3
        )
        flow.save(cls.model_dir)
        cls.x = _draw(2000, seed=7)
        joblib.dump(MinMaxScaler().fit(cls.x), cls.model_dir / "scaler.gz")
        bounds = [(None, None), (0.125, 1.0), (0.0, 5000.0)]
        cls.new = bcp.NFDist(names=NAMES, flow_dir=str(cls.model_dir), bounds=bounds)
        cls.old = ReferenceNFDist(names=NAMES, flow_dir=str(cls.model_dir), bounds=bounds)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_no_record(self):
        self.assertIsNone(load_preprocessing(self.model_dir))
        self.assertIsNone(self.new.preprocessing)

    def test_ln_prob_bit_identical(self):
        x = np.vstack([self.x, [[1.19, 1.1, 300.0], [1.19, 0.8, -1.0]]])
        np.testing.assert_array_equal(self.new.ln_prob(x), self.old.ln_prob(x))

    def test_rescale_bit_identical(self):
        u = np.random.default_rng(8).uniform(size=(2000, 3))
        np.testing.assert_array_equal(self.new.rescale(u), self.old.rescale(u))

    def test_sample_bit_identical(self):
        torch.manual_seed(9)
        a = self.new._sample(500)
        torch.manual_seed(9)
        b = self.old._sample(500)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
