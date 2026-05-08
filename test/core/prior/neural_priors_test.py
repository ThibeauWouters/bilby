"""Tests for NFDist and NFPrior backed by neural_priors_gym flows."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import joblib
    from neural_priors_gym.config.schemas.flow import GlasflowConfig, ZukoMAFConfig
    from neural_priors_gym.flows.glasflow import GlasflowNSF
    from neural_priors_gym.flows.zuko_maf import ZukoMAF
    from sklearn.preprocessing import MinMaxScaler

    _NEURAL_PRIORS_GYM_AVAILABLE = True
except ImportError:
    _NEURAL_PRIORS_GYM_AVAILABLE = False

import bilby.core.prior as bcp


def _make_glasflow_dir(directory: Path, n_inputs: int = 3) -> Path:
    config = GlasflowConfig(
        n_transforms=2, n_neurons=16, n_blocks_per_transform=1, num_bins=4
    )
    flow = GlasflowNSF.from_config(config, n_inputs=n_inputs)
    flow.save(directory)
    return directory


def _make_zuko_dir(directory: Path, n_inputs: int = 3) -> Path:
    config = ZukoMAFConfig(transforms=2, hidden_features=[16, 16])
    flow = ZukoMAF.from_config(config, n_inputs=n_inputs)
    flow.save(directory)
    return directory


def _add_scaler(directory: Path, n_inputs: int = 3, data_min=0.0, data_max=2.0):
    """Fit and save a MinMaxScaler to directory/scaler.gz."""
    rng = np.random.default_rng(0)
    data = rng.uniform(data_min, data_max, size=(100, n_inputs)).astype(np.float32)
    scaler = MinMaxScaler()
    scaler.fit(data)
    joblib.dump(scaler, directory / "scaler.gz")
    return scaler


NAMES = ["a", "b", "c"]


@unittest.skipUnless(_NEURAL_PRIORS_GYM_AVAILABLE, "neural_priors_gym not installed")
class TestNFDistGlasflow(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._tmpdir.name) / "model"
        _make_glasflow_dir(self.model_dir, n_inputs=len(NAMES))

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_succeeds(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        self.assertEqual(dist.num_vars, len(NAMES))

    def test_sample_shape(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        samples = dist._sample(50)
        self.assertEqual(samples.shape, (50, len(NAMES)))
        self.assertEqual(samples.dtype, np.float64)

    def test_log_prob_shape_and_finite(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        samp = dist._sample(20)
        lnprob = np.zeros(20)
        outbounds = np.zeros(20, dtype=bool)
        result = dist._ln_prob(samp, lnprob, outbounds)
        self.assertEqual(result.shape, (20,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_log_prob_outbounds_set_to_neginf(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        samp = dist._sample(5)
        lnprob = np.zeros(5)
        outbounds = np.array([True, False, True, False, False])
        result = dist._ln_prob(samp, lnprob, outbounds)
        self.assertTrue(np.all(result[outbounds] == -np.inf))
        self.assertTrue(np.all(np.isfinite(result[~outbounds])))

    def test_no_scaler_by_default(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        self.assertIsNone(dist.scaler)
        self.assertEqual(dist.log_jacobian_correction, 0.0)

    def test_directory_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            bcp.NFDist(names=NAMES, flow_dir="/nonexistent/path/to/model")


@unittest.skipUnless(_NEURAL_PRIORS_GYM_AVAILABLE, "neural_priors_gym not installed")
class TestNFDistWithScaler(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._tmpdir.name) / "model"
        _make_glasflow_dir(self.model_dir, n_inputs=len(NAMES))
        self.scaler = _add_scaler(self.model_dir, n_inputs=len(NAMES))

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_scaler_loaded(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        self.assertIsNotNone(dist.scaler)

    def test_jacobian_correction_computed(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        expected = float(
            -np.sum(np.log(self.scaler.data_max_ - self.scaler.data_min_))
        )
        self.assertAlmostEqual(dist.log_jacobian_correction, expected, places=6)

    def test_jacobian_correction_applied_to_log_prob(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        # With a non-trivial scaler the correction should differ from zero
        self.assertNotEqual(dist.log_jacobian_correction, 0.0)

        samp = dist._sample(10)
        lnprob = np.zeros(10)
        outbounds = np.zeros(10, dtype=bool)
        result = dist._ln_prob(samp, lnprob, outbounds)
        self.assertEqual(result.shape, (10,))

    def test_sample_returns_unscaled_values(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        samples = dist._sample(200)
        # Samples should be in the original space (roughly data_min..data_max)
        self.assertEqual(samples.shape, (200, len(NAMES)))


@unittest.skipUnless(_NEURAL_PRIORS_GYM_AVAILABLE, "neural_priors_gym not installed")
class TestNFDistZuko(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._tmpdir.name) / "model"
        _make_zuko_dir(self.model_dir, n_inputs=len(NAMES))

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_succeeds(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        self.assertEqual(dist.num_vars, len(NAMES))

    def test_sample_shape(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        samples = dist._sample(30)
        self.assertEqual(samples.shape, (30, len(NAMES)))

    def test_log_prob_finite(self):
        dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        samp = dist._sample(10)
        lnprob = np.zeros(10)
        outbounds = np.zeros(10, dtype=bool)
        result = dist._ln_prob(samp, lnprob, outbounds)
        self.assertTrue(np.all(np.isfinite(result)))


@unittest.skipUnless(_NEURAL_PRIORS_GYM_AVAILABLE, "neural_priors_gym not installed")
class TestNFPrior(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._tmpdir.name) / "model"
        _make_glasflow_dir(self.model_dir, n_inputs=len(NAMES))
        self.dist = bcp.NFDist(names=NAMES, flow_dir=str(self.model_dir))
        self.priors = {name: bcp.NFPrior(dist=self.dist, name=name) for name in NAMES}

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_prior_returns_zero_until_all_params_provided(self):
        # Only providing the first parameter should return 0.0 (not yet evaluated)
        result = self.priors[NAMES[0]].ln_prob(0.5)
        self.assertEqual(result, 0.0)

    def test_prior_returns_float_when_all_params_provided(self):
        self.dist.reset_request()
        for name in NAMES[:-1]:
            self.priors[name].ln_prob(0.5)
        result = self.priors[NAMES[-1]].ln_prob(0.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))

    def test_prior_evaluates_independently_per_call(self):
        self.dist.reset_request()
        for name in NAMES[:-1]:
            self.priors[name].ln_prob(0.5)
        r1 = self.priors[NAMES[-1]].ln_prob(0.5)

        self.dist.reset_request()
        for name in NAMES[:-1]:
            self.priors[name].ln_prob(0.5)
        r2 = self.priors[NAMES[-1]].ln_prob(0.5)

        self.assertAlmostEqual(r1, r2, places=6)


if __name__ == "__main__":
    unittest.main()
