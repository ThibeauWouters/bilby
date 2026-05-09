"""
Normalizing flow-based prior classes backed by `neural_priors_gym` normalizing flows.

The source code is available here: https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym
"""

from pathlib import Path

import numpy as np

from .joint import BaseJointPriorDist, JointPrior, MultivariateGaussianDist
from ..utils import logger

import joblib
from neural_priors_gym.flows.base import FlowBase

class NFDist(BaseJointPriorDist):
    """Multivariate joint prior distribution backed by a neural_priors_gym flow.

    Loads a flow trained with neural_priors_gym's FlowTrainer from a model
    directory containing model.pt, model_kwargs.json, and optionally scaler.gz.

    Parameters
    ----------
    names : list[str]
        Parameter names in the same order the flow was trained on.
    flow_dir : str
        Path to the model directory produced by neural_priors_gym's FlowTrainer.
    bounds : list[tuple] or None
        Parameter bounds as ``[(min, max), ...]`` aligned with ``names``. Either
        element may be ``None`` to mean ±inf, e.g. ``(None, 5.0)``. If ``None``,
        all parameters are unbounded.
    """

    def __init__(self, names: list, flow_dir: str, bounds=None):

        flow_dir_path = Path(flow_dir)
        if not flow_dir_path.is_dir():
            raise FileNotFoundError(f"Flow directory does not exist: {flow_dir}")

        self.flow_dir = str(flow_dir)
        self._flow: FlowBase = FlowBase.load_from_dir(flow_dir_path)

        # Load the MinMaxScaler produced during training, if present
        scaler_path = flow_dir_path / "scaler.gz"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.log_jacobian_correction = FlowBase.compute_log_jacobian_correction(
                self.scaler
            )
            logger.info(f"Loaded MinMaxScaler from {scaler_path}")
        else:
            self.scaler = None
            self.log_jacobian_correction = 0.0
            logger.info("No scaler found — assuming flow was trained on unscaled data")

        if bounds is None:
            bounds = [(-np.inf, np.inf)] * len(names)

        # Resolve None to inf; _bounds_dict is used internally for clipping.
        self._bounds_dict = {
            name: (
                -np.inf if lo is None else lo,
                np.inf if hi is None else hi,
            )
            for name, (lo, hi) in zip(names, bounds)
        }
        resolved_bounds = [self._bounds_dict[name] for name in names]

        super().__init__(names=names, bounds=resolved_bounds)

        # Standard normal base distribution used by _rescale for nested samplers
        mvg_names = [f"x{i}" for i in range(1, len(names) + 1)]
        self._mvg = MultivariateGaussianDist(
            names=mvg_names,
            mus=[[0.0] * len(names)],
            corrcoefs=[[[1.0 if i == j else 0.0 for j in range(len(names))]
                        for i in range(len(names))]],
            sigmas=[[1.0] * len(names)],
        )

        logger.info(
            f"Loaded NFDist with n_dim={self.num_vars} from {flow_dir}"
        )

    def _clip_samples(self, samples: np.ndarray) -> np.ndarray:
        """Clip samples to the declared bounds (column order matches self.names).

        Columns whose bound is ±inf are left untouched. This is a soft clip —
        samples are moved to the boundary rather than rejected.
        """
        samples = samples.copy()
        for col, name in enumerate(self.names):
            lo, hi = self._bounds_dict[name]
            samples[:, col] = np.clip(samples[:, col], lo, hi)
        return samples

    def _sample(self, size, **_kwargs):
        """Draw samples from the flow in the original (unscaled) parameter space."""
        raw = self._flow.sample(int(size))
        if self.scaler is not None:
            raw = self.scaler.inverse_transform(raw)
        raw = self._clip_samples(raw.astype(np.float64))
        return raw

    def _ln_prob(self, samp, lnprob, outbounds):
        """Evaluate log probability of samples.

        Transforms to the scaled space used during training, evaluates the flow
        log-prob, and applies the Jacobian correction for the scaler.
        """
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        if self.scaler is not None:
            scaled = self.scaler.transform(samp)
        else:
            scaled = samp

        log_probs = self._flow.log_prob(scaled) + self.log_jacobian_correction

        lnprob[~outbounds] = log_probs[~outbounds]
        lnprob[outbounds] = -np.inf
        return lnprob

    def _rescale(self, samp, **_kwargs):
        """Rescale from the unit hypercube to the parameter space.

        Maps through a standard normal (via MultivariateGaussianDist) and then
        through the flow's generative direction (latent → data). Required by
        nested samplers such as dynesty.
        """
        mvg_samp = np.array(self._mvg.rescale(samp))
        if len(mvg_samp.shape) == 1:
            mvg_samp = mvg_samp.reshape(1, self.num_vars)

        # Map from the latent space (standard normal) to the data space using the flow's generative direction
        flow_samp = self._flow.latent_to_data(mvg_samp)

        # In case we scaled training data, then rescale back to the very original parameter space
        if self.scaler is not None:
            flow_samp = self.scaler.inverse_transform(flow_samp)

        return self._clip_samples(flow_samp.astype(np.float64))


class NFPrior(JointPrior):
    """Single-parameter bilby prior backed by a shared NFDist.

    Each parameter in the joint flow distribution gets its own NFPrior instance.
    All instances for the same flow must share the same NFDist object.

    Example
    -------
    >>> dist = NFDist(
    ...     names=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"],
    ...     flow_dir="/path/to/model/",
    ... )
    >>> priors = {name: NFPrior(dist=dist, name=name) for name in dist.names}
    """

    def ln_prob(self, val):
        """Return the log prior probability for this parameter.

        Accumulates values across all parameters in the joint distribution and
        evaluates the full log-prob once all have been provided (bilby's
        joint-prior protocol).
        """
        try:
            val = float(val)
        except Exception as e:
            logger.debug(f"Could not cast {val} to float in NFPrior.ln_prob: {e}")

        self.dist.requested_parameters[self.name] = val

        if self.dist.filled_request():
            values = list(self.dist.requested_parameters.values())

            for i in range(len(self.dist) - 1):
                if isinstance(values[i], (list, np.ndarray)) or isinstance(
                    values[i + 1], (list, np.ndarray)
                ):
                    if isinstance(values[i], (list, np.ndarray)) and isinstance(
                        values[i + 1], (list, np.ndarray)
                    ):
                        if len(values[i]) != len(values[i + 1]):
                            raise ValueError(
                                "Each parameter must have the same number of "
                                "requested values."
                            )
                    else:
                        raise ValueError(
                            "Each parameter must have the same number of "
                            "requested values."
                        )

            lnp = np.atleast_1d(self.dist.ln_prob(np.asarray(values).T).squeeze())
            try:
                lnp = lnp.item()
            except ValueError:
                pass  # lnp is a multi-element array (vectorised call); keep as-is

            self.dist.reset_request()
            return lnp
        else:
            if isinstance(val, (float, int, np.integer)):
                return 0.0
            else:
                try:
                    len(val)
                except Exception as e:
                    raise TypeError(f"Invalid type for ln_prob: {e}")
                return np.zeros_like(val)
