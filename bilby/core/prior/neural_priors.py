"""
Normalizing flow-based prior classes backed by `neural_priors_gym` normalizing flows.

The source code is available here: https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym
"""

from pathlib import Path

import numpy as np

from .joint import BaseJointPriorDist, JointPrior, MultivariateGaussianDist
from .dict import PriorDict
from ..utils import logger

import joblib
from neural_priors_gym.flows.base import FlowBase

try:
    from neural_priors_gym.preprocessing import (
        flow_to_physical,
        load_preprocessing,
        physical_log_prob,
    )
except ImportError:  # neural_priors_gym without the preprocessing record
    load_preprocessing = None

class NFDist(BaseJointPriorDist):
    """Multivariate joint prior distribution backed by a neural_priors_gym flow.

    Loads a flow trained with neural_priors_gym's FlowTrainer from a model
    directory containing model.pt, model_kwargs.json, and optionally scaler.gz.

    Two model formats are supported:

    - **With a preprocessing record** (the ``"preprocessing"`` entry of
      ``model_kwargs.json``, written by neural_priors_gym for every backend):
      the flow was trained on ``y = T(x)``, with ``T`` an optional ln(x) or
      ln(1 - x) per parameter followed by an affine scaler (min-max or
      standard). ``ln_prob`` is the flow density at ``T(x)`` plus
      ``log|det dT/dx|``; ``rescale`` and ``sample`` map flow output back with
      ``T^-1``. Points where a transform is undefined (``x <= 0`` for ln(x),
      ``x >= 1`` for ln(1 - x)) get ``-inf`` and are never produced. No
      clipping is applied, so ``rescale``/``sample`` and ``ln_prob`` describe
      the same density (there is no point mass at the bounds). Samples
      outside the declared ``bounds`` get ``-inf`` in ``ln_prob``; at
      construction the flow mass outside the bounds is estimated and a
      warning is logged when it is non-zero, because ``rescale`` does not
      truncate.
    - **Without a record** (older models, including those of
      arXiv:2511.22987): the pickled MinMaxScaler in ``scaler.gz`` is used and
      samples are clipped to the bounds, exactly as before.

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

        self.preprocessing = (
            None if load_preprocessing is None else load_preprocessing(flow_dir_path)
        )
        if self.preprocessing is not None:
            record_names = list(self.preprocessing["parameter_names"])
            if record_names != list(names):
                raise ValueError(
                    f"NFDist names {list(names)} do not match the parameter names "
                    f"the flow in {flow_dir} was trained on: {record_names}"
                )
            # The record replaces scaler.gz (which may also be present).
            self.scaler = None
            self.log_jacobian_correction = 0.0
            logger.info(
                f"Using the preprocessing record of {flow_dir}: scaler "
                f"{(self.preprocessing.get('scaler') or {}).get('type')}, "
                f"log {self.preprocessing.get('log_transform')}, "
                f"log(1 - x) {self.preprocessing.get('log1m_transform')}"
            )
        # Old format: load the MinMaxScaler produced during training, if present
        elif (scaler_path := flow_dir_path / "scaler.gz").exists():
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
        if self.preprocessing is not None:
            self._warn_mass_outside_bounds()

    def _outside_bounds(self, samples: np.ndarray) -> np.ndarray:
        """Rows with any coordinate outside the declared bounds (or NaN)."""
        out = np.zeros(samples.shape[0], dtype=bool)
        for col, name in enumerate(self.names):
            lo, hi = self._bounds_dict[name]
            s = samples[:, col]
            out |= ~((s >= lo) & (s <= hi))
        return out

    def _warn_mass_outside_bounds(self, n: int = 20_000) -> None:
        """Estimate the flow mass outside the declared bounds (new format only)."""
        frac = float(np.mean(self._outside_bounds(self._sample(n))))
        if frac > 0:
            logger.warning(
                f"NFDist: about {100 * frac:.3g}% of the flow mass in {self.flow_dir} "
                "lies outside the declared bounds. rescale/sample do not truncate, "
                "while ln_prob is -inf there; widen the bounds to make them consistent."
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
        if self.preprocessing is not None:
            return flow_to_physical(raw, self.preprocessing)
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

        if self.preprocessing is not None:
            # BaseJointPriorDist._check_samp only reports the out-of-bounds rows
            # of one parameter, so recompute them over all parameters.
            inside = ~(outbounds | self._outside_bounds(samp))
            lnprob[:] = -np.inf
            if np.any(inside):
                lnprob[inside] = physical_log_prob(
                    self._flow.log_prob, samp[inside], self.preprocessing
                )
            return lnprob

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

        if self.preprocessing is not None:
            # No clipping: flow_to_physical maps every finite point into the
            # support of the transforms (e.g. q < 1 for ln(1 - q)).
            return flow_to_physical(flow_samp, self.preprocessing)

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

        Handles three call patterns:

        1. Normal scalar accumulation (e.g. from priors.ln_prob with scalar
           params): each NFPrior stores its value and the last one to be
           called evaluates the full joint log-prob.

        2. Empty-list placeholder from the bilby joint-prior rescale protocol
           (dynesty prior_transform path): PriorDict.rescale returns [] for
           the first N-1 parameters and the full (N,) rescaled sample for the
           Nth.  Empty-list calls return 0.0 immediately.

        3. Full-sample array (the Nth parameter in case 2): evaluate the
           joint log-prob directly using the complete rescaled sample.
        """
        # Case 2: empty-list placeholder from the rescale protocol
        if isinstance(val, list) and len(val) == 0:
            return 0.0

        # Case 3: full rescaled sample array from the rescale protocol.
        # The last parameter in the joint group receives the complete (N,)
        # array produced by _rescale; evaluate log_prob directly.
        if (
            isinstance(val, np.ndarray)
            and val.ndim == 1
            and len(val) == self.dist.num_vars
        ):
            samp = val.reshape(1, -1).astype(np.float64)
            lnp = np.atleast_1d(self.dist.ln_prob(samp).squeeze())
            self.dist.reset_request()
            try:
                return lnp.item()
            except ValueError:
                return lnp

        # Case 1: normal scalar path
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
                pass  # vectorised call; keep as-is

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


class NFPriorDict(PriorDict):
    """PriorDict subclass that correctly handles NFPrior in nested samplers.

    bilby's standard PriorDict.rescale uses a joint-prior accumulation protocol
    that returns [] for the first N-1 parameters in a joint group and the full
    (N,) rescaled array for the last.  The resulting list
    ``[[], [], ..., array([a, b, ..., n])]`` breaks evaluate_constraints and
    log_likelihood because they expect a flat list of scalar parameter values.

    This subclass post-processes the raw rescale output and flattens each joint
    prior group into individual scalar values so that downstream bilby / dynesty
    code works without modification.

    Use this dict whenever NFPrior objects are part of a nested-sampler run:

        dist = NFDist(names=[...], flow_dir=..., bounds=[...])
        priors = NFPriorDict({name: NFPrior(dist=dist, name=name) for name in dist.names})
        bilby.run_sampler(likelihood=..., priors=priors, sampler="dynesty", ...)
    """

    def rescale(self, keys, theta):
        raw = super().rescale(keys, theta)

        # Fast path: no joint-prior placeholders present.
        if not any(isinstance(r, list) and len(r) == 0 for r in raw):
            return raw

        # Flatten: the bilby joint-prior protocol produces
        #   [[], [], ..., array([v0, v1, ..., vN-1])]
        # for each group of N joint parameters (empties for 0..N-2, full array
        # for N-1).  Replace each group with individual scalar values.
        result = []
        pending_empties = 0
        for r in raw:
            if isinstance(r, list) and len(r) == 0:
                pending_empties += 1
                result.append(r)  # temporary placeholder
            elif (
                isinstance(r, np.ndarray)
                and r.ndim == 1
                and pending_empties == len(r) - 1
            ):
                for i in range(pending_empties):
                    result[-(pending_empties - i)] = float(r[i])
                result.append(float(r[-1]))
                pending_empties = 0
            else:
                pending_empties = 0
                result.append(r)
        return result
