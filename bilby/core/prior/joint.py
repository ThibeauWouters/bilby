import re
import os

import numpy as np
import scipy.stats
from scipy.special import erfinv

from .base import Prior, PriorException
from ..utils import logger, infer_args_from_method, get_dict_with_properties
from ..utils import random

import json

### glasflow (optional — only needed for NFDist/NFPrior)
try:
    from glasflow.flows import RealNVP
    from glasflow.flows.nsf import CouplingNSF
    import torch
    from sklearn.preprocessing import MinMaxScaler
    import joblib
except ImportError:
    RealNVP = None
    CouplingNSF = None
    torch = None
    MinMaxScaler = None
    joblib = None

class BaseJointPriorDist(object):
    def __init__(self, names, bounds=None):
        """
        A class defining JointPriorDist that will be overwritten with child
        classes defining the joint prior distributions between given parameters,


        Parameters
        ==========
        names: list (required)
            A list of the parameter names in the JointPriorDist. The
            listed parameters must have the same order that they appear in
            the lists of statistical parameters that may be passed in child class
        bounds: list (optional)
            A list of bounds on each parameter. The defaults are for bounds at
            +/- infinity.
        """
        self.distname = "joint_dist"
        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names

        self.num_vars = len(self.names)

        # set the bounds for each parameter
        if isinstance(bounds, list):
            if len(bounds) != len(self):
                raise ValueError("Wrong number of parameter bounds")

            # check bounds
            for bound in bounds:
                if isinstance(bounds, (list, tuple, np.ndarray)):
                    if len(bound) != 2:
                        raise ValueError(
                            "Bounds must contain an upper and lower value."
                        )
                    else:
                        if bound[1] <= bound[0]:
                            raise ValueError("Bounds are not properly set")
                else:
                    raise TypeError("Bound must be a list")
        else:
            bounds = [(-np.inf, np.inf) for _ in self.names]
        self.bounds = {name: val for name, val in zip(self.names, bounds)}

        self._current_sample = {}  # initialise empty sample
        self._uncorrelated = None
        self._current_lnprob = None

        # a dictionary of the parameters as requested by the prior
        self.requested_parameters = dict()
        self.reset_request()

        # a dictionary of the rescaled parameters
        self.rescale_parameters = dict()
        self.reset_rescale()

        # a list of sampled parameters
        self.reset_sampled()

    def reset_sampled(self):
        self.sampled_parameters = []
        self.current_sample = {}

    def filled_request(self):
        """
        Check if all requested parameters have been filled.
        """

        return not np.any([val is None for val in self.requested_parameters.values()])

    def reset_request(self):
        """
        Reset the requested parameters to None.
        """

        for name in self.names:
            self.requested_parameters[name] = None

    def filled_rescale(self):
        """
        Check if all the rescaled parameters have been filled.
        """

        return not np.any([val is None for val in self.rescale_parameters.values()])

    def reset_rescale(self):
        """
        Reset the rescaled parameters to None.
        """

        for name in self.names:
            self.rescale_parameters[name] = None

    def get_instantiation_dict(self):
        subclass_args = infer_args_from_method(self.__init__)
        dict_with_properties = get_dict_with_properties(self)
        instantiation_dict = dict()
        for key in subclass_args:
            if isinstance(dict_with_properties[key], list):
                value = np.asarray(dict_with_properties[key]).tolist()
            else:
                value = dict_with_properties[key]
            instantiation_dict[key] = value
        return instantiation_dict

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        """Overrides the special method __repr__.

        Returns a representation of this instance that resembles how it is instantiated.
        Works correctly for all child classes

        Returns
        =======
        str: A string representation of this instance

        """
        dist_name = self.__class__.__name__
        instantiation_dict = self.get_instantiation_dict()
        args = ", ".join(
            [
                "{}={}".format(key, repr(instantiation_dict[key]))
                for key in instantiation_dict
            ]
        )
        return "{}({})".format(dist_name, args)

    @classmethod
    def from_repr(cls, string):
        """Generate the distribution from its __repr__"""
        return cls._from_repr(string)

    @classmethod
    def _from_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)

        string = string.replace(" ", "")
        kwargs = cls._split_repr(string)
        for key in kwargs:
            val = kwargs[key]
            if key not in subclass_args:
                raise AttributeError(
                    "Unknown argument {} for class {}".format(key, cls.__name__)
                )
            else:
                kwargs[key.strip()] = Prior._parse_argument_string(val)

        return cls(**kwargs)

    @classmethod
    def _split_repr(cls, string):
        string = string.replace(",", ", ")
        # see https://stackoverflow.com/a/72146415/1862861
        args = re.findall(r"(\w+)=(\[.*?]|{.*?}|\S+)(?=\s*,\s*\w+=|\Z)", string)
        kwargs = dict()
        for key, arg in args:
            kwargs[key.strip()] = arg
        return kwargs

    def prob(self, samp):
        """
        Get the probability of a sample. For bounded priors the
        probability will not be properly normalised.
        """

        return np.exp(self.ln_prob(samp))

    def _check_samp(self, value):
        """
        Get the log-probability of a sample. For bounded priors the
        probability will not be properly normalised.

        Parameters
        ==========
        value: array_like
            A 1d vector of the sample, or 2d array of sample values with shape
            NxM, where N is the number of samples and M is the number of
            parameters.

        Returns
        =======
        samp: array_like
            returns the input value as a sample array
        outbounds: array_like
            Boolean Array that selects samples in samp that are out of given bounds
        """
        samp = np.array(value)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        if len(samp.shape) != 2:
            raise ValueError("Array is the wrong shape")
        elif samp.shape[1] != self.num_vars:
            raise ValueError("Array is the wrong shape")

        # check sample(s) is within bounds
        outbounds = np.ones(samp.shape[0], dtype=bool)
        for s, bound in zip(samp.T, self.bounds.values()):
            outbounds = (s < bound[0]) | (s > bound[1])
            if np.any(outbounds):
                break
        return samp, outbounds

    def ln_prob(self, value):
        """
        Get the log-probability of a sample. For bounded priors the
        probability will not be properly normalised.

        Parameters
        ==========
        value: array_like
            A 1d vector of the sample, or 2d array of sample values with shape
            NxM, where N is the number of samples and M is the number of
            parameters.
        """

        samp, outbounds = self._check_samp(value)
        lnprob = -np.inf * np.ones(samp.shape[0])
        lnprob = self._ln_prob(samp, lnprob, outbounds)
        if samp.shape[0] == 1:
            return lnprob[0]
        else:
            return lnprob

    def _ln_prob(self, samp, lnprob, outbounds):
        """
        Get the log-probability of a sample. For bounded priors the
        probability will not be properly normalised. **this method needs overwritten by child class**

        Parameters
        ==========
        samp: vector
            sample to evaluate the ln_prob at
        lnprob: vector
            of -inf passed in with the same shape as the number of samples
        outbounds: array_like
            boolean array showing which samples in lnprob vector are out of the given bounds

        Returns
        =======
        lnprob: vector
            array of lnprob values for each sample given
        """
        """
        Here is where the subclass where overwrite ln_prob method
        """
        return lnprob

    def sample(self, size=1, **kwargs):
        """
        Draw, and set, a sample from the Dist, accompanying method _sample needs to overwritten

        Parameters
        ==========
        size: int
            number of samples to generate, defaults to 1
        """

        if size is None:
            size = 1
        samps = self._sample(size=size, **kwargs)
        for i, name in enumerate(self.names):
            if size == 1:
                self.current_sample[name] = samps[:, i].flatten()[0]
            else:
                self.current_sample[name] = samps[:, i].flatten()

    def _sample(self, size, **kwargs):
        """
        Draw, and set, a sample from the joint dist (**needs to be ovewritten by child class**)

        Parameters
        ==========
        size: int
            number of samples to generate, defaults to 1
        """
        samps = np.zeros((size, len(self)))
        """
        Here is where the subclass where overwrite sampling method
        """
        return samps

    def rescale(self, value, **kwargs):
        """
        Rescale from a unit hypercube to JointPriorDist. Note that no
        bounds are applied in the rescale function. (child classes need to
        overwrite accompanying method _rescale().

        Parameters
        ==========
        value: array
            A 1d vector sample (one for each parameter) drawn from a uniform
            distribution between 0 and 1, or a 2d NxM array of samples where
            N is the number of samples and M is the number of parameters.
        kwargs: dict
            All keyword args that need to be passed to _rescale method, these keyword
            args are called in the JointPrior rescale methods for each parameter

        Returns
        =======
        array:
            An vector sample drawn from the multivariate Gaussian
            distribution.
        """
        samp = np.array(value)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        if len(samp.shape) != 2:
            raise ValueError("Array is the wrong shape")
        elif samp.shape[1] != self.num_vars:
            raise ValueError("Array is the wrong shape")

        samp = self._rescale(samp, **kwargs)
        return np.squeeze(samp)

    def _rescale(self, samp, **kwargs):
        """
        rescale a sample from a unit hypercybe to the joint dist (**needs to be ovewritten by child class**)

        Parameters
        ==========
        samp: numpy array
            this is a vector sample drawn from a uniform distribution to be rescaled to the distribution
        """
        """
        Here is where the subclass where overwrite rescale method
        """
        return samp

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self.get_instantiation_dict() == other.get_instantiation_dict()

class MultivariateGaussianDist(BaseJointPriorDist):
    def __init__(
        self,
        names,
        nmodes=1,
        mus=None,
        sigmas=None,
        corrcoefs=None,
        covs=None,
        weights=None,
        bounds=None,
    ):
        """
        A class defining a multi-variate Gaussian, allowing multiple modes for
        a Gaussian mixture model.

        Note: if using a multivariate Gaussian prior, with bounds, this can
        lead to biases in the marginal likelihood estimate and posterior
        estimate for nested samplers routines that rely on sampling from a unit
        hypercube and having a prior transform, e.g., nestle, dynesty and
        MultiNest.

        Parameters
        ==========
        names: list
            A list of the parameter names in the multivariate Gaussian. The
            listed parameters must have the same order that they appear in
            the lists of means, standard deviations, and the correlation
            coefficient, or covariance, matrices.
        nmodes: int
            The number of modes for the mixture model. This defaults to 1,
            which will be checked against the shape of the other inputs.
        mus: array_like
            A list of lists of means of each mode in a multivariate Gaussian
            mixture model. A single list can be given for a single mode. If
            this is None then means at zero will be assumed.
        sigmas: array_like
            A list of lists of the standard deviations of each mode of the
            multivariate Gaussian. If supplying a correlation coefficient
            matrix rather than a covariance matrix these values must be given.
            If this is None unit variances will be assumed.
        corrcoefs: array
            A list of square matrices containing the correlation coefficients
            of the parameters for each mode. If this is None it will be assumed
            that the parameters are uncorrelated.
        covs: array
            A list of square matrices containing the covariance matrix of the
            multivariate Gaussian.
        weights: list
            A list of weights (relative probabilities) for each mode of the
            multivariate Gaussian. This will default to equal weights for each
            mode.
        bounds: list
            A list of bounds on each parameter. The defaults are for bounds at
            +/- infinity.
        """
        super(MultivariateGaussianDist, self).__init__(names=names, bounds=bounds)
        for name in self.names:
            bound = self.bounds[name]
            if bound[0] != -np.inf or bound[1] != np.inf:
                logger.warning(
                    "If using bounded ranges on the multivariate "
                    "Gaussian this will lead to biased posteriors "
                    "for nested sampling routines that require "
                    "a prior transform."
                )
        self.distname = "mvg"
        self.mus = []
        self.covs = []
        self.corrcoefs = []
        self.sigmas = []
        self.logprodsigmas = []   # log of product of sigmas, needed for "standard" multivariate normal
        self.weights = []
        self.eigvalues = []
        self.eigvectors = []
        self.sqeigvalues = []  # square root of the eigenvalues
        self.mvn = []  # list of multivariate normal distributions

        # put values in lists if required
        if nmodes == 1:
            if mus is not None:
                if len(np.shape(mus)) == 1:
                    mus = [mus]
                elif len(np.shape(mus)) == 0:
                    raise ValueError("Must supply a list of means")
            if sigmas is not None:
                if len(np.shape(sigmas)) == 1:
                    sigmas = [sigmas]
                elif len(np.shape(sigmas)) == 0:
                    raise ValueError("Must supply a list of standard deviations")
            if covs is not None:
                if isinstance(covs, np.ndarray):
                    covs = [covs]
                elif isinstance(covs, list):
                    if len(np.shape(covs)) == 2:
                        covs = [np.array(covs)]
                    elif len(np.shape(covs)) != 3:
                        raise TypeError("List of covariances the wrong shape")
                else:
                    raise TypeError("Must pass a list of covariances")
            if corrcoefs is not None:
                if isinstance(corrcoefs, np.ndarray):
                    corrcoefs = [corrcoefs]
                elif isinstance(corrcoefs, list):
                    if len(np.shape(corrcoefs)) == 2:
                        corrcoefs = [np.array(corrcoefs)]
                    elif len(np.shape(corrcoefs)) != 3:
                        raise TypeError(
                            "List of correlation coefficients the wrong shape"
                        )
                elif not isinstance(corrcoefs, list):
                    raise TypeError("Must pass a list of correlation coefficients")
            if weights is not None:
                if isinstance(weights, (int, float)):
                    weights = [weights]
                elif isinstance(weights, list):
                    if len(weights) != 1:
                        raise ValueError("Wrong number of weights given")

        for val in [mus, sigmas, covs, corrcoefs, weights]:
            if val is not None and not isinstance(val, list):
                raise TypeError("Value must be a list")
            else:
                if val is not None and len(val) != nmodes:
                    raise ValueError("Wrong number of modes given")

        # add the modes
        self.nmodes = 0
        for i in range(nmodes):
            mu = mus[i] if mus is not None else None
            sigma = sigmas[i] if sigmas is not None else None
            corrcoef = corrcoefs[i] if corrcoefs is not None else None
            cov = covs[i] if covs is not None else None
            weight = weights[i] if weights is not None else 1.0

            self.add_mode(mu, sigma, corrcoef, cov, weight)

    def add_mode(self, mus=None, sigmas=None, corrcoef=None, cov=None, weight=1.0):
        """
        Add a new mode.
        """

        # add means
        if mus is not None:
            try:
                self.mus.append(list(mus))  # means
            except TypeError:
                raise TypeError("'mus' must be a list")
        else:
            self.mus.append(np.zeros(self.num_vars))

        # add the covariances if supplied
        if cov is not None:
            self.covs.append(np.asarray(cov))

            if len(self.covs[-1].shape) != 2:
                raise ValueError("Covariance matrix must be a 2d array")

            if (
                self.covs[-1].shape[0] != self.covs[-1].shape[1]
                or self.covs[-1].shape[0] != self.num_vars
            ):
                raise ValueError("Covariance shape is inconsistent")

            # check matrix is symmetric
            if not np.allclose(self.covs[-1], self.covs[-1].T):
                raise ValueError("Covariance matrix is not symmetric")

            self.sigmas.append(np.sqrt(np.diag(self.covs[-1])))  # standard deviations

            # convert covariance into a correlation coefficient matrix
            D = self.sigmas[-1] * np.identity(self.covs[-1].shape[0])
            Dinv = np.linalg.inv(D)
            self.corrcoefs.append(np.dot(np.dot(Dinv, self.covs[-1]), Dinv))
        elif corrcoef is not None and sigmas is not None:
            self.corrcoefs.append(np.asarray(corrcoef))

            if len(self.corrcoefs[-1].shape) != 2:
                raise ValueError(
                    "Correlation coefficient matrix must be a 2d array."
                )

            if (
                self.corrcoefs[-1].shape[0] != self.corrcoefs[-1].shape[1]
                or self.corrcoefs[-1].shape[0] != self.num_vars
            ):
                raise ValueError(
                    "Correlation coefficient matrix shape is inconsistent"
                )

            # check matrix is symmetric
            if not np.allclose(self.corrcoefs[-1], self.corrcoefs[-1].T):
                raise ValueError("Correlation coefficient matrix is not symmetric")

            # check diagonal is all ones
            if not np.all(np.diag(self.corrcoefs[-1]) == 1.0):
                raise ValueError("Correlation coefficient matrix is not correct")

            try:
                self.sigmas.append(list(sigmas))  # standard deviations
            except TypeError:
                raise TypeError("'sigmas' must be a list")

            if len(self.sigmas[-1]) != self.num_vars:
                raise ValueError(
                    "Number of standard deviations must be the "
                    "same as the number of parameters."
                )

            # convert correlation coefficients to covariance matrix
            D = self.sigmas[-1] * np.identity(self.corrcoefs[-1].shape[0])
            self.covs.append(np.dot(D, np.dot(self.corrcoefs[-1], D)))
        else:
            # set unit variance uncorrelated covariance
            self.corrcoefs.append(np.eye(self.num_vars))
            self.covs.append(np.eye(self.num_vars))
            self.sigmas.append(np.ones(self.num_vars))

        # compute log of product of sigmas, needed for "standard" multivariate normal
        self.logprodsigmas.append(np.log(np.prod(self.sigmas[-1])))

        # get eigen values and vectors
        try:
            evals, evecs = np.linalg.eig(self.corrcoefs[-1])
            self.eigvalues.append(evals)
            self.eigvectors.append(evecs)
        except Exception as e:
            raise RuntimeError(
                "Problem getting eigenvalues and vectors: {}".format(e)
            )

        # check eigenvalues are positive
        if np.any(self.eigvalues[-1] <= 0.0):
            raise ValueError(
                "Correlation coefficient matrix is not positive definite"
            )
        self.sqeigvalues.append(np.sqrt(self.eigvalues[-1]))

        # set the weights
        if weight is None:
            self.weights.append(1.0)
        else:
            self.weights.append(weight)

        # set the cumulative relative weights
        self.cumweights = np.cumsum(self.weights) / np.sum(self.weights)

        # add the mode
        self.nmodes += 1

        # add "standard" multivariate normal distribution
        # - when the typical scales of the parameters are very different,
        #   multivariate_normal() may complain that the covariance matrix is singular
        # - instead pass zero means and correlation matrix instead of covariance matrix
        #   to get the equivalent of a standard normal distribution in higher dimensions
        # - this modifies the multivariate normal PDF as follows:
        #     multivariate_normal(mean=mus, cov=cov).logpdf(x)
        #     = multivariate_normal(mean=0, cov=corrcoefs).logpdf((x - mus)/sigmas) - logprodsigmas
        self.mvn.append(
            scipy.stats.multivariate_normal(mean=np.zeros(self.num_vars), cov=self.corrcoefs[-1])
        )

    def _rescale(self, samp, **kwargs):
        try:
            mode = kwargs["mode"]
        except KeyError:
            mode = None

        if mode is None:
            if self.nmodes == 1:
                mode = 0
            else:
                mode = np.argwhere(self.cumweights - random.rng.uniform(0, 1) > 0)[0][0]

        samp = erfinv(2.0 * samp - 1) * 2.0 ** 0.5

        # rotate and scale to the multivariate normal shape
        samp = self.mus[mode] + self.sigmas[mode] * np.einsum(
            "ij,kj->ik", samp * self.sqeigvalues[mode], self.eigvectors[mode]
        )
        return samp

    def _sample(self, size, **kwargs):
        try:
            mode = kwargs["mode"]
        except KeyError:
            mode = None

        if mode is None:
            if self.nmodes == 1:
                mode = 0
            else:
                if size == 1:
                    mode = np.argwhere(self.cumweights - random.rng.uniform(0, 1) > 0)[0][0]
                else:
                    # pick modes
                    mode = [
                        np.argwhere(self.cumweights - r > 0)[0][0]
                        for r in random.rng.uniform(0, 1, size)
                    ]

        samps = np.zeros((size, len(self)))
        for i in range(size):
            inbound = False
            while not inbound:
                # sample the multivariate Gaussian keys
                vals = random.rng.uniform(0, 1, len(self))

                if isinstance(mode, list):
                    samp = np.atleast_1d(self.rescale(vals, mode=mode[i]))
                else:
                    samp = np.atleast_1d(self.rescale(vals, mode=mode))
                samps[i, :] = samp

                # check sample is in bounds (otherwise perform another draw)
                outbound = False
                for name, val in zip(self.names, samp):
                    if val < self.bounds[name][0] or val > self.bounds[name][1]:
                        outbound = True
                        break

                if not outbound:
                    inbound = True

        return samps

    def _ln_prob(self, samp, lnprob, outbounds):
        for j in range(samp.shape[0]):
            # loop over the modes and sum the probabilities
            for i in range(self.nmodes):
                # self.mvn[i] is a "standard" multivariate normal distribution; see add_mode()
                z = (samp[j] - self.mus[i]) / self.sigmas[i]
                lnprob[j] = np.logaddexp(lnprob[j], self.mvn[i].logpdf(z) - self.logprodsigmas[i])

        # set out-of-bounds values to -inf
        lnprob[outbounds] = -np.inf
        return lnprob

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        for key in self.__dict__:
            if key == "mvn":
                if len(self.__dict__[key]) != len(other.__dict__[key]):
                    return False
                for thismvn, othermvn in zip(self.__dict__[key], other.__dict__[key]):
                    if not isinstance(
                        thismvn, scipy.stats._multivariate.multivariate_normal_frozen
                    ) or not isinstance(
                        othermvn, scipy.stats._multivariate.multivariate_normal_frozen
                    ):
                        return False
            elif isinstance(self.__dict__[key], (np.ndarray, list)):
                thisarr = np.asarray(self.__dict__[key])
                otherarr = np.asarray(other.__dict__[key])
                if thisarr.dtype == float and otherarr.dtype == float:
                    fin1 = np.isfinite(np.asarray(self.__dict__[key]))
                    fin2 = np.isfinite(np.asarray(other.__dict__[key]))
                    if not np.array_equal(fin1, fin2):
                        return False
                    if not np.allclose(thisarr[fin1], otherarr[fin2], atol=1e-15):
                        return False
                else:
                    if not np.array_equal(thisarr, otherarr):
                        return False
            else:
                if not self.__dict__[key] == other.__dict__[key]:
                    return False
        return True


class MultivariateNormalDist(MultivariateGaussianDist):
    """A synonym for the :class:`~bilby.core.prior.MultivariateGaussianDist` distribution."""


class JointPrior(Prior):
    def __init__(self, dist, name=None, latex_label=None, unit=None):
        """This defines the single parameter Prior object for parameters that belong to a JointPriorDist

        Parameters
        ==========
        dist: ChildClass of BaseJointPriorDist
            The shared JointPriorDistribution that this parameter belongs to
        name: str
            Name of this parameter. Must be contained in dist.names
        latex_label: str
            See superclass
        unit: str
            See superclass
        """
        if not isinstance(dist, BaseJointPriorDist):
            raise TypeError(
                "Must supply a JointPriorDist object instance to be shared by all joint params"
            )

        if name not in dist.names:
            raise ValueError(
                "'{}' is not a parameter in the JointPriorDist".format(name)
            )

        self.dist = dist
        super(JointPrior, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=dist.bounds[name][0],
            maximum=dist.bounds[name][1],
        )

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum
        self.dist.bounds[self.name] = (minimum, self.dist.bounds[self.name][1])

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum
        self.dist.bounds[self.name] = (self.dist.bounds[self.name][0], maximum)

    def rescale(self, val, **kwargs):
        """
        Scale a unit hypercube sample to the prior.

        Parameters
        ==========
        val: array_like
            value drawn from unit hypercube to be rescaled onto the prior
        kwargs: dict
            all kwargs passed to the dist.rescale method
        Returns
        =======
        float:
            A sample from the prior parameter.
        """

        self.dist.rescale_parameters[self.name] = val

        if self.dist.filled_rescale():
            values = np.array(list(self.dist.rescale_parameters.values())).T
            samples = self.dist.rescale(values, **kwargs)
            self.dist.reset_rescale()
            return samples
        else:
            return []  # return empty list

    def sample(self, size=1, **kwargs):
        """
        Draw a sample from the prior.

        Parameters
        ==========
        size: int, float (defaults to 1)
            number of samples to draw
        kwargs: dict
            kwargs passed to the dist.sample method
        Returns
        =======
        float:
            A sample from the prior parameter.
        """

        if self.name in self.dist.sampled_parameters:
            logger.warning(
                "You have already drawn a sample from parameter "
                "'{}'. The same sample will be "
                "returned".format(self.name)
            )

        if len(self.dist.current_sample) == 0:
            # generate a sample
            self.dist.sample(size=size, **kwargs)

        sample = self.dist.current_sample[self.name]

        if self.name not in self.dist.sampled_parameters:
            self.dist.sampled_parameters.append(self.name)

        if len(self.dist.sampled_parameters) == len(self.dist):
            # reset samples
            self.dist.reset_sampled()
        self.least_recently_sampled = sample
        return sample

    def ln_prob(self, val):
        """
        Return the natural logarithm of the prior probability. Note that this
        will not be correctly normalised if there are bounds on the
        distribution.

        Parameters
        ==========
        val: array_like
            value to evaluate the prior log-prob at
        Returns
        =======
        float:
            the logp value for the prior at given sample
        """
        self.dist.requested_parameters[self.name] = val

        if self.dist.filled_request():
            # all required parameters have been set
            values = list(self.dist.requested_parameters.values())

            # check for the same number of values for each parameter
            for i in range(len(self.dist) - 1):
                if isinstance(values[i], (list, np.ndarray)) or isinstance(
                    values[i + 1], (list, np.ndarray)
                ):
                    if isinstance(values[i], (list, np.ndarray)) and isinstance(
                        values[i + 1], (list, np.ndarray)
                    ):
                        if len(values[i]) != len(values[i + 1]):
                            raise ValueError(
                                "Each parameter must have the same "
                                "number of requested values."
                            )
                    else:
                        raise ValueError(
                            "Each parameter must have the same "
                            "number of requested values."
                        )

            lnp = self.dist.ln_prob(np.asarray(values).T)

            # reset the requested parameters
            self.dist.reset_request()
            return lnp
        else:
            # if not all parameters have been requested yet, just return 0
            if isinstance(val, (float, int)):
                return 0.0
            else:
                try:
                    # check value has a length
                    len(val)
                except Exception as e:
                    raise TypeError("Invalid type for ln_prob: {}".format(e))

                if len(val) == 1:
                    return 0.0
                else:
                    return np.zeros_like(val)

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ==========
        val: array_like
            value to evaluate the prior prob at

        Returns
        =======
        float:
            the p value for the prior at given sample
        """

        return np.exp(self.ln_prob(val))


class MultivariateGaussian(JointPrior):
    def __init__(self, dist, name=None, latex_label=None, unit=None):
        if not isinstance(dist, MultivariateGaussianDist):
            raise JointPriorDistError(
                "dist object must be instance of MultivariateGaussianDist"
            )
        super(MultivariateGaussian, self).__init__(
            dist=dist, name=name, latex_label=latex_label, unit=unit
        )


class MultivariateNormal(MultivariateGaussian):
    """A synonym for the :class:`bilby.core.prior.MultivariateGaussian`
    prior distribution."""


class JointPriorDistError(PriorException):
    """Class for Error handling of JointPriorDists for JointPriors"""



class NFDist(BaseJointPriorDist):
    """Class with normalizing flow as distribution for multivariate prior with glasflow"""
    
    def __init__(self, 
                 names,
                 flow_filename: str,
                 use_tilde: bool = False,
                 use_component_masses: bool = False,
                 ):
        
        # Create the bounds here for simplicity since we kind of hard code this prior
        my_bounds = {
            "chirp_mass_source": (1e-2, 5.0),
            "mass_ratio": (0.05, 1.0),
            "lambda_1": (1e-2, 100_000),
            "lambda_2": (1e-2, 100_000),
            "chirp_mass": (1e-2, 5.0),
            "luminosity_distance": (1.0, 100_000.0), # FIXME: this might give unexpected behavior, we assume LVK and BNS, but we need to be more careful
        }
        bounds = [my_bounds[name] for name in names]
        logger.info(f"Using bounds: {bounds} for names: {names}")
        
        super(NFDist, self).__init__(names=names, bounds=bounds)

        self.flow_filename = flow_filename
        self.use_tilde = use_tilde
        self.use_component_masses = use_component_masses
        
        # Check if the filename exists:
        if not os.path.isfile(flow_filename):
            raise FileNotFoundError(f"File {flow_filename} does not exist.")
        
        # Load kwargs
        kwargs_filename = flow_filename.replace(".pt", "_kwargs.json")
        with open(kwargs_filename, "r") as f:
            kwargs = json.load(f)
        self.kwargs = kwargs
        
        self.source_type = kwargs["source_type"]
        
        logger.info("Using glasflow backend")
        
        # Load the scaler
        model_dir = os.path.dirname(flow_filename)
        scaler_path = os.path.join(model_dir, "scaler.gz")
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from {scaler_path}")
            self.scaler: MinMaxScaler = joblib.load(scaler_path)
            
            # Store parameter ranges for clipping
            self.param_min = list(self.scaler.data_min_)
            self.param_max = list(self.scaler.data_max_)
            
            # Compute log Jacobian correction for the MinMaxScaler transformation
            # The Jacobian determinant is ∏_i (1/(x_max_i - x_min_i))
            # So log|J| = ∑_i log(1/(x_max_i - x_min_i)) = -∑_i log(x_max_i - x_min_i)
            param_ranges = [self.param_max[i] - self.param_min[i] for i in range(len(self.names))]
            self.log_jacobian_correction = -np.sum(np.log(param_ranges))
            
            logger.info("Parameter ranges from scaler:")
            for i in range(len(self.names)):
                param_name = self.names[i]
                logger.info(f"  {param_name}: [{self.param_min[i]:.6f}, {self.param_max[i]:.6f}]")
            logger.info(f"Log Jacobian correction: {self.log_jacobian_correction:.6f}")
        else:
            logger.info("No scaler found - assuming input was not scaled during training")
            self.scaler = None
            # Create lists of None values for no clipping
            self.param_min = [None] * len(self.names)
            self.param_max = [None] * len(self.names)
            # No Jacobian correction needed if no scaling
            self.log_jacobian_correction = 0.0

        # Set up glasflow model
        self._setup_glasflow_model()
        
        # Define the n-dimensional standard normal distribution for easier rescaling later on
        names = [f"x{i}" for i in range(1, self.num_vars + 1)]

        mu = [[0.0] * self.num_vars]
        sigmas = [[1.0] * self.num_vars]
        corrcoef = [[[1.0 if i == j else 0.0 for j in range(self.num_vars)] for i in range(self.num_vars)]]

        self.mvg = MultivariateGaussianDist(
            names=names,
            mus=mu,
            corrcoefs=corrcoef,
            sigmas=sigmas,
        )
        
        logger.info(f"Loaded NFDist prior with n_dim = {self.num_vars} from flow_filename = {self.flow_filename}")
        
    def _setup_glasflow_model(self):
        """Set up glasflow model."""
        
        if self.kwargs["model_type"] == "CouplingNSF":
            flow = CouplingNSF(
                n_inputs=self.kwargs["n_inputs"],
                n_transforms=self.kwargs["n_transforms"],
                n_neurons=self.kwargs["n_neurons"],
                n_blocks_per_transform=self.kwargs["n_blocks_per_transform"],
                num_bins=self.kwargs["num_bins"]
            )
        elif self.kwargs["model_type"] == "MaskedAutoregressiveFlow":
            raise NotImplementedError("MaskedAutoregressiveFlow is not implemented in bilby yet.")
            
        elif self.kwargs["model_type"] == "Triangular":
            raise NotImplementedError("Triangular flow is not implemented in bilby yet.")
            
        else:
            raise ValueError(f"Unsupported glasflow model type: {self.kwargs['model_type']}")
        
        # Load model weights with CPU mapping for compatibility
        flow.load_state_dict(torch.load(self.flow_filename, map_location="cpu"))
        self.nf = flow
        self.nf.eval()
        
        # Test sample
        with torch.inference_mode():
            self.nf.sample(100)
    
    def nf_sample_glasflow(self, size):
        """Glasflow sampling function."""
        with torch.inference_mode():
            flow_samp = self.nf.sample(int(size))
            return flow_samp.cpu().numpy()
            
    def nf_ln_prob_glasflow(self, samp):
        """Glasflow log probability function."""
        with torch.inference_mode():
            samp_tensor = torch.tensor(samp, dtype=torch.float32)
            log_probs = self.nf.log_prob(samp_tensor)
            return log_probs.cpu().numpy()
            
    def nf_inverse_glasflow(self, samp):
        """Glasflow inverse function."""
        with torch.inference_mode():
            samp_tensor = torch.tensor(samp, dtype=torch.float32)
            flow_samp, _ = self.nf.inverse(samp_tensor)
            return flow_samp.cpu().numpy()
        
    def clean_samples(self, samp):
        """
        The NF might have some "leakage" into unphysical regions, so we need to clean the samples to ensure they are within the expected bounds. 
        Uses the scaler's parameter ranges if available, otherwise no clipping is applied.
        """
        
        for i in range(self.num_vars):
            samp[:, i] = np.clip(samp[:, i], self.param_min[i], self.param_max[i])
            
        return samp
        
    def _ln_prob(self, samp, lnprob, outbounds):
        
        # TODO: we do outbounds manually here, that is not the best method (but it works for now), change later
        
        # Check for lambda_1 > lambda_2, which is unphysical, and return -inf for those samples
        unphysical_lambda = np.zeros(samp.shape[0], dtype=bool)
        if "lambda_1" in self.names and "lambda_2" in self.names:
            lambda_1_idx = self.names.index("lambda_1")
            lambda_2_idx = self.names.index("lambda_2")
            lambda_1 = samp[:, lambda_1_idx]
            lambda_2 = samp[:, lambda_2_idx]
            unphysical_lambda = lambda_1 > lambda_2 # unphysical if lambda_1 > lambda_2
        
        # Check for lambda_1 < 0 or lambda_2 < 0, which is unphysical, and return -inf for those samples
        if "lambda_1" in self.names:
            lambda_1_idx = self.names.index("lambda_1")
            lambda_1 = samp[:, lambda_1_idx]
            unphysical_lambda |= lambda_1 < 0.0
        
        if "lambda_2" in self.names:
            lambda_2_idx = self.names.index("lambda_2")
            lambda_2 = samp[:, lambda_2_idx]
            unphysical_lambda |= lambda_2 < 0.0
        
        # Check for lambda_tilde < 0 or delta_lambda_tilde < 0, which is unphysical, and return -inf for those samples
        if "lambda_tilde" in self.names:
            lambda_tilde_idx = self.names.index("lambda_tilde")
            lambda_tilde = samp[:, lambda_tilde_idx]
            unphysical_lambda |= lambda_tilde < 0.0
            
        if "delta_lambda_tilde" in self.names:
            delta_lambda_tilde_idx = self.names.index("delta_lambda_tilde")
            delta_lambda_tilde = samp[:, delta_lambda_tilde_idx]
            unphysical_lambda |= delta_lambda_tilde < 0.0
        
        # Ensure the shape is correct
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)
            
        # Use the scaler to transform to the preprocessed space for the NF
        if self.scaler is not None:
            samp = self.scaler.transform(samp)
            
        # Get the log-probability using the glasflow function
        log_probs = self.nf_ln_prob_glasflow(samp)
        
        # Apply Jacobian correction for the rescaling transformation
        log_probs = log_probs + self.log_jacobian_correction
            
        # Set in-bounds values to log_probs, out-of-bounds and unphysical to -inf
        valid_samples = ~outbounds & ~unphysical_lambda
        lnprob[valid_samples] = log_probs[valid_samples]
        lnprob[~valid_samples] = -np.inf
        return lnprob
    
    def _sample(self, size, **kwargs):
        # Generate samples using the glasflow function
        flow_samp = self.nf_sample_glasflow(int(size))
        
        # Rescale the samples with sklearn's MinMaxScaler
        if self.scaler is not None:
            flow_samp = self.scaler.inverse_transform(flow_samp)
        
        # Clean the samples -- return as float64 to not break dynesty
        flow_samp = self.clean_samples(flow_samp)
        
        return flow_samp
    
    def _rescale(self, samp, **kwargs):
        # Rescale them with MultivariateGaussianDist from unit hypercube to Gaussian (base dist)
        mvg_samp = self.mvg.rescale(samp)
        
        # Ensure the shape is correct
        mvg_samp = np.array(mvg_samp)
        if len(mvg_samp.shape) == 1:
            mvg_samp = mvg_samp.reshape(1, self.num_vars)
        
        # Use the glasflow inverse function
        flow_samp = self.nf_inverse_glasflow(mvg_samp)
        
        # Rescale the samples with sklearn's MinMaxScaler
        if self.scaler is not None:
            flow_samp = self.scaler.inverse_transform(flow_samp)
    
        # FIXME: if the NF is trained better, this should no longer be necessary!
        # Clean the samples -- return as float64 to not break dynesty
        flow_samp = self.clean_samples(flow_samp)
    
        return flow_samp


class NFPrior(JointPrior):
    """This is partly inspired by Ann-Kristin Malz's code, glitchflow, available at https://zenodo.org/records/15316399"""
    
    def ln_prob(self, val):
        """
        Return the natural logarithm of the prior probability. Note that this
        will not be correctly normalised if there are bounds on the
        distribution.

        Parameters
        ==========
        val: array_like
            value to evaluate the prior log-prob at
        Returns
        =======
        float:
            the logp value for the prior at given sample
        """
        try:
            val = float(val)
        except Exception as e:
            print(f"At this point, we cannot create a float from {val}, error: {e}")
        self.dist.requested_parameters[self.name] = val

        if self.dist.filled_request():
            # all required parameters have been set
            values = list(self.dist.requested_parameters.values())

            # check for the same number of values for each parameter
            for i in range(len(self.dist) - 1):
                if isinstance(values[i], (list, np.ndarray)) or isinstance(
                    values[i + 1], (list, np.ndarray)
                ):
                    if isinstance(values[i], (list, np.ndarray)) and isinstance(
                        values[i + 1], (list, np.ndarray)
                    ):
                        if len(values[i]) != len(values[i + 1]):
                            raise ValueError(
                                "Each parameter must have the same "
                                "number of requested values."
                            )
                    else:
                        raise ValueError(
                            "Each parameter must have the same "
                            "number of requested values."
                        )

            lnp = np.atleast_1d(self.dist.ln_prob(np.asarray(values).T).squeeze())
            try:
                lnp = float(lnp)
            except Exception as e:
                print(f"Skipped taking the float value from {lnp}, error: {e}")

            # reset the requested parameters
            self.dist.reset_request()
            return lnp
        else:
            # if not all parameters have been requested yet, just return 0
            if isinstance(val, (float, int, np.int32)):
                return 0.0
            else:
                try:
                    # check value has a length
                    len(val)
                except Exception as e:
                    raise TypeError("Invalid type for ln_prob: {}".format(e))

                ret = np.zeros_like(val)
                return ret


class NFDistJester(BaseJointPriorDist):
    """Normalizing flow distribution backed by a jester-trained flowjax model.

    Unlike NFDist (which uses glasflow/PyTorch), this class loads flows trained
    with jester's train_jester_flow pipeline (JAX/flowjax/equinox). The Flow
    object handles all standardization and Jacobian corrections internally.

    Note: rescaling from the unit hypercube (needed by nested samplers such as
    dynesty) is not supported. Use MCMC-based samplers (e.g. bilby's emcee or
    nessai) with this prior.

    Parameters
    ----------
    names : list[str]
        Parameter names. Must match the order in the trained flow's metadata.
    flow_dir : str
        Path to the directory produced by jester's train_jester_flow, containing
        flow_weights.eqx, flow_kwargs.json, and metadata.json.
    bounds : list[tuple] or None
        Parameter bounds as [(min, max), ...]. If None, bounds are inferred from
        the flow's training data statistics (mean ± 5*std for z-score flows, or
        [data_min, data_max] for min-max flows).
    seed : int
        Base seed for the JAX PRNG key. The key is incremented on each call to
        _sample so that successive calls return independent samples.
    """

    def __init__(
        self,
        names: list,
        flow_dir: str,
        bounds=None,
        seed: int = 0,
    ):
        try:
            import jax
            import jax.numpy as jnp
            from jesterTOV.inference.flows import Flow
        except ImportError as e:
            raise ImportError(
                "NFDistJester requires jax and jesterTOV to be installed. "
                f"Original error: {e}"
            )

        if not os.path.isdir(flow_dir):
            raise FileNotFoundError(f"Flow directory does not exist: {flow_dir}")

        logger.info(f"Loading jester flow from {flow_dir}")
        self._flow = Flow.from_directory(flow_dir)

        # Read parameter names from metadata to validate ordering
        import json as _json
        metadata_path = os.path.join(flow_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = _json.load(f)
        flow_param_names = metadata.get("parameter_names", None)
        if flow_param_names is not None and list(names) != list(flow_param_names):
            raise ValueError(
                f"Provided names {names} do not match the flow's parameter names "
                f"{flow_param_names}. The order must match exactly."
            )

        # Infer bounds from flow metadata if not provided
        if bounds is None:
            bounds = self._infer_bounds(metadata)
            logger.info(f"Inferred bounds from flow metadata: {bounds}")

        self.flow_dir = flow_dir
        self.seed = seed

        super(NFDistJester, self).__init__(names=names, bounds=bounds)

        # JAX PRNG state — incremented on each _sample call
        self._jax_key = jax.random.key(seed)
        self._jax = jax
        self._jnp = jnp

        logger.info(
            f"Loaded NFDistJester with n_dim={self.num_vars}, "
            f"standardization={self._flow.standardization_method}"
        )

    def _infer_bounds(self, metadata: dict) -> list:
        """Infer parameter bounds from flow training metadata."""
        import numpy as _np

        if "data_mean" in metadata and "data_std" in metadata:
            # Z-score flow: use mean ± 5*std as generous bounds
            means = _np.array(metadata["data_mean"])
            stds = _np.array(metadata["data_std"])
            return [(float(means[i] - 5 * stds[i]), float(means[i] + 5 * stds[i]))
                    for i in range(len(means))]
        elif "data_bounds_min" in metadata and "data_bounds_max" in metadata:
            # Min-max flow: use training data bounds
            mins = _np.array(metadata["data_bounds_min"])
            maxs = _np.array(metadata["data_bounds_max"])
            return [(float(mins[i]), float(maxs[i])) for i in range(len(mins))]
        else:
            raise ValueError(
                "Cannot infer bounds: metadata missing both "
                "(data_mean, data_std) and (data_bounds_min, data_bounds_max). "
                "Please provide bounds explicitly."
            )

    def _sample(self, size, **kwargs):
        """Draw samples from the flow in original parameter space."""
        # Advance the PRNG key so successive calls are independent
        self._jax_key, subkey = self._jax.random.split(self._jax_key)
        samples = self._flow.sample(subkey, (int(size),))
        return np.array(samples, dtype=np.float64)

    def _ln_prob(self, samp, lnprob, outbounds):
        """Evaluate log probability using the jester flow.

        The flow's log_prob already accounts for the standardization Jacobian.
        Out-of-bounds samples (as determined by the parent class) are set to -inf.
        """
        jnp = self._jnp

        # Reshape if needed (single sample)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        x = jnp.array(samp, dtype=jnp.float64)
        log_probs = np.array(self._flow.log_prob(x), dtype=np.float64)

        lnprob[~outbounds] = log_probs[~outbounds]
        lnprob[outbounds] = -np.inf
        return lnprob

    def _rescale(self, samp, **kwargs):
        raise NotImplementedError(
            "NFDistJester does not support rescaling from the unit hypercube. "
            "Use an MCMC-based sampler (e.g. emcee or nessai) instead of "
            "a nested sampler (dynesty)."
        )


class NFPriorJester(JointPrior):
    """Single-parameter prior for use with NFDistJester.

    Each parameter in the joint flow distribution gets its own NFPriorJester
    instance that shares a single NFDistJester object. Create one instance per
    parameter and pass the same dist object to all of them.

    Example
    -------
    >>> from jesterTOV.inference.flows import Flow
    >>> dist = NFDistJester(
    ...     names=["chirp_mass_source", "mass_ratio", "lambda_1", "lambda_2"],
    ...     flow_dir="/path/to/uniform_radio/",
    ... )
    >>> priors = {name: NFPriorJester(dist=dist, name=name) for name in dist.names}
    """

    def ln_prob(self, val):
        """Return the log prior probability for this parameter.

        Accumulates values across all parameters in the joint distribution and
        evaluates once all have been provided (bilby's joint-prior protocol).
        """
        try:
            val = float(val)
        except Exception as e:
            logger.debug(f"Could not cast {val} to float in NFPriorJester.ln_prob: {e}")

        self.dist.requested_parameters[self.name] = val

        if self.dist.filled_request():
            values = list(self.dist.requested_parameters.values())

            # Validate consistent lengths for array-valued inputs
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
                lnp = float(lnp)
            except Exception as e:
                logger.debug(f"Could not cast lnp to float in NFPriorJester: {e}")

            self.dist.reset_request()
            return lnp
        else:
            # Not all parameters have been set yet — return 0 as placeholder
            if isinstance(val, (float, int, np.int32)):
                return 0.0
            else:
                try:
                    len(val)
                except Exception as e:
                    raise TypeError("Invalid type for ln_prob: {}".format(e))
                return np.zeros_like(val)