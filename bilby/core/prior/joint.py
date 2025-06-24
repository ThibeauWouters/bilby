import re
import os

import numpy as np
import scipy.stats
from scipy.special import erfinv

from .base import Prior, PriorException
from ..utils import logger, infer_args_from_method, get_dict_with_properties
from ..utils import random

### glasflow
import json
from glasflow.flows import RealNVP
from glasflow.flows.nsf import CouplingNSF
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib

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
                 flow_filename: str):
        
        super(NFDist, self).__init__(names=names)
        self.flow_filename = flow_filename
        
        # Check if the filename exists:
        if not os.path.isfile(flow_filename):
            raise FileNotFoundError(f"File {flow_filename} does not exist.")
        
        kwargs_filename = flow_filename.replace(".pt", "_kwargs.json")
        with open(kwargs_filename, "r") as f:
            kwargs = json.load(f)
            
        flow = CouplingNSF(n_inputs=self.num_vars,
                           n_transforms=kwargs["n_transforms"],
                           n_neurons=kwargs["n_neurons"],
                           n_blocks_per_transform=kwargs["n_blocks_per_transform"]
        )
        
        # Load the scaler:
        scaler_name = flow_filename.replace(".pt", "_scaler.gz")
        self.scaler: MinMaxScaler = joblib.load(scaler_name)

        # Load model weights
        flow.load_state_dict(torch.load(flow_filename))
        self.nf = flow
        self.nf.eval()
        self.nf.compile()
        # Test sample
        with torch.inference_mode():
            self.nf.sample(100)
        
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
        
    def clean_samples(self, samp):
        """
        Sometimes the NF seemingly returns something slightly unphysical. Need to clip it or change some
        """
        
        # First, fix the masses: make sure m1 > m2:
        m1_samp = samp[:, 0]
        m2_samp = samp[:, 1]
        
        # Make sure the masses are not too crazy out of the usual training bounds
        m1_samp = np.clip(m1_samp, 0.5, 10.0)
        m2_samp = np.clip(m2_samp, 0.5, 10.0)
        
        m1 = np.maximum(m1_samp, m2_samp)
        m2 = np.minimum(m1_samp, m2_samp)
        
        # Make sure lambdas are OK, after which we rebuild samp per dimensional case
        if self.num_vars == 3:
            lambda_2_samp = samp[:, 2]
            lambda_2 = np.clip(lambda_2_samp, 0.0, None)
            samp = np.column_stack((m1, m2, lambda_2))
        
        else:
            lambda_1_samp = samp[:, 2]
            lambda_2_samp = samp[:, 3]
            
            lambda_1_samp = np.clip(lambda_1_samp, 0.0, None)
            lambda_2_samp = np.clip(lambda_2_samp, 0.0, None)
            
            # Make sure lambda_2 > lambda_1 for the 4D case
            lambda_1 = np.minimum(lambda_1_samp, lambda_2_samp)
            lambda_2 = np.maximum(lambda_1_samp, lambda_2_samp)
            
            samp = np.column_stack((m1, m2, lambda_1, lambda_2))
            
        return samp
        
    def _ln_prob(self, samp, lnprob, outbounds):
        with torch.inference_mode():
            # Ensure the shape is correct
            if len(samp.shape) == 1:
                samp = samp.reshape(1, self.num_vars)
                
            # Use the scaler to transform to the preprocessed space for the NF
            samp = self.scaler.transform(samp)
                
            # Get the log-probability of the sample, passing to Torch tensor first
            samp = torch.tensor(samp, dtype=torch.float32)
            log_probs = self.nf.log_prob(samp)
            log_probs = np.atleast_2d(log_probs.cpu().numpy())
        
        return log_probs
    
    def _sample(self, size, **kwargs):
        with torch.inference_mode():
            flow_samp = self.nf.sample(size)
            flow_samp = flow_samp.cpu().numpy()
            
            # Rescale the samples with sklearn's MinMaxScaler
            flow_samp = self.scaler.inverse_transform(flow_samp)
            
            # Clean the samples -- return as float64 to not break dynesty
            flow_samp = self.clean_samples(flow_samp) # .astype(np.float32)
            
        return flow_samp
    
    def _rescale(self, samp, **kwargs):
        with torch.inference_mode():
            # Rescale them with MultivariateGaussianDist from unit hypercube to Gaussian (base dist)
            mvg_samp = self.mvg.rescale(samp)
            
            # Ensure the shape is correct
            mvg_samp = np.array(mvg_samp)
            if len(mvg_samp.shape) == 1:
                mvg_samp = mvg_samp.reshape(1, self.num_vars)
            
            # Then use the flow map to transform 
            mvg_samp = torch.tensor(mvg_samp, dtype=torch.float32)

            # Pass through the normalizing flow (note: inverse outputs something in data space!) -- this returns samples and log determinant Jacobian, but ignore the latter here
            flow_samp, _ = self.nf.inverse(mvg_samp)

            # Convert the result back to NumPy if needed
            flow_samp = flow_samp.cpu().numpy()
            
            # Rescale the samples with sklearn's MinMaxScaler
            flow_samp = self.scaler.inverse_transform(flow_samp)
        
            # Clean the samples -- return as float64 to not break dynesty
            flow_samp = self.clean_samples(flow_samp) # .astype(np.float32)
        
        return flow_samp

class NFPrior(JointPrior):
    """This is taken from Ann-Kristin Malz's code, glitchflow, available at https://zenodo.org/records/15316399"""
    
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
        val = float(val)
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
            lnp = float(lnp)

            # reset the requested parameters
            self.dist.reset_request()
            return lnp
        else:
            # if not all parameters have been requested yet, just return 0
            if isinstance(val, (float, int)):
                # return np.array([0.0]) # this is for nessai
                return 0.0
            else:

                try:
                    # check value has a length
                    len(val)
                except Exception as e:
                    raise TypeError("Invalid type for ln_prob: {}".format(e))

                ret = np.zeros_like(val)
                return ret


class NFDistConditional(BaseJointPriorDist):
    """
    Conditional normalizing flow distribution for gravitational wave inference.
    
    This class implements p(chirp_mass, mass_ratio, luminosity_distance, lambda_1, lambda_2) where:
    - chirp_mass, mass_ratio, luminosity_distance are sampled from base priors (using bilby conventions)
    - lambda_1, lambda_2 are sampled from conditional NF: p(lambda_1, lambda_2 | m_1_source, m_2_source)
    
    The workflow is:
    1. Sample chirp_mass, mass_ratio, luminosity_distance from base priors
    2. Convert luminosity_distance to redshift using bilby.gw.conversion.luminosity_distance_to_redshift
    3. Convert detector frame chirp_mass to source frame: Mc_source = Mc_detector / (1 + z)
    4. Convert chirp_mass_source, mass_ratio to component masses using bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses
    5. Use conditional NF to sample lambda_1, lambda_2 given m_1_source, m_2_source
    """
    
    def __init__(self, 
                 names: list[str],
                 flow_filename: str,
                 Mc_bounds: tuple = (1.0, 3.0),
                 q_bounds: tuple = (0.125, 1.0),
                 dL_bounds: tuple = (1.0, 500.0)):
        """
        Initialize the conditional normalizing flow distribution.

        Args:
            names (list[str]): Names of the parameters in the distribution. # FIXME: redundant, remove this
            1. chirp_mass
            2. mass_ratio
            3. luminosity_distance
            4. lambda_1
            5. lambda_2
            flow_filename (str): _description_
            Mc_bounds (tuple, optional): Bounds for uniform chirp mass prior. Defaults to (1.0, 3.0).
            q_bounds (tuple, optional): Bounds for uniform mass range prior. Defaults to (0.125, 1.0).
            dL_bounds (tuple, optional): Bounds for UniformComovingVolume distance prior. Defaults to (1.0, 500.0).
        """
        
        from .analytical import Uniform
        from bilby.gw.prior import UniformComovingVolume
        
        names = ["chirp_mass", "mass_ratio", "luminosity_distance", "lambda_1", "lambda_2"]
        super(NFDistConditional, self).__init__(names=names)
        self.flow_filename = flow_filename
        
        # Load the conditional flow model
        self.nf = self._load_conditional_flow_model(flow_filename, device='cpu')
        
        # Initialize base priors using bilby's GW-specific priors
        self.base_priors = {}
        self.base_priors["chirp_mass"] = Uniform(minimum=Mc_bounds[0], maximum=Mc_bounds[1], name="chirp_mass", latex_label='$M_c$')
        self.base_priors["mass_ratio"] = Uniform(minimum=q_bounds[0], maximum=q_bounds[1], name="mass_ratio", latex_label='$q$') 
        self.base_priors["luminosity_distance"] = UniformComovingVolume(minimum=dL_bounds[0], maximum=dL_bounds[1], name='luminosity_distance', latex_label='$D_L$')
        
        # Define the 2-dimensional standard normal distribution for rescaling lambda parameters only
        mvg_names = ["lambda_1", "lambda_2"]
        mu = [[0.0, 0.0]]
        sigmas = [[1.0, 1.0]]
        corrcoef = [[[1.0, 0.0], [0.0, 1.0]]]

        self.mvg_lambda = MultivariateGaussianDist(
            names=mvg_names,
            mus=mu,
            corrcoefs=corrcoef,
            sigmas=sigmas,
        )
        
        # Initialize storage for rescaled results coordination
        self.rescaled_results = {}
        
        logger.info(f"Loaded conditional NFDist prior with n_dim = {self.num_vars} from flow_filename = {self.flow_filename}")
        
    def _load_conditional_flow_model(self, flow_filename: str, device: str='cpu'):
        """
        Load and configure the conditional normalizing flow model.
        
        Parameters
        ==========
        flow_filename : str
            Path to the conditional flow model (.pt file)
        device : str, optional
            Device to load the model on ('cpu' or 'cuda'). Default is 'cpu'.
            
        Returns
        =======
        flow : RealNVP
            Loaded and configured conditional normalizing flow model
        """
        # Check if the filename exists
        if not os.path.isfile(flow_filename):
            raise FileNotFoundError(f"File {flow_filename} does not exist.")
        
        # Load model configuration
        kwargs_filename = flow_filename.replace(".pt", "_kwargs.json")
        with open(kwargs_filename, "r") as f:
            kwargs = json.load(f)
        
        # Verify this is a conditional model (2 inputs for masses, 2 outputs for Lambdas)
        if kwargs.get("names") != ["lambda_1", "lambda_2"]:
            raise ValueError(f"Expected conditional model with ['lambda_1', 'lambda_2'], got {kwargs.get('names')}")
        if kwargs.get("names_conditional") != ["m_1", "m_2"]:
            raise ValueError(f"Expected conditional inputs ['m_1', 'm_2'], got {kwargs.get('names_conditional')}")
            
        # Create conditional NF for lambda_1, lambda_2 (2D output, 2D conditioning)
        flow = RealNVP(n_inputs=2,  # lambda_1, lambda_2
                       n_conditional_inputs=2,  # m_1_source, m_2_source
                       n_transforms=kwargs["n_transforms"],
                       n_neurons=kwargs["n_neurons"],
                       n_blocks_per_transform=kwargs["n_blocks_per_transform"],
                       batch_norm_between_transforms=True,
        )

        # Load model weights on specified device
        flow.load_state_dict(torch.load(flow_filename, map_location=torch.device(device)))
        flow.to(device)
        flow.eval()
        flow.compile()
        
        return flow
        
    def _convert_to_source_masses(self, chirp_mass_det, mass_ratio, luminosity_distance):
        """
        Convert detector frame parameters to source frame component masses.
        Now fully vectorized to handle arrays efficiently.
        
        Parameters
        ==========
        chirp_mass_det: float or array
            Detector frame chirp mass
        mass_ratio: float or array
            Mass ratio q = m2/m1
        luminosity_distance: float or array
            Luminosity distance in Mpc
            
        Returns
        =======
        m1_source: float or array
            Source frame mass of primary
        m2_source: float or array  
            Source frame mass of secondary
        """
        from bilby.gw.conversion import (luminosity_distance_to_redshift, 
                                         chirp_mass_and_mass_ratio_to_component_masses)
        
        # Convert luminosity distance to redshift (handles arrays)
        redshift = luminosity_distance_to_redshift(luminosity_distance)
        
        # Convert detector frame chirp mass to source frame (vectorized)
        chirp_mass_source = chirp_mass_det / (1.0 + redshift)
        
        # Convert source frame chirp mass and mass ratio to component masses (handles arrays)
        m1_source, m2_source = chirp_mass_and_mass_ratio_to_component_masses(
            chirp_mass_source, mass_ratio)
            
        return m1_source, m2_source
        
    def _ln_prob(self, samp, lnprob, outbounds):
        """
        Calculate log probability for conditional prior.
        Fully vectorized implementation for improved performance.
        
        Expected sample format: [chirp_mass, mass_ratio, luminosity_distance, lambda_1, lambda_2]
        """
        # Ensure the shape is correct
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)
            
        n_samples = samp.shape[0]
        
        # Set out-of-bounds samples to -inf
        lnprob[outbounds] = -np.inf
        
        # Only process in-bounds samples
        inbounds_mask = ~outbounds
        if not np.any(inbounds_mask):
            return lnprob
            
        inbounds_samp = samp[inbounds_mask]
        
        # Extract base parameters and lambdas (vectorized)
        chirp_mass = inbounds_samp[:, 0]
        mass_ratio = inbounds_samp[:, 1] 
        luminosity_distance = inbounds_samp[:, 2]
        lambda_1 = inbounds_samp[:, 3]
        lambda_2 = inbounds_samp[:, 4]
        
        # Calculate base prior log probabilities (vectorized)
        base_ln_prob = np.zeros(len(inbounds_samp))
        for i, (mc, q, dl) in enumerate(zip(chirp_mass, mass_ratio, luminosity_distance)):
            base_ln_prob[i] = (self.base_priors["chirp_mass"].ln_prob(mc) +
                              self.base_priors["mass_ratio"].ln_prob(q) +
                              self.base_priors["luminosity_distance"].ln_prob(dl))
        
        # Convert to source frame masses for conditioning (vectorized)
        m1_source, m2_source = self._convert_to_source_masses(chirp_mass, mass_ratio, luminosity_distance)
            
        # Calculate conditional NF log probability (batched)
        with torch.inference_mode():
            lambda_tensor = torch.tensor(np.column_stack([lambda_1, lambda_2]), dtype=torch.float32)
            condition_tensor = torch.tensor(np.column_stack([m1_source, m2_source]), dtype=torch.float32)
            nf_ln_prob = self.nf.log_prob(lambda_tensor, conditional=condition_tensor).cpu().numpy()
        
        # Total log probability is sum of base priors and conditional NF (vectorized)
        total_ln_prob = base_ln_prob + nf_ln_prob
        
        # Store results back to the full lnprob array
        lnprob[inbounds_mask] = total_ln_prob
        
        return lnprob
    
    def _sample(self, size, **kwargs):
        """
        Hierarchical sampling: base priors -> mass conversion -> conditional NF
        
        Returns samples in format: [chirp_mass, mass_ratio, luminosity_distance, lambda_1, lambda_2]
        """
        samples = np.zeros((size, self.num_vars))
        
        for i in range(size):
            # Step 1: Sample base parameters
            chirp_mass = self.base_priors["chirp_mass"].sample()
            mass_ratio = self.base_priors["mass_ratio"].sample()
            luminosity_distance = self.base_priors["luminosity_distance"].sample()
            
            # Step 2: Convert to source frame masses
            m1_source, m2_source = self._convert_to_source_masses(chirp_mass, mass_ratio, luminosity_distance)
                
            # Step 3: Sample lambdas from conditional NF
            with torch.inference_mode():
                condition_tensor = torch.tensor([[m1_source, m2_source]], dtype=torch.float32)
                lambda_samples = self.nf.sample(1, conditional=condition_tensor).cpu().numpy()[0]
                lambda_1, lambda_2 = lambda_samples[0], lambda_samples[1]
                
                # Ensure physical values
                lambda_1 = max(0.0, lambda_1)
                lambda_2 = max(0.0, lambda_2)
                
            # Store sample
            samples[i] = [chirp_mass, mass_ratio, luminosity_distance, lambda_1, lambda_2]
            
        return samples
    
    def _rescale(self, samp, **kwargs):
        """
        Rescale from unit hypercube to conditional prior.
        Fully vectorized implementation for improved performance.
        
        Input: unit hypercube samples [0,1]^5
        Output: [chirp_mass, mass_ratio, luminosity_distance, lambda_1, lambda_2]
        """
        # Ensure the shape is correct
        samp = np.array(samp)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)
            
        n_samples = samp.shape[0]
        samples = np.zeros_like(samp)
        
        # Step 1: Vectorized rescaling of base parameters from unit hypercube
        chirp_mass = np.array([self.base_priors["chirp_mass"].rescale(u) for u in samp[:, 0]])
        mass_ratio = np.array([self.base_priors["mass_ratio"].rescale(u) for u in samp[:, 1]])
        luminosity_distance = np.array([self.base_priors["luminosity_distance"].rescale(u) for u in samp[:, 2]])
        
        # Step 2: Vectorized conversion to source frame masses for conditioning
        m1_source, m2_source = self._convert_to_source_masses(chirp_mass, mass_ratio, luminosity_distance)
            
        # Step 3: Vectorized rescaling of lambda parameters using conditional NF
        with torch.inference_mode():
            # First rescale all lambda unit samples to Gaussian space in batch
            lambda_gaussian_samples = np.array([self.mvg_lambda.rescale(samp[i, 3:5]) for i in range(n_samples)])
            lambda_gaussian_tensor = torch.tensor(lambda_gaussian_samples, dtype=torch.float32)
            
            # Then use conditional NF inverse transform in batch
            condition_tensor = torch.tensor(np.column_stack([m1_source, m2_source]), dtype=torch.float32)
            lambda_samples, _ = self.nf.inverse(lambda_gaussian_tensor, conditional=condition_tensor)
            lambda_samples = lambda_samples.cpu().numpy()
            
            # Ensure physical values (vectorized clipping)
            lambda_1 = np.maximum(0.0, lambda_samples[:, 0])
            lambda_2 = np.maximum(0.0, lambda_samples[:, 1])
        
        # Store rescaled samples (vectorized assignment)
        samples[:, 0] = chirp_mass
        samples[:, 1] = mass_ratio
        samples[:, 2] = luminosity_distance
        samples[:, 3] = lambda_1
        samples[:, 4] = lambda_2
            
        return np.squeeze(samples)


class NFPriorConditional(JointPrior):
    """This is inspired by Ann-Kristin Malz's code, glitchflow, available at https://zenodo.org/records/15316399"""
    
    def rescale(self, val, **kwargs):
        """
        Scale a unit hypercube sample to the prior.
        For conditional priors, we need to handle the case where individual parameters
        are being rescaled, which requires coordination through the joint distribution.
        """
        # Store this parameter's value for joint rescaling
        self.dist.rescale_parameters[self.name] = val
        
        # If all parameters have been set, do the joint rescaling
        if self.dist.filled_rescale():
            # Get all parameter values in correct order
            param_values = []
            for param_name in self.dist.names:
                param_values.append(self.dist.rescale_parameters[param_name])
            
            # Do joint rescaling
            joint_rescaled = self.dist.rescale(np.array(param_values))
            
            # Store results for each parameter
            for i, param_name in enumerate(self.dist.names):
                self.dist.rescaled_results[param_name] = joint_rescaled[i]
            
            # Reset rescale parameters for next time
            self.dist.reset_rescale()
            
            # Return this parameter's result
            return self.dist.rescaled_results[self.name]
        else:
            # Not all parameters set yet, return None or some placeholder
            # This will be handled by bilby's sampling logic
            return None
    
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
        val = float(val)
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
            lnp = float(lnp)
            
            # reset the requested parameters
            self.dist.reset_request()
            return lnp
        else:
            # if not all parameters have been requested yet, just return 0
            if isinstance(val, (float, int)):
                # return np.array([0.0]) # this is for nessai
                return 0.0
            else:

                try:
                    # check value has a length
                    len(val)
                except Exception as e:
                    raise TypeError("Invalid type for ln_prob: {}".format(e))

                ret = np.zeros_like(val)
                return ret


class ConditionalGWPriorLoader:
    """
    Hybrid conditional/individual prior system for gravitational wave inference.
    All conditional parameter bounds must be specified in the prior file which is passed to this class to properly handle the conditional normalizing flow.
    
    This class combines:
    - Conditional NF for 5 parameters: ["chirp_mass", "mass_ratio", "luminosity_distance", "lambda_1", "lambda_2"]
    - Individual priors for all other parameters from a bilby prior file
    
    Usage:
        loader = ConditionalGWPriorLoader(
            prior_file_path="GW190425/common.prior",
            nf_model_path="conditional_bns/model.pt"
        )
        
        # Example prior file must contain conditional parameters:
        # chirp_mass = Uniform(minimum=1.485, maximum=1.490, name="chirp_mass")
        # mass_ratio = Uniform(minimum=0.25, maximum=1.0, name="mass_ratio") 
        # luminosity_distance = UniformComovingVolume(minimum=1.0, maximum=500.0, name='luminosity_distance')
        # geocent_time = Uniform(minimum=1240215503.017147-0.1, maximum=1240215503.017147+0.1, name='geocent_time')
        # a_1 = Uniform(minimum=0.0, maximum=0.05, name='a_1')
        # ...
        
        # Bilby integration
        def prior_transform(unit_cube): return loader.rescale(unit_cube)
        def log_prior(parameters): return loader.ln_prob(parameters)
    """
    
    def __init__(self, prior_file_path, nf_model_path):
        """
        Initialize the hybrid conditional/individual prior system.
        
        Parameters:
        -----------
        prior_file_path : str
            Path to bilby prior file (*.prior) containing ALL parameter priors.
            Must include conditional parameters: chirp_mass, mass_ratio, luminosity_distance
            Example conditional parameter definitions:
            - chirp_mass = Uniform(minimum=1.485, maximum=1.490, name="chirp_mass")
            - mass_ratio = Uniform(minimum=0.25, maximum=1.0, name="mass_ratio")
            - luminosity_distance = UniformComovingVolume(minimum=1.0, maximum=500.0, name='luminosity_distance')
        nf_model_path : str
            Path to conditional normalizing flow model (.pt file)
        
        Raises:
        -------
        ValueError
            If any conditional parameters are missing from the prior file
        """
        # import importlib.util # TODO: remove me if OK without it
        from bilby.core.prior import PriorDict
        
        self.prior_file_path = prior_file_path
        self.nf_model_path = nf_model_path
        
        # Conditional parameters (handled by normalizing flow)
        self.conditional_params = ["chirp_mass", "mass_ratio", "luminosity_distance", "lambda_1", "lambda_2"]
        
        # Parse prior file and extract individual priors
        self.individual_priors, self.conditional_bounds = self._parse_prior_file()
        
        # Validate that all conditional parameters have bounds from prior file
        self._validate_conditional_bounds()
        
        # Create conditional NF distribution
        self.conditional_dist = NFDistConditional(
            names=self.conditional_params,
            flow_filename=nf_model_path,
            Mc_bounds=self.conditional_bounds.get("chirp_mass"),
            q_bounds=self.conditional_bounds.get("mass_ratio"),
            dL_bounds=self.conditional_bounds.get("luminosity_distance")
        )
        
        # Create PriorDict for individual parameters
        self.individual_prior_dict = PriorDict(self.individual_priors)
        
        # Total parameter count and ordering
        self.all_param_names = self.conditional_params + list(self.individual_priors.keys())
        self.n_conditional = len(self.conditional_params)
        self.n_individual = len(self.individual_priors)
        self.n_total = self.n_conditional + self.n_individual
        
        print(f"ConditionalGWPriorLoader initialized:")
        print(f"  Conditional parameters ({self.n_conditional}): {self.conditional_params}")
        print(f"  Individual parameters ({self.n_individual}): {list(self.individual_priors.keys())}")
        print(f"  Total parameters: {self.n_total}")
    
    def _parse_prior_file(self):
        """
        Safely parse bilby prior file and extract individual priors.
        
        Returns:
        --------
        individual_priors : dict
            Dictionary of individual prior objects (not conditional parameters)
        conditional_bounds : dict
            Extracted bounds for conditional parameters found in prior file
        """
        import numpy as np
        from bilby.core.prior.analytical import Uniform, Sine, Cosine
        from bilby.gw.prior import UniformComovingVolume
        
        # Create safe namespace for executing prior file
        safe_globals = {
            '__builtins__': {
                'abs': abs, 'min': min, 'max': max, 'round': round,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sum': sum,
                'any': any, 'all': all, 'sorted': sorted, 'list': list,
                'dict': dict, 'tuple': tuple, 'set': set
            },
            'np': np,
            'Uniform': Uniform,
            'Sine': Sine, 
            'Cosine': Cosine,
            'UniformComovingVolume': UniformComovingVolume
        }
        
        # Execute prior file
        local_vars = {}
        try:
            with open(self.prior_file_path, 'r') as f:
                prior_code = f.read()
            exec(prior_code, safe_globals, local_vars)
        except Exception as e:
            raise RuntimeError(f"Failed to parse prior file {self.prior_file_path}: {e}")
        
        # Extract priors and bounds
        individual_priors = {}
        conditional_bounds = {}
        
        for var_name, var_value in local_vars.items():
            if hasattr(var_value, 'minimum') and hasattr(var_value, 'maximum'):
                # This is a prior object
                if var_name in self.conditional_params:
                    # Extract bounds for conditional parameters
                    conditional_bounds[var_name] = (var_value.minimum, var_value.maximum)
                else:
                    # Keep as individual prior
                    individual_priors[var_name] = var_value
        
        return individual_priors, conditional_bounds
    
    def _validate_conditional_bounds(self):
        """Ensure all conditional parameters have bounds defined in the prior file."""
        
        # Required conditional parameters (lambda_1, lambda_2 are generated by NF, no bounds needed)
        required_params = ["chirp_mass", "mass_ratio", "luminosity_distance"]
        missing_params = []
        
        for param in required_params:
            if param not in self.conditional_bounds:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(
                f"Missing conditional parameters in prior file: {missing_params}\n"
                f"The prior file must contain bounds for all conditional parameters.\n"
                f"Example definitions needed:\n"
                f"  chirp_mass = Uniform(minimum=1.485, maximum=1.490, name='chirp_mass')\n"
                f"  mass_ratio = Uniform(minimum=0.25, maximum=1.0, name='mass_ratio')\n"
                f"  luminosity_distance = UniformComovingVolume(minimum=1.0, maximum=500.0, name='luminosity_distance')"
            )
    
    def rescale(self, unit_cube):
        """
        Transform from unit hypercube to parameter space.
        
        Parameters:
        -----------
        unit_cube : array_like
            Array of shape (n_total,) or (n_samples, n_total) with values in [0,1]
        
        Returns:
        --------
        parameters : np.ndarray
            Rescaled parameters with shape matching input
        """
        unit_cube = np.asarray(unit_cube)
        
        # Handle both single sample and batch
        if unit_cube.ndim == 1:
            unit_cube = unit_cube.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        n_samples = unit_cube.shape[0]
        
        # Split unit cube: first n_conditional for conditional, rest for individual
        conditional_unit = unit_cube[:, :self.n_conditional]
        individual_unit = unit_cube[:, self.n_conditional:]
        
        # Rescale conditional parameters
        conditional_rescaled = self.conditional_dist.rescale(conditional_unit)
        
        # Ensure conditional_rescaled is 2D
        if conditional_rescaled.ndim == 1:
            conditional_rescaled = conditional_rescaled.reshape(1, -1)
        
        # Rescale individual parameters
        if self.n_individual > 0:
            individual_rescaled = np.zeros((n_samples, self.n_individual))
            individual_param_names = list(self.individual_priors.keys())
            
            for i, sample_unit in enumerate(individual_unit):
                individual_list = self.individual_prior_dict.rescale(individual_param_names, sample_unit)
                individual_rescaled[i] = individual_list
        else:
            individual_rescaled = np.empty((n_samples, 0))
        
        # Combine results
        all_rescaled = np.column_stack([conditional_rescaled, individual_rescaled])
        
        # Return original shape
        if single_sample:
            return all_rescaled.squeeze(0)
        else:
            return all_rescaled
    
    def ln_prob(self, parameters):
        """
        Evaluate log probability for parameters.
        
        Parameters:
        -----------
        parameters : array_like
            Parameters with shape (n_total,) or (n_samples, n_total)
        
        Returns:
        --------
        ln_prob : float or np.ndarray
            Log probability values
        """
        parameters = np.asarray(parameters)
        
        # Handle both single sample and batch
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        n_samples = parameters.shape[0]
        
        # Split parameters: first n_conditional for conditional, rest for individual
        conditional_params = parameters[:, :self.n_conditional]
        individual_params = parameters[:, self.n_conditional:]
        
        # Evaluate conditional log probability
        conditional_ln_prob = self.conditional_dist.ln_prob(conditional_params)
        
        # Evaluate individual log probabilities
        if self.n_individual > 0:
            individual_ln_prob = np.zeros(n_samples)
            individual_param_names = list(self.individual_priors.keys())
            
            for i, sample_params in enumerate(individual_params):
                individual_dict = {name: sample_params[j] for j, name in enumerate(individual_param_names)}
                individual_ln_prob[i] = self.individual_prior_dict.ln_prob(individual_dict)
        else:
            individual_ln_prob = np.zeros(n_samples)
        
        # Sum log probabilities
        total_ln_prob = conditional_ln_prob + individual_ln_prob
        
        # Return original shape
        if single_sample:
            return float(total_ln_prob.squeeze(0))
        else:
            return total_ln_prob
    
    def sample(self, size=1):
        """
        Generate samples from the hybrid prior.
        
        Parameters:
        -----------
        size : int
            Number of samples to generate
        
        Returns:
        --------
        samples : dict
            Dictionary with parameter names as keys and sample arrays as values
        """
        # Sample from conditional distribution
        self.conditional_dist.sample(size=size)
        conditional_samples = self.conditional_dist.current_sample
        
        # Sample from individual priors
        if self.n_individual > 0:
            individual_samples = self.individual_prior_dict.sample(size)
        else:
            individual_samples = {}
        
        # Combine samples
        all_samples = {**conditional_samples, **individual_samples}
        
        return all_samples


class ConditionalPriorDict:
    """
    A PriorDict-compatible wrapper for ConditionalGWPriorLoader.
    This allows the conditional prior system to work seamlessly with bilby's sampling.
    This is just a wrapper around ConditionalGWPriorLoader to make it compatible with bilby's prior system.
    """
    
    def __init__(self, loader):
        self.loader = loader
        self.keys = loader.all_param_names
        
    def rescale(self, keys, unit_cube):
        """Rescale from unit hypercube to parameter space"""
        return self.loader.rescale(unit_cube)
        
    def ln_prob(self, parameters_dict):
        """Evaluate log probability for parameters"""
        # Extract parameter values in the correct order
        param_array = [parameters_dict[key] for key in self.keys]
        return self.loader.ln_prob(param_array)
        
    def __len__(self):
        """Return total number of parameters"""
        return self.loader.n_total
        
    def __getitem__(self, key):
        """Allow bilby to access individual priors if needed"""
        return self
        
    def sample(self, size=1):
        """Generate samples from the hybrid prior"""
        return self.loader.sample(size)