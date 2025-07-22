import json
import os
import re
from importlib import import_module
from io import open as ioopen

import numpy as np

from .analytical import DeltaFunction
from .base import Prior, Constraint
from .joint import JointPrior, BaseJointPriorDist
from ..utils import (
    logger,
    check_directory_exists_and_if_not_mkdir,
    BilbyJsonEncoder,
    decode_bilby_json,
)

# TODO: This is for the NF priors -- might have to replace this in a separate file, for now, keep it here
from bilby.gw.conversion import (
            luminosity_distance_to_redshift,
            chirp_mass_and_mass_ratio_to_component_masses
        )
import torch
from scipy.stats import norm
from glasflow.flows import RealNVP
from glasflow.flows.autoregressive import MaskedAffineAutoregressiveFlow

class PriorDict(dict):
    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """A dictionary of priors

        Parameters
        ==========
        dictionary: Union[dict, str, None]
            If given, a dictionary to generate the prior set.
        filename: Union[str, None]
            If given, a file containing the prior to generate the prior set.
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            Default is no conversion.
        """
        super(PriorDict, self).__init__()
        if isinstance(dictionary, dict):
            self.from_dictionary(dictionary)
        elif type(dictionary) is str:
            logger.debug(
                'Argument "dictionary" is a string.'
                + " Assuming it is intended as a file name."
            )
            self.from_file(dictionary)
        elif type(filename) is str:
            self.from_file(filename)
        elif dictionary is not None:
            raise ValueError("PriorDict input dictionary not understood")
        self._cached_normalizations = {}

        self.convert_floats_to_delta_functions()

        if conversion_function is not None:
            self.conversion_function = conversion_function
        else:
            self.conversion_function = self.default_conversion_function

    def evaluate_constraints(self, sample):
        out_sample = self.conversion_function(sample)
        prob = 1
        for key in self:
            if isinstance(self[key], Constraint) and key in out_sample:
                prob *= self[key].prob(out_sample[key])
        return prob

    def default_conversion_function(self, sample):
        """
        Placeholder parameter conversion function.

        Parameters
        ==========
        sample: dict
            Dictionary to convert

        Returns
        =======
        sample: dict
            Same as input
        """
        return sample

    def to_file(self, outdir, label):
        """Write the prior distribution to file.

        Parameters
        ==========
        outdir: str
            output directory name
        label: str
            Output file naming scheme
        """

        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}.prior".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        joint_dists = []
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                if JointPrior in self[key].__class__.__mro__:
                    distname = "_".join(self[key].dist.names) + "_{}".format(
                        self[key].dist.distname
                    )
                    if distname not in joint_dists:
                        joint_dists.append(distname)
                        outfile.write("{} = {}\n".format(distname, self[key].dist))
                    diststr = repr(self[key].dist)
                    priorstr = repr(self[key])
                    outfile.write(
                        "{} = {}\n".format(key, priorstr.replace(diststr, distname))
                    )
                else:
                    outfile.write("{} = {}\n".format(key, self[key]))

    def _get_json_dict(self):
        self.convert_floats_to_delta_functions()
        total_dict = {key: json.loads(self[key].to_json()) for key in self}
        total_dict["__prior_dict__"] = True
        total_dict["__module__"] = self.__module__
        total_dict["__name__"] = self.__class__.__name__
        return total_dict

    def to_json(self, outdir, label):
        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}_prior.json".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        with open(prior_file, "w") as outfile:
            json.dump(self._get_json_dict(), outfile, cls=BilbyJsonEncoder, indent=2)

    def from_file(self, filename):
        """Reads in a prior from a file specification

        Parameters
        ==========
        filename: str
            Name of the file to be read in

        Notes
        =====
        Lines beginning with '#' or empty lines will be ignored.
        Priors can be loaded from:

        - bilby.core.prior as, e.g.,    :code:`foo = Uniform(minimum=0, maximum=1)`
        - floats, e.g.,                 :code:`foo = 1`
        - bilby.gw.prior as, e.g.,      :code:`foo = bilby.gw.prior.AlignedSpin()`
        - other external modules, e.g., :code:`foo = my.module.CustomPrior(...)`

        """

        comments = ["#", "\n"]
        prior = dict()
        with ioopen(filename, "r", encoding="unicode_escape") as f:
            for line in f:
                if line[0] in comments:
                    continue
                line.replace(" ", "")
                elements = line.split("=")
                key = elements[0].replace(" ", "")
                val = "=".join(elements[1:]).strip()
                prior[key] = val
        self.from_dictionary(prior)

    @classmethod
    def _get_from_json_dict(cls, prior_dict):
        try:
            class_ = getattr(
                import_module(prior_dict["__module__"]), prior_dict["__name__"]
            )
        except ImportError:
            logger.debug(
                "Cannot import prior module {}.{}".format(
                    prior_dict["__module__"], prior_dict["__name__"]
                )
            )
            class_ = cls
        except KeyError:
            logger.debug("Cannot find module name to load")
            class_ = cls
        for key in ["__module__", "__name__", "__prior_dict__"]:
            if key in prior_dict:
                del prior_dict[key]
        obj = class_(prior_dict)
        return obj

    @classmethod
    def from_json(cls, filename):
        """Reads in a prior from a json file

        Parameters
        ==========
        filename: str
            Name of the file to be read in
        """
        with open(filename, "r") as ff:
            obj = json.load(ff, object_hook=decode_bilby_json)

        # make sure priors containing JointDists are properly handled and point
        # to the same object when required
        jointdists = {}
        for key in obj:
            if isinstance(obj[key], JointPrior):
                for name in obj[key].dist.names:
                    jointdists[name] = obj[key].dist
        # set dist for joint values so that they point to the same object
        for key in obj:
            if isinstance(obj[key], JointPrior):
                obj[key].dist = jointdists[key]

        return obj

    def from_dictionary(self, dictionary):
        jpdkwargs = {}
        for key in list(dictionary.keys()):
            val = dictionary[key]
            if isinstance(val, Prior):
                continue
            elif isinstance(val, (int, float)):
                dictionary[key] = DeltaFunction(peak=val)
            elif isinstance(val, str):
                cls = val.split("(")[0]
                args = "(".join(val.split("(")[1:])[:-1]
                try:
                    dictionary[key] = DeltaFunction(peak=float(cls))
                    logger.debug("{} converted to DeltaFunction prior".format(key))
                    continue
                except ValueError:
                    pass
                if "." in cls:
                    module = ".".join(cls.split(".")[:-1])
                    cls = cls.split(".")[-1]
                else:
                    module = __name__.replace(
                        "." + os.path.basename(__file__).replace(".py", ""), ""
                    )
                try:
                    cls = getattr(import_module(module), cls, cls)
                except ModuleNotFoundError:
                    logger.error(
                        "Cannot import prior class {} for entry: {}={}".format(
                            cls, key, val
                        )
                    )
                    raise
                if key.lower() in ["conversion_function", "condition_func"]:
                    setattr(self, key, cls)
                elif isinstance(cls, str):
                    if "(" in val:
                        raise TypeError("Unable to parse prior class {}".format(cls))
                    else:
                        continue
                elif issubclass(cls, BaseJointPriorDist):
                    dictionary.pop(key)
                    if key not in jpdkwargs:
                        jpdkwargs[key] = cls.from_repr(args)
                elif issubclass(cls, JointPrior):
                    jpkwargs = {
                        item[0].strip(): cls._parse_argument_string(item[1])
                        for item in cls._split_repr(
                            ", ".join(
                                [arg for arg in args.split(",") if "dist=" not in arg]
                            )
                        ).items()
                    }
                    keymatch = re.match(r"dist=(?P<distkey>\S+),", args)
                    if keymatch is None:
                        raise ValueError(
                            "'dist' argument for JointPrior is not specified"
                        )

                    if keymatch["distkey"] not in jpdkwargs:
                        raise ValueError(
                            f"BaseJointPriorDist {keymatch['distkey']} must be defined before {cls.__name__}"
                        )

                    jpkwargs["dist"] = jpdkwargs[keymatch["distkey"]]
                    dictionary[key] = cls(**jpkwargs)
                else:
                    try:
                        dictionary[key] = cls.from_repr(args)
                    except TypeError as e:
                        raise TypeError(
                            "Unable to parse prior, bad entry: {} "
                            "= {}. Error message {}".format(key, val, e)
                        )
            elif isinstance(val, dict):
                try:
                    _class = getattr(
                        import_module(val.get("__module__", "none")),
                        val.get("__name__", "none"),
                    )
                    dictionary[key] = _class(**val.get("kwargs", dict()))
                except ImportError:
                    logger.debug(
                        "Cannot import prior module {}.{}".format(
                            val.get("__module__", "none"), val.get("__name__", "none")
                        )
                    )
                    logger.warning(
                        "Cannot convert {} into a prior object. "
                        "Leaving as dictionary.".format(key)
                    )
                    continue
            else:
                raise TypeError(
                    "Unable to parse prior, bad entry: {} "
                    "= {} of type {}".format(key, val, type(val))
                )
        self.update(dictionary)

    def convert_floats_to_delta_functions(self):
        """Convert all float parameters to delta functions"""
        for key in self:
            if isinstance(self[key], Prior):
                continue
            elif isinstance(self[key], float) or isinstance(self[key], int):
                self[key] = DeltaFunction(self[key])
                logger.debug("{} converted to delta function prior.".format(key))
            else:
                logger.debug(
                    "{} cannot be converted to delta function prior.".format(key)
                )

    def fill_priors(self, likelihood, default_priors_file=None):
        """
        Fill dictionary of priors based on required parameters of likelihood

        Any floats in prior will be converted to delta function prior. Any
        required, non-specified parameters will use the default.

        Note: if `likelihood` has `non_standard_sampling_parameter_keys`, then
        this will set-up default priors for those as well.

        Parameters
        ==========
        likelihood: bilby.likelihood.GravitationalWaveTransient instance
            Used to infer the set of parameters to fill the prior with
        default_priors_file: str, optional
            If given, a file containing the default priors.


        Returns
        =======
        prior: dict
            The filled prior dictionary

        """

        self.convert_floats_to_delta_functions()

        missing_keys = set(likelihood.parameters) - set(self.keys())

        for missing_key in missing_keys:
            if not self.test_redundancy(missing_key):
                default_prior = create_default_prior(missing_key, default_priors_file)
                if default_prior is None:
                    set_val = likelihood.parameters[missing_key]
                    logger.warning(
                        "Parameter {} has no default prior and is set to {}, this"
                        " will not be sampled and may cause an error.".format(
                            missing_key, set_val
                        )
                    )
                else:
                    self[missing_key] = default_prior

        for key in self:
            self.test_redundancy(key)

    def sample(self, size=None):
        """Draw samples from the prior set

        Parameters
        ==========
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        dict: Dictionary of the samples
        """
        return self.sample_subset_constrained(keys=list(self.keys()), size=size)

    def sample_subset_constrained_as_array(self, keys=iter([]), size=None):
        """Return an array of samples

        Parameters
        ==========
        keys: list
            A list of keys to sample in
        size: int
            The number of samples to draw

        Returns
        =======
        array: array_like
            An array of shape (len(key), size) of the samples (ordered by keys)
        """
        samples_dict = self.sample_subset_constrained(keys=keys, size=size)
        samples_dict = {key: np.atleast_1d(val) for key, val in samples_dict.items()}
        samples_list = [samples_dict[key] for key in keys]
        return np.array(samples_list)

    def sample_subset(self, keys=iter([]), size=None):
        """Draw samples from the prior set for parameters which are not a DeltaFunction

        Parameters
        ==========
        keys: list
            List of prior keys to draw samples from
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        dict: Dictionary of the drawn samples
        """
        self.convert_floats_to_delta_functions()
        samples = dict()
        for key in keys:
            if isinstance(self[key], Constraint):
                continue
            elif isinstance(self[key], Prior):
                samples[key] = self[key].sample(size=size)
            else:
                logger.debug("{} not a known prior.".format(key))
        return samples

    @property
    def non_fixed_keys(self):
        keys = self.keys()
        keys = [k for k in keys if isinstance(self[k], Prior)]
        keys = [k for k in keys if self[k].is_fixed is False]
        keys = [k for k in keys if k not in self.constraint_keys]
        return keys

    @property
    def fixed_keys(self):
        return [
            k for k, p in self.items() if (p.is_fixed and k not in self.constraint_keys)
        ]

    @property
    def constraint_keys(self):
        return [k for k, p in self.items() if isinstance(p, Constraint)]

    def sample_subset_constrained(self, keys=iter([]), size=None):
        efficiency_warning_was_issued = False

        def check_efficiency(n_tested, n_valid):
            nonlocal efficiency_warning_was_issued
            if efficiency_warning_was_issued:
                return
            efficiency = n_valid / float(n_tested)
            if n_tested >= 1e3 and efficiency < 1e-3:
                logger.warning("Prior sampling efficiency is very low, please verify its validity.")
                efficiency_warning_was_issued = True

        n_tested_samples, n_valid_samples = 0, 0
        if size is None or size == 1:
            while True:
                sample = self.sample_subset(keys=keys, size=size)
                is_valid = self.evaluate_constraints(sample)
                n_tested_samples += 1
                n_valid_samples += int(is_valid)
                check_efficiency(n_tested_samples, n_valid_samples)
                if is_valid:
                    return sample
        else:
            needed = np.prod(size)
            for key in keys.copy():
                if isinstance(self[key], Constraint):
                    del keys[keys.index(key)]
            all_samples = {key: np.array([]) for key in keys}
            _first_key = list(all_samples.keys())[0]
            while len(all_samples[_first_key]) < needed:
                samples = self.sample_subset(keys=keys, size=needed)
                keep = np.array(self.evaluate_constraints(samples), dtype=bool)
                for key in keys:
                    all_samples[key] = np.hstack(
                        [all_samples[key], samples[key][keep].flatten()]
                    )
                n_tested_samples += needed
                n_valid_samples += np.sum(keep)
                check_efficiency(n_tested_samples, n_valid_samples)
            all_samples = {
                key: np.reshape(all_samples[key][:needed], size) for key in keys
            }
            return all_samples

    def normalize_constraint_factor(
        self, keys, min_accept=10000, sampling_chunk=50000, nrepeats=10
    ):
        if keys in self._cached_normalizations.keys():
            return self._cached_normalizations[keys]
        else:
            factor_estimates = [
                self._estimate_normalization(keys, min_accept, sampling_chunk)
                for _ in range(nrepeats)
            ]
            factor = np.mean(factor_estimates)
            if np.std(factor_estimates) > 0:
                decimals = int(-np.floor(np.log10(3 * np.std(factor_estimates))))
                factor_rounded = np.round(factor, decimals)
            else:
                factor_rounded = factor
            self._cached_normalizations[keys] = factor_rounded
            return factor_rounded

    def _estimate_normalization(self, keys, min_accept, sampling_chunk):
        samples = self.sample_subset(keys=keys, size=sampling_chunk)
        keep = np.atleast_1d(self.evaluate_constraints(samples))
        if len(keep) == 1:
            self._cached_normalizations[keys] = 1
            return 1
        all_samples = {key: np.array([]) for key in keys}
        while np.count_nonzero(keep) < min_accept: # TODO: need to watch out for this while loop, might get stuck
            samples = self.sample_subset(keys=keys, size=sampling_chunk)
            for key in samples:
                all_samples[key] = np.hstack([all_samples[key], samples[key].flatten()])
            keep = np.array(self.evaluate_constraints(all_samples), dtype=bool)
        factor = len(keep) / np.count_nonzero(keep)
        return factor

    def prob(self, sample, **kwargs):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.prod`

        Returns
        =======
        float: Joint probability of all individual sample probabilities

        """
        prob = np.prod([self[key].prob(sample[key]) for key in sample], **kwargs)

        return self.check_prob(sample, prob)

    def check_prob(self, sample, prob):
        ratio = self.normalize_constraint_factor(tuple(sample.keys()))
        if np.all(prob == 0.0):
            return prob * ratio
        else:
            if isinstance(prob, float):
                if self.evaluate_constraints(sample):
                    return prob * ratio
                else:
                    return 0.0
            else:
                constrained_prob = np.zeros_like(prob)
                keep = np.array(self.evaluate_constraints(sample), dtype=bool)
                constrained_prob[keep] = prob[keep] * ratio
                return constrained_prob

    def ln_prob(self, sample, axis=None, normalized=False): # FIXME: changed from default True to False!
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which to calculate the log probability
        axis: None or int
            Axis along which the summation is performed
        normalized: bool
            When False, disables calculation of constraint normalization factor
            during prior probability computation. Default value is True.

        Returns
        =======
        float or ndarray:
            Joint log probability of all the individual sample probabilities

        """
        ln_prob = np.sum([self[key].ln_prob(sample[key]) for key in sample], axis=axis)
        return self.check_ln_prob(sample, ln_prob,
                                  normalized=normalized)

    def check_ln_prob(self, sample, ln_prob, normalized=True):
        if normalized:
            ratio = self.normalize_constraint_factor(tuple(sample.keys()))
        else:
            ratio = 1
        if np.all(np.isinf(ln_prob)):
            return ln_prob
        else:
            if isinstance(ln_prob, float):
                if self.evaluate_constraints(sample):
                    return ln_prob + np.log(ratio)
                else:
                    return -np.inf
            else:
                constrained_ln_prob = -np.inf * np.ones_like(ln_prob)
                keep = np.array(self.evaluate_constraints(sample), dtype=bool)
                constrained_ln_prob[keep] = ln_prob[keep] + np.log(ratio)
                return constrained_ln_prob

    def cdf(self, sample):
        """Evaluate the cumulative distribution function at the provided points

        Parameters
        ----------
        sample: dict, pandas.DataFrame
            Dictionary of the samples of which to calculate the CDF

        Returns
        -------
        dict, pandas.DataFrame: Dictionary containing the CDF values

        """
        return sample.__class__(
            {key: self[key].cdf(sample) for key, sample in sample.items()}
        )

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ==========
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        =======
        list: List of floats containing the rescaled sample
        """
        theta = list(theta)
        samples = []
        for key, units in zip(keys, theta):
            samps = self[key].rescale(units)
            samples += list(np.asarray(samps).flatten())
        return samples

    def test_redundancy(self, key, disable_logging=False):
        """Empty redundancy test, should be overwritten in subclasses"""
        return False

    def test_has_redundant_keys(self):
        """
        Test whether there are redundant keys in self.

        Returns
        =======
        bool: Whether there are redundancies or not
        """
        redundant = False
        for key in self:
            if isinstance(self[key], Constraint):
                continue
            temp = self.copy()
            del temp[key]
            if temp.test_redundancy(key, disable_logging=True):
                logger.warning(
                    "{} is a redundant key in this {}.".format(
                        key, self.__class__.__name__
                    )
                )
                redundant = True
        return redundant

    def copy(self):
        """
        We have to overwrite the copy method as it fails due to the presence of
        defaults.
        """
        return self.__class__(dictionary=dict(self))


class PriorDictException(Exception):
    """General base class for all prior dict exceptions"""


class ConditionalPriorDict(PriorDict):
    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """

        Parameters
        ==========
        dictionary: dict
            See parent class
        filename: str
            See parent class
        """
        self._conditional_keys = []
        self._unconditional_keys = []
        self._rescale_keys = []
        self._rescale_indexes = []
        self._least_recently_rescaled_keys = []
        super(ConditionalPriorDict, self).__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=conversion_function,
        )
        self._resolved = False
        self._resolve_conditions()

    def _resolve_conditions(self):
        """
        Resolves how priors depend on each other and automatically
        sorts them into the right order.
        1. All unconditional priors are put in front in arbitrary order
        2. We loop through all the unsorted conditional priors to find
        which one can go next
        3. We repeat step 2 len(self) number of times to make sure that
        all conditional priors will be sorted in order
        4. We set the `self._resolved` flag to True if all conditional
        priors were added in the right order
        """
        self._unconditional_keys = [
            key for key in self.keys() if not hasattr(self[key], "condition_func")
        ]
        conditional_keys_unsorted = [
            key for key in self.keys() if hasattr(self[key], "condition_func")
        ]
        self._conditional_keys = []
        for _ in range(len(self)):
            for key in conditional_keys_unsorted[:]:
                if self._check_conditions_resolved(key, self.sorted_keys):
                    self._conditional_keys.append(key)
                    conditional_keys_unsorted.remove(key)

        self._resolved = True
        if len(conditional_keys_unsorted) != 0:
            self._resolved = False

    def _check_conditions_resolved(self, key, sampled_keys):
        """Checks if all required variables have already been sampled so we can sample this key"""
        conditions_resolved = True
        for k in self[key].required_variables:
            if k not in sampled_keys:
                conditions_resolved = False
        return conditions_resolved

    def sample_subset(self, keys=iter([]), size=None):
        self.convert_floats_to_delta_functions()
        add_delta_keys = [
            key
            for key in self.keys()
            if key not in keys and isinstance(self[key], DeltaFunction)
        ]
        use_keys = add_delta_keys + list(keys)
        subset_dict = ConditionalPriorDict({key: self[key] for key in use_keys})
        if not subset_dict._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolvable conditions."
            )
        samples = dict()
        for key in subset_dict.sorted_keys:
            if key not in keys or isinstance(self[key], Constraint):
                continue
            if isinstance(self[key], Prior):
                try:
                    samples[key] = subset_dict[key].sample(
                        size=size, **subset_dict.get_required_variables(key)
                    )
                except ValueError:
                    # Some prior classes can not handle an array of conditional parameters (e.g. alpha for PowerLaw)
                    # If that is the case, we sample each sample individually.
                    required_variables = subset_dict.get_required_variables(key)
                    samples[key] = np.zeros(size)
                    for i in range(size):
                        rvars = {
                            key: value[i] for key, value in required_variables.items()
                        }
                        samples[key][i] = subset_dict[key].sample(**rvars)
            else:
                logger.debug("{} not a known prior.".format(key))
        return samples

    def get_required_variables(self, key):
        """Returns the required variables to sample a given conditional key.

        Parameters
        ==========
        key : str
            Name of the key that we want to know the required variables for

        Returns
        =======
        dict: key/value pairs of the required variables
        """
        return {
            k: self[k].least_recently_sampled
            for k in getattr(self[key], "required_variables", [])
        }

    def prob(self, sample, **kwargs):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.prod`

        Returns
        =======
        float: Joint probability of all individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        prob = np.prod(res, **kwargs)
        return self.check_prob(sample, prob)

    def ln_prob(self, sample, axis=None, normalized=False): # FIXME: changed from default True to False!
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the log probability of
        axis: Union[None, int]
            Axis along which the summation is performed
        normalized: bool
            When False, disables calculation of constraint normalization factor
            during prior probability computation. Default value is True.

        Returns
        =======
        float: Joint log probability of all the individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].ln_prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        ln_prob = np.sum(res, axis=axis)
        return self.check_ln_prob(sample, ln_prob,
                                  normalized=normalized)

    def cdf(self, sample):
        self._prepare_evaluation(*zip(*sample.items()))
        res = {
            key: self[key].cdf(sample[key], **self.get_required_variables(key))
            for key in sample
        }
        return sample.__class__(res)

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ==========
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        =======
        list: List of floats containing the rescaled sample
        """
        keys = list(keys)
        theta = list(theta)
        self._check_resolved()
        self._update_rescale_keys(keys)
        result = dict()
        for key, index in zip(
            self.sorted_keys_without_fixed_parameters, self._rescale_indexes
        ):
            result[key] = self[key].rescale(
                theta[index], **self.get_required_variables(key)
            )
            self[key].least_recently_sampled = result[key]
        samples = []
        for key in keys:
            samples += list(np.asarray(result[key]).flatten())
        return np.array(samples)

    def _update_rescale_keys(self, keys):
        if not keys == self._least_recently_rescaled_keys:
            self._rescale_indexes = [
                keys.index(element)
                for element in self.sorted_keys_without_fixed_parameters
            ]
            self._least_recently_rescaled_keys = keys

    def _prepare_evaluation(self, keys, theta):
        self._check_resolved()
        for key, value in zip(keys, theta):
            self[key].least_recently_sampled = value

    def _check_resolved(self):
        if not self._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolveable conditions."
            )

    @property
    def conditional_keys(self):
        return self._conditional_keys

    @property
    def unconditional_keys(self):
        return self._unconditional_keys

    @property
    def sorted_keys(self):
        return self.unconditional_keys + self.conditional_keys

    @property
    def sorted_keys_without_fixed_parameters(self):
        return [
            key
            for key in self.sorted_keys
            if not isinstance(self[key], (DeltaFunction, Constraint))
        ]

    def __setitem__(self, key, value):
        super(ConditionalPriorDict, self).__setitem__(key, value)
        self._resolve_conditions()

    def __delitem__(self, key):
        super(ConditionalPriorDict, self).__delitem__(key)
        self._resolve_conditions()


class DirichletPriorDict(ConditionalPriorDict):
    def __init__(self, n_dim=None, label="dirichlet_"):
        from .conditional import DirichletElement

        self.n_dim = n_dim
        self.label = label
        super(DirichletPriorDict, self).__init__(dictionary=dict())
        for ii in range(n_dim - 1):
            self[label + "{}".format(ii)] = DirichletElement(
                order=ii, n_dimensions=n_dim, label=label
            )

    def copy(self, **kwargs):
        return self.__class__(n_dim=self.n_dim, label=self.label)

    def _get_json_dict(self):
        total_dict = dict()
        total_dict["__prior_dict__"] = True
        total_dict["__module__"] = self.__module__
        total_dict["__name__"] = self.__class__.__name__
        total_dict["n_dim"] = self.n_dim
        total_dict["label"] = self.label
        return total_dict

    @classmethod
    def _get_from_json_dict(cls, prior_dict):
        try:
            cls == getattr(
                import_module(prior_dict["__module__"]), prior_dict["__name__"]
            )
        except ImportError:
            logger.debug(
                "Cannot import prior module {}.{}".format(
                    prior_dict["__module__"], prior_dict["__name__"]
                )
            )
        except KeyError:
            logger.debug("Cannot find module name to load")
        for key in ["__module__", "__name__", "__prior_dict__"]:
            if key in prior_dict:
                del prior_dict[key]
        obj = cls(**prior_dict)
        return obj


class NFConditionalPrior(Prior):
    """
    A conditional prior that uses a normalizing flow to model tidal deformability
    parameters (lambda_1, lambda_2) conditional on neutron star masses.
    
    This class integrates with bilby's ConditionalPriorDict to provide
    p(Lambda_i | m_1, m_2) using a trained normalizing flow model.
    """
    
    def __init__(self, nf_model_path, target_param, name=None, latex_label=None, 
                 unit=None, boundary=None, minimum=0.0, maximum=10000.0):
        """
        Parameters
        ==========
        nf_model_path: str
            Path to the trained normalizing flow model (.pt file)
        target_param: str
            Which lambda parameter this prior represents ('lambda_1' or 'lambda_2')
        name: str, optional
            Name of the parameter
        latex_label: str, optional
            LaTeX label for plotting
        unit: str, optional
            Unit of the parameter
        boundary: str, optional
            Boundary condition
        minimum: float
            Minimum value for the parameter
        maximum: float
            Maximum value for the parameter
        """
        # Load the NF model
        if not os.path.isfile(nf_model_path):
            raise FileNotFoundError(f"NF model file {nf_model_path} does not exist.")
            
        kwargs_file = nf_model_path.replace(".pt", "_kwargs.json")
        with open(kwargs_file, "r") as f:
            kwargs = json.load(f)
            
        # Validate model configuration and determine source type
        supported_configs = {
            "bns": {
                "names": ["lambda_1", "lambda_2"],
                "names_conditional": ["m_1", "m_2"],
                "n_inputs": 2,
                "n_conditional_inputs": 2
            },
            "nsbh": {
                "names": ["lambda_2"],
                "names_conditional": ["m_2"],
                "n_inputs": 1,
                "n_conditional_inputs": 1
            }
        }
        
        # Determine source type from model configuration
        self.source_type = None
        for src_type, config in supported_configs.items():
            if (kwargs["names"] == config["names"] and 
                kwargs["names_conditional"] == config["names_conditional"]):
                self.source_type = src_type
                break
                
        if self.source_type is None:
            raise ValueError(
                f"Unsupported model configuration: names={kwargs['names']}, "
                f"names_conditional={kwargs['names_conditional']}. "
                f"Supported configurations: {supported_configs}"
            )
            
        logger.info(f"Detected {self.source_type} model configuration for NF prior.")
            
        # Load the appropriate flow model based on source type
        config = supported_configs[self.source_type]
        
        if self.source_type == "bns":
            self.nf = RealNVP(
                n_inputs=config["n_inputs"],
                n_conditional_inputs=config["n_conditional_inputs"],
                n_transforms=kwargs["n_transforms"],
                n_neurons=kwargs["n_neurons"],
                n_blocks_per_transform=kwargs["n_blocks_per_transform"],
                batch_norm_between_transforms=True
            )
        elif self.source_type == "nsbh":
            self.nf = MaskedAffineAutoregressiveFlow(
                n_inputs=config["n_inputs"],
                n_conditional_inputs=config["n_conditional_inputs"],
                n_transforms=kwargs["n_transforms"],
                n_neurons=kwargs["n_neurons"],
                n_blocks_per_transform=kwargs["n_blocks_per_transform"]
            )
        self.nf.load_state_dict(torch.load(nf_model_path, map_location="cpu"))
        self.nf.eval()
        
        # Store configuration
        self.nf_model_path = nf_model_path
        self.target_param = target_param
        self.take_log_lambda = kwargs.get("take_log_lambda", "False") == "True"
        
        # Set target index for BNS case
        if self.source_type == "bns":
            self.target_index = 0 if target_param == 'lambda_1' else 1
        else:  # NSBH case - always lambda_2
            self.target_index = 0  # Single dimension output
        
        # Initialize the base prior (we override the key methods)
        super().__init__(
            name=name or target_param,
            latex_label=latex_label or f"$\\{target_param.replace('_', '_{') + '}'}$",
            unit=unit,
            boundary=boundary,
            minimum=minimum,
            maximum=maximum
        )
        
        # Set required variables for ConditionalPriorDict based on source type
        self.required_variables = ['chirp_mass', 'mass_ratio', 'luminosity_distance']
        
        # Bind source-type-specific methods for efficiency
        if self.source_type == "bns":
            self.sample = self._sample_bns
            self.rescale = self._rescale_bns
            self.ln_prob = self._ln_prob_bns
        else:
            self.sample = self._sample_nsbh
            self.rescale = self._rescale_nsbh
            self.ln_prob = self._ln_prob_nsbh
        
    def _convert_to_source_masses(self, chirp_mass, mass_ratio, dL):
        """Convert detector-frame parameters to source-frame masses"""
        z = luminosity_distance_to_redshift(dL)
        mc_source = chirp_mass / (1 + z)
        return chirp_mass_and_mass_ratio_to_component_masses(mc_source, mass_ratio)
        
    def _sample_bns(self, size=None, **required_variables):
        """Sample from BNS conditional NF distribution"""
        
        # Check for required variables
        if not all(var in required_variables for var in self.required_variables):
            missing = [var for var in self.required_variables if var not in required_variables]
            raise ValueError(f"Missing required variables: {missing}")
        
        # Extract required variables
        chirp_mass = required_variables['chirp_mass']
        mass_ratio = required_variables['mass_ratio']
        dL = required_variables['luminosity_distance']
        
        # Convert to source masses
        m1, m2 = self._convert_to_source_masses(chirp_mass, mass_ratio, dL)
        
        # Handle vectorized inputs
        if np.isscalar(chirp_mass):
            batch_size = 1 if size is None else size
            m1 = np.full(batch_size, m1)
            m2 = np.full(batch_size, m2)
        else:
            batch_size = len(chirp_mass)
            
        # Sample from NF
        with torch.inference_mode():
            cond = torch.tensor(np.column_stack([m1, m2]), dtype=torch.float32)
            lambdas = self.nf.sample(batch_size, conditional=cond).cpu().numpy()
            
            # Extract target parameter and enforce positivity
            target_values = np.maximum(0.0, lambdas[:, self.target_index])
            
            # FIXME: not done here for prob reasons, using the constraint priors for that
            # # Enforce bounds
            # target_values = np.clip(target_values, self.minimum, self.maximum)
            
        result = target_values[0] if batch_size == 1 else target_values
        self.least_recently_sampled = result
        return result
        
    def _sample_nsbh(self, size=None, **required_variables):
        """Sample from NSBH conditional NF distribution"""
        
        # Check for required variables
        if not all(var in required_variables for var in self.required_variables):
            missing = [var for var in self.required_variables if var not in required_variables]
            raise ValueError(f"Missing required variables: {missing}")
        
        # Extract required variables
        chirp_mass = required_variables['chirp_mass']
        mass_ratio = required_variables['mass_ratio']
        dL = required_variables['luminosity_distance']
        
        # Convert to source masses and extract NS mass (m2)
        _, m2 = self._convert_to_source_masses(chirp_mass, mass_ratio, dL)
        
        # Handle vectorized inputs
        if np.isscalar(chirp_mass):
            batch_size = 1 if size is None else size
            m2 = np.full(batch_size, m2)
        else:
            batch_size = len(chirp_mass)
            
        # Sample from NF - NSBH model expects 1D conditioning on m_2
        with torch.inference_mode():
            cond = torch.tensor(m2.reshape(-1, 1), dtype=torch.float32)
            lambdas = self.nf.sample(batch_size, conditional=cond).cpu().numpy()
            
            # For NSBH, convert from log space and enforce bounds
            if hasattr(self, 'take_log_lambda') and self.take_log_lambda:
                target_values = np.exp(lambdas.flatten())
            else:
                target_values = lambdas.flatten()
                
            # FIXME: doing this in constraints
            # target_values = np.maximum(0.0, target_values)
            # target_values = np.clip(target_values, self.minimum, self.maximum)
            
        result = target_values[0] if batch_size == 1 else target_values
        self.least_recently_sampled = result
        return result
        
    def _rescale_bns(self, val, **required_variables):
        """Rescale from unit hypercube using inverse NF transform for BNS"""
        
        # Check for required variables
        if not all(var in required_variables for var in self.required_variables):
            return super().rescale(val)  # Fallback to uniform rescaling
        
        # Extract required variables
        chirp_mass = required_variables['chirp_mass']
        mass_ratio = required_variables['mass_ratio']  
        dL = required_variables['luminosity_distance']
        
        # Convert to source masses
        m1, m2 = self._convert_to_source_masses(chirp_mass, mass_ratio, dL)
        
        # Handle vectorized inputs
        val = np.atleast_1d(val)
        if np.isscalar(chirp_mass):
            m1 = np.full(len(val), m1)
            m2 = np.full(len(val), m2)
        
        with torch.inference_mode():
            # Convert uniform samples to standard normal
            normal_samples = norm.ppf(np.clip(val, 1e-10, 1-1e-10))
            
            # Create 2D normal samples (we need both lambda components for NF inverse)
            # For the target component, use the provided val
            # For the other component, use a neutral value (0.0 standard normal)
            if self.target_index == 0:
                normal_2d = np.column_stack([normal_samples, np.zeros(len(val))])
            else:
                normal_2d = np.column_stack([np.zeros(len(val)), normal_samples])
                
            normal_tensor = torch.tensor(normal_2d, dtype=torch.float32)
            cond = torch.tensor(np.column_stack([m1, m2]), dtype=torch.float32)
            
            # Use NF inverse transform
            lambdas, _ = self.nf.inverse(normal_tensor, conditional=cond)
            lambdas = lambdas.cpu().numpy()
            
            # Extract target parameter and enforce bounds
            target_values = lambdas[:, self.target_index]
            
            # FIXME: doing this with constraints
            # target_values = np.maximum(0.0, target_values)
            # target_values = np.clip(target_values, self.minimum, self.maximum)
            
        return target_values[0] if len(target_values) == 1 else target_values
        
    def _rescale_nsbh(self, val, **required_variables):
        """Rescale from unit hypercube using inverse NF transform for NSBH"""
        
        # Check for required variables
        if not all(var in required_variables for var in self.required_variables):
            return super().rescale(val)  # Fallback to uniform rescaling
        
        # Extract required variables
        chirp_mass = required_variables['chirp_mass']
        mass_ratio = required_variables['mass_ratio']  
        dL = required_variables['luminosity_distance']
        
        # Convert to source masses and extract NS mass (m2)
        _, m2 = self._convert_to_source_masses(chirp_mass, mass_ratio, dL)
        
        # Handle vectorized inputs
        val = np.atleast_1d(val)
        if np.isscalar(chirp_mass):
            m2 = np.full(len(val), m2)
        
        with torch.inference_mode():
            # Convert uniform samples to standard normal
            normal_samples = norm.ppf(np.clip(val, 1e-10, 1-1e-10))
            normal_tensor = torch.tensor(normal_samples.reshape(-1, 1), dtype=torch.float32)
            cond = torch.tensor(m2.reshape(-1, 1), dtype=torch.float32)
            
            # Use NF inverse transform
            lambdas, _ = self.nf.inverse(normal_tensor, conditional=cond)
            lambdas = lambdas.cpu().numpy().flatten()
            
            # Convert from log space if needed and enforce bounds
            if self.take_log_lambda:
                target_values = np.exp(lambdas)
            else:
                target_values = lambdas
                
            # FIXME: doing this with constraints
            # target_values = np.maximum(0.0, target_values)
            # target_values = np.clip(target_values, self.minimum, self.maximum)
            
        return target_values[0] if len(target_values) == 1 else target_values
        
    def _ln_prob_bns(self, val, **required_variables):
        """Compute log probability using the BNS NF"""
        
        # FIXME: doing this in constraints
        # # Check bounds first
        # val = np.atleast_1d(val)
        # out_of_bounds = (val < self.minimum) | (val > self.maximum)
        # if np.any(out_of_bounds):
        #     return np.full_like(val, -np.inf, dtype=float)
            
        # TODO: check if this is necessary or if fails?
        # # Check for required variables
        # if not all(var in required_variables for var in self.required_variables):
        #     return np.zeros_like(val, dtype=float)  # Return neutral log prob
        
        # Extract required variables
        chirp_mass = required_variables['chirp_mass']
        mass_ratio = required_variables['mass_ratio']
        dL = required_variables['luminosity_distance']
        
        # Convert to source masses
        m1, m2 = self._convert_to_source_masses(chirp_mass, mass_ratio, dL)
        
        # Handle vectorized inputs
        if np.isscalar(chirp_mass):
            m1 = np.full(len(val), m1)
            m2 = np.full(len(val), m2)
            
        with torch.inference_mode():
            # For NF log_prob, we need both lambda values
            # We approximate by using the target value and a reference value for the other
            if self.target_index == 0:
                # val is lambda_1, use a reference lambda_2 (median value)
                lambda_pairs = np.column_stack([val, np.full(len(val), 1000.0)])
            else:
                # val is lambda_2, use a reference lambda_1
                lambda_pairs = np.column_stack([np.full(len(val), 1000.0), val])
                
            x = torch.tensor(lambda_pairs, dtype=torch.float32)
            cond = torch.tensor(np.column_stack([m1, m2]), dtype=torch.float32)
            
            # Get log prob from NF (this is the joint log prob)
            joint_logp = self.nf.log_prob(x, conditional=cond).cpu().numpy()
            
        return joint_logp[0] if len(joint_logp) == 1 else joint_logp
        
    def _ln_prob_nsbh(self, val, **required_variables):
        """Compute log probability using the NSBH NF"""
        
        # FIXME: doing this in constraints
        # # Check bounds first
        # val = np.atleast_1d(val)
        # out_of_bounds = (val < self.minimum) | (val > self.maximum)
        # if np.any(out_of_bounds):
        #     return np.full_like(val, -np.inf, dtype=float)
            
        # TODO: check if this is necessary or if fails?
        # # Check for required variables
        # if not all(var in required_variables for var in self.required_variables):
        #     return np.zeros_like(val, dtype=float)  # Return neutral log prob
        
        # Extract required variables
        chirp_mass = required_variables['chirp_mass']
        mass_ratio = required_variables['mass_ratio']
        dL = required_variables['luminosity_distance']
        
        # Convert to source masses and extract NS mass (m2)
        _, m2 = self._convert_to_source_masses(chirp_mass, mass_ratio, dL)
        
        # Handle vectorized inputs
        if np.isscalar(chirp_mass):
            m2 = np.full(len(val), m2)
            
        with torch.inference_mode():
            # Convert to log space if needed
            if self.take_log_lambda:
                x_input = np.log(np.maximum(val, 1e-10))  # Avoid log(0)
            else:
                x_input = val
                
            x = torch.tensor(x_input.reshape(-1, 1), dtype=torch.float32)
            cond = torch.tensor(m2.to_numpy().reshape(-1, 1), dtype=torch.float32)
            
            # Get log prob from NF
            logp = self.nf.log_prob(x, conditional=cond).cpu().numpy()
            
            # Apply Jacobian correction if we took log of lambda
            if self.take_log_lambda:
                # d(log(lambda))/d(lambda) = 1/lambda
                jacobian_correction = -np.log(np.maximum(val, 1e-10))
                logp += jacobian_correction
            
        return logp[0] if len(logp) == 1 else logp
        
    def prob(self, val, **required_variables):
        """Return the prior probability"""
        return np.exp(self.ln_prob(val, **required_variables))
        
    def get_instantiation_dict(self):
        """Return dictionary for reconstructing this prior"""
        instantiation_dict = super().get_instantiation_dict()
        instantiation_dict['nf_model_path'] = self.nf_model_path
        instantiation_dict['target_param'] = self.target_param
        return instantiation_dict


class ConditionalPriorDictException(PriorDictException):
    """General base class for all conditional prior dict exceptions"""


def create_default_prior(name, default_priors_file=None):
    """Make a default prior for a parameter with a known name.

    Parameters
    ==========
    name: str
        Parameter name
    default_priors_file: str, optional
        If given, a file containing the default priors.

    Returns
    =======
    prior: Prior
        Default prior distribution for that parameter, if unknown None is
        returned.
    """

    if default_priors_file is None:
        logger.debug("No prior file given.")
        prior = None
    else:
        default_priors = PriorDict(filename=default_priors_file)
        if name in default_priors.keys():
            prior = default_priors[name]
        else:
            logger.debug("No default prior found for variable {}.".format(name))
            prior = None
    return prior


class IllegalConditionsException(ConditionalPriorDictException):
    """Exception class to handle prior dicts that contain unresolvable conditions."""
