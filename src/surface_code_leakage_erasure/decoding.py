from collections.abc import Mapping
from collections import Counter
from copy import deepcopy
import inspect
import itertools
from dataclasses import dataclass
import time
import heapq
import numpy as np
# from line_profiler import profile
from functools import lru_cache, cached_property
import stim
import pymatching

from .circuit_builder import (
    CircuitBuilder,
    SurfaceCodeCircuitBuilder,
    iterate_erasure_checks,
)


PYMATCHING_FORK_URL = "git+https://github.com/Allenator/PyMatching.git"

_REWEIGHT_METHODS = ("decode", "decode_to_edges_array")


@lru_cache(maxsize=1)
def pymatching_supports_edge_reweights() -> bool:
    """ Whether the installed PyMatching honours the `edge_reweights` argument.

    Erasure decoding reweights the matching graph per shot, so this is a hard
    requirement. Upstream PyMatching declares `decode(self, z, *, ...,
    **kwargs)`, so passing `edge_reweights` to it raises nothing and is
    silently discarded -- decoding then proceeds as if no erasure information
    existed. We therefore check for an explicit `edge_reweights` parameter
    rather than relying on a TypeError, which upstream never raises.
    """
    for method_name in _REWEIGHT_METHODS:
        method = getattr(pymatching.Matching, method_name, None)
        if method is None:
            return False
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):  # unintrospectable native method
            return False
        if "edge_reweights" not in params:
            return False
    return True


def require_pymatching_edge_reweights():
    """ Raise if the installed PyMatching cannot do per-shot edge reweighting. """
    if pymatching_supports_edge_reweights():
        return
    raise RuntimeError(
        "The installed PyMatching does not support per-shot edge reweighting, "
        "which this package requires for erasure decoding.\n"
        "\n"
        "Upstream PyMatching accepts `edge_reweights` through **kwargs and "
        "silently ignores it, so decoding would run to completion while "
        "discarding all erasure information -- producing plausible-looking but "
        "incorrect logical error rates.\n"
        "\n"
        "Install the required fork:\n"
        f"    pip install --force-reinstall 'pymatching @ {PYMATCHING_FORK_URL}'"
    )

def get_blank_dem_str(n_detectors : int, n_observables : int) -> str:
    """ Create a blank DEM with no errors. """
    detector_list = [f"detector D{idx}" for idx in range(n_detectors)]
    observable_list = [f"logical_observable L{idx}" for idx in range(n_observables)]

    return "\n".join(detector_list + observable_list)


def dem_to_dict(dem : stim.DetectorErrorModel) -> dict:
    """ Convert a stim DEM to a dictionary where the keys are tuples of tuples representing the (hyper)edges and the values are the probabilities. This is used for easier manipulation of DEMs, especially for combining them. """
    dem_dict = {}
    for line in dem:
        if line.type == "error":
            hyperedge = []
            edge = []
            for targ in line.targets_copy():
                if targ.is_relative_detector_id():
                    edge.append(targ.val)
                elif targ.is_logical_observable_id():
                    edge.append(str(targ))
                elif targ.is_separator():
                    hyperedge.append(edge_to_tuple(edge))
                    edge = []
            if edge:
                hyperedge.append(edge_to_tuple(edge))
            dem_dict[tuple(hyperedge)] = line.args_copy()[0]
        else:
            break  # stop at the first non-error line, which should be the end of the error list

    return dem_dict


def dem_to_dict_graphlike(dem : stim.DetectorErrorModel, return_edge_logicals=False) -> dict:
    """ Convert a stim DEM to a dictionary where the keys are tuples of tuples representing the edges and the values are the probabilities. Unlike dem_to_dict, breaks apart hyperedges into individual edges. Still using tuples of tuples for consistency with the hyperedge representation, but the outer tuples will always have length 1. """
    dem_dict = {}
    edge_logicals = {}
    for line in dem:
        if line.type == "error":
            edge = []
            logicals = []
            for targ in line.targets_copy():
                if targ.is_relative_detector_id():
                    edge.append(targ.val)
                elif targ.is_logical_observable_id():
                    logicals.append(targ.val)
                elif targ.is_separator():
                    add_hyperedge_prob_inplace(
                        dem_dict, (edge_to_tuple(edge),), line.args_copy()[0])
                    edge = []
                    logicals = []
            if edge:
                edge_tuple = edge_to_tuple(edge)
                add_hyperedge_prob_inplace(
                    dem_dict, (edge_tuple,), line.args_copy()[0])
                edge_logicals[edge_tuple] = frozenset(logicals)
        else:
            break  # stop at the first non-error line, which should be the end of the error list

    if return_edge_logicals:
        return dem_dict, edge_logicals
    return dem_dict


def add_hyperedge_prob_inplace(dem_dict, hyperedge, new_prob):
    """ Add new probability for a hyperedge to dem_dict, treating error as independent from existing errors. """
    old_prob = dem_dict.get(hyperedge, 0.0)
    dem_dict[hyperedge] = old_prob*(1-new_prob) + (1-old_prob)*new_prob


def edge_to_tuple(edge):
    """ Convert an edge represented as a list to a sorted tuple. """
    return tuple(sorted(edge, key=lambda x: (isinstance(x, str), x)))


def single_leak_loc(rounds, leak_rnd, leak_gate, qubit):
    leak_locs = {
        rnd: {step: set() for step in range(4)} for rnd in range(rounds)}
    leak_locs[leak_rnd][leak_gate].add(qubit)
    return leak_locs


def combine_disjoint_dems(dem_list : list["DetectorErrorModelDict"], weights=None) -> "DetectorErrorModelDict":
    if len(dem_list) == 0:
        return DetectorErrorModelDict({})
    if weights is None:
        weights = [1.0/len(dem_list)]*len(dem_list)
    else:  # make sure correct length and normalized
        if len(weights) != len(dem_list):
            raise ValueError("Length of weights must match length of DEM list")
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]

    combined_dict = {}

    for dem, weight in zip(dem_list, weights):
        for dets, prob in dem.items():
            if dets in combined_dict:
                combined_dict[dets] += prob * weight
            else:
                combined_dict[dets] = prob * weight

    return DetectorErrorModelDict(combined_dict)


def pop_nth_erasure_check(erasure_checks, n):
    """ Remove the nth erasure check from the erasure_checks, returning the removed check by its (round, gate, qubit) and the new erasure_checks in the usual list of lists of sets format. """
    new_ec = deepcopy(erasure_checks)
    for (ec_rnd, ec_gate, ec_qubit) in iterate_erasure_checks(erasure_checks):
        if n == 0:
            new_ec[ec_rnd][ec_gate].remove(ec_qubit)
            return ((ec_rnd, ec_gate, ec_qubit), new_ec)
        n -= 1  # count down until we reach the nth erasure check

    raise ValueError("n is larger than the total number of erasure checks")


class DetectorErrorModelDict(Mapping):
    """ My class for representing a DEM as a dictionary where the keys are tuples of tuples representing the (hyper)edges and the values are the probabilities. This is used for easier manipulation of DEMs, especially for combining them. """
    _det_token_cache: dict = {}
    _prob_str_cache: dict = {}

    @classmethod
    def _det_token(cls, d):
        c = cls._det_token_cache
        s = c.get(d)
        if s is None:
            s = f"D{d}" if type(d) is int else d
            c[d] = s
        return s

    @classmethod
    def _prob_str(cls, p):
        c = cls._prob_str_cache
        s = c.get(p)
        if s is None:
            s = repr(p)
            c[p] = s
        return s

    def __init__(self, dem = None, graphlike=False):
        """ Instatiate a DEM from a dict or a stim DEM. Can call with no arguments to create an empty DEM. """
        if isinstance(dem, str):
            dem = stim.DetectorErrorModel(dem)
        if isinstance(dem, stim.DetectorErrorModel):
            if graphlike:
                dem_dict = dem_to_dict_graphlike(dem)
            else:
                dem_dict = dem_to_dict(dem)
        elif isinstance(dem, dict):
            dem_dict = dem
        elif dem is None:
            dem_dict = {}
        else:
            raise TypeError("Invalid type for 'dem' argument")

        self._data = dem_dict
        self.graphlike = graphlike

    def __str__(self):
        return f"DetectorErrorModelDict(num_errors={len(self._data)})"

    def __repr__(self):
        return f"DetectorErrorModelDict{self._data}"

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_str(self) -> str:
        """ Convert my format back to a string that can be parsed by stim. """
        tok = self._det_token
        prob_str = self._prob_str
        return "\n".join([
            f"error({prob_str(prob)}) " + " ^ ".join(
                [" ".join([tok(d) for d in edge]) for edge in hyperedge]
            )
            for hyperedge, prob in self.items()
        ])

    def to_tuples(self) -> frozenset:
        """ Convert my format to a set of tuples for use in constraint-checking algorithm. Drops the logicals. """
        tuples = []
        for hyperedge in self.keys():
            for edge in hyperedge:
                tuples.append(tuple(d for d in edge if type(d) is int))

        return frozenset(tuples)

    def increment_dem(self, shift) -> "DetectorErrorModelDict":
        """Increment the detector ids in the DEM by a certain amount. This is used to calculate a DEM with the same leakage in a different round.

        Arguments:
        shift: the amount to increment the detector ids by

        Returns:
        A new DetectorErrorModelDict representing the incremented DEM
        """
        new_dem_dict = {}

        # get errors and increment detector ids until we hit the first non-error line (which should be the end of the error list)
        for hyperedge, prob in self.items():
            new_hyperedge = []
            for edge in hyperedge:
                new_edge = tuple(d+shift if type(d) is int else d for d in edge)
                new_hyperedge.append(new_edge)
            new_dem_dict[tuple(new_hyperedge)] = prob

        new_dem = DetectorErrorModelDict(new_dem_dict, graphlike=self.graphlike)
        return new_dem

    def __rmul__(self, factor : float) -> "DetectorErrorModelDict":
        """ Scale the probabilities in the DEM by a certain amount. This is used to calculate a DEM with different error rates but the same leakage.

        Arguments:
        factor: the amount to scale the probabilities by

        Returns:
        A new DetectorErrorModelDict representing the scaled DEM
        """
        new_dem_dict = {hyperedge: prob*factor for hyperedge, prob in self.items()}
        new_dem = DetectorErrorModelDict(new_dem_dict, graphlike=self.graphlike)
        return new_dem

    def __add__(self, other : "DetectorErrorModelDict") -> "DetectorErrorModelDict":
        """ Add two DEMs together. Simply adds the probabilities so not physically meaningful.

        Arguments:
        other: the other DEM to add

        Returns:
        A new DetectorErrorModelDict representing the combined DEM
        """
        new_dem_dict = dict(self._data)  # start with a copy of the first DEM's dict

        for hyperedge, prob in other.items():
            if hyperedge in new_dem_dict:
                new_dem_dict[hyperedge] += prob
            else:
                new_dem_dict[hyperedge] = prob

        new_dem = DetectorErrorModelDict(new_dem_dict, graphlike=(self.graphlike and other.graphlike))
        return new_dem


class ErasureDecoder:
    """ Class for decoding surface code circuits with leakage and erasures by building DEMs for them """
    def __init__(self, circuit_builder, cache_size=4096, max_queue_size=4096):
        """
        Arguments:
        circuit_builder: the CircuitBuilder used to construct circuits and DEMs
        cache_size: maxsize of the LRU caches on the DEM-building methods. 0 disables caching
        max_queue_size: default cap on the size of the branch-and-bound priority queue in
          `find_valid_solution`. Unrelated to `cache_size`; can be overridden per call
        """
        require_pymatching_edge_reweights()
        if not isinstance(circuit_builder, CircuitBuilder):
            raise TypeError("Invalid type of input `circuit_builder`")
        self.circuit_builder = circuit_builder
        self.mode = "static" if isinstance(circuit_builder, SurfaceCodeCircuitBuilder) else "walking"
        self.cache_size = cache_size
        self.max_queue_size = max_queue_size

        # LRU caches
        if cache_size > 0:
            self.get_Pauli_dem_str = lru_cache(maxsize=cache_size)(self.get_Pauli_dem_str)
            self.get_Pauli_matcher_graphdict_edgedict = lru_cache(maxsize=cache_size)(self.get_Pauli_matcher_graphdict_edgedict)
            self.get_single_leakloc_dem = lru_cache(maxsize=cache_size)(self.get_single_leakloc_dem)
            self.get_single_leakloc_edge_set = lru_cache(maxsize=cache_size)(self.get_single_leakloc_edge_set)
            self.get_single_ec_average_dem = lru_cache(maxsize=cache_size)(self.get_single_ec_average_dem)
            self.get_single_ec_edge_sets = lru_cache(maxsize=cache_size)(self.get_single_ec_edge_sets)

    def __repr__(self):
        return f"ErasureDecoder({self.circuit_builder})"

    ## Pauli DEMs
    def get_Pauli_dem_str(self, rounds, p, **circuit_kwargs) -> str:
        """ Get the DEM for the Pauli errors. """
        circuit, _ = self.circuit_builder.get_circuit(
            rounds, p, **circuit_kwargs
        )
        dem =  circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True)

        return str(dem)

    def get_Pauli_matcher_graphdict_edgedict(self, rounds, p, lower_p_bound=1e-9, **circuit_kwargs):
        """ Get the pymatching.Matching object for just the Pauli errors. This matcher is as the baseline for reweighting. """
        p = max(p, lower_p_bound)  # need to have any errors for reweighting to work
        circuit, _ = self.circuit_builder.get_circuit(
            rounds, p, **circuit_kwargs
        )
        dem =  circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)

        graphdict, edgedict = dem_to_dict_graphlike(dem, return_edge_logicals=True)

        return matcher, graphdict, edgedict

    ## erasure DEMs
    def check_can_use_increment(self, rounds, event_rnd, event_type="leak", **circuit_kwargs):
        """ Checks if can generate DEM by incrementing a baseline DEM, based on the circuit location of the event (leakage or erasure check). If event is in first or last round, need to generate circuit directly to get correct DEM. If event is in baseline round(s) used for calculating later rounds' DEMs, also need to generate circuit directly.

        Arguments:
        rounds: total number of rounds in the circuit
        event_rnd: the round in which the event (leakage or erasure check) occurs
        event_type: type of event ("leak" or "ec"/"erasure"), used to determine which rounds are baseline rounds that require direct circuit generation
        circuit_kwargs: kwargs used in circuit construction. See CircuitBuilder. The kwarg ec_sched affects which rounds can use cache.

        Returns:
        True if can use increment strategy, False if need to generate circuit directly to get correct DEM
        """
        ec_sched = circuit_kwargs.get("ec_sched", CircuitBuilder.kwargs_options["ec_sched"][0])

        # if leakage is in first round, may affect special first round detectors
        # for erasure checks and ec_sched 8, round 1 may also affect special first round detectors due to delayed erasure check
        # baseline round = this is used to calculate bulk detectors later---need to be directly calculated
        # for static, round 1 is baseline for later rounds
        # for walking, round 1 is baseline for odd rounds and round 2 is baseline for even rounds
        # for erasure checks with ec_sched 8, round 1 can't be baseline because leakage may have come from round 0, so round 2 is baseline for even rounds and round 3 is baseline for odd rounds
        first_or_base_round_flag = event_rnd <= 1 or (
            event_rnd == 2 and self.mode == "walking") or (
            event_rnd <= 3 and event_type.startswith("e") and ec_sched == 8)

        # if leakage is in last round, may affect special final round detectors. If leakage is in 2nd-to-last round and ec_sched is 8, may also affect these detectors
        last_round_flag = event_rnd == rounds-1 or (ec_sched == 8 and event_rnd == rounds-2)

        return not (first_or_base_round_flag or last_round_flag)

    # @profile
    def get_single_leakloc_dem(self, rounds, leak_rnd, leak_gate, qubit, **circuit_kwargs) -> DetectorErrorModelDict:
        """ Construct the DEM for a single leakage location. This is the basic building block for all DEMs with leakage, since we can combine the DEMs for different leakage locations to get the DEM for multiple leakage locations or to get the DEM for an imprecise erasure check. """
        circuit_kwargs = self.circuit_builder.validate_circuit_kwargs(circuit_kwargs, decode=True)
        if not self.check_can_use_increment(rounds, leak_rnd, event_type="leak", **circuit_kwargs):

            # need to generate circuit directly to get DEM
            leak_locs = single_leak_loc(rounds, leak_rnd, leak_gate, qubit)
            cir, _ = self.circuit_builder.get_circuit(
                rounds, 0.0, leakage_circuit_locations=leak_locs,
                **circuit_kwargs
            )
            dem = DetectorErrorModelDict(
                cir.detector_error_model(decompose_errors=True),
                graphlike=True
            )
            return dem

        elif self.mode == "static":
            base_rnd = 1
        else:  # self.mode == "walking"
            base_rnd = 2 - leak_rnd % 2  # 1 if leak_rnd is odd, 2 if leak_rnd is even

        dem = self.get_single_leakloc_dem(
            rounds, base_rnd, leak_gate, qubit, **circuit_kwargs
        ).increment_dem(
            shift=(leak_rnd-base_rnd)*self.circuit_builder.detectors_per_round)

        return dem

    def get_single_leakloc_edge_set(self, rounds, leak_rnd, leak_gate, qubit, **circuit_kwargs) -> frozenset[tuple]:
        """ Get the set of edges corresponding to the DEM for a single leakage location. """
        dem = self.get_single_leakloc_dem(rounds, leak_rnd, leak_gate, qubit, **circuit_kwargs)
        tuples = []
        for hyperedge in dem.keys():
            for edge in hyperedge:
                tuples.append(tuple(d for d in edge if type(d) is int))

        return frozenset(tuples)

    def get_single_ec_dem_list(self, rounds, ec_rnd, ec_gate, ec_qubit, **circuit_kwargs) -> list[DetectorErrorModelDict]:
        """ Get the list of DEMs for a single erasure check, one DEM per leakage location that could cause that check to trigger. """
        circuit_builder = self.circuit_builder
        circuit_kwargs = circuit_builder.validate_circuit_kwargs(circuit_kwargs, decode=True)

        # find possible leakage locations that could cause an error on this check
        possible_leak_locs = circuit_builder.single_ec_traceback(
            ec_rnd, ec_gate, ec_qubit,
            circuit_kwargs["ec_sched"], circuit_kwargs["leak_locations"])

        # for each leakage location, generate the circuit and get the DEM
        dem_list = []
        for [leak_rnd, leak_gate, qubit] in possible_leak_locs:
            dem = self.get_single_leakloc_dem(
                rounds, leak_rnd, leak_gate, qubit,
                **circuit_kwargs
            )
            dem_list.append(dem)

        return dem_list

    def get_single_ec_edge_sets(self, rounds, ec_rnd, ec_gate, ec_qubit, **circuit_kwargs) -> tuple[frozenset[tuple]]:
        """ Get the list of sets of edges for a single erasure check, one set per leakage location that could cause that check to trigger. """
        circuit_kwargs = self.circuit_builder.validate_circuit_kwargs(circuit_kwargs, decode=True)

        # find possible leakage locations that could cause an error on this check
        possible_leak_locs = self.circuit_builder.single_ec_traceback(
            ec_rnd, ec_gate, ec_qubit,
            circuit_kwargs["ec_sched"], circuit_kwargs["leak_locations"])

        # get the edge set for each leakage location's DEM
        possible_edge_sets = []
        for [leak_rnd, leak_gate, qubit] in possible_leak_locs:
            possible_edge_sets.append(
                self.get_single_leakloc_edge_set(
                    rounds, leak_rnd, leak_gate, qubit, **circuit_kwargs))

        return tuple(possible_edge_sets)

    def get_single_ec_average_dem(self, rounds, ec_rnd, ec_gate, ec_qubit, **circuit_kwargs) -> DetectorErrorModelDict:
        """ Get the average DEM for a single erasure check by combining the DEMs for different leakage locations that could cause that check to trigger. """
        #TODO: add p_leak argument to weight the DEMs by the probability of leakage at that location, instead of just averaging them equally
        circuit_kwargs = self.circuit_builder.validate_circuit_kwargs(circuit_kwargs, decode=True)

        if not self.check_can_use_increment(rounds, ec_rnd, event_type="ec", **circuit_kwargs):
            # need to combine DEMs for different leakage locations directly to get DEM
            dem_list = self.get_single_ec_dem_list(rounds, ec_rnd, ec_gate, ec_qubit, **circuit_kwargs)
            dem = combine_disjoint_dems(dem_list)
            return dem

        elif circuit_kwargs["ec_sched"] == 8:
            base_rnd = 2 + ec_rnd % 2  # 2 if ec_rnd is even, 3 if ec_rnd is odd
        elif self.mode == "static":
            base_rnd = 1
        else:  # self.mode == "walking" and ec_sched != 8
            base_rnd = 2 - ec_rnd % 2  # 1 if ec_rnd is odd, 2 if ec_rnd is even

        dem = self.get_single_ec_average_dem(
            rounds, base_rnd, ec_gate, ec_qubit, **circuit_kwargs
        ).increment_dem(
            shift=(ec_rnd-base_rnd)*self.circuit_builder.detectors_per_round)

        return dem

    # @profile
    def get_full_average_dem_str(self, erasure_checks, **circuit_kwargs) -> str:
        """ Get the average DEM for all erasure checks. Combines DEMs for different leakage locations that but same erasure check as disjoint. Concatenates DEMs for different erasure checks to let them be parsed and combined independently by stim/pymatching. """
        rounds = len(erasure_checks)
        dem_str_list = []
        for ec_rnd, ec_gate, ec_qubit in iterate_erasure_checks(erasure_checks):
            dem_str_list.append(self.get_single_ec_average_dem(
                rounds=rounds,
                ec_rnd=ec_rnd,
                ec_gate=ec_gate,
                ec_qubit=ec_qubit,
                **circuit_kwargs
            ).to_str())

        return "\n".join(dem_str_list)

    # @profile
    # TODO: could be caching this in _ValidSolutionSearchContext
    def get_table_of_dem_edges(self, erasure_checks, **circuit_kwargs) -> list:
        """ Return of list of lists representing the edges of the decoding graph.
        1. The top level of lists correspond to the erasure checks
        2. The next level of lists correspond to leakage locations which could trigger that erasure check
        3. The elements of the 2nd level of lists are sets of tuples corresponding to the edges that could have been triggered due to a specific leakage event """
        rounds = len(erasure_checks)

        list_of_edge_sets = []
        coverage = {}

        ec_idx = 0
        for ec_rnd, ec_gate, ec_qubit in iterate_erasure_checks(erasure_checks):
            edge_sets = self.get_single_ec_edge_sets(
                rounds=rounds, ec_rnd=ec_rnd, ec_gate=ec_gate, ec_qubit=ec_qubit,
                **circuit_kwargs
                    )
            list_of_edge_sets.append(edge_sets)
            self.update_edge_coverage_inplace(coverage, edge_sets, ec_idx)
            ec_idx +=1

        return list_of_edge_sets, coverage

    def update_edge_coverage_inplace(self, coverage, edge_sets, ec_idx):
        for leak_idx, leak_edge_set in enumerate(edge_sets):
            for e in leak_edge_set:
                if e not in coverage:
                    coverage[e] = []
                coverage[e].append((ec_idx, leak_idx))

    ## simple decoding
    # @profile
    def calculate_edge_updates(self, graphdict_orig, *, erasure_checks=None, leakloc_tuples=None, leakloc_probs=None, rounds=None, **circuit_kwargs):
        """ Calculate changed edge probabilities from the erasure checks and/or leakage locations, to be passed to the edge_reweights argument of pymatching.Matching.decode. Errors are added as independent events.

        Arguments:
        graphdict_orig: original dict for the Pauli DEM, which has had its hyperedges decomposed.
        erasure_checks: list of erasure checks, in format returned by get_circuit
        leakloc_tuples: flat list of leakage circuit locations, as
          (leak_rnd, leak_gate, qubit) tuples. Note this is a different format
          from the `leakage_circuit_locations` argument of
          `CircuitBuilder.get_circuit`, which is a nested dict.
        leakloc_probs: probability of leakage at each entry of leakloc_tuples;
          must be the same length as leakloc_tuples.
        rounds: number of rounds in the circuit, required if erasure_checks is not given.

        Returns:
        A numpy array of shape (num_updated_edges, 3) where each row is of the form [detector1, detector2, new_weight] representing the updated edge weights to be passed to the edge_reweights argument of pymatching.Matching.decode
        """
        if erasure_checks is None:
            erasure_checks = []
        if leakloc_tuples is None:
            leakloc_tuples = []
        if leakloc_probs is None:
            leakloc_probs = []

        rounds = rounds or len(erasure_checks)
        if rounds == 0:
            raise ValueError("Must provide rounds if erasure_checks is not given")
        if len(leakloc_tuples) != len(leakloc_probs):
            raise ValueError("leakloc_tuples and leakloc_probs must have the same length")

        updated_probs = {}
        for ec_rnd, ec_gate, ec_qubit in iterate_erasure_checks(erasure_checks):
            dem = self.get_single_ec_average_dem(rounds, ec_rnd, ec_gate, ec_qubit, **circuit_kwargs)
            for (hyperedge, new_prob) in dem.items():
                for edge in hyperedge:
                    old_prob = updated_probs.get(edge, graphdict_orig.get((edge,), 0.0))
                    updated_probs[edge] = old_prob*(1-new_prob) + (1-old_prob)*new_prob
        for (leak_rnd, leak_gate, qubit), leak_prob in zip(leakloc_tuples, leakloc_probs):
            dem = self.get_single_leakloc_dem(rounds, leak_rnd, leak_gate, qubit, **circuit_kwargs)
            for (hyperedge, new_prob) in dem.items():
                new_prob = leak_prob * new_prob  # weight the error by the probability of leakage at that location
                for edge in hyperedge:
                    old_prob = updated_probs.get(edge, graphdict_orig.get((edge,), 0.0))
                    updated_probs[edge] = old_prob*(1-new_prob) + (1-old_prob)*new_prob

        # convert to numpy array format required by edge_reweights argument of pymatching.Matching.decode
        update_array = np.empty((len(updated_probs), 3))
        for i, (edge, new_prob) in enumerate(updated_probs.items()):
            dets_only = [term for term in edge if isinstance(term, int)]
            if len(dets_only) == 1:
                update_array[i] = [dets_only[0], -1, new_prob]
            else:  # len(dets_only) == 2
                update_array[i] = [dets_only[0], dets_only[1], new_prob]

        # calculation edge weight from probabilities
        update_array[:, 2] = np.log(
            (1-update_array[:, 2]) / update_array[:, 2]
        )

        return update_array

    ## validity checking & quasi-MLE decoding
    # @profile
    def check_solution_validity(self, decoded_edges, erasure_checks, chosen_leak_locs=None, **circuit_kwargs):
        """
        This function checks if the decoded edges are compatible with the fact that the leakage locations are disjoint.

        Arguments:
        decoded_edges: a set of edges (tuples of detectors) that were decoded from the syndrome using a marginal DEM
        erasure_checks: list of erasure checks, in format returned by get_circuit
        chosen_leak_locs: list of leakage locations that have been chosen for decoding

        Returns:
        (True, set()) if there is a combination of DEMs from the dem_table that can explain the decoded edges, (False, conflict_witness) otherwise. The conflict_witness is the smallest set of edges which were unable to be covered given the disjointness constraints, which can be used to guide the quasi-MLE decoding by telling us which edges to prioritize covering/fixing.

        """
        if chosen_leak_locs is None:
            chosen_leak_locs = []

        # no need to check validity when leak_effect does not include skipped gates they do not lead to disjoint errors
        if "skip" not in circuit_kwargs.get("leak_effect", ""):
            return (True, set())
        # make sure decoded_edges is correct format and type
        decoded_edges = set(tuple(sorted(d for d in edge if d != -1)) for edge in decoded_edges)

        # remove decoded_edges explained by chosen_leak_locs
        rounds = len(erasure_checks)
        leak_edges = set().union(
            *(self.get_single_leakloc_edge_set(rounds, *loc, **circuit_kwargs) for loc in chosen_leak_locs))
        decoded_edges = decoded_edges - leak_edges

        # construct DEMs and coverage dict from erasure checks
        dem_table, coverage = self.get_table_of_dem_edges(
            erasure_checks, **circuit_kwargs)
        # if an edge in decoded_edges is not covered by any DEM in the entire table, that means it came from a Pauli error, so there is no concern of disjointness and we can ignore it for the validity check
        decoded_edges = decoded_edges & coverage.keys()
        options_per_row = [0]*len(dem_table)  # how much overlap do DEMs in each row have with decoded_edges
        for e in decoded_edges:
            for (ec_idx, _) in coverage[e]:
                options_per_row[ec_idx] += 1

        # Order dem lists by most-constrained first (fewest useful edges), dropping DEM lists with no useful edges
        useful_idxs = np.argsort(options_per_row)[options_per_row.count(0):]
        dem_lists_sorted = [dem_table[idx] for idx in useful_idxs]
        dem_union_list = [set().union(*dem_list) for dem_list in dem_lists_sorted]

        # Backtrack
        return self.backtrack(dem_lists_sorted, uncovered=decoded_edges, dem_unions=dem_union_list)

    # @profile
    def backtrack(self, remaining_dems, uncovered, dem_unions):
        """Backtrack to find a valid combination of DEMs that can explain the decoded edges. Subroutine for check_solution_validity

        Arguments:
        remaining_dems: the remaining DEM lists of the dem_table to consider, ordered by most-constrained first
        uncovered: the edges that still need to be explained
        dem_unions: precomputed unions of DEMs in each row, used for forward checking to prune branches early

        """
        # print("entered `backtrack`")
        if not uncovered:  # no unaccounted for edges left -> found a valid combination of DEMs that can explain the decoded edges
            return (True, set())
        if not remaining_dems:   # no more DEMs to choose from, but still have uncovered edges -> this combination of DEMs can't explain the decoded edges
            # print("no remaining dems!")
            return (False, uncovered)

        # Forward check: any edge in none of the remaining DEMs?
        # Can probably get rid of this since solution is valid >99% of the time, so maybe not necessary to do early pruning
        # for e in uncovered:
        #     if not any(e in dem_union for dem_union in dem_unions):
        #         return (False, {e})   # prune immediately

        cur_dem_list = remaining_dems[0]
        leftover_dems = remaining_dems[1:]
        leftover_unions = dem_unions[1:]

        # Prioritize DEMs that explain the most uncovered elements (greedy heuristic)
        coverage_per_dem = [len(uncovered & dem) for dem in cur_dem_list]
        # sort DEMs by most coverage of uncovered edges, and filter out DEMs that don't cover any uncovered edges
        candidate_dems = [cur_dem_list[i] for i in np.argsort(coverage_per_dem)[::-1] if coverage_per_dem[i] > 0]

        conflict_witness = uncovered  # to keep track of which edge(s) caused the problem

        if not candidate_dems:
            # print("no useful dems in this row!")
            return self.backtrack(leftover_dems, uncovered, leftover_unions)

        for (dem_idx, edge_set) in enumerate(candidate_dems):
            # print(f"check candidate DEM {dem_idx} with edges {edge_set}")
            new_uncovered = uncovered - edge_set
            (valid, witness) = self.backtrack(leftover_dems, new_uncovered, leftover_unions)
            if valid:
                return (valid, witness)  # should be (True, set()) in this case
            if len(witness) < len(conflict_witness):
                # new smallest conflict witness found, update
                conflict_witness = witness
            if len(conflict_witness & set().union(*candidate_dems[(dem_idx+1):])) == 0:
                # if no remaining candidate DEM can explain any of the edges in the conflict witness, then no need to continue checking the remaining candidate DEMs for this row (because they have even less coverage than the current one), can prune immediately
                break

        return (False, conflict_witness)

    def find_disjoint_edges(self, uncovered_edges, covered_edges, dem_table, coverage):
        """ Given conflict witness of uncovered edges, find the edges which are included in the solution but can't be mutually explained given the disjointness constraints. """
        # uncovered is typically {b} — one stranded element
        conflicting_edges = []
        for e in uncovered_edges:
            dem_coverage_for_e = coverage.get(e, [])  # find which DEMs include edge e
            for dem_pos in dem_coverage_for_e:
                dem_row = dem_table[dem_pos[0]]  # the row of the DEM table
                all_disjoint_edges = set().union(*dem_row) - dem_row[dem_pos[1]]  # edges disjoint from e in this row
                problem_edges = covered_edges & all_disjoint_edges  # edges disjoint from e that average decoding invalidly included in solution
                if problem_edges:
                    conflicting_edges.append((e, *problem_edges))

        return conflicting_edges

    # @profile
    def find_valid_solution(self, erasure_checks, root_node, *, deadline=None, max_queue_size=None, verbose=False, **circuit_kwargs):
        """ This is a decoding function which tries to find a valid solution by fixing some of the leakage locations (removing disjoint DEMs from the average graph) based on the conflict witness from the validity check. Much more efficient than brute-force checking all combinations of leakage locations!

        Arguments:
        erasure_checks: the erasure checks for the circuit, in the usual list of lists of sets format
        root_node: a ValidityCheckNode representing the root of the search tree, which should be initialized with no fixed leak locations and the solution from the marginal decoding as the decoded_edges
        deadline: a time (in monotonic time) at which to stop the search and give up, returning None. If None, will search entire space until solution is found or space is exhausted
        max_queue_size: cap on the size of the search priority queue; whenever it is exceeded the queue is trimmed to the `max_queue_size` lowest-weight nodes. If None, uses the decoder's `self.max_queue_size`
        verbose: whether to print out debugging info about the search process
        circuit_kwargs: kwargs used in circuit construction. See CircuitBuilder

        Returns:
        best_solution, best_solution_weight
        """

        if max_queue_size is None:
            max_queue_size = self.max_queue_size
        priority_queue = []  # heap queue
        heapq.heappush(priority_queue, root_node)  # push root node onto priority queue to start the search

        while priority_queue:
            if deadline is not None and time.monotonic() > deadline:
                return None, float('inf')
            candidate = heapq.heappop(priority_queue)
            if verbose:
                print(f"checking candidate with fixed leak locs {candidate.fixed_leak_locs} and weight {candidate.weight}...", end="")
            if candidate.check_validity():
                if verbose:
                    print("valid solution found!")
                return candidate.solution, candidate.weight

            if verbose:
                print("invalid.")
            num_children = 0
            for child in candidate.generate_children():
                heapq.heappush(priority_queue, child)
                num_children += 1
            if verbose:
                print(f"    generated {num_children} children.")
            if len(priority_queue) > max_queue_size:
                # nsmallest returns an ascending-sorted list, which is already a valid min-heap
                priority_queue = heapq.nsmallest(max_queue_size, priority_queue)
                if verbose:
                    print(f"    queue trimmed to {max_queue_size}.")

        if verbose:
            print("no valid solution found within the search space!")
        return None, float('inf')

    ## decoding with validity checking and jumping to branch-and-bound search if necessary
    # @profile
    def decode(self, erasure_checks, syndrome, p_Pauli, branch_and_bound=False, mle_time_limit=10, max_queue_size=None, verbose=False, **circuit_kwargs):
        """ Decode the QEC memory experiment using the marginal/averaged strategy, optionally switching to branch-and-bound decoding if the solution from the marginal decoding is not valid and branch_and_bound is enabled.

        Arguments:
        erasure_checks: triggered erasure checks (generated during circuit construction)
        syndrome: Pauli syndrome (generated during circuit sampling)
        p_Pauli: probability of Pauli errors
        branch_and_bound: whether to use the branch-and-bound search decoder if the marginal decoding is not valid
        mle_time_limit: time limit for the branch-and-bound search, in seconds. None means no time limit. Only applies if branch_and_bound is True.
        max_queue_size: cap on the branch-and-bound search queue size. None means use the decoder's `self.max_queue_size`. Only applies if branch_and_bound is True.
        circuit_kwargs: kwargs used in circuit construction. See CircuitBuilder

        Returns:
        decoded_result [np.array] : whether the logical outcome is flipped, according to marginal decoding
        valid_solution [np.array] : the decoding result from the branch-and-bound search if the marginal decoding is not valid, or the same as decoded_result if the marginal decoding is valid. None if branch_and_bound=False

        """
        ## do marginal decoding first

        # get no-leakage matcher and graphdict
        rounds = len(erasure_checks)
        matcher, graphdict, edgedict = self.get_Pauli_matcher_graphdict_edgedict(
            rounds, p_Pauli, **circuit_kwargs)

        # get reweighting info from erasure checks
        edge_updates = self.calculate_edge_updates(
            graphdict, erasure_checks=erasure_checks, **circuit_kwargs)

        # the decoding
        decoded_result, avg_soln_weight = matcher.decode(syndrome, edge_reweights=edge_updates, return_weight=True)
        if not branch_and_bound:
            return (decoded_result, None)
        if "skip" not in circuit_kwargs.get("leak_effect", ""):
            # if leak_effect does not have skipped gates, then we don't have disjointness constraints and no need to do special decoding
            return (decoded_result, decoded_result)
        # if circuit_kwargs.get("ec_sched", 8) == 1:
        #     # erasure check every CNOT, no disjointness
        #     return (decoded_result, decoded_result)
        ## do branch-and-bound decoding

        # setup for branch-and-bound search for valid solution
        circuit_kwargs = self.circuit_builder.validate_circuit_kwargs(circuit_kwargs, decode=True)
        context = _ValidSolutionSearchContext(
            decoder=self, matcher=matcher, graphdict=graphdict,
            edge_data = edgedict, num_observables = matcher.num_fault_ids,
            syndrome=syndrome, p_Pauli=p_Pauli,
            circuit_kwargs=circuit_kwargs)
        root_node = ValidSolutionSearchNode(
            context=context,
            erasure_checks=erasure_checks,
            fixed_leak_locs=[],
            fixed_leakloc_probs=[],
        )
        root_node.edge_updates = edge_updates
        root_node.solution = decoded_result
        root_node.weight = avg_soln_weight

        # do the guided search for a valid solution, with an optional time limit
        deadline = time.monotonic() + mle_time_limit if mle_time_limit is not None else None
        valid_solution = self.find_valid_solution(
            erasure_checks, root_node = root_node,
            deadline=deadline, max_queue_size=max_queue_size,
            verbose=verbose, **circuit_kwargs)[0]

        return (decoded_result, valid_solution)


@dataclass
class _ValidSolutionSearchContext:
    decoder: ErasureDecoder
    matcher: pymatching.Matching
    graphdict: dict
    edge_data: dict
    num_observables: int
    syndrome: np.ndarray
    p_Pauli: float
    circuit_kwargs: dict


class ValidSolutionSearchNode:
    """ Class to represent a node in the search tree for the guided search quasi-MLE decoding. Each node corresponds to fixing some particular leakage locations and removing the corresponding DEM from the average graph. """
    counter = itertools.count()  # unique sequence number used for tie-breaking in the priority queue

    def __init__(self, context, erasure_checks, fixed_leak_locs, fixed_leakloc_probs):
        self.context = context  # context object

        self.erasure_checks = erasure_checks
        self.fixed_leak_locs = fixed_leak_locs
        self.fixed_leakloc_probs = fixed_leakloc_probs

        self.valid = None
        self.id = next(ValidSolutionSearchNode.counter)

    def __lt__(self, other):
        # order by weight, then by id to break ties
        return (self.weight, self.id) < (other.weight, other.id)

    @cached_property
    def edge_updates(self):
        edge_updates = self.context.decoder.calculate_edge_updates(
            self.context.graphdict,
            erasure_checks=self.erasure_checks,
            leakloc_tuples=self.fixed_leak_locs,
            leakloc_probs=self.fixed_leakloc_probs,
            **self.context.circuit_kwargs)

        return edge_updates

    # @profile
    def do_marginal_decode(self):
        """ Do the marginal decoding using this node's erasure checks and fixed leakage locations, which gives us a solution and weight that we can use to prioritize this node in the search. """
        context = self.context
        try:
            sol, weight = context.matcher.decode(
                context.syndrome, edge_reweights=self.edge_updates, return_weight=True)
            self.solution = sol
            self.weight = weight
        except ValueError as e:  # no solution found for this graph
            if str(e).startswith("No perfect matching"):
                self.solution = None
                self.weight = float('inf')
            else:
                raise e

        return sol, weight

    # @profile
    def check_validity(self):
        """ Check if the solution for this node is valid, i.e. if the decoded edges are compatible with the fact that the leakage locations are disjoint. """
        if self.valid is not None:
            return self.valid  # return cached result if already checked
        context = self.context
        decoded_edges = context.matcher.decode_to_edges_array(
            context.syndrome, edge_reweights=self.edge_updates)
        valid, conflict_witness = context.decoder.check_solution_validity(
            decoded_edges, self.erasure_checks, self.fixed_leak_locs, **context.circuit_kwargs)

        # recalculate logical outcome
        if valid:
            edge_tuples = [tuple(sorted(d for d in edge if d != -1)) for edge in decoded_edges]
            edge_data = context.edge_data
            flipped_observables = set()
            for edge in edge_tuples:
                flipped_observables ^= edge_data[edge]
            sol = np.zeros(context.num_observables, np.uint8)
            for to_flip in flipped_observables:
                sol[to_flip] = 1
            self.solution = sol

        # store these for later use in generating child nodes
        matched_edge_set = set(tuple(sorted(d for d in edge if d != -1)) for edge in decoded_edges)
        self.decoded_edges = matched_edge_set
        self.valid = valid
        self.conflict_witness = conflict_witness

        return valid

    # @profile
    def generate_children(self):
        """ Generate child nodes by fixing one more leakage location based on the conflict witness from the validity check. """
        # local references
        context = self.context
        decoder = context.decoder
        erasure_checks = self.erasure_checks
        circuit_kwargs = context.circuit_kwargs
        already_fixed_leak_locs = self.fixed_leak_locs
        already_fixed_leakloc_probs = self.fixed_leakloc_probs

        dem_table, dem_coverage = decoder.get_table_of_dem_edges(
            erasure_checks, **circuit_kwargs)
        conflict_pairs = decoder.find_disjoint_edges(
            self.conflict_witness, self.decoded_edges, dem_table, dem_coverage)

        # these are the leakage locations we want to try fixing/clamping---the ones that cover the edges in the conflict pairs. We will to try removing DEMs which are disjoint with these leakage locations and see if we can get a (valid) MWPM solution on the pared graph
        leak_loc_idxs_to_fix = [
            dem_pos for bad_pair in conflict_pairs for edge in bad_pair for dem_pos in dem_coverage[edge]
            ]

        # Branch on a single erasure check: fixing EC A then EC B reaches the same partially
        # constrained graph as B then A, so branching on several at once duplicates subtrees.
        # Pick the EC most implicated in the conflict pairs, tie-broken by index for determinism.
        ec_counts = Counter(idx[0] for idx in leak_loc_idxs_to_fix)
        if not ec_counts:
            return
        ec_idx = min(ec_counts, key=lambda idx: (-ec_counts[idx], idx))

        # retrieve erasure check indexed by ec_idx and remove it from erasure_checks
        ((ec_rnd, ec_gate, ec_qubit), new_ec) = pop_nth_erasure_check(
            erasure_checks, ec_idx)

        # all possible leakage locations that could trigger this erasure check
        possible_leak_locs = decoder.circuit_builder.single_ec_traceback(
            ec_rnd, ec_gate, ec_qubit,
            circuit_kwargs["ec_sched"], circuit_kwargs["leak_locations"])
        n_leak_locs = len(possible_leak_locs)
        leak_loc_prob = 1/n_leak_locs  # assume uniform probability

        # the possible leak locs for this EC are mutually exclusive and exhaustive, so branching
        # over all of them keeps the search complete
        for ll_idx in range(n_leak_locs):
            leak_loc = tuple(possible_leak_locs[ll_idx])
            new_fixed_leak_locs = [*already_fixed_leak_locs, leak_loc]
            new_fixed_leakloc_probs = [*already_fixed_leakloc_probs, leak_loc_prob]

            new_node = ValidSolutionSearchNode(
                context, new_ec, new_fixed_leak_locs, new_fixed_leakloc_probs)
            new_sol, _ = new_node.do_marginal_decode()
            if new_sol is not None:  # only yield child node if its graph is matchable
                yield new_node
