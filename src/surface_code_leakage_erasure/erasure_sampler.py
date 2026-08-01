import time
import numpy as np
from warnings import warn
from dataclasses import dataclass
from collections import Counter
from dataclasses import asdict
from joblib import Parallel, delayed, cpu_count
# from line_profiler import profile

import stim
import pymatching
import sinter

from .surface_code import SurfaceCodeLayout
from .circuit_builder import SurfaceCodeCircuitBuilder
from .walking_surface_code import WalkingSurfaceCodeLayout
from .walking_circuit_builder import WalkingSCCircuitBuilder
from .decoding import ErasureDecoder, require_pymatching_edge_reweights

## for functions that will be used in parallel, define outside the class for pickling
# @profile
def _single_erasure_sample(circuitbuilder, decoder, pams : 'SamplerParameters', leak_locs = None) -> tuple[int, int]:
    # run single erasure sample
    if leak_locs is None:
        if pams.n_leak is not None:
            total_leak_locs, _, _ = circuitbuilder.count_leakage_locations(
                pams.rounds, leak_locations=pams.circuit_kwargs.leak_locations)
            leak_locs_unraveled = circuitbuilder.rng.choice(
                total_leak_locs, size=pams.n_leak, replace=False)
            leak_locs = circuitbuilder.ravel_leakage_locations(
                pams.rounds, leak_locs_unraveled,
                leak_locations=pams.circuit_kwargs.leak_locations)
        else:
            leak_locs = circuitbuilder.generate_leakage_locations(
                pams.rounds, pams.p_leak,
                leak_locations=pams.circuit_kwargs.leak_locations, atleast_1=True)
    circuit, ec = circuitbuilder.get_circuit(
        pams.rounds, p=pams.p_Pauli, leakage_circuit_locations=leak_locs,
        **asdict(pams.circuit_kwargs))

    fs = stim.FlipSimulator(batch_size=1, num_qubits=circuit.num_qubits)
    fs.do(circuit)

    detection_events = fs.get_detector_flips(bit_packed=True)
    observable_flips = fs.get_observable_flips(bit_packed=True)

    ## decode
    predictions = decoder.decode(
        ec, detection_events, pams.p_Pauli,
        branch_and_bound=pams.use_branch_and_bound,
        mle_time_limit=pams.mle_time_limit,
        **asdict(pams.circuit_kwargs))

    num_failures_marginal = int(np.sum(predictions[0] != observable_flips))
    if pams.use_branch_and_bound:
        if predictions[1] is not None:
            num_failures_MLE = int(np.sum(predictions[1] != observable_flips))
        else:
            num_failures_MLE = num_failures_marginal
    else:
        num_failures_MLE = -1

    return num_failures_marginal, num_failures_MLE


def _erasure_sample_batch(
    layout: SurfaceCodeLayout | WalkingSurfaceCodeLayout, batch_size: int,
    pams : 'SamplerParameters', deadline: float | None = None,
) -> tuple[int, int, int]:
    # implement serial batch of erasure samples
    if isinstance(layout, WalkingSurfaceCodeLayout):
        circuitbuilder = WalkingSCCircuitBuilder(layout)
    else:
        circuitbuilder = SurfaceCodeCircuitBuilder(layout)
    decoder = ErasureDecoder(
        circuitbuilder, cache_size=pams.max_cache_size,
        max_queue_size=pams.max_queue_size)

    # handle trivial/edge cases
    if batch_size <= 0:
        return 0, 0, 0

    # initialize counters
    failures_marginal = 0
    failures_MLE = 0
    num_done = 0

    for i in range(batch_size):
        if deadline is not None and time.time() >= deadline:
            break
        marg, mle = _single_erasure_sample(circuitbuilder, decoder, pams)
        failures_marginal += marg
        failures_MLE += mle
        num_done += 1

    return failures_marginal, failures_MLE, num_done


class SurfaceCodeErasureSampler:
    """ Sampler for estimating logical error rates of surface codes with leakage and erasure checks """
    def __init__(self, layout: SurfaceCodeLayout | WalkingSurfaceCodeLayout, n_jobs: int = 1):
        # check up front so we fail here rather than once per joblib worker,
        # since the decoder is only constructed inside the workers
        require_pymatching_edge_reweights()
        self.layout = layout
        if isinstance(layout, WalkingSurfaceCodeLayout):
            self.circuit_builder_class = WalkingSCCircuitBuilder
            self.mode = "walking"
            self.swap_time = layout.swap_time
        else:
            self.circuit_builder_class = SurfaceCodeCircuitBuilder
            self.mode = "static"
            self.swap_time = None
        if n_jobs == -1:
            n_jobs = cpu_count()  # use all available CPUs
        self.n_jobs = n_jobs
        self.parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
        ## maybe cache some circuits or matchers here

    def single_erasure_sample(self, pams : 'SamplerParameters') -> tuple[int, int]:
        circuitbuilder = self.circuit_builder_class(self.layout)
        decoder = ErasureDecoder(
            circuitbuilder, cache_size=pams.max_cache_size,
            max_queue_size=pams.max_queue_size)
        return _single_erasure_sample(circuitbuilder, decoder, pams)

    def erasure_sample_batch(self, batch_size: int, pams : 'SamplerParameters') -> tuple[int, int, int]:
        return _erasure_sample_batch(self.layout, batch_size, pams)

    def parallel_erasure_samples(
        self, pams: 'SamplerParameters', num_samples: int | None = None,
        deadline: float | None = None,
    ) -> tuple[int, int, int]:
        """ Collect samples with leakage in parallel batches using joblib.

        Returns (failures_marginal, failures_MLE, num_samples_done). If
        deadline (a wall-clock time.time() value) is given, workers stop
        sampling once it passes and return whatever they have so far.
        """
        if num_samples is None:
            num_samples = pams.min_samples
        if num_samples <= 0:
            return 0, 0, 0

        # calculate batch sizes
        n_batches = self.n_jobs
        q, r = divmod(num_samples, n_batches)
        batch_size_list = [q + (1 if i < r else 0) for i in range(n_batches)]

        # initialize counters
        failures_marginal = 0
        failures_MLE = 0
        num_done = 0

        # dispatch to module-level worker
        results = self.parallel(
            delayed(_erasure_sample_batch)(self.layout, batch_size, pams, deadline)
            for batch_size in batch_size_list
        )

        # aggregate results as they arrive
        for res in results:
            failures_marginal += res[0]
            failures_MLE += res[1]
            num_done += res[2]

        return failures_marginal, failures_MLE, num_done

    def sample_without_leakage(self, pams: 'SamplerParameters', num_samples: int | None = None) -> int:
        """ Collect samples without leakage using stim's built-in sampler """
        if num_samples is None:
            num_samples = pams.min_samples
        if num_samples <= 0:
            return 0

        # get samples which have no leakage
        circuitbuilder = self.circuit_builder_class(self.layout)
        circuit, _ = circuitbuilder.get_circuit(
            pams.rounds, p=pams.p_Pauli, p_leak=0.0, **asdict(pams.circuit_kwargs))

        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            num_samples, separate_observables=True, bit_packed=True)

        matcher = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(
                decompose_errors=True, approximate_disjoint_errors=True))

        predictions = matcher.decode_batch(detection_events, bit_packed_shots=True)
        num_failures = int(np.sum(predictions != observable_flips))

        return num_failures

    def sample_LER(
        self, rounds: int, p_Pauli: float, p_leak: float,
        min_samples: int, target_failure_count: int = 0,
        max_samples: int = 1_000_000, *,
        verbose=False,
        use_branch_and_bound=False, mle_time_limit=1,
        max_cache_size=8192, max_queue_size=4096,
        time_limit_hours: float | None = None, deadline: float | None = None,
        n_leak: int | None = None,
        **circuit_kwargs):
        """ Estimate logical error rate by sampling leakage and non-leakage events

        Shots are split into a leakage population and a no-leakage population
        according to the probability that no leakage occurs anywhere in the
        circuit. The no-leakage population is sampled with stim's batch sampler
        and decoded by plain matching; the leakage population is sampled one
        shot at a time and decoded by ErasureDecoder.

        Arguments:
        rounds: number of error correction rounds to simulate
        p_Pauli: probability of Pauli errors occurring
        p_leak: probability of leakage occurring
        min_samples: minimum number of samples to take
        target_failure_count: target number of failures to obtain. If > 0, a
          preliminary run estimates how many samples are needed to reach it;
          the result is clamped to [min_samples, max_samples]. 0 disables the
          estimate and takes exactly min_samples.
        max_samples: maximum number of samples to take
        verbose: whether to print progress messages

        use_branch_and_bound: if True, additionally decode with the
          branch-and-bound search for a solution consistent with the
          disjointness of leakage locations, and report it as the MLE result.
          If False, `failures_MLE` is not computed and `pfail_MLE` is None.
          Only has an effect when leak_effect involves skipped gates; other
          leak_effect values impose no disjointness constraints.
        mle_time_limit: per-shot wall-clock budget, in seconds, for the
          branch-and-bound search. None means no limit. Ignored unless
          use_branch_and_bound is True.
        max_cache_size: maxsize of the ErasureDecoder's LRU caches on its
          DEM-building methods, per worker. 0 disables caching.
        max_queue_size: cap on the size of the branch-and-bound search queue;
          it is trimmed to the lowest-weight nodes whenever it is exceeded.
          Ignored unless use_branch_and_bound is True.

        time_limit_hours: stop sampling after this many hours and return
          whatever has been collected. The no-leakage budget is scaled down
          proportionally so the leak / no-leak mix still reflects prob_no_leak.
        deadline: same as time_limit_hours but given as an absolute
          time.time() value. Takes precedence if both are given.

        n_leak: if set, every shot is given exactly this many leakage
          locations, chosen uniformly at random, instead of sampling each
          location at rate p_leak. Intended for studying the response to a
          fixed leakage count. Note this forces p_Pauli = 0 and routes all
          shots through the leakage path.

        circuit_kwargs: keyword arguments for circuit construction -- basis,
          ec_sched, Pauli_locations, leak_locations, leak_effect. See
          CircuitParameters for defaults and CircuitBuilder.get_circuit for the
          meaning of each value.

        Returns:
        ErasureSamplerResult dataclass with results and statistics
        """

        if min_samples > max_samples:
            warn(f"min_samples {min_samples} is greater than max_samples {max_samples}. Clamping min_samples to max_samples.")
            min_samples = max_samples

        if n_leak is not None and p_Pauli != 0.0:
            warn(f"n_leak={n_leak} is set, so p_Pauli is forced to 0.0 (was {p_Pauli}).")
            p_Pauli = 0.0

        pams = SamplerParameters(
            rounds=rounds, p_Pauli=p_Pauli, p_leak=p_leak,
            min_samples=min_samples,
            target_failure_count=target_failure_count,
            max_samples=max_samples,
            circuit_kwargs=CircuitParameters(**circuit_kwargs),
            mode = self.mode,
            swap_time=self.swap_time,
            use_branch_and_bound=use_branch_and_bound,
            mle_time_limit=mle_time_limit,
            max_cache_size=max_cache_size,
            max_queue_size=max_queue_size,
            n_leak=n_leak,
        )

        # calculate probability of no leakage occurring
        # (force prob_no_leak=0 when sampling at fixed n_leak so all shots go through the leakage path)
        if n_leak is not None:
            prob_no_leak = 0.0
        else:
            num_leaklocs, _, _ = self.circuit_builder_class(
                self.layout).count_leakage_locations(
                    rounds, leak_locations=pams.circuit_kwargs.leak_locations)
            prob_no_leak = (1 - p_leak) ** num_leaklocs

        if deadline is None:  # prioritize deadline over time_limit_hours if both given
            deadline = (time.time() + time_limit_hours * 3600
                        if time_limit_hours is not None else None)

        # estimate required number of samples
        if verbose and target_failure_count > 0:
            print("Estimating required number of samples...")
        num_samples = self._estimate_reqd_samples(pams, prob_no_leak, deadline=deadline)
        num_leakage_samples, num_noleak_samples = self._subdivide_numsamples(
            num_samples, prob_no_leak)
        if verbose:
            print(f"Will take {num_samples} = {num_leakage_samples} + {num_noleak_samples} samples")

        # get leakage samples
        if p_leak > 0 or n_leak is not None:
            if verbose:
                print("Getting leakage samples...")
            leak_failures_marginal, leak_failures_MLE, num_leakage_done = (
                self.parallel_erasure_samples(pams, num_leakage_samples, deadline=deadline))
            # if the timeout cut us short, scale down the no-leakage budget
            # proportionally so the leak / no-leak mix still reflects prob_no_leak
            if num_leakage_samples > 0 and num_leakage_done < num_leakage_samples:
                if verbose:
                    print(f"Warning: Leakage sampling budget of {num_leakage_samples} cut short by deadline to {num_leakage_done} samples. Adjusting no-leakage samples accordingly.")
                num_noleak_samples = int(round(
                    num_noleak_samples * num_leakage_done / num_leakage_samples))
                num_samples = num_leakage_done + num_noleak_samples  # update total sample count to reflect actual samples taken
            num_leakage_samples = num_leakage_done
        else:
            num_leakage_samples = 0
            leak_failures_marginal = 0
            leak_failures_MLE = 0

        # get no-leakage samples
        if p_Pauli > 0:
            if verbose:
                print("Getting no-leakage samples...")
            num_noleak_failures = self.sample_without_leakage(pams, num_noleak_samples)
        else:
            num_noleak_samples = 0
            num_noleak_failures = 0

        # estimate LER
        total_failures_marginal = leak_failures_marginal + num_noleak_failures
        total_failures_MLE = leak_failures_MLE + num_noleak_failures
        pfail_marginal = sinter.fit_binomial(
            num_shots=num_samples, num_hits=total_failures_marginal, max_likelihood_factor=9.0)
        if pams.use_branch_and_bound:
            pfail_MLE = sinter.fit_binomial(
                num_shots=num_samples, num_hits=total_failures_MLE, max_likelihood_factor=9.0)
        else:
            pfail_MLE = None

        return ErasureSamplerResult(
            parameters=pams,
            prob_no_leak=prob_no_leak,
            pfail_marginal=pfail_marginal,
            pfail_MLE=pfail_MLE,
            num_samples=num_samples,
            failures_marginal=total_failures_marginal,
            failures_MLE=total_failures_MLE,
            num_leakage_samples=num_leakage_samples,
            num_noleak_samples=num_noleak_samples,
            leak_failures_marginal=leak_failures_marginal,
            leak_failures_MLE=leak_failures_MLE,
            num_noleak_failures=num_noleak_failures,
        )

    def _estimate_reqd_samples(self, pams: 'SamplerParameters', prob_no_leak: float, deadline: float | None = None) -> int:
        """ Estimate required number of samples based on parameters and preliminary sampling.
        Returned number of samples always between min_samples and max_samples. """
        if pams.target_failure_count < 1:
            return pams.min_samples
        if pams.min_samples >= pams.max_samples:
            return pams.max_samples

        # draw samples until we reach at least 2 failures
        failure_count = 0
        num_samples = 0
        while (failure_count < 2) and (num_samples < pams.max_samples//2):
            if deadline is not None and time.time() >= deadline:
                break
            tmp_num_samples = max(pams.min_samples, num_samples)
            n_leak, n_noleak = self._subdivide_numsamples(tmp_num_samples, prob_no_leak)
            # get leakage samples
            nf, _, n_leak_done = self.parallel_erasure_samples(pams, n_leak, deadline=deadline)
            failure_count += nf
            # if the deadline cut leakage sampling short, scale no-leak budget to match
            if n_leak > 0 and n_leak_done < n_leak:
                n_noleak = int(round(n_noleak * n_leak_done / n_leak))
            # get no-leakage samples
            failure_count += self.sample_without_leakage(pams, n_noleak)
            num_samples += n_leak_done + n_noleak

        # estimate required samples to reach target failure count based on observed failure count
        if failure_count >= 2:
            p_est = (failure_count-1) / (num_samples-1)
            reqd_samples = int(np.ceil(pams.target_failure_count / p_est))
        else:
            reqd_samples = pams.max_samples

        # make sure have at least num_samples and no more than max_samples
        reqd_samples = max(pams.min_samples, reqd_samples)
        reqd_samples = min(reqd_samples, pams.max_samples)
        return reqd_samples

    def _subdivide_numsamples(self, num_samples: int, prob_no_leak: int):
        """ Divide num_samples into leakage and no-leakage samples according to prob_no_leak """
        if prob_no_leak >= 1.0:
            return 0, num_samples
        elif prob_no_leak <= 0.0:
            return num_samples, 0
        fractional_n_noleak = prob_no_leak * num_samples
        rem = fractional_n_noleak % 1
        n_noleak = int(fractional_n_noleak) + (1 if np.random.random() < rem else 0)
        n_leak = num_samples - n_noleak
        return n_leak, n_noleak


@dataclass
class CircuitParameters:
    basis: str = "Z"
    ec_sched: int = 8
    Pauli_locations: str = "2-qubit gates"
    leak_locations: str = "2-qubit gates"
    leak_effect: str = "depolarize"


@dataclass
class SamplerParameters:
    rounds: int
    p_Pauli: float
    p_leak: float

    min_samples: int
    target_failure_count: int
    max_samples: int
    use_branch_and_bound: bool
    mle_time_limit: int | None
    max_cache_size: int
    max_queue_size: int

    circuit_kwargs: CircuitParameters
    mode: str  # "static" or "walking"
    swap_time: str  # walking SC only: "late" = walking, "early" = moonwalking

    n_leak: int | None = None  # if set, sample with exactly this many leakage locations per shot


@dataclass
class ErasureSamplerResult:
    # main results
    parameters: SamplerParameters
    pfail_marginal: sinter.Fit
    pfail_MLE: sinter.Fit | None

    # intermediate results
    prob_no_leak: float
    num_samples: int
    failures_marginal: int
    failures_MLE: int

    # sampler stats
    num_leakage_samples: int
    num_noleak_samples: int

    leak_failures_marginal: int
    leak_failures_MLE: int
    num_noleak_failures: int

