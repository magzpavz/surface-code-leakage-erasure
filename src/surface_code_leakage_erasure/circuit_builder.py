import numpy as np
from abc import ABC, abstractmethod
from warnings import warn
from functools import cache
# from line_profiler import profile
import stim
from .surface_code import SurfaceCodeLayout
from .tracker import MeasurementTracker


def iterate_erasure_checks(erasure_checks : list[list[set[int]]]):
    """ Iterates through erasure checks in format of list of lists of sets of qubit indices, yielding (round number, step number, qubit index) for each erasure check """
    for ec_rnd, ec_rnd_list in enumerate(erasure_checks):
        for ec_gate, ec_qubit_set in enumerate(ec_rnd_list):
            for ec_qubit in sorted(ec_qubit_set):
                yield ec_rnd, ec_gate, ec_qubit


class CircuitBuilder(ABC):
    """ Abstract base class for building error correction circuits

    Subclasses supply the layout-specific pieces (qubit index strings, CNOT
    schedules, detectors, observable) by implementing the abstract methods
    below. The shared, layout-independent circuit construction lives in the
    `_*_impl` helpers, which subclasses call via `super()` after filling in
    their layout-derived arguments. """
    # options for kwargs in get_circuit, etc
    kwargs_options = {
        "basis": ["Z", "X"],
        # first entry is the default used by validate_circuit_kwargs; keep it in
        # sync with the CircuitParameters dataclass in erasure_sampler.py
        "Pauli_locations": ["2-qubit gates", "gates", "all"],
        "leak_locations": ["2-qubit gates", "all"],
        "ec_sched": [8,4,2,1],
        "leak_effect": [
            "depolarize",
            "tailored",
            "skip gates",
            "decode skip gates",
            "tailored then skip",
            "decode tailored then skip"]
    }

    def __init__(self, layout, seed = None):
        self.layout = layout

        self.circuit = None  # only initialize the circuit when building it
        self.tracker = MeasurementTracker()
        self.rng = np.random.default_rng(seed)
        self.leaked = set()  # keep track of which qubits have leaked

        # whether the tracker needs to be (re)populated during the current build.
        # only True when the final observable text for this build's key isn't cached yet.
        self._track_measurements = True
        self._built_observable_keys = set()  # observable cache keys already fully built

        # cache frequently used attributes
        self.qubits = sorted(self.layout.qubits)
        self.qubit_set = set(qubit.index for qubit in self.qubits)
        self.n_qubits = len(self.qubits)

        # set up caches for no-leakage circuits
        self._no_leakage_stab_circuit_cache : dict[tuple, list[str]] = {}
        self._no_leakage_stab_meas_list_cache : dict[tuple, list[int]] = {}
        self._no_leakage_final_circuit_cache : dict[tuple, list[str]] = {}
        self._no_leakage_final_meas_list_cache : dict[tuple, list[int]] = {}

    def __eq__(self, other):
        ## let them be equal even if RNGs are different
        return (type(self) == type(other) and
                self.layout == other.layout and
                self.tracker == other.tracker and
                self.circuit == other.circuit and
                self.leaked == other.leaked
                )

    def _observable_cache_key(self, rounds, basis):
        """ Key identifying the final observable text for a given build.
        The base/walking observable depends on the number of rounds; subclasses
        whose observable is rounds-independent should override this. """
        return (rounds, basis)

    def single_plaquette_cnot_indexes(self, plaquette, t, reverse=False):
        dq = plaquette.ordered_data_qubits[t]
        if dq is None:
            return []
        if plaquette.type == 'X':
            cnot_list = [plaquette.ancilla.index, dq.index]
        else:
            cnot_list = [dq.index, plaquette.ancilla.index]
        if reverse:
            cnot_list = cnot_list[::-1]
        return cnot_list

    def setup_circuit(self, p, *, Pauli_locations, basis):
        self.circuit = [] #stim.Circuit()
        self.tracker = MeasurementTracker()
        self.leaked = set()

        self.circuit.extend(f"QUBIT_COORDS({qubit.coord[0]}, {qubit.coord[1]}) {qubit.index} " for qubit in self.qubits)

        self.initialize_data_qubits(p, Pauli_locations=Pauli_locations, basis=basis)

    @abstractmethod
    def initialize_data_qubits(self, p, *, Pauli_locations : str, basis : str):
        """ Reset the data qubits into the given basis (implemented in subclass) """
        raise NotImplementedError("initialize_data_qubits must be implemented in subclass")

    @abstractmethod
    def measure_stabilizers(
        self, p, rnd : int, *, Pauli_locations : str, basis : str,
        ec_sched : int, leak_effect : str,
        leakage_circuit_locations : dict | None = None,
        use_cache : bool = True):
        """ Measure stabilizers for one round (implemented in subclass)

        The subclass gathers its layout-derived qubit strings/CNOT schedule for
        this round and delegates to `_measure_stabilizers_impl`, or bounces to
        `measure_stabilizers_no_leakage` when the cached no-leakage circuit
        applies. """
        raise NotImplementedError("measure_stabilizers must be implemented in subclass")

    # @profile
    def _measure_stabilizers_impl(
        self, p, rnd : int, *,
        reset_indexes_str : str, dq_indexes_str: str, rx_indexes_str : str,
        idling_qubits_str : list[str], cnot_qubits : list[dict[int, tuple]], cnot_str : list[str],
        measure_set: set, mx_indexes_str: str,
        Pauli_locations : str, basis : str, ec_sched : int, leak_effect : str,
        leakage_circuit_locations : dict | None = None):
        """ Shared implementation of one round of stabilizer measurement

        Arguments:
        p : physical error rate
        rnd : round number (only matters if we're in round 0 or not)

        reset_indexes_str, dq_indexes_str, rx_indexes_str, idling_qubits_str,
            cnot_qubits, cnot_str, measure_set, mx_indexes_str: to be provided by subclass method

        Pauli_locations : where Pauli errors may occur
        basis : "Z" or "X" for logical state being preserved
        ec_sched: frequency of erasure checks
        leak_effect: effect of leaked qubit on other qubit during 2-qubit gate

        leakage_circuit_locations : dict mapping from time step (0-3) to set of qubit indices which leak during that step
        """
        if leakage_circuit_locations is None:
            leakage_circuit_locations = {}

        # reset ancillas
        self.circuit.append(f"R {reset_indexes_str}")

        if Pauli_locations in ["all", "gates"] and p > 0.0:
            self.circuit.append(
                f"X_ERROR({p}) {reset_indexes_str}\nDEPOLARIZE1({p}) {dq_indexes_str}")

        # rotate X ancillas
        self.circuit.append(f"H {rx_indexes_str}")

        if Pauli_locations in ["all", "gates"] and p > 0.0:
            self.circuit.append(f"DEPOLARIZE1({p}) {rx_indexes_str}")

        self.circuit.append("TICK")

        erasure_checks = [set() for _ in range(4)]
        for t in range(4):
            cnot_dict = cnot_qubits[t]

            # create new leakages according to given indexes
            newly_leaked = leakage_circuit_locations.get(t, set()) - self.leaked
            self.leaked.update(newly_leaked)
            # depolarize newly-leaked qubits if necessary
            if len(newly_leaked) > 0 and leak_effect in ["decode skip gates", "decode tailored then skip"]:
                for qubit in newly_leaked:
                    self.circuit.append(f"DEPOLARIZE1(0.75) {qubit}")
            # apply tailored noise from new-leaked qubits if necessary
            if len(newly_leaked) > 0 and leak_effect in ["tailored then skip", "decode tailored then skip"]:
                affected_qubits = newly_leaked & cnot_dict.keys()  # qubits involved in a gate that just leaked
                for qubit in affected_qubits:
                    (other_qubit, idx_in_cnot, _) = cnot_dict[qubit]
                    (q0, q1) = (qubit, other_qubit) if idx_in_cnot%2 == 0 else (other_qubit, qubit)
                    # determine if one or both are leaked
                    p0 = q0 in self.leaked
                    p1 = q1 in self.leaked
                    self.circuit.extend(
                        self.apply_leakage_noise_model("tailored", q0, q1, p0, p1))

            # make the gates
            idling_qubits = idling_qubits_str[t] if Pauli_locations == "all" else ""
            self.make_stabilizer_gates(
                p, cnot_dict, cnot_str[t], idling_qubits,
                leak_effect)
            # print(f"end of round {rnd}, step {t}, leaked qubits: {sorted(self.leaked)}")  # for debugging

            # record and reset erasures where applicable
            if t < 3 and ((t+1) % ec_sched == 0):
                erasure_checks[t] = self.do_erasure_checks(erasure_checks[t])

        # record and reset erasures at final measurement
        erasure_checks[3] = self.do_erasure_checks(
            erasure_checks[3], measure_set if ec_sched == 8 else None)

        ## measure ancillas
        self.circuit.append(f"H {mx_indexes_str}")
        if Pauli_locations in ["all", "gates"] and p > 0.0:
            self.circuit.append(f"DEPOLARIZE1({p}) {mx_indexes_str}")

        p_meas = p if Pauli_locations in ["all", "gates"] else 0.0
        self.three_way_measurement(p_meas, measure_set & self.qubit_set, erasure_checks[-1])

        if Pauli_locations == "all" and p > 0.0:
            self.circuit.append(f"DEPOLARIZE1({p}) {dq_indexes_str}")

        self.circuit.append("TICK")

        self.define_detectors(rnd, basis)

        return erasure_checks

    # @profile
    def measure_stabilizers_no_leakage(self, p, rnd_eff : int, *, Pauli_locations : str, basis : str, ec_sched : int):
        # use cached circuit for no-leakage case if available, otherwise build it and cache it

        # print("In parent class measure_stabilizers_no_leakage")

        if (p, rnd_eff, Pauli_locations, basis, ec_sched) in self._no_leakage_stab_circuit_cache:

            circuit_text = self._no_leakage_stab_circuit_cache[(p, rnd_eff, Pauli_locations, basis, ec_sched)]
            self.circuit.append(circuit_text)
            if self._track_measurements:
                measurement_list = self._no_leakage_stab_meas_list_cache[(p, rnd_eff, Pauli_locations, basis)]
                self.tracker.add_measurement_list(measurement_list)
        else:  # extract the circuit and tracker for cache

            # save the current length of the circuit and tracker measurement list so that I can extract and cache the new portion at end
            circuit_starting_len = len(self.circuit)
            tracker_starting_len = len(self.tracker.measurement_list)

            self.measure_stabilizers(
                p, rnd_eff, Pauli_locations=Pauli_locations, basis=basis,
                ec_sched=ec_sched, leak_effect="depolarize", use_cache=False)
            # leak_effect doesn't matter since there's no leakage, just need to put something for the function to run

            # extract and cache the circuit and tracker measurement list for no-leakage case
            circuit_text = "\n".join(self.circuit[circuit_starting_len:])
            self._no_leakage_stab_circuit_cache[(p, rnd_eff, Pauli_locations, basis, ec_sched)] = circuit_text

            measurement_list = self.tracker.measurement_list[tracker_starting_len:]
            self._no_leakage_stab_meas_list_cache[(p, rnd_eff, Pauli_locations, basis)] = measurement_list

        return [set() for _ in range(4)]  # no erasure checks since no leakage

    # @profile
    def make_stabilizer_gates(
        self, p,
        cnot_dict: dict, cnot_str: str, idling_qubits_str: str,
        leak_effect : str):

        leakage_noise_strs = []

        # entangling gates
        if len(self.leaked) > 0:
            if leak_effect in ["skip gates", "tailored then skip"]:
                extra_idling_qubits = []
                to_remove = cnot_dict.keys() & self.leaked  # qubits that are involved in a gate and are leaked
                str_idxs_to_remove = []
                for qubit in to_remove:
                    (other_qubit, idx_in_cnot, _) = cnot_dict[qubit]
                    str_idxs_to_remove.append(8 * (idx_in_cnot//2))
                    extra_idling_qubits.extend([qubit, other_qubit])

                str_idxs_to_remove.sort()
                str_idxs_to_remove.insert(0, -8)
                str_idxs_to_remove.append(None)
                cnot_str = "".join(
                    cnot_str[(str_idxs_to_remove[i]+8):str_idxs_to_remove[i+1]]
                    for i in range(len(str_idxs_to_remove)-1))

                if len(idling_qubits_str) > 0:
                    # add idling qubits from skipped gates iff there were already idling qubits (i.e. we are including idling noise)
                    extra_idling_qubits = set(extra_idling_qubits)  # remove duplicates
                    idling_qubits_str += " " + " ".join(str(q) for q in extra_idling_qubits)
            elif leak_effect in ["decode skip gates", "decode tailored then skip"]:
                affected_qubits = cnot_dict.keys() & self.leaked  # qubits that are involved in a gate and are leaked
                # print(f"Leaked qubits: {sorted(self.leaked)}, involved in gates: {sorted(to_remove)}")  # for debugging
                for qubit in affected_qubits:
                    role_change = cnot_dict[qubit][-1]
                    if role_change:
                        leakage_noise_strs.append(f"DEPOLARIZE1(0.75) {qubit}")

        # Pauli errors
        Pauli_error_strs = []
        if p > 0.0:
            Pauli_error_strs.append(
                f"DEPOLARIZE2({p}) {cnot_str}")

            if len(idling_qubits_str) > 0:
                    Pauli_error_strs.append(
                    f"DEPOLARIZE1({p}) {idling_qubits_str}")

        # implement effect of leakages during 2-qubit gates
        if leak_effect in ["depolarize","tailored"]:
            affected_qubits = self.leaked & cnot_dict.keys()  # qubits involved in a gate that are leaked
            already_added = []  # to avoid double counting noise from pairs of leaked qubits
            for qubit in affected_qubits:
                if qubit in already_added:
                    continue
                (other_qubit, idx_in_cnot, _) = cnot_dict[qubit]
                (q0, q1) = (qubit, other_qubit) if idx_in_cnot%2 == 0 else (other_qubit, qubit)
                # actual or probable leakages
                p0 = q0 in self.leaked
                p1 = q1 in self.leaked
                leakage_noise_strs.extend(
                    self.apply_leakage_noise_model(leak_effect, q0, q1, p0, p1))
                already_added.extend([q0, q1])

        # change order of operations depending on leak effect model
        if leak_effect in ["decode skip gates", "decode tailored then skip"]:
            # first do leakage noise, then gates, then Pauli noise
            self.circuit.append(
                "\n".join(leakage_noise_strs + [f"CX {cnot_str}"] + Pauli_error_strs))
        else:
            # first do gates, then Pauli noise, then leakage noise
            self.circuit.append(
                "\n".join([f"CX {cnot_str}"] + Pauli_error_strs + leakage_noise_strs))

        self.circuit.append("TICK")

    def apply_leakage_noise_model(self, leak_effect : str, q0 : int, q1 : int, p0 : bool, p1 : bool):
        """ Apply leakage noise model to a pair of qubits given their leakage probabilities

        Arguments:
        leak_effect : "depolarize" or "tailored" for leakage model
        q0 : index of first qubit (control)
        q1 : index of second qubit (target)
        p0 : if first qubit is leaked
        p1 : if second qubit is leaked

        Returns:
        list of stim program strings to append to circuit
        """
        noise_strs = []
        if p0 and (not p1):
            if leak_effect == "depolarize":  # depolarizing channel on target
                noise_strs.append(f"DEPOLARIZE1({3/4}) {q1}")
            elif leak_effect == "tailored":  # X error channel on target
                noise_strs.append(f"X_ERROR({1/2}) {q1}")

        if (not p0) and p1:
            if leak_effect == "depolarize":  # depolarizing channel on control
                noise_strs.append(f"DEPOLARIZE1({3/4}) {q0}")
            elif leak_effect == "tailored":  # Z error channel on control
                noise_strs.append(f"Z_ERROR({1/2}) {q0}")

        return noise_strs

    def do_erasure_checks(self, erasure_checks_t : set, ec_subset : set | None = None):
        # record erasure checks
        if ec_subset is None:
            erasure_checks_t |= self.leaked  # use union-equals or else end up with erasure_checks_t being a reference to the same set as self.leaked
        else:
            erasure_checks_t |= self.leaked & ec_subset

        # reset leakages
        self.leaked -= erasure_checks_t
        erased_qubits = erasure_checks_t & self.qubit_set  # make sure only real qubits, not erasure ancillas
        if len(erased_qubits) > 0:
            erased_qubits = ' '.join(str(q) for q in erased_qubits)
            self.circuit.append(f"DEPOLARIZE1(0.75) {erased_qubits}")

        return erasure_checks_t

    @abstractmethod
    def define_detectors(self, rnd : int, basis : str = "Z"):
        """ Append this round's detectors to the circuit (implemented in subclass) """
        raise NotImplementedError("define_detectors must be implemented in subclass")

    # @profile
    def finish_circuit(
        self, rounds : int, p : float, *, Pauli_locations : str, basis : str,
        use_cache : bool = True):

        erasure_checks = self.final_measurement(p, Pauli_locations, basis, use_cache=use_cache)
        self.final_detectors(basis=basis)
        self.final_observable(rounds=rounds, basis=basis)
        return erasure_checks

    @abstractmethod
    def final_measurement(self, p, Pauli_locations : str, basis : str = "Z",
                          use_cache : bool = True):
        """ Measure the data qubits at the end of the circuit (implemented in subclass)

        The subclass looks up its data qubit set/index string and delegates to
        `_final_measurement_impl`.

        Returns:
        erasure_checks : set of data qubit indices that were leaked at readout """
        raise NotImplementedError("final_measurement must be implemented in subclass")

    def _final_measurement_impl(self, p, dq_set, dq_str,
                                Pauli_locations : str, basis : str = "Z",
                                use_cache : bool = True):
        # shared implementation of the final measurement of data qubits

        # check if there is leakage or erasure checks. if not, bounce to no-leakage version of final_measurement which can be cached
        if len(self.leaked) == 0 and use_cache:
            return self.final_measurement_no_leakage(p, Pauli_locations=Pauli_locations, basis=basis)

        if basis == "X":
            # apply Hadamards to data qubits to switch back to Z basis for measurement
            self.circuit.append(f"H {dq_str}")

        erasure_checks = self.leaked & dq_set  # record erasure checks

        p_meas = p if Pauli_locations in ["all", "gates"] else 0.0
        self.three_way_measurement(p_meas, dq_set, erasure_checks)

        return erasure_checks

    # @profile
    def final_measurement_no_leakage(self, p, Pauli_locations : str, basis : str = "Z"):

        if (p, Pauli_locations, basis) in self._no_leakage_final_circuit_cache:
            circuit_text = self._no_leakage_final_circuit_cache[(p, Pauli_locations, basis)]
            self.circuit.append(circuit_text)
            if self._track_measurements:
                measurement_list = self._no_leakage_final_meas_list_cache[(p, Pauli_locations, basis)]
                self.tracker.add_measurement_list(measurement_list)

        else:  # extract the circuit for cache
            circuit_starting_len = len(self.circuit)
            tracker_starting_len = len(self.tracker.measurement_list)

            self.final_measurement(
                p, Pauli_locations, basis, use_cache=False)

            circuit_text = "\n".join(self.circuit[circuit_starting_len:])
            self._no_leakage_final_circuit_cache[(p, Pauli_locations, basis)] = circuit_text

            measurement_list = self.tracker.measurement_list[tracker_starting_len:]
            self._no_leakage_final_meas_list_cache[(p, Pauli_locations, basis)] = measurement_list

        return set()  # no erasure checks since no leakage

    @abstractmethod
    def final_detectors(self, basis : str):
        """ Append the final round of detectors to the circuit (implemented in subclass) """
        raise NotImplementedError("final_detectors must be implemented in subclass")

    @abstractmethod
    def final_observable(self, rounds : int, basis : str = "Z"):
        """ Append the logical observable to the circuit (implemented in subclass)

        `rounds` is the total number of rounds; codes whose observable only
        involves the final round can ignore it. """
        raise NotImplementedError("final_observable must be implemented in subclass")

    # @profile
    def get_circuit(
        self, rounds : int, p : float, p_leak : float = 0.0,
        leakage_circuit_locations : dict | None = None, use_cache = True, **circuit_kwargs):
        """ Construct the error correction circuit

        Arguments:
        rounds: number of rounds of error correction
        p: Pauli error rate
        p_leak: probability of erasure per 2-qubit gate
        leakage_circuit_locations: specific circuit locations to put leakages
          if this is given, p_leak is ignored
        use_cache: if True (default), reuse cached no-leakage sub-circuits across
          rounds and builds for faster construction. Set to False to force a full
          rebuild without the cache (useful for debugging)
        circuit_kwargs: keyword arguments for circuit construction (first is default)
          basis: "Z" or "X" basis for logical qubit preparation and measurement
          Pauli_locations: where Pauli errors may occur
            "gates": after every gate, including reset and measurement
            "all": at all spacetime locations, including idling,
            "2-qubit gates": only after 2-qubit gates
          leak_locations: where leakage may occur
            "2-qubit gates": only during (before) 2-qubit gates
            "all": idling qubits may also leak during (before) each 2-qubit gate
          ec_sched: how frequently erasure checks are performed
            8: only on ancilla qubits during stabilizer measurements
            4: on all qubits during stabilizer measurements
            2: on all qubits every other step
            1: on all qubits every step
          leak_effect: effect of leaked qubit on other qubit during 2-qubit gate
            "depolarize": gate still applied; the partner of a leaked qubit is
              depolarized (p=3/4)
            "tailored": gate still applied; the partner of a leaked qubit gets an
              X error (p=1/2) if it is the target, or a Z error (p=1/2) if it is
              the control
            "skip gates": any gate involving a leaked qubit is removed from the
              circuit, and no error is applied to either qubit
            "tailored then skip": on the step where a qubit first leaks, its gate
              partner gets the "tailored" error; on every later step, gates
              involving the leaked qubit are skipped as in "skip gates"
            Only the three models above are physical noise models intended for
            sampling. The two below are their decoding counterparts: skipped
            gates cannot be represented in a stim detector error model, so they
            are replaced by a depolarizing channel on the leaked qubit at
            appropriate circuit locations.
            validate_circuit_kwargs(..., decode=True) substitutes them
            automatically, so you normally pass "skip gates" or "tailored then
            skip" and never these directly.
            "decode skip gates": decoding counterpart of "skip gates"
            "decode tailored then skip": decoding counterpart of
              "tailored then skip"
        """

        if leakage_circuit_locations is None:
            leakage_circuit_locations = {}

        circuit_kwargs = self.validate_circuit_kwargs(circuit_kwargs)
        leak_locations = circuit_kwargs.pop("leak_locations")

        if p_leak > 0.0:
            if len(leakage_circuit_locations) > 0:
                warn("Ignoring p_leak since leakage_circuit_locations is provided.")
                p_leak = 0.0
            else:
                # generate leakage locations
                leakage_circuit_locations = self.generate_leakage_locations(
                    rounds, p_leak, leak_locations=leak_locations)
        erasure_checks = []

        # setup
        self.setup_circuit(
            p, Pauli_locations=circuit_kwargs["Pauli_locations"], basis=circuit_kwargs["basis"])

        # only (re)populate the tracker if the final observable text for this build
        # isn't cached yet (e.g. this rounds value hasn't been built before)
        observable_key = self._observable_cache_key(rounds, circuit_kwargs["basis"])
        self._track_measurements = observable_key not in self._built_observable_keys

        # rounds
        for rnd in range(rounds):
            ec_round = self.measure_stabilizers(
                p, rnd, **circuit_kwargs,
                leakage_circuit_locations=leakage_circuit_locations.get(rnd, {}),
                use_cache=use_cache)
            erasure_checks.append(ec_round)

        # end
        erasure_checks[-1][-1] |= self.finish_circuit(
            rounds, p, Pauli_locations=circuit_kwargs["Pauli_locations"], basis=circuit_kwargs["basis"], use_cache=use_cache)

        # observable text for this key is now cached; future builds with the same
        # key don't need to repopulate the tracker
        self._built_observable_keys.add(observable_key)

        return stim.Circuit("\n".join(self.circuit)), erasure_checks

    # @profile
    def three_way_measurement(self, p, qubit_set, leaked_set):

        qubit_list = sorted(list(qubit_set))  # sorted to ensure consistent order
        self.tracker.add_measurement_list(qubit_list)

        p_str = f"({p})" if (p > 0.0) else ""
        measure_str = "\n".join(
            f"M{'(0.5)' if q in leaked_set else p_str} {q}" for q in qubit_list)
        self.circuit.append(measure_str)

    def validate_circuit_kwargs(self, circuit_kwargs, decode=False):
        validated_kwargs = {k: vlist[0] for k, vlist in self.kwargs_options.items()}
        validated_kwargs.update(circuit_kwargs)
        for name, value in validated_kwargs.items():
            if name not in self.kwargs_options:
                raise ValueError(f"Unknown circuit kwarg name: {name}.")
            elif value not in self.kwargs_options[name]:
                raise ValueError(f"Invalid value {value} for circuit kwarg {name}. "
                                 f"Valid options are: {self.kwargs_options[name]}")

        if decode:
            leak_effect = validated_kwargs["leak_effect"]
            if ("skip" in leak_effect) and ("decode" not in leak_effect):
                validated_kwargs["leak_effect"] = "decode " + leak_effect
        return validated_kwargs

    ## below here is all related to leakage/erasure and decoding circuit construction
    @abstractmethod
    def single_ec_traceback(self, ec_rnd : int, ec_gate : int, ec_index : int, ec_sched : int, leak_locations : str = "2-qubit gates"):
        """ Trace back a single erasure check to find when it may have leaked (implemented in subclass)

        The subclass supplies its CNOT schedule and the qubit the erasure may
        have been swapped from, then delegates to
        `_single_ec_traceback_impl`.

        Returns:
        possible_leak_locs : list of [round number, step number, qubit index] for each possible leakage location that could have caused the erasure check """
        raise NotImplementedError("single_ec_traceback must be implemented in subclass")

    def _single_ec_traceback_impl(self, ec_rnd : int, ec_gate : int, ec_index : int, ec_sched : int, leak_locations : str, cnot_qubits : list[list[int]], ec_index_prev : int):
        """ Shared implementation of tracing back a single erasure check

        Arguments:
        ec_rnd : round number of erasure check
        ec_gate : gate step number of erasure check (0-3)
        ec_index : qubit index of erasure check
        ec_sched : frequency of erasure checks (1,2,4,8: most -> least frequent)
        leak_locations : where leakage may occur ("2-qubit gates" or "all")
        cnot_qubits: list of CNOT qubit indexes per step per round (provided by subclass)
        ec_index_prev: qubit index that erasure may have come from in previous round (provided by subclass)

        Returns:
        possible_leak_locs : list of [round number, step number, qubit index] for each possible leakage location that could have caused the erasure check
        """

        if leak_locations == "2-qubit gates":
            round_leak_locs = self._trace_qubit_interactions(
                ec_gate, ec_index, ec_sched, cnot_qubits[ec_rnd%2])
        elif leak_locations == "all":
            round_leak_locs = [
                [t, ec_index, idx] for (idx, t)
                in enumerate(range(ec_gate, max(ec_gate - ec_sched, -1), -1))]
        else:
            raise ValueError(f"Invalid leak_locations value: {leak_locations}")

        leak_loc_count = len(round_leak_locs)
        possible_leak_locs = [[ec_rnd] + e[:-1] for e in round_leak_locs]  # don't need interaction count for this version since we're not calculating probabilities here

        # if ec_sched == 8, ancilla erasure may have come from data qubit erasure in previous round
        if (ec_sched == 8) and (ec_rnd > 0) and (ec_index_prev is not None):
            if leak_locations == "2-qubit gates":
                round_leak_locs = self._trace_qubit_interactions(
                    ec_gate, ec_index_prev, ec_sched, cnot_qubits[(ec_rnd-1)%2])
            elif leak_locations == "all":
                round_leak_locs = [
                    [t, ec_index_prev, idx] for (idx, t)
                    in enumerate(range(ec_gate, max(ec_gate - ec_sched, -1), -1))]

            # this is so we know how likely it is that data qubit was leaked in previous round
            possible_leak_locs.extend([[ec_rnd-1] + e[:-1] for e in round_leak_locs])
            leak_loc_count += len(round_leak_locs)

        # reverse so earliest possible leakage locations are first in the list, which are also most likely
        return possible_leak_locs[::-1]

    def _trace_qubit_interactions(self, gate_index : int, qubit_index : int, ec_sched : int, cnot_qubits : list[int]):
        """ Helper function to count qubit interactions back until is was last reset for determining when it may have leaked

        Arguments:
        gate_index : gate step number of erasure check (0-3)
        qubit_index : qubit index of erasure check
        ec_sched : frequency of erasure checks (1,2,4,8: most -> least frequent)
        cnot_qubits : list of CNOT qubit indexes per step

        Returns:
        possible_leak_locs : list of [gate index, qubit index, interaction count]
        """

        possible_leak_locs = [] # list of (gate index, qubit indexes, interaction count)

        gate_range = range(gate_index, gate_index - ec_sched, -1)  # relevant list of gate indexes
        gate_slice = slice(gate_range.start, gate_range.stop if gate_range.stop >= 0 else None, -1)  # slice to get relevant portion of cnot_qubits

        interaction_count = 0  # number of interactions this erased qubit could have had
        for (t, interaction_list) in zip(gate_range, cnot_qubits[gate_slice]):
            if qubit_index in interaction_list:
                possible_leak_locs.append([t, qubit_index, interaction_count])
                interaction_count += 1

        return possible_leak_locs

    def generate_bernoulli_indexes(self, p : float, num_samples : int, atleast_1 : bool = False):
        """ Generate indexes according to Bernoulli process
        Arguments:
        p : probability of hit
        num_samples : number of samples to draw from
        atleast_1 : whether to guarantee at least one hit

        Returns:
        idx_list : list of indexes of hits
        """
        idx_list = []
        idx = self.rng.geometric(p) - 1
        if atleast_1:
            idx %= num_samples
        while idx < num_samples:
            idx_list.append(idx)
            idx += self.rng.geometric(p)
        return idx_list

    def generate_leakage_locations(
        self, rounds : int, p_leak : float,
        atleast_1 : bool = False, leak_locations : str = "2-qubit gates"):
        """ Generate leakage locations for the circuit

        Arguments:
        rounds : number of rounds of error correction
        p_leak : probability of leakage per leakage location
        atleast_1 : whether to guarantee at least one leakage location
        leak_locations : where leakages may occur
          "2-qubit gates": assign a leakage with probability p_leak before each 2-qubit gate
            Which of the two qubits leaks is chosen randomly
          "all": assign a leakage with probability p_leak to each qubit before
            each of the 4 CNOT steps, whether or not it is involved in a gate

        Returns:
        leak_locs : dict[dict[set]] {round number: {step number: {leaked qubit indices}}}
        """
        total_leak_locs, _, _ = self.count_leakage_locations(rounds, leak_locations)

        # generate leakage locations
        leak_locs_unraveled = self.generate_bernoulli_indexes(
            p_leak, total_leak_locs, atleast_1=atleast_1)

        return self.ravel_leakage_locations(rounds, leak_locs_unraveled, leak_locations=leak_locations)

    @abstractmethod
    def count_leakage_locations(self, rounds : int, leak_locations : str = "2-qubit gates"):
        """ Count the leakage locations in the circuit (implemented in subclass)

        Arguments:
        rounds : number of rounds of error correction
        leak_locations : where leakages may occur ("2-qubit gates" or "all")

        Returns:
        (total_leakage_locations, leakage_locations_per_round, leakage_locations_per_step) """
        raise NotImplementedError("count_leakage_locations must be implemented in subclass")

    @abstractmethod
    def ravel_leakage_locations(self, rounds : int, leak_locs_unraveled : list, leak_locations : str = "2-qubit gates"):
        """ Convert unraveled leakage locations into per-round, per-step leakage locations (implemented in subclass)

        The subclass supplies its CNOT schedule and delegates to
        `_ravel_leakage_locations_impl`.

        Returns:
        leak_locs : dict[dict[set]] {round number: {step number: {leaked qubit indices}}} """
        raise NotImplementedError("ravel_leakage_locations must be implemented in subclass")

    def _ravel_leakage_locations_impl(self, rounds : int, leak_locs_unraveled : list, leak_locations : str, cnot_qubits : list):
        """ Shared implementation of converting unraveled leakage locations into per-round, per-step leakage locations

        Arguments:
        rounds : total number of rounds of error correction
        leak_locs_unraveled : list of unraveled leakage location indexes
        leak_locations : where leakages may occur ("2-qubit gates" or "all")
        cnot_qubits : list of CNOT qubit indexes per step per round (provided by subclass)

        Returns:
        leak_locs : dict[dict[set]] {round number: {step number: {leaked qubit indices}}}
        """

        total_leak_locs, leak_locs_per_round, leak_locs_per_step = \
            self.count_leakage_locations(rounds, leak_locations)

        leak_locs = {rnd: {step: set() for step in range(4)} for rnd in range(rounds)}
        for loc in leak_locs_unraveled:
            rnd = loc // leak_locs_per_round
            idx_in_round = loc % leak_locs_per_round
            step = np.argmax(idx_in_round < np.cumsum(leak_locs_per_step))
            if leak_locations == "2-qubit gates":
                gate = idx_in_round - sum(leak_locs_per_step[:step])
                qubit_idx = cnot_qubits[rnd%2][step][gate*2 + self.rng.choice(2)]
            else:
                qubit_idx = idx_in_round - sum(leak_locs_per_step[:step])
            leak_locs[rnd][step].add(qubit_idx)

        return leak_locs


class SurfaceCodeCircuitBuilder(CircuitBuilder):
    """ Circuit builder for the rotated surface code, handling leakage and erasure checks
    Main function: get_circuit """
    # TODO:
    # handle imperfect erasure check measurements
    def __init__(self, d : int | SurfaceCodeLayout, seed = None):
        if isinstance(d, SurfaceCodeLayout):
            layout = d
            d = layout.d
        else:
            layout = SurfaceCodeLayout(d)

        super().__init__(layout, seed)

        # cache frequently used attributes
        self.dq_indexes = sorted([qubit.index for qubit in self.layout.data_qubits])
        self.dq_indexes_str = " ".join(str(q) for q in self.dq_indexes)
        self.n_dq = len(self.dq_indexes)
        self.z_plaquettes = sorted(self.layout.z_plaquettes)  # need to be sorted for consistent detector definitions between get_circuit and get_decoding_circuit!
        self.x_plaquettes = sorted(self.layout.x_plaquettes)
        self.plaquettes = self.x_plaquettes + self.z_plaquettes
        z_indexes = [plaquette.ancilla.index for plaquette in self.z_plaquettes]
        self.z_indexes_str = " ".join(str(q) for q in z_indexes)
        x_indexes = [plaquette.ancilla.index for plaquette in self.x_plaquettes]
        self.x_indexes_str = " ".join(str(q) for q in x_indexes)
        self.stab_ancillas_set = frozenset(x_indexes + z_indexes)

        # precompute per-step CNOT index lists to avoid rebuilding them each round
        self.cnot_lists : list[list[int]] = []
        self.cnot_dicts : list[dict[int, tuple]] = []
        self.cnot_strs : list[str] = []
        self.idling_qubits_str: list[str] = []
        all_qubit_indexes = set(q.index for q in self.qubits)
        qubit_roles = {}  # {qubit index: [0 for control, 1 for target per CNOT it participates in]}
        first_gate_per_qubit = {}  # {qubit index: step index of first gate it participates in}
        for t in range(4):
            list_index = 0  # index of qubit in list of CNOT qubits for this step
            cnot_qubits_list = []
            cnot_qubits_dict = {}
            for plaquette in self.plaquettes:
                cnot_qubits = self.single_plaquette_cnot_indexes(plaquette, t)
                cnot_qubits_list.extend(cnot_qubits)
                for (role, qubit) in enumerate(cnot_qubits):
                    if qubit not in first_gate_per_qubit:
                        first_gate_per_qubit[qubit] = t

                    if qubit in qubit_roles:
                        prev_role = qubit_roles[qubit][-1]
                        qubit_roles[qubit].append(role)
                        role_change = role != prev_role
                    else:
                        qubit_roles[qubit] = [role]
                        role_change = None  # will fix this after by comparing first role to last role
                        # ignore the first-round "data qubit surely 0" thing for now because I don't expect it to affect decoding performance much
                        # qubit_is_0 = role == 0 if qubit in self.dq_indexes else False  #
                    cnot_qubits_dict[qubit] = (cnot_qubits[1-role], list_index, role_change)
                    list_index += 1

            self.cnot_lists.append(cnot_qubits_list)
            self.cnot_dicts.append(cnot_qubits_dict)
            self.cnot_strs.append(' '.join(f"{q:3d}" for q in cnot_qubits_list))
            idling_qubits = all_qubit_indexes - set(cnot_qubits_list)
            self.idling_qubits_str.append(' '.join(str(q) for q in idling_qubits))
        # figure out if role-change in first gate per round for each qubit
        for qubit, t in first_gate_per_qubit.items():
            first_role = qubit_roles[qubit][0]
            last_role = qubit_roles[qubit][-1]
            self.cnot_dicts[t][qubit] = (
                self.cnot_dicts[t][qubit][0],
                self.cnot_dicts[t][qubit][1],
                last_role != first_role)

        # precompute the pairs that are used for erasure swaps
        self.erasure_swaps : dict[int, int] = {} # map from data qubit index to its erasure swap partner
        self.erasure_unswaps : dict[int, int] = {} # map from ancilla index to its data qubit partner
        for plaquette in self.plaquettes:
            dq = plaquette.ordered_data_qubits[0]
            if dq is not None:
                self.erasure_swaps[dq.index] = plaquette.ancilla.index
                self.erasure_unswaps[plaquette.ancilla.index] = dq.index
        remaining_qubits = list(set(self.dq_indexes) - set(self.erasure_swaps.keys()))
        self.erasure_ancillas = list(range(self.n_qubits, self.n_qubits+d+1))
            # indexes of extra ancillas used only for erasure checks (not part of stim circuit)
        self.erasure_swaps.update(zip(remaining_qubits, self.erasure_ancillas))
        self.erasure_unswaps.update(zip(self.erasure_ancillas, remaining_qubits))

        # set up caches for detector definitions
        self._get_detector_text = cache(self._get_detector_text)
        self._get_final_detector_text = cache(self._get_final_detector_text)
        self._get_final_observable_text = cache(self._get_final_observable_text)

        self.count_leakage_locations = cache(self.count_leakage_locations)

    def __repr__(self):
        return f"SurfaceCodeCircuitBuilder(d={self.layout.d})"

    def _observable_cache_key(self, rounds, basis):
        # static surface code observable only uses the final round, so it's
        # rounds-independent
        return (basis,)

    def initialize_data_qubits(self, p, * , Pauli_locations, basis):
        circuit = self.circuit  # local reference
        circuit.append(f"R {self.dq_indexes_str}")
        if Pauli_locations in ["all", "gates"] and p > 0.0:
            circuit.append(f"X_ERROR({p}) {self.dq_indexes_str}")
        if basis == "X":
            circuit.append(f"H {self.dq_indexes_str}")

    # @profile
    def measure_stabilizers(
        self, p, rnd : int, *, Pauli_locations : str, basis : str,
        ec_sched : int, leak_effect : str,
        leakage_circuit_locations : dict | None = None,
        use_cache = True):
        if leakage_circuit_locations is None:
            leakage_circuit_locations = {}

        # check if there is or will be leakage. if not, bounce to no-leakage version of measure_stabilizers which can be cached
        if (len(self.leaked) == 0 and not any(leakage_circuit_locations.values())
            and use_cache):
            rnd_eff = 0 if rnd == 0 else 1  # the circuit is the same for all rounds >= 1
            return super().measure_stabilizers_no_leakage(
                p, rnd_eff, Pauli_locations=Pauli_locations, basis=basis, ec_sched=ec_sched)

        qubit_kwargs = dict(
            reset_indexes_str=self.x_indexes_str + " " + self.z_indexes_str,
            dq_indexes_str=self.dq_indexes_str,
            rx_indexes_str=self.x_indexes_str,
            idling_qubits_str=self.idling_qubits_str,
            cnot_qubits=self.cnot_dicts,
            cnot_str=self.cnot_strs,
            measure_set=self.stab_ancillas_set | set(self.erasure_ancillas),
            mx_indexes_str=self.x_indexes_str,
        )
        circuit_kwargs = dict(
            Pauli_locations=Pauli_locations, basis=basis, ec_sched=ec_sched, leak_effect=leak_effect,
            leakage_circuit_locations = leakage_circuit_locations)

        # swap erasures from data to ancillas if applicable
        if ec_sched == 8:
            # need to depolarize qubits that are erasure-swapped since they are effectively reset
            if len(self.leaked) > 0:
                self.circuit.append(
                    f"DEPOLARIZE1(0.75) {' '.join(str(q) for q in self.leaked)}")

            # perform erasure swaps
            self.leaked = {self.erasure_swaps[q] for q in self.leaked}

        erasure_checks = super()._measure_stabilizers_impl(p, rnd, **qubit_kwargs, **circuit_kwargs)

        return erasure_checks

    # @profile
    def define_detectors(self, rnd : int, basis : str = "Z"):
        detector_text = self._get_detector_text(rnd, basis)
        self.circuit.append(detector_text)

    def _get_detector_text(self, rnd : int, basis : str):
        recs_list = []
        if rnd == 0:
            plaq_list = self.z_plaquettes if basis == "Z" else self.x_plaquettes
            for plaquette in plaq_list:
                idx = self.tracker.get_curr_meas(plaquette.index)
                recs_list.append(f"rec[{idx}]")
            self.detectors_in_round_0 = len(recs_list)
        elif rnd == 1:
            for plaquette in self.plaquettes:
                idx_curr = self.tracker.get_curr_meas(plaquette.index)
                idx_prev = self.tracker.get_prev_meas(plaquette.index)
                recs_list.append(f"rec[{idx_curr}] rec[{idx_prev}]")
            self.detectors_per_round = len(recs_list)
        else:
            return self._get_detector_text(1, basis)  # detectors are the same for all rounds >= 1

        return '\n'.join(f"DETECTOR {recs}" for recs in recs_list)

    # @profile
    def final_measurement(self, p, Pauli_locations : str, basis : str = "Z",
                          use_cache=True):
        # final measurement of data qubits

        dq_set = set(self.dq_indexes)
        return super()._final_measurement_impl(
            p, dq_set, self.dq_indexes_str, Pauli_locations, basis, use_cache=use_cache)

    # @profile
    def final_detectors(self, basis : str = "Z"):
        detector_text = self._get_final_detector_text(basis)
        self.circuit.append(detector_text)

    def _get_final_detector_text(self, basis : str):
        recs_list = []
        plaq_list = self.z_plaquettes if basis == "Z" else self.x_plaquettes
        for plaquette in plaq_list:
            idx_list = []
            plaquette_time = self.tracker.get_curr_meas(plaquette.index)
            idx_list.append(plaquette_time)
            for qubit in plaquette.ordered_data_qubits:
                if qubit is not None:
                    qubit_time = self.tracker.get_curr_meas(qubit.index)
                    idx_list.append(qubit_time)
            recs_list.append(" ".join(f"rec[{idx}]" for idx in idx_list))
        detector_text = '\n'.join(f"DETECTOR {recs}" for recs in recs_list)
        return detector_text

    def final_observable(self, rounds = 0, basis : str = "Z"):
        # final measurement of logical observable
        # total number of rounds is irrelevant for static surface code, but we have it here for consistency with walking surface code
        self.circuit.append(self._get_final_observable_text(basis))

    def _get_final_observable_text(self, basis : str):
        idx_list = []
        observable = self.layout.z_logical if basis == "Z" else self.layout.x_logical
        for qubit in sorted(observable):
            idx_list.append(self.tracker.get_curr_meas(qubit.index))
        recs = " ".join(f"rec[{idx}]" for idx in idx_list)

        return f"OBSERVABLE_INCLUDE(0) {recs}"

    def single_ec_traceback(self, ec_rnd : int, ec_gate : int, ec_index : int, ec_sched : int, leak_locations : str = "2-qubit gates"):

        possible_leak_locs = super()._single_ec_traceback_impl(
            ec_rnd, ec_gate, ec_index, ec_sched, leak_locations,
            [self.cnot_lists]*2, self.erasure_unswaps.get(ec_index, None))

        return possible_leak_locs

    def count_leakage_locations(self, rounds : int, leak_locations : str = "2-qubit gates"):
        if leak_locations == "2-qubit gates":
            leakage_locations_per_step = [
                len(self.cnot_lists[t])//2 for t in range(4)]  # total 2-qubit gates per step
        elif leak_locations == "all":
            leakage_locations_per_step = [self.n_qubits for _ in range(4)] # all qubits per step
        else:
            raise ValueError(f"Invalid leak_locations value: {leak_locations}")
        leakage_locations_per_round = sum(leakage_locations_per_step)
        total_leakage_locations = rounds*leakage_locations_per_round
        return total_leakage_locations, leakage_locations_per_round, leakage_locations_per_step

    def ravel_leakage_locations(self, rounds : int, leak_locs_unraveled : list, leak_locations : str = "2-qubit gates"):
        return super()._ravel_leakage_locations_impl(
            rounds, leak_locs_unraveled, leak_locations,
            cnot_qubits=[self.cnot_lists]*2)
