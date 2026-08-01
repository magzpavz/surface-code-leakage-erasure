# surface-code-leakage-erasure

Simulation of static, walking, and moonwalking surface codes with **leakage** 
and **erasure checks**, 
built on [Stim](https://github.com/quantumlib/Stim) and a fork of 
PyMatching ([original](https://github.com/oscarhiggott/PyMatching); 
[fork](https://github.com/Allenator/PyMatching)).

Leakage events are sampled at circuit locations, directly modifying the circuit.
Erasure checks are used to decode by either a marginal (averaged) matching graph
or with a branch-and-bound search that respects the disjointness constraints between
leakage locations.

This is the code accompanying **"Erasure surface code circuit without
mid-circuit erasure checks"**, by Margaret Pavlovich, Ivan Rojkov, Chen Wang,
and Shruti Puri — arXiv:2608.XXXXX <TODO: arXiv URL>. 
<!-- See [Citation](#citation) below. -->

## Requirements

- **Python 3.10 or newer.**
- **A C++ toolchain.** The required PyMatching fork (see below) has no
  pre-built wheels, so pip compiles it from source. On macOS this means Xcode
  command line tools (`xcode-select --install`); on Linux, a recent `g++`.

Runtime dependencies, installed automatically:
`numpy`, `stim`, `sinter`, `joblib`, `pymatching` (fork!)

Plotting and data analysis (`analysis.py`, `plotting.py`) additionally need
`pandas`, `scipy`, and `matplotlib`. These are *not* required to run
simulations, so they live behind an optional `analysis` extra.

## Installation

```bash
git clone https://github.com/magzpavz/surface-code-leakage-erasure.git
cd surface-code-leakage-erasure
pip install .
```

To include the analysis and plotting helpers:

```bash
pip install ".[analysis]"
```

For development, install in editable mode:

```bash
pip install -e ".[analysis]"
```

Expect the install to take a few minutes: pip clones and compiles the
PyMatching fork.

## Quickstart

```python
from surface_code_leakage_erasure import SurfaceCodeErasureSampler, SurfaceCodeLayout

sampler = SurfaceCodeErasureSampler(SurfaceCodeLayout(3), n_jobs=1)
result = sampler.sample_LER(
    rounds=4, p_Pauli=1e-3, p_leak=1e-3, min_samples=100,
)
print(result.num_samples, result.failures_marginal)
```

### The PyMatching fork

**This package will not work with PyMatching from PyPI.**

Erasure decoding reweights the matching graph on every shot, passing the
per-shot weights through the `edge_reweights` argument of `Matching.decode`
and `Matching.decode_to_edges_array`. Upstream PyMatching does not implement
this, so `Matching.decode` absorbs and silently discards the kwarg (`Matching.decode_to_edges_array` raises a TypeError). Marginal decoding 
then runs to completion while ignoring every erasure check, yielding 
plausible-looking but meaningless logical error rates.

The required fork is
[`Allenator/PyMatching`](https://github.com/Allenator/PyMatching), which
implements `edge_reweights` on both methods.

`pip install .` pulls the fork in automatically — it is declared as a direct
reference in `pyproject.toml`, so no extra step is needed in a clean
environment. **If you already have upstream PyMatching installed**, pip may
consider the requirement satisfied and keep it. Replace it explicitly:

```bash
pip install --force-reinstall \
  "pymatching @ git+https://github.com/Allenator/PyMatching.git"
```

Because both builds report version `2.4.0`, a version specifier cannot
distinguish them; the direct URL is the only way to select the fork.

As a safeguard, the package checks for this capability when you construct a
`SurfaceCodeErasureSampler` or an `ErasureDecoder`, and raises a `RuntimeError`
with installation instructions if the installed PyMatching cannot honor
`edge_reweights`. You can query it directly:

```python
from surface_code_leakage_erasure.decoding import pymatching_supports_edge_reweights
assert pymatching_supports_edge_reweights()
```

## Concepts

A simulation is assembled from four pieces:

| Piece | Classes | Role |
| --- | --- | --- |
| **Layout** | `SurfaceCodeLayout`, `WalkingSurfaceCodeLayout` | Qubit coordinates, plaquettes, gate order, logical operators. Built once per distance; construction is slow, so reuse it. `WalkingSurfaceCodeLayout` takes a `swap_time` argument selecting the walking or moonwalking circuit — see [Layout parameters](#layout-parameters). |
| **Circuit builder** | `SurfaceCodeCircuitBuilder`, `WalkingSCCircuitBuilder` | Turns a layout into a stim circuit for a given round count and noise model, and reports which erasure checks fired. |
| **Decoder** | `ErasureDecoder` | Builds detector error models for leakage events and decodes syndromes. |
| **Sampler** | `SurfaceCodeErasureSampler` | Drives the whole loop and estimates logical failure probability. |

## Parameter reference

### Layout parameters

Passed to the layout constructor, not to `sample_LER`. `SurfaceCodeLayout`
takes only the code distance `d`; `WalkingSurfaceCodeLayout` additionally takes
`swap_time`, which selects which of the two circuits from the paper is built.

| Parameter | Values | Meaning |
| --- | --- | --- |
| `d` | — | Code distance. Must be odd for the walking layouts. |
| `swap_time` | `"late"`, `"early"` | `"late"` builds the conventional **walking** surface code; `"early"` builds the **moonwalking** surface code.  |

The layout carries its `swap_time` with it, so a builder or sampler constructed
from a layout inherits it — you do not pass it again. `WalkingSCCircuitBuilder`
accepts `swap_time` directly only when given a bare distance instead of a
layout.

### Circuit parameters

Passed as keyword arguments to `sample_LER`, (and to
`CircuitBuilder.get_circuit`, etc.). Defaults are the first value listed.

| Parameter | Values | Meaning |
| --- | --- | --- |
| `basis` | `"Z"`, `"X"` | Basis in which the logical qubit is prepared and measured. |
| `ec_sched` | `8`, `4`, `2`, `1` | How often erasure checks run. `8` = ancillas only, during stabilizer measurement; `4` = all qubits, during stabilizer measurement; `2` = all qubits every other step; `1` = all qubits every step. When using EC schedule 8 with the static surface code, a leakage-SWAP is applied between each data qubit and a neighboring ancilla (requiring extra ancillas) at the beginning of each syndrome extraction round. |
| `Pauli_locations` | `"2-qubit gates"`, `"gates"`, `"all"` | Where Pauli noise is applied. Pauli noise is always applied after gates. `"gates"` includes reset, measurement, and Hadamards; `"all"` adds idling locations. |
| `leak_locations` | `"2-qubit gates"`, `"all"` | Where leakage may occur. Leakage is always applied before gates. `"all"` lets idling qubits leak too. |
| `leak_effect` | see below | What a leaked qubit does to its gate partner. |

`leak_effect` selects the leaked-qubit effect:

| Value | Effect |
| --- | --- |
| `"depolarize"` | Gate still applied; the partner of a leaked qubit is fully depolarized (p = 3/4). |
| `"tailored"` | Gate still applied; the partner gets an X error (p = 1/2) if it is the target, or a Z error (p = 1/2) if it is the control. |
| `"skip gates"` | Any gate involving a leaked qubit is removed; neither qubit receives an error. |
| `"tailored then skip"` | On the step where a qubit first leaks, its partner gets the `"tailored"` error; on later steps the gate is skipped. |

Two further values, `"decode skip gates"` and `"decode tailored then skip"`,
are the decoding counterparts (Pauli envelopes) of the two skipping models. 
A skipped gate is a Clifford error and has no
representation in a stim detector error model, so for decoding it is replaced by
a depolarizing channel on the leaked qubit at appropriate circuit locations (see paper). 
These are substituted automatically when detector error models are built — pass 
`"skip gates"` or `"tailored then skip"` and let the decoder handle it.

### Sampling parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `rounds` | — | Number of error correction rounds. Must be even for the walking circuits. |
| `p_Pauli` | — | Physical Pauli error rate. |
| `p_leak` | — | Leakage probability per leakage location. |
| `min_samples` | — | Lower bound on shots taken. |
| `target_failure_count` | `0` | If > 0, a preliminary run estimates the number of shots needed to observe this many failures, clamped to `[min_samples, max_samples]`. `0` takes exactly `min_samples`. |
| `max_samples` | `1_000_000` | Upper bound on shots taken. |
| `use_branch_and_bound` | `False` | Also decode with the branch-and-bound search and report it as the MLE result. Only differs from the marginal result when `leak_effect` skips gates. |
| `mle_time_limit` | `1` | Per-shot budget in seconds for that search. `None` for no limit. |
| `time_limit_hours` | `None` | Stop after this many hours and return what was collected. |
| `deadline` | `None` | The same, as an absolute `time.time()` value. Wins if both are given. |
| `n_leak` | `None` | Give every shot exactly this many leakage locations instead of sampling at rate `p_leak`. Forces `p_Pauli = 0`. |
| `verbose` | `False` | Print progress. |

- **Marginal vs. branch-and-bound decoding** — a fired erasure check does not
  say *when* the qubit leaked, only that it did. Marginal decoding averages
  the detector error models (Pauli envelopes) over every leakage location consistent with the
  check. Branch-and-bound decoding instead searches for a solution in which
  the chosen Pauli errors are consistent with leakage location disjointness, which matters when
  leaked qubits cause gates to be skipped. `sample_LER` reports the marginal
  result always, and the branch-and-bound result when
  `use_branch_and_bound=True`.

`sample_LER` returns an `ErasureSamplerResult`, whose `pfail_marginal` and
`pfail_MLE` fields are `sinter` fit objects carrying `.best`, `.low`, and
`.high`. `pfail_MLE` is `None` unless `use_branch_and_bound=True`.

> **`pfail` is not the LER.** These fields are the probability that a *whole
> shot* fails — `failures / num_samples` — not a per-round rate. To get a
> logical error rate per round, divide by `rounds`. 

## Known issues

**Leakage-explained edges are not filtered in the branch-and-bound decoder.**
While preparing this code for public release I noticed a bug in
[`decoding.py`](src/surface_code_leakage_erasure/decoding.py): the set of edges
already explained by the leakage locations fixed at a branch-and-bound node is
built with `set().union(generator)` instead of `set().union(*(...))`. The
missing unpacking star makes the result a set of frozensets rather than the
union of the edges, so the subsequent subtraction removes nothing and those
edges are still passed to the consistency check.

The effect is that valid solutions can be reported as invalid, but never the
reverse — the search over-branches and may miss an MLE solution, but it never
accepts an inconsistent one. I therefore do not expect this to change the
headline results. I intend to fix it and update the paper with new simulations;
for now I am leaving the code as it was run for v1 of the paper. This bug only
affects the branch-and-bound (`use_branch_and_bound=True`) path; marginal
decoding is unaffected.

**Sampling is not seedable end-to-end.** `SurfaceCodeErasureSampler` exposes no
seed: each joblib worker builds its own `SurfaceCodeCircuitBuilder(layout)` with
`seed=None`, and one leakage-sampling site in
[`erasure_sampler.py`](src/surface_code_leakage_erasure/erasure_sampler.py) draws
from the global `np.random.random()` rather than from a passed generator. As a
result individual runs cannot be reproduced exactly, only statistically. 

**There are no tests.** The repository ships no test suite, sorry!

## Citation

If you use this code, please cite:

TBA
