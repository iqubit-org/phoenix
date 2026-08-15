# 🐦‍🔥 PHOENIX: Pauli-based High-level Optimization ENgine for Instruction eXecution on NISQ Devices

[![PyPI](https://img.shields.io/pypi/v/phoenix-quantum?logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/phoenix-quantum/) [![Python](https://img.shields.io/badge/Python-3.9--3.12-3776AB?logo=python&logoColor=FFD43B)](https://pypi.org/project/phoenix-quantum/) [![License](https://img.shields.io/pypi/l/phoenix-quantum?logo=opensourceinitiative&logoColor=white&color=3DA639)](./LICENSE) [![CI](https://github.com/iqubit-org/phoenix/actions/workflows/ci.yml/badge.svg)](https://github.com/iqubit-org/phoenix/actions/workflows/ci.yml) [![Slides](https://img.shields.io/badge/Slides-PPTX-orange?logo=files&logoColor=white)](https://youngcius.github.io/docs/slides/phoenix_dac2025.pdf) [![Conference](https://img.shields.io/badge/Conference-DAC%202025-7B2CBF?logo=acm&logoColor=white)](https://arxiv.org/abs/2504.03529) [![arXiv](https://img.shields.io/badge/arXiv-2608.11579-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.11579)


## Overview

Phoenix is a Qiskit-based compiler package for generic Hamiltonian-simulation workloads, including product-formula simulation, VQE/UCCSD, and QAOA. It accepts a weighted sequence of Pauli strings, optimizes the sequence at the Pauli intermediate-representation (Pauli-IR) level, and lowers it to a Qiskit `QuantumCircuit`.

The package contains two generations of the compiler behind one public API:

| Compiler | `grouping` value | Experiment key | Description |
|----------|------------------|----------------|-------------|
| **PHOENIX++/Symphony** ([arXiv:2608.11579](https://arxiv.org/abs/2608.11579)) | `None` or `"holistic"` (default) | `phoenixpp` | Grouping-free, holistic simplification of one global BSF tableau, followed by adaptive block emission and commutativity-aware ASAP scheduling |
| **PHOENIX** ([arXiv:2504.03529, DAC '25](https://arxiv.org/abs/2504.03529)) | `"support"` | `phoenix` | Exact same-support grouping, per-group BSF simplification, and TSP-based inter-group circuit ordering |

The newer compiler is named **Symphony** in the paper. `Phoenix++`/`phoenixpp` is retained only as the historical name used by the benchmarking targets, output directories, and result CSVs in this repository.

Both compilers optimize Pauli exponentials through Clifford conjugation in binary symplectic form (BSF) before gate-level optimization. The repository also contains the benchmark suites, raw circuits, result CSVs, plotting scripts, and baseline integrations used to compare against [Qiskit-Rustiq](https://arxiv.org/abs/2404.03280), [TKET](https://arxiv.org/abs/2103.08602), [Paulihedral](https://arxiv.org/abs/2109.03371), [Tetris](https://arxiv.org/abs/2309.01905), [PauliOpt](https://github.com/hashberg-io/pauliopt), and [QuCLEAR](https://arxiv.org/abs/2408.13316).


## Quick Start

```python
import phoenix

ham = phoenix.Hamiltonian(
    ["YYYXZI", "ZZYXZI", "IYYZXY", "IZYZXX"],
    [0.01, 0.01, 0.01, 0.01],
)

# PHOENIX++ (Symphony): the default holistic compiler, arxiv:2608.11579
symphony_circuit = phoenix.compile_hamiltonian_simulation(ham)

# PHOENIX: the first-generation phoenix compiler, arXiv2504.03529 (DAC '25)
phoenix_circuit = phoenix.compile_hamiltonian_simulation(
    ham,
    grouping="support",
)

phoenix.utils.print_circ_info(symphony_circuit, title="Symphony")
```

`Hamiltonian` extends Qiskit's `SparsePauliOp`, so Pauli labels follow Qiskit's qubit-order convention. See [`examples/phoenix_pass.py`](examples/phoenix_pass.py) for JSON input, topology mapping, circuit statistics, equivalence checks, and QASM export.


## Compilation Pipelines

### PHOENIX++/Symphony (default `grouping="holistic"` mode)

Symphony removes the a-priori support partition used by the original PHOENIX pipeline:

1. **Global BSF tableau** -- keep all unresolved Pauli strings in one binary symplectic tableau.
2. **Forward-frame peeling** -- select a minimum-weight target row and apply a CNOT-equivalent controlled-Pauli Clifford that strictly reduces that target while maximizing whole-tableau benefit.
3. **Adaptive emission** -- emit weight-1 rotations immediately and emit weight-2 blocks when the active-tableau density falls at or below `rho_threshold`.
4. **Commutativity-aware ASAP scheduling** -- preserve frame-induced dependencies while exposing parallelism among compatible Clifford moves and emitted Pauli blocks.
5. **Terminal-frame closure** -- replay or resynthesize the accumulated Clifford frame, then apply the common Qiskit circuit-level post-optimization pipeline.

Each peeling move strictly decreases the active target, so the core loop has guaranteed progress without a visited set or long-horizon search. When `rho_threshold=None`, the current public default compiles five candidates (`0.0`, `0.25`, `0.5`, `0.75`, and `1.0`) in parallel and selects the circuit with the lexicographically smallest pre-optimization `(2Q gate count, 2Q depth)`.

### PHOENIX (DAC '25 `grouping="support"` mode)

1. **Support-based grouping** -- partition Pauli terms by exact non-identity support.
2. **Per-group BSF simplification** -- search for Clifford conjugations that reduce each group.
3. **Circuit construction** -- lower the simplified groups to `QuantumCircuit` blocks.
4. **Inter-group ordering** -- use trivial, greedy, or TSP-based (default) ordering to reduce depth and expose cancellation.
5. **Post-optimization** -- apply the same Qiskit circuit-level optimization including Clifford subcircuit resynthesis, gate cancellation, and unitary resynthesis pipeline used for Symphony.


## API Options

Key parameters of `compile_hamiltonian_simulation`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grouping` | `None` | `None`/`"holistic"` selects Symphony; `"support"` selects DAC '25 PHOENIX |
| `terminal` | `"auto"` | Symphony only: `"auto"`, `"replay"`, `"synth"`, or `"absorb"`; auto chooses the cheaper replay/resynthesis tail by 2Q count |
| `rho_threshold` | `None` | Symphony only: scan five thresholds automatically, or provide one value in `[0, 1]` to compile a single candidate |
| `optimize` | `True` | Apply common Qiskit post-optimization to the generated circuit |
| `parallel_search` | `True` | PHOENIX only: use the vectorized parallel Clifford search within each support group |
| `order_method` | `None` (`"tsp"`) | PHOENIX only: `"trivial"`, `"greedy"`, or `"tsp"` inter-group ordering |
| `backend` | `"sequential"` | PHOENIX only: group-level execution via `"sequential"`, `"concurrent.futures"`, or `"joblib"` |
| `search_patience` | `None` | PHOENIX only: optional stall patience for the legacy BSF search safety net |

Set a numeric `rho_threshold` to avoid the five-candidate auto-tuning cost. The committed Symphony benchmark results below use `rho_threshold=0.35`, matching [`experiments/scripts/bench_utils.py`](experiments/scripts/bench_utils.py).

`terminal="absorb"` omits the terminal Clifford and is valid only when the caller also transforms measurement observables classically. `terminal="synth"` and the synthesized branch of `"auto"` preserve the unitary up to global phase; use `terminal="replay"` when the compiled circuit will be controlled or otherwise requires exact global phase.

The public compiler performs logical, all-to-all high-level synthesis. Hardware mapping is a separate step; the experiment harness uses Qiskit optimization level 3 to map topology-agnostic outputs to square and heavy-hex coupling maps.


## Illustration

Shared algebraic primitive: simultaneous BSF simplification through Clifford conjugation.

![](./assets/bsf_simp_example.svg)


## Evaluation Snapshot

The reported **optimization rate** is the geometric mean of `compiled / naive` for the given metric, so lower is better. Compiler outputs are normalized to the same CNOT plus single-qubit basis and receive the common circuit-level optimization pipeline; the naive reference receives basis translation without optimization. Symphony is recorded under the `phoenixpp` key in the committed result files.

### HamLib

The main suite contains 100 representative [HamLib](https://arxiv.org/abs/2306.13126) Hamiltonians across binary optimization, discrete optimization, chemistry, and condensed matter. These results use logical all-to-all connectivity. QuCLEAR timed out on four instances, so its aggregate is computed over the completed subset.

```
>>> Num2Q Opt Rate
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|       Num2Q Opt Rate      | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | PHOENIX | PHOENIX++ |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|  binaryoptimization (15)  | 1.252  | 0.886 |    0.669    | 0.715  |  1.598  |  0.718  |   0.732   |
| discreteoptimization (15) | 0.542  | 0.556 |    0.524    | 0.751  |  0.588  |  0.753  |   0.688   |
|       chemistry (35)      | 0.292  | 0.276 |    0.346    | 0.496  |  0.354  |  0.327  |   0.226   |
|    condensedmatter (35)   | 1.084  | 0.855 |    0.517    | 0.678  |   0.83  |  0.493  |   0.467   |
|         All (100)         |  0.63  | 0.542 |    0.468    | 0.622  |  0.639  |  0.482  |   0.411   |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+

>>> Depth2Q Opt Rate
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|      Depth2Q Opt Rate     | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | PHOENIX | PHOENIX++ |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|  binaryoptimization (15)  | 1.277  | 0.626 |    0.309    | 0.529  |  1.603  |  0.261  |   0.206   |
| discreteoptimization (15) | 0.595  |  0.35 |    0.413    | 1.728  |  0.641  |  0.377  |   0.179   |
|       chemistry (35)      | 0.218  | 0.182 |    0.354    | 0.397  |  0.273  |  0.236  |   0.134   |
|    condensedmatter (35)   | 0.357  | 0.799 |    0.621    | 0.123  |  0.724  |  0.069  |   0.031   |
|         All (100)         | 0.393  | 0.405 |    0.432    | 0.342  |  0.563  |  0.167  |    0.09   |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
```

Across HamLib, Symphony reduces the naive 2Q gate count by 59% and 2Q depth by 91% in geometric mean. The repository also contains the paper's pairwise scatter plots, Pareto plot, UCCSD runtime study, adaptive-emission and scheduling ablations, and early fault-tolerant analysis. In the latter, Symphony reduces geometric-mean T-depth by 2.73x over TKET, 2.15x over Qiskit, 2.30x over QuCLEAR, and 1.22x over PHOENIX while T-count remains approximately compiler-invariant.


### UCCSD

Six synthetic UCCSD Hamiltonians (`UCC-10` through `UCC-35`) contain 10--35 qubits and 800--9,800 Pauli terms. Logical circuits are evaluated directly on all-to-all connectivity and after routing to square and heavy-hex topologies.

```
>>> Num2Q Opt Rate
+----------------+--------+-------+-------------+--------+---------+---------+-----------+
| Num2Q Opt Rate | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | PHOENIX | PHOENIX++ |
+----------------+--------+-------+-------------+--------+---------+---------+-----------+
|    All2all     | 0.256  | 0.175 |    0.299    | 0.311  |  0.228  |  0.133  |   0.131   |
|     Square     | 1.117  | 0.691 |    0.427    |  0.61  |   0.95  |  0.395  |   0.452   |
|      HHex      | 1.976  | 1.168 |    0.914    | 0.764  |  1.625  |  0.571  |    0.76   |
+----------------+--------+-------+-------------+--------+---------+---------+-----------+

>>> Depth2Q Opt Rate
+------------------+--------+-------+-------------+--------+---------+---------+-----------+
| Depth2Q Opt Rate | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | PHOENIX | PHOENIX++ |
+------------------+--------+-------+-------------+--------+---------+---------+-----------+
|     All2all      | 0.153  | 0.059 |    0.307    | 0.309  |  0.102  |  0.112  |   0.051   |
|      Square      | 0.575  | 0.309 |    0.325    | 0.438  |  0.455  |  0.276  |    0.21   |
|       HHex       | 0.917  |  0.47 |    0.724    |  0.54  |  0.702  |  0.362  |    0.32   |
+------------------+--------+-------+-------------+--------+---------+---------+-----------+
```

Symphony gives the lowest 2Q depth rate on all three topologies. PHOENIX retains the lowest mapped 2Q gate-count rate on square and heavy-hex, illustrating the routing trade-off between support preservation and aggressive holistic simplification.


## Installation

### 1. From PyPI

```bash
pip install phoenix-quantum
```


### 2. From Source

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```


## Requirements

Phoenix supports Python 3.9--3.12. Core dependencies are installed automatically:

- `qiskit >= 1.0.0`
- `cirq >= 1.0.0`
- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `joblib >= 1.1.0`
- `prettytable >= 3.0.0`
- `rustworkx >= 0.12.0`



## Reproducing the Experiments

Benchmarks live under [`benchmarks/`](benchmarks/). The committed result CSVs and paper figures are under [`experiments/results/`](experiments/results/) and [`experiments/figures/`](experiments/figures/), respectively. The runnable harness is under [`experiments/scripts/`](experiments/scripts/).

| Script | Description |
|--------|-------------|
| [`bench_single.py`](experiments/scripts/bench_single.py) | Compile one JSON Hamiltonian with a selected compiler and topology |
| [`bench_uccsd.py`](experiments/scripts/bench_uccsd.py) | Batch-compile the six UCCSD workloads |
| [`bench_hamlib.py`](experiments/scripts/bench_hamlib.py) | Batch-compile one HamLib category |
| [`bench_utils.py`](experiments/scripts/bench_utils.py) | Common compiler adapters and post-processing used for all methods |
| [`uccsd_all2all_to_limited.py`](experiments/scripts/uccsd_all2all_to_limited.py) | Map all-to-all UCCSD outputs to square or heavy-hex connectivity |


The batch Makefile targets also require [GNU Parallel](https://www.gnu.org/software/parallel/). From the `experiments` directory, `phoenixpp` means Symphony and `phoenix` means the original DAC '25 compiler.

UCCSD:

```bash
cd experiments
make naive
make phoenix phoenixpp
make phoenix_square phoenix_hhex
make phoenixpp_square phoenixpp_hhex
make sum_result
make disp_result
```

HamLib:

```bash
cd experiments
make -f Makefile-Hamlib naive
make -f Makefile-Hamlib phoenix phoenixpp
make -f Makefile-Hamlib sum_result
make -f Makefile-Hamlib disp_result
```

The scripts skip or reuse existing outputs where supported. `make clean` removes generated compiler QASM files but retains the naive reference circuits; `make clean_naive` removes those references as well.

Additional analyses under [`experiments/analysis/`](experiments/analysis/) reproduce the HamLib pairwise/Pareto plots, the UCCSD runtime comparison, Clifford+T resource estimates, and the emission/scheduling ablations.

## Citation

The two papers below form a continuous line of work and share the same BSF/Clifford-simplification foundation. We therefore recommend citing both when using this package. The DAC '25 PHOENIX paper introduces the original support-grouped compiler (`grouping="support"`), while the Symphony paper develops the default holistic compiler (`grouping=None` or `"holistic"`, called PHOENIX++ in this repository's experiments).

```
@inproceedings{yang2025phoenix,
  author={Yang, Zhaohui and Ding, Dawei and Zhu, Chenghong and Chen, Jianxin and Xie, Yuan},
  booktitle={2025 62nd ACM/IEEE Design Automation Conference (DAC)}, 
  title={PHOENIX: Pauli-Based High-Level Optimization Engine for Instruction Execution on NISQ Devices}, 
  year={2025},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/DAC63849.2025.11133028}
}

@article{yang2026efficient,
  title={Efficient Compilation for Hamiltonian Simulation via Global Binary Symplectic Form Simplification},
  author={Yang, Zhaohui and Han, Yuwei and Ding, Dawei and Chen, Jianxin and Feng, Yuan and Xie, Yuan}, 
  journal={arXiv preprint arXiv:2608.11579},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 -- see the [LICENSE](LICENSE) file for details.
