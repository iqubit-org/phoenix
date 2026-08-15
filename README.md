# 🐦‍🔥 PHOENIX: Pauli-based High-level Optimization ENgine for Instruction eXecution on NISQ Devices

[![PyPI](https://img.shields.io/pypi/v/phoenix-quantum?logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/phoenix-quantum/) [![Python](https://img.shields.io/badge/Python-3.9--3.12-3776AB?logo=python&logoColor=FFD43B)](https://pypi.org/project/phoenix-quantum/) [![License](https://img.shields.io/pypi/l/phoenix-quantum?logo=opensourceinitiative&logoColor=white&color=3DA639)](./LICENSE) [![CI](https://github.com/iqubit-org/phoenix/actions/workflows/ci.yml/badge.svg)](https://github.com/iqubit-org/phoenix/actions/workflows/ci.yml) [![Slides](https://img.shields.io/badge/Slides-PPTX-orange?logo=files&logoColor=white)](https://youngcius.github.io/docs/slides/phoenix_dac2025.pdf) [![Conference](https://img.shields.io/badge/Conference-DAC%202025-7B2CBF?logo=acm&logoColor=white)](https://arxiv.org/abs/2504.03529) [![arXiv](https://img.shields.io/badge/arXiv-2608.11579-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.11579)


## Overview

Phoenix is a high-level application-specific for generic Hamiltonian simulation programs (e.g., VQA, QAOA) built on top of Qiskit framework. It compiles Hamiltonian simulation circuits by exploiting global optimization opportunities through the BSF (binary symplectic form) representation of Pauli exponentiations and Clifford transformation.

Different from ZX calculus or Phase polynomial like approaches (e.g., [TKet](https://github.com/CQCL/pytket-docs), [PauliOpt](https://github.com/hashberg-io/pauliopt)) and local peephole optimization approaches (e.g., [Paulihedral](https://arxiv.org/abs/2109.03371), [Tetris](https://arxiv.org/abs/2309.01905)), Phoenix performs global optimization on the Pauli-string IR level before lowering to gate-level circuits.


This repo also includes benchmarking scripts and results with other SOTA baselines -- [Qiskit-Rustiq (Goubault de Brugière et al. 2024)](https://arxiv.org/abs/2404.03280), [TKet (Schmitz et al. 2021, Paykin et al. 2023)](https://arxiv.org/abs/2103.08602), [Paulihedral (Li et al. 2022)](https://arxiv.org/abs/2109.03371), [Tetris (Jin et al. 2024)](https://arxiv.org/abs/2309.01905), [PauliOpt (Winderl et al. 2023)](https://github.com/hashberg-io/pauliopt), and [QuCLEAR (Liu et al. 2025)](https://arxiv.org/abs/2408.13316).


## Usage

```python
import phoenix

ham = phoenix.Hamiltonian(['XXIII', 'YYIII', 'ZZIII'], [0.5, 0.5, 0.5])
qc = phoenix.compile_hamiltonian_simulation(ham)
phoenix.utils.print_circ_info(qc)
```

*Also, see [examples/phoenix_pass.py](examples/phoenix_pass.py) for reference.*

The compiler pipeline:

1. **Grouping** -- group Pauli terms by their non-trivial support
2. **Simplification** -- simplify each group via Clifford conjugation in binary symplectic form
3. **Circuit construction** -- convert simplified configurations into `QuantumCircuit` blocks
4. **Ordering** -- schedule blocks to minimize depth overhead and maximize gate cancellation (TSP-based or greedy)
5. **Post-optimization** -- apply Qiskit transpiler passes for Clifford cancellation and unitary resynthesis

Key parameters of `compile_hamiltonian_simulation`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grouping` | `True` | Group Pauli terms by non-trivial support before simplification |
| `optimize` | `True` | Apply Qiskit post-optimization |
| `parallel_search` | `False` | Perform Clifford search in parallel or sequentially on each BSF |
| `order_method` | `None` (=`'tsp'`) | Block ordering: `'trivial'`, `'greedy'`, or `'tsp'` |
| `backend` | `'sequential'` | Parallelization: `'sequential'`, `'concurrent.futures'`, or `'joblib'` |


## Illustration

Core optimization strategy (BSF simplification via Clifford conjugation):

![](./assets/bsf_simp_example.svg)


## Benchmarking Results

### UCCSD Benchmarks (6 synthetic Hamiltonians, generated from sampling UCCSD Pauli terms, 3 device topologies)

```
>>> Num2Q Opt Rate
+----------------+--------+-------+-------------+--------+---------+---------+-----------+
| Num2Q Opt Rate | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | Phoenix | Phoenix++ |
+----------------+--------+-------+-------------+--------+---------+---------+-----------+
|    All2all     | 0.256  | 0.175 |    0.299    | 0.311  |  0.228  |  0.133  |   0.131   |
|     Square     | 1.117  | 0.691 |    0.427    |  0.61  |   0.95  |  0.395  |   0.452   |
|      HHex      | 1.976  | 1.168 |    0.914    | 0.764  |  1.625  |  0.571  |    0.76   |
+----------------+--------+-------+-------------+--------+---------+---------+-----------+

>>> Depth2Q Opt Rate
+------------------+--------+-------+-------------+--------+---------+---------+-----------+
| Depth2Q Opt Rate | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | Phoenix | Phoenix++ |
+------------------+--------+-------+-------------+--------+---------+---------+-----------+
|     All2all      | 0.153  | 0.059 |    0.307    | 0.309  |  0.102  |  0.112  |   0.051   |
|      Square      | 0.575  | 0.309 |    0.325    | 0.438  |  0.455  |  0.276  |    0.21   |
|       HHex       | 0.917  |  0.47 |    0.724    |  0.54  |  0.702  |  0.362  |    0.32   |
+------------------+--------+-------+-------------+--------+---------+---------+-----------+
```

Lower is better. Opt Rate = geometric mean of (optimized / original) across all benchmarks.

### [HamLib Benchmarks](https://arxiv.org/abs/2306.13126) (100 programs across 4 categories)

```
>>> Num2Q Opt Rate
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|       Num2Q Opt Rate      | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | Phoenix | Phoenix++ |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|  binaryoptimization (15)  | 1.252  | 0.886 |    0.669    | 0.715  |  1.598  |  0.718  |   0.732   |
| discreteoptimization (15) | 0.542  | 0.556 |    0.524    | 0.751  |  0.588  |  0.753  |   0.688   |
|       chemistry (35)      | 0.292  | 0.276 |    0.346    | 0.496  |  0.354  |  0.327  |   0.226   |
|    condensedmatter (35)   | 1.084  | 0.855 |    0.517    | 0.678  |   0.83  |  0.493  |   0.467   |
|         All (100)         |  0.63  | 0.542 |    0.468    | 0.622  |  0.639  |  0.482  |   0.411   |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+

>>> Depth2Q Opt Rate
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|      Depth2Q Opt Rate     | Qiskit |  TKet | Paulihedral | Tetris | QuCLEAR | Phoenix | Phoenix++ |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+
|  binaryoptimization (15)  | 1.277  | 0.626 |    0.309    | 0.529  |  1.603  |  0.261  |   0.206   |
| discreteoptimization (15) | 0.595  |  0.35 |    0.413    | 1.728  |  0.641  |  0.377  |   0.179   |
|       chemistry (35)      | 0.218  | 0.182 |    0.354    | 0.397  |  0.273  |  0.236  |   0.134   |
|    condensedmatter (35)   | 0.357  | 0.799 |    0.621    | 0.123  |  0.724  |  0.069  |   0.031   |
|         All (100)         | 0.393  | 0.405 |    0.432    | 0.342  |  0.563  |  0.167  |    0.09   |
+---------------------------+--------+-------+-------------+--------+---------+---------+-----------+

```


## Installation


### 1. From PyPI

```bash
pip install phoenix-quantum
```


### 2. From Source

```bash
pip install .
```

or for development:

```bash
pip install -e .
```


## Requirements

Core dependencies (automatically installed):

- `qiskit >= 1.0.0`
- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `matplotlib >= 3.0.0`



## Benchmarking Scripts

All benchmarking scripts are under [`./experiments/`](./experiments/).

| Script | Description |
|--------|-------------|
| `bench_single.py` | Run a single benchmark given a compiler and input JSON file |
| `bench_uccsd.py` | Batch benchmark on UCCSD suite for a given compiler |
| `bench_hamlib.py` | Batch benchmark on HamLib suite for a given compiler and category |
| `bench_utils.py` | Standard compilation passes for Phoenix and all baselines |
| `uccsd_all2all_to_limited.py` | Map all-to-all results to limited-connectivity topologies (square, heavy-hex) |


Use `make` targets for batch execution across UCCSB benchmark suite while use `make -f Makefile-Hamlib` for HamLib benchmarking:

For example,
```bash
cd experiments
make phoenix          # Run Phoenix on UCCSD (all2all)
make phoenix_square   # Map Phoenix results to square topology
make sum_result       # Summarize all results to CSV
make disp_result      # Display comparison tables
```

## Citation

If you make use of Phoenix in your work, please cite the following publication:

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
