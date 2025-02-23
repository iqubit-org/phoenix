# 🐦‍🔥 𝑷𝑯𝑶𝑬𝑵𝑰𝑿: Pauli-based High-level Optimization ENgine for Instruction eXecution on NISQ Devices

[![](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE) [![](https://img.shields.io/badge/build-passing-green)]() ![](https://img.shields.io/badge/Python-3.8--3.12-blue) ![](https://img.shields.io/badge/dev-v1.0.0-blue)

## Overview

Phoenix is a highly-effective VQA (variational quantum algorithm) application-specifc compiler based on BSF (binary symplectic form) of Pauli exponentiations and Clifford formalism. Different from ZX-calculus-like approaches (e.g., [TKet](https://github.com/CQCL/pytket-docs), [PauliOpt](https://github.com/hashberg-io/pauliopt)) and local peephole optimization approaches (e.g., [Paulihedral](https://arxiv.org/abs/2109.03371), [TKet](https://arxiv.org/abs/2309.01905v2)), Phoenix exploits global optimization opportunities for VQA programs to the largest extent, when representing Pauli strings as BSF and employing Clifford formalism on the higher-level IR.

This repo includes benchmarking scripts and results with other SOTA baselines -- TKet, Paulihedral, Tetris, and Rustiq (). Code of Paulihedral and Tetris are refactored and integrated in this repo.

If you make sure of Phoenix in your work, please cite the following publication:

```
......
```

## Features

- High-level compilation
- Global optimization
- ISA-independent

**E.g., Hardware-agnostic compilation:**

![](./assets/num_2q_gates_all2all.png)

**E.g., Hardware-aware compilation:**

![](./assets/num_2q_gates_manhattan.png)


## Requirements

Basic library requirements are lists in `requirements.txt`.

- We align with the `1.2.4` version of  `qiskit`  across the published benchmarking results. Since `qiskit`'s O1/O2 has different built-in workflows, note that version 1.0+ are suitable for Phoenix.
- Originally, Paulihedral and Tetris require version 0.23.x and version 0.43.x of Qiskit. In this code repo, they can also be soomthly tested under Qiskit-1.2.4.

## Benchmarking description

**Benchmark suites:**

- Hamlib: 100 generic Hamiltonian simulation programs ( a subset of the original [Hamlib](https://arxiv.org/abs/2306.13126)) from IBM's latest [benchpress]() toolkit, including "binaryoptimization", "discreteoptimization", "chemistry", and "condensedmatter" four categories of Hamiltonian simulation programs.
- UCCSD: 16 molecule simulation programs from benchmarks from [TKet benchmarking](https://github.com/CQCL/tket_benchmarking). We use this suite for fine-grain benchmarking and real system analysis.

**Benchmarking scripts:** (under [./experiments/scripts/](./experiments/scripts/))

- `bench_single.py`: Benchmarking given a compiler and input file (`.qasm` file for TKet, `.json` file for Paulihedral/Tetris/Phoenix)
- `bench_hamlib.py`: Benchmarking given a compiler and a category of Hamlib benchmarks
- `bench_uccsd.py`: Benchmarking given a compiler and a physical-qubit topology type (E.g., all2all, manhattan, sycamore) (Since TKet/Phoenix does no make hardware-ware co-optimization, manually set topology execept "all2all" are invalid. Instead, the hardware-ware compilation are conducted by executing another script `uccsd_all2all_to_limited.py`)
- `bench_utils.py`: Utils used in benchmarking, within which the standard compilation passes of Phoenix and baselines are specified
- `hamlib_cnot_to_su4.py`: Rebase CNOT-based output circuits to SU(4)-based circuits given a compiler and a category of Hamlib benchmarks
- `uccsd_all2lall_to_limited.py`: Compile all-to-all compilation results from TKet/Phoenix for UCCSD benchmarks to limited-connectivity topology (manhattan, sycamore)
- `uccsd_all2all_qiskit_opt.py`: Further perform Qiskit O3 for logical-level synthesis results of Paulihedral/Tetris/Phoenix

**Result files:**

- `./experiments/output_hamlib/<compiler>/<category>/`: Output circuits by some `<compiler>` (e.g., Tetris, Phoenix) from some `<category>` of the Hamlib benchmark suite
- `./experiments/output_hamlib/<compiler>/<category>_su4/`: Output circuits that are further rebased to SU(4) ISA *(For Phoenix compiler, the SU(4)-based circuits can be natively generated that are the same as the rebased results from CNOT-based output circuits)*

- `./experiments/output_uccsd/<compiler>/<device>`: Output circuits by some `<compiler>` (E.g., Tetris, Phoenix) for some kind of `<device>` (E.g.,  all2all,  manhattan) from the UCCSD benchmark suite

- `./experiments/output_uccsd/<compiler>_opt/all2all`: Output circuits by some `<compiler>` (E.g., Tetris, Phoenix) when performing its logical-level synthesis with Qiskit O3 optimization procedure on logical circuits

*Remarks: We developed in the early beginning of 2024. Suprisingly the BSF algorithm within Phoenix utilized similar data structures and heuristic-optimization philosophy to Rustiq. Despite the accidential similarly,  our simplification goal and search strategy are distinct and involves a more sophisticated cost function. Even the BSF simplification algorithm alone outperforms Rustiq according to our further benchmarking in [./experiments/rebuttal.ipynb](./experiments/rebuttal.ipynb).*

## Copyright and License

This project is licensed under the Apache License 2.0 -- see the [LICENSE](LICENSE) file for details.
