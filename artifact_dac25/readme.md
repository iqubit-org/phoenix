This folder is the artifact for the [DAC'25 paper -- PHOENIX: Pauli-Based High-Level Optimization Engine for Instruction Execution on NISQ Devices](https://arxiv.org/abs/2504.03529).



## Benchmarking guidelines

You can use the `Makefile` in the current folder or Python test scripts in the [./experiments/](./experiments/) folder for benchmarking.

For example:

- Benchmarking for Phoenix and baselins

  ```Shell
  make phoenix # Logical-level compilation by Phoenix
  make tket # Logical-level compilation by TKet
  make paulihedral # Logical-level compilation by Paulihedral
  make paulihedral_hhex # Hardware-aware compilation with heavy-hex topology by Paulihedral
  make paulihedral_square # Hardware-aware compilation with square topology by Paulihedral
  make tetris # Logical-level compilation by Tetris
  make tetris_hhex # Hardware-aware compilation with heavy-hex topology by Tetris
  make tetris_square # Hardware-aware compilation with square topology by Tetris
  ```

- Perform local optimization by Qiskit O3 after high-level compilation, especially for `Phoenix`, `Paulihedral` and `Tetris` compilers

  ```Shell
  make O3
  ```

- Perform hardware-aware benchmarking sine some VQA compilers (`Phoenix`, `TKet`) are not topology-aware

  ```shell
  make topology
  ```

- Summarize all benchmarking results according to generated .qasm files

  ```Shell
  make sum_result
  ```

- Display benchmarking results (#2Q and Depth2Q optimization rates) in format of a table

  ``` Shell
  make disp_result
  
  # >>> Num2Q Opt Rate
  # +----------------+-------+-------------+--------+---------+
  # | Num2Q Opt Rate |  TKet | Paulihedral | Tetris | Phoenix |
  # +----------------+-------+-------------+--------+---------+
  # |    All2all     | 0.315 |    0.284    | 0.537  |  0.187  |
  # |   All2all O3   | 0.315 |    0.257    | 0.367  |  0.185  |
  # |     Square     |  0.82 |    0.434    | 0.611  |   0.41  |
  # |      HHex      | 1.302 |    0.868    | 0.723  |  0.546  |
  # +----------------+-------+-------------+--------+---------+
  
  # >>> Depth2Q Opt Rate
  # +------------------+-------+-------------+--------+---------+
  # | Depth2Q Opt Rate |  TKet | Paulihedral | Tetris | Phoenix |
  # +------------------+-------+-------------+--------+---------+
  # |     All2all      | 0.289 |    0.291    | 0.533  |   0.16  |
  # |    All2all O3    | 0.289 |    0.263    | 0.364  |  0.158  |
  # |      Square      | 0.592 |    0.374    | 0.504  |  0.332  |
  # |       HHex       | 0.845 |     0.77    | 0.603  |  0.425  |
  # +------------------+-------+-------------+--------+---------+
  ```
  
