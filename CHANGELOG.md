# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-18

### Added
- Initial public release on PyPI as `phoenix-quantum`.
- `compile_hamiltonian_simulation` end-to-end pipeline: grouping →
  simplification → ordering → Qiskit post-optimization.
- `simplify_hamiltonian` with optional vectorized parallel search
  (`search_best_clifford_par`, compressed-representation implementation
  covering wide-active-qubit groups efficiently).
- BSF heuristic cost `heuristic_bsf_cost` with matmul-based pairwise-OR
  counting.
- Ordering strategies: `trivial`, `greedy` (with lookahead), and `tsp`
  (exact Held–Karp DP for small `n`; vectorized 2-opt for larger).
- PEP 561 typing marker (`py.typed`).
- Apache-2.0 license.

[Unreleased]: https://github.com/iqubit-org/phoenix/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/iqubit-org/phoenix/releases/tag/v0.1.0
