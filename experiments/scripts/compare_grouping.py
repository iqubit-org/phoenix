#!/usr/bin/env python
"""Compare grouping strategies (support / peel) on one benchmark:
compile time split by stage + final circuit metrics.

Usage: python compare_grouping.py <benchmark.json> <mode> [mode ...]
"""

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import phoenix
from phoenix.compiler import optimize_phoenix_circuit_by_qiskit


def metrics(qc):
    ops2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    return {
        "gates": sum(qc.count_ops().values()),
        "2q": ops2q,
        "depth": qc.depth(),
        "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits == 2),
    }


def main():
    fname, modes = sys.argv[1], sys.argv[2:]
    with open(fname) as f:
        data = json.load(f)

    for mode in modes:
        ham = phoenix.Hamiltonian(data["paulis"], data["coeffs"])

        t0 = time.perf_counter()
        qc = phoenix.compile_hamiltonian_simulation(ham, grouping=mode, optimize=False)
        t_simp = time.perf_counter() - t0
        m_pre = metrics(qc)

        t0 = time.perf_counter()
        qc = optimize_phoenix_circuit_by_qiskit(qc)
        t_o3 = time.perf_counter() - t0
        m_post = metrics(qc)

        print(
            f"RESULT mode={mode} t_simplify={t_simp:.1f}s t_o3={t_o3:.1f}s "
            f"pre[2q={m_pre['2q']} depth2q={m_pre['depth_2q']}] "
            f"post[gates={m_post['gates']} 2q={m_post['2q']} depth={m_post['depth']} depth2q={m_post['depth_2q']}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
