"""Teaching demo + shared helpers for the UCCSD cluster-operator benchmarks.

This file shows the molecule-provenance path for one small example (H2 via PySCF) and provides
the two helpers (:func:`uccsd_pauli_operator`, :func:`dump_qubit_operator_json`) reused by the
bulk generator ``qiskit_nature_uccsd_operators.py``. That generator is driver-free, because at
a uniform amplitude the operator depends only on (num_spatial_orbitals, num_particles, mapper);
the demo below also checks that the molecule-derived operator equals the driver-free one built
straight from those numbers.

The dumped object is the Pauli decomposition of the UCCSD cluster operator t·(T - T†)/i at a
uniform amplitude t = 0.1 (VQE-ansatz compilation benchmark): singles contribute 2 strings at
±t/2, doubles contribute same-support families of 8 strings at ±t/8, every string carries an
odd number of Y letters. It is NOT the molecular electronic Hamiltonian (Hermitian combination,
even-Y strings plus a pure-Z diagonal block; that family lives in
``benchmarks/hamlib/chemistry``).

Requirements:
    pip install qiskit-nature pyscf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

if not hasattr(np, "in1d"):  # numpy>=2.0 removed in1d; pyscf still calls it
    np.in1d = np.isin

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit

# Uniform cluster amplitude plugged in for every excitation; 0.1 reproduces the
# legacy family's coefficient classes (singles ±0.05, doubles ±0.0125).
UCCSD_AMPLITUDE = 0.1


def uccsd_pauli_operator(ansatz: UCCSD, amplitude: float = UCCSD_AMPLITUDE) -> SparsePauliOp:
    """Collect the ansatz's excitation generators into one real-coefficient operator.

    Each element of ``ansatz.operators`` is the qubit image of the excitation
    generator (T_k - T_k†)/i — qiskit-nature 0.7 already folds the i, so the
    Pauli coefficients are real (singles ±1/2, doubles ±1/8). For robustness
    against the raw anti-Hermitian convention (purely imaginary coefficients)
    we rotate by -i when needed. Summing, scaling by the uniform ``amplitude``
    and simplifying merges duplicate labels across excitations.
    """
    ops = [op if isinstance(op, SparsePauliOp) else op.primitive for op in ansatz.operators]
    total = SparsePauliOp.sum(ops).simplify()
    if np.abs(np.real(total.coeffs)).max() < 1e-10:  # raw anti-Hermitian image
        total = total * (-1j)
    real_op = (total * amplitude).simplify()
    if np.abs(np.imag(real_op.coeffs)).max() > 1e-10:
        raise ValueError("Cluster operator must map to real Pauli coefficients (up to a global i).")
    return real_op


def dump_qubit_operator_json(
    qubit_operator: SparsePauliOp,
    output_path: str | Path,
    strip_identity: bool = True,
) -> None:
    """Serialize a real-coefficient qubit operator to the benchmark JSON schema.

    ``strip_identity`` drops the all-identity term (matching the
    hamlib/uccsd convention); cluster operators contain none anyway.
    """
    paulis = []
    coeffs = []
    for label, coeff in zip(qubit_operator.paulis.to_labels(), qubit_operator.coeffs):
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Expected a real-valued qubit operator, got coefficient {coeff!r}.")
        if strip_identity and set(label) == {"I"}:
            continue
        paulis.append(label)
        coeffs.append(float(coeff.real))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_qubits": qubit_operator.num_qubits,
                "paulis": paulis,
                "coeffs": coeffs,
            },
            f,
            indent=2,
        )


def main() -> None:
    mapper = JordanWignerMapper()

    # Molecule-provenance path: H2 @ STO-3G -> (num_spatial_orbitals, num_particles).
    problem = PySCFDriver(
        atom="H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    ).run()
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles
    hartree_fock = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    uccsd = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=hartree_fock)
    cluster_op = uccsd_pauli_operator(uccsd)

    # Driver-free equivalence: the operator depends only on (orbitals, particles, mapper).
    driver_free = uccsd_pauli_operator(UCCSD(num_spatial_orbitals, num_particles, mapper))
    assert cluster_op == driver_free, "molecule-derived operator must equal the driver-free one"

    print("Molecule: H2 @ 0.735 A, basis=STO-3G")
    print(f"Spatial orbitals: {num_spatial_orbitals} | particles: {num_particles} | qubits: {uccsd.num_qubits}")
    print(f"UCCSD excitations (parameters): {uccsd.num_parameters}")
    print("Driver-free operator from (orbitals, particles, mapper) matches.")
    print()
    print(f"Cluster operator t*(T - T+)/i at t={UCCSD_AMPLITUDE} (SparsePauliOp):")
    print(cluster_op)
    # This is a print-only provenance demo; it does NOT seed examples/uccsd/.
    # The benchmark ladder is generated by qiskit_nature_uccsd_operators.py.


if __name__ == "__main__":
    main()
