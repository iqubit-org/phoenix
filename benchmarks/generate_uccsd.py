"""Generate a clean ladder of UCCSD cluster operators (14-20 qubits) as JSON benchmarks.

At a uniform cluster amplitude the UCCSD operator t·(T - T†)/i depends ONLY on
(num_spatial_orbitals, num_particles, mapper) — the molecule, geometry and basis set never
enter (they would only matter through the variational amplitudes, which we fix). So these
benchmarks are built driver-free, straight from (electrons, orbitals) tuples: no PySCF, no
molecule, no basis. Two physically different molecules with the same (e, o) signature produce
byte-identical operators, which is why the files are named by their true determinants.

The dumped operator is a VQE-ansatz compilation benchmark (singles -> 2 strings at ±t/2,
doubles -> same-support families of 8 strings at ±t/8, every string odd-Y), NOT a molecular
electronic Hamiltonian. JW and BK give very different Pauli patterns at the same (e, o): JW
carries contiguous Z-chains (max weight = #qubits, full 8-string same-support families), BK
trades them for O(log n)-weight parity sets (lower weight, more fragmented supports).

Ladder (fixed 10 closed-shell electrons, virtual orbitals increasing; string counts identical
for JW/BK):

    ucc_10e_7o   14q / 1000     ucc_10e_8o   16q / 2340
    ucc_10e_9o   18q / 4240     ucc_10e_10o  20q / 6700
"""

from __future__ import annotations

import numpy as np

if not hasattr(np, "in1d"):  # numpy>=2.0 removed in1d; pyscf still calls it
    np.in1d = np.isin
    
import json
from dataclasses import dataclass
from pathlib import Path
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

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

@dataclass(frozen=True)
class UCCCase:
    num_electrons: int  # total electrons, closed-shell -> (n/2 alpha, n/2 beta)
    num_spatial_orbitals: int  # qubit count = 2 * num_spatial_orbitals

    @property
    def num_particles(self) -> tuple[int, int]:
        half = self.num_electrons // 2
        return (half, half)


# Fixed 10 electrons, virtual orbitals 7->10. Each (e, o) class is realizable by real
# closed-shell molecules at STO-3G, e.g. H2O/NH3/CH4 (complete) and C2H2 (frozen-core),
# but the operator does not depend on which.
CASES: tuple[UCCCase, ...] = (
    UCCCase(num_electrons=10, num_spatial_orbitals=7),
    UCCCase(num_electrons=10, num_spatial_orbitals=8),
    UCCCase(num_electrons=10, num_spatial_orbitals=9),
    UCCCase(num_electrons=10, num_spatial_orbitals=10),
)

MAPPERS = {"JW": JordanWignerMapper(), "BK": BravyiKitaevMapper()}


def build_case(case: UCCCase) -> None:
    for encoding, mapper in MAPPERS.items():
        ansatz = UCCSD(case.num_spatial_orbitals, case.num_particles, mapper)
        cluster_op = uccsd_pauli_operator(ansatz)
        assert cluster_op.num_qubits == 2 * case.num_spatial_orbitals

        name = f"ucc_{case.num_electrons}e_{case.num_spatial_orbitals}o_{encoding}"
        output_path = Path(__file__).resolve().parent / "uccsd" / f"{name}.json"
        dump_qubit_operator_json(cluster_op, output_path)

        labels = cluster_op.paulis.to_labels()
        max_weight = max(sum(c != "I" for c in lbl) for lbl in labels)
        print(
            f"{name}: {cluster_op.num_qubits}q, {ansatz.num_parameters} excitations, "
            f"{len(labels)} strings, max_weight {max_weight} -> {output_path.name}"
        )


def main() -> None:
    for case in CASES:
        build_case(case)


if __name__ == "__main__":
    main()
