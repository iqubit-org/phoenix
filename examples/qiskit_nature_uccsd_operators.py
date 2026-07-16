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

from dataclasses import dataclass
from pathlib import Path

from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

from qiskit_nature_uccsd import dump_qubit_operator_json, uccsd_pauli_operator


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
