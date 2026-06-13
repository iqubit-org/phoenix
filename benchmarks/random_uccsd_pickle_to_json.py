"""Extract tetris/data/random/*.pickle into benchmarks/random/*.json.

The pickles store tetris ``pauliString`` blocks (size-8 UCCSD double-excitation groups) with
ZERO coefficients — pure Pauli structure. We flatten the blocks into the flat benchmark schema
``{num_qubits, paulis, coeffs}`` and assign each string a random rotation angle in (0, 2*pi),
seeded for reproducibility (the structure benchmark's gate count is angle-independent; a random
nonzero angle just makes the circuit non-trivial and avoids special cancellations).

Pauli strings are kept in their native pickle orientation.
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(HERE, "..", "tetris", "data", "random")
OUTPUT_DIR = os.path.join(HERE, "uccsd")
SEED = 42

# The pickles contain tetris ``benchmark.mypauli.pauliString`` instances, so the
# ``benchmark`` package (tetris/benchmark) must be importable to unpickle them.
sys.path.insert(0, os.path.join(HERE, "..", "tetris"))
import benchmark.mypauli  # noqa: F401  (registers the class for unpickling)


def convert(pickle_path: str, output_path: str, rng: np.random.Generator) -> None:
    with open(pickle_path, "rb") as f:
        blocks = pickle.load(f)

    paulis = [p.ps for block in blocks for p in block]
    num_qubits = len(paulis[0])
    assert all(len(p) == num_qubits for p in paulis), "inconsistent Pauli string lengths"

    coeffs = rng.uniform(-0.1, 0.1, size=len(paulis)).tolist()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"num_qubits": num_qubits, "paulis": paulis, "coeffs": coeffs}, f, indent=2)
    print(f"{os.path.basename(pickle_path)} -> {os.path.basename(output_path)}: "
          f"{num_qubits}q, {len(blocks)} blocks, {len(paulis)} strings")


def main() -> None:
    rng = np.random.default_rng(SEED)
    names = sorted(
        f for f in os.listdir(PICKLE_DIR) if f.endswith(".pickle")
    )
    for name in names:
        convert(
            os.path.join(PICKLE_DIR, name),
            os.path.join(OUTPUT_DIR, name.replace(".pickle", ".json").replace("random_", "uccsd_")),
            rng,
        )


if __name__ == "__main__":
    main()
