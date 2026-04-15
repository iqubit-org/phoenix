"""
Holistic SMT-based Hamiltonian simulation compilation demo.

Compares the SMT solver (globally optimal CNOT count) against the greedy
heuristic, verifying correctness via infidelity checks.
"""

import numpy as np
from qiskit.quantum_info import Operator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import phoenix
from phoenix.primitive.simplification_smt import compile_hamiltonian_smt

console = Console()

# 2-qubit gates that each cost ~2 CX to implement on hardware
_2Q_ROTATION_GATES = {"rxx", "ryy", "rzz", "rzx", "rxy", "rxz", "ryz"}


def infidelity(u_exact, u_circ):
    """Compute process infidelity between two unitaries."""
    return phoenix.utils.infidelity(u_exact, u_circ)


def effective_cx(qc) -> int:
    """Count CX + 2-qubit rotation gates (each ~2 CX on hardware)."""
    ops = qc.count_ops()
    total = ops.get("cx", 0)
    for gate in _2Q_ROTATION_GATES:
        total += ops.get(gate, 0) * 2
    return total


def _infidelity_style(val):
    if val < 1e-6:
        return "bold green"
    if val < 1e-3:
        return "yellow"
    return "bold red"


def _smt_progress(depth, status):
    """Rich-formatted callback for the SMT solver's incremental search.

    The solver tries T=1, 2, 3, ... CNOTs until it finds a satisfying assignment.
      - UNSAT       = Z3 proved that T CNOTs are NOT enough (lower-bound certificate)
      - SAT         = Z3 found a valid Clifford conjugation with T CNOTs (optimal!)
      - skip        = trivially weight <= 2, no CNOTs needed
      - optimizing  = minimizing weight at optimal T via Z3 Optimize
      - trying_extra = trying T+1 with weight optimization to reduce total CX
    """
    if status == "skip":
        console.print(f"    T=0  [green]already weight <= 2, no CNOT needed[/]")
    elif status == "unsat":
        console.print(f"    T={depth}  [red]UNSAT[/] [dim]-- {depth} CNOT(s) provably insufficient[/]")
    elif status == "sat":
        console.print(f"    T={depth}  [bold green]SAT[/] [green]-- optimal solution found: {depth} CNOT(s)[/]")
    elif status == "optimizing":
        console.print(f"    T={depth}  [bold yellow]OPT[/]  [dim]-- minimizing weight at depth {depth}...[/]")
    elif status == "trying_extra":
        console.print(f"    T={depth}  [bold yellow]OPT+1[/] [dim]-- trying depth {depth} with weight optimization...[/]")


def run_test(name, pauli_strings, coeffs):
    ham = phoenix.Hamiltonian(pauli_strings, coeffs)
    ham.print_tableau()
    u_exact = ham.unitary_evolution()

    console.rule(f"[bold cyan]{name}[/]")
    console.print(
        f"  [dim]{len(pauli_strings)} Pauli strings on {ham.num_qubits} qubits[/]"
    )

    results = {}  # method -> (cx, eff_cx, infidelity, ops_dict)

    # -- SMT (globally optimal) --
    console.print("\n  [bold magenta]SMT solver[/]  [dim]incremental search for minimum CNOT depth:[/]")
    qc_smt = compile_hamiltonian_smt(ham, progress_callback=_smt_progress)
    u_smt = Operator(qc_smt).to_matrix()
    infid_smt = infidelity(u_exact, u_smt)
    ops_smt = dict(qc_smt.count_ops())
    cx_smt = ops_smt.get("cx", 0)
    eff_smt = effective_cx(qc_smt)
    results["SMT"] = (cx_smt, eff_smt, infid_smt, ops_smt)

    # -- Greedy (heuristic) --
    console.print("\n  [bold magenta]Greedy heuristic[/]")
    qc_greedy = phoenix.compile_hamiltonian_simulation(ham, optimize=True)
    u_greedy = Operator(qc_greedy).to_matrix()
    infid_greedy = infidelity(u_exact, u_greedy)
    ops_greedy = dict(qc_greedy.count_ops())
    cx_greedy = ops_greedy.get("cx", 0)
    eff_greedy = effective_cx(qc_greedy)
    results["Greedy"] = (cx_greedy, eff_greedy, infid_greedy, ops_greedy)

    # -- Via compiler.py method="smt" --
    console.print("\n  [bold magenta]compile_hamiltonian_simulation(method='smt')[/]")
    qc_via = phoenix.compile_hamiltonian_simulation(
        ham, method="smt", optimize=True,
        smt_verbose=False,
    )
    u_via = Operator(qc_via).to_matrix()
    infid_via = infidelity(u_exact, u_via)
    ops_via = dict(qc_via.count_ops())
    cx_via = ops_via.get("cx", 0)
    eff_via = effective_cx(qc_via)
    results["SMT+opt"] = (cx_via, eff_via, infid_via, ops_via)

    # -- Results table --
    table = Table(title="Results", show_header=True, header_style="bold")
    table.add_column("Method", style="cyan", min_width=12)
    table.add_column("CX", justify="right", min_width=6)
    table.add_column("eff. CX", justify="right", min_width=9)
    table.add_column("Infidelity", justify="right", min_width=14)
    table.add_column("2Q rotation gates", min_width=20)

    best_eff = min(v[1] for v in results.values())

    for method, (cx, eff, infid, ops) in results.items():
        eff_style = "bold green" if eff == best_eff else ""
        infid_style = _infidelity_style(infid)
        # list non-cx 2Q rotation gates
        rot_parts = []
        for g in sorted(_2Q_ROTATION_GATES):
            cnt = ops.get(g, 0)
            if cnt:
                rot_parts.append(f"{g}={cnt}")
        rot_str = ", ".join(rot_parts) if rot_parts else "[dim]--[/]"

        table.add_row(
            method,
            str(cx),
            Text(str(eff), style=eff_style),
            Text(f"{infid:.2e}", style=infid_style),
            rot_str,
        )

    console.print()
    console.print(table)
    console.print(
        "  [dim]eff. CX = cx + 2 * (rxx + ryy + rzz + rzx + ...)"
        "  (each 2Q rotation ~ 2 CX on hardware)[/]"
    )

    if eff_smt < eff_greedy:
        console.print(
            Panel(
                f"[bold green]SMT saved {eff_greedy - eff_smt} effective CX vs greedy![/]",
                border_style="green",
            )
        )

    return eff_smt, eff_greedy


if __name__ == "__main__":
    console.print(
        Panel(
            "[bold]Phoenix SMT vs Greedy Compilation Benchmark[/]",
            style="bold blue",
            expand=False,
        )
    )

    # Test 1: Trivial case (already weight <= 2)
    run_test(
        "Trivial: weight-1 Pauli strings",
        ["XI", "IZ"],
        [0.5, 0.3],
    )

    # Test 2: Three 8-qubit Pauli strings (known optimal = 4 CNOTs from SMT)
    run_test(
        "8-qubit, 3 Pauli strings",
        ["XXXZIYZI", "YXXZIYYI", "ZXXZIYZI"],
        [-0.0125, -0.0125, -0.0125],
    )

    # Test 3: Four 4-qubit strings (SMT beats greedy)
    run_test(
        "4-qubit, 4 Pauli strings (SMT < greedy)",
        ["XYZX", "ZXZY", "YXYZ", "ZYXY"],
        [-0.01, 0.02, -0.015, 0.005],
    )

    # Test 4: Six 8-qubit strings
    run_test(
        "8-qubit, 6 Pauli strings",
        [
            "XXXZIYZI", "YXXZIYYI", "ZXXZIYZI",
            "IXXZIYZZ", "IXXZIYYY", "IXXZIYZY",
        ],
        [-0.0125] * 6,
    )

    console.rule("[bold green]All tests complete[/]")
