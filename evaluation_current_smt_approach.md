# Review of the Current SMT Approach

This note reviews the current implementation in [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py) after the recent changes.

## Findings

### 1. High severity: the peeling semantics is still inconsistent with the emitted circuit

The transition system still freezes a Pauli string once it becomes weight `<= 1`:

- state update: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L262)
- active-mask update: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L266)
- `peeled_at` extraction: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L316)

However, `build_circuit()` still constructs one global Clifford sandwich and places all Pauli evolutions in one center block:

- circuit construction: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L558)

So the solver semantics is:

- some strings stop evolving early,

while the emitted circuit semantics is:

- every original string evolves through the full final Clifford.

Those are not the same model.

I re-checked the counterexample:

- Hamiltonian: `['IIXX', 'XXZX']`
- solver result with `optimize_total_cx=True`: depth `3`, `peeled_at = [[0], [], [1]]`
- built-circuit infidelity: about `1.98e-4`

That means the correctness bug is still present.

### 2. High severity: the current total-CX optimization can actively prefer semantically invalid solutions

The post-processing cost function only looks at the **final active strings**:

- rotation-cost accounting: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L478)
- total score: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L499)

This would be fine only if peeled strings were truly removed from the emitted circuit. But they are not: `build_circuit()` still emits center evolutions for all original terms.

So when peeling happens, the optimizer is minimizing a cost model that matches the SMT state machine, not the actual emitted circuit.

The same counterexample shows the problem clearly:

- with `optimize_total_cx=False`, the solver returns depth `2`
- with `optimize_total_cx=True`, the solver returns depth `3`
- the depth-`3` solution is preferred because the cost model counts peeled strings as gone
- but the emitted circuit is still incorrect, so that apparent improvement is not trustworthy

This is more serious than “objective is only a surrogate”. Right now the objective can be inconsistent with the compiled artifact itself.

### 3. Medium severity: the `T+1` tie-handling logic and its comment do not match

The updated comparison is now:

- [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L195)

It accepts the `T+1` solution whenever

`_compute_total_cx_cost(result_t1) <= _compute_total_cx_cost(result)`.

But the comment says:

- on equal cost, prefer the higher-depth solution with fewer weight-2 strings

The code does **not** check “fewer weight-2 strings”. It only checks `<=`.

So the actual behavior is:

- any equal-scoring `T+1` solution replaces the shallower one,

not:

- equal-scoring `T+1` solutions replace the shallower one only if they have a better secondary structure.

This matters because it biases the search toward deeper solutions without enforcing the property claimed in the comment.

### 4. Medium severity: the fixed-depth optimization objective is still a surrogate, not the exact hardware objective

At fixed depth, `_solve_for_depth_opt_weight()` minimizes:

1. number of qubit pairs carrying terminal weight-2 strings
2. total final Pauli weight

See:

- [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L404)

That is a useful heuristic, but it is still not the exact hardware objective you ultimately care about.

In particular:

- pair count is coarser than the interaction-matrix-aware rotation cost
- total final weight is only a tie-breaker proxy
- the search only probes `T` and `T+1`, not all depths that could reduce total 2Q cost

So the “global total-CX optimization” remains heuristic.

### 5. Medium severity: the encoding still favors research agility over propagation strength

The current encoding is compact and expressive, but not especially strong from a solver-performance standpoint:

- `ctrl/targ` are integer selectors: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L223)
- `_z3_select()` builds nested `If` chains: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L282)
- each single-qubit Clifford is encoded by four Boolean matrix entries plus one symplecticity constraint: [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L244)

This is one reason Z3 is still a natural fit today: the model is really an SMT-flavored symbolic transition system over Booleans, XORs, and conditionals.

But this encoding is weaker than a more explicit combinatorial formulation, for example:

- one-hot directed-edge variables for the chosen CNOT
- one-hot 6-state variables for the single-qubit Clifford on each qubit and each step

That kind of reformulation would help both Z3 and CP-SAT.

### 6. Lower severity: the depth search still rebuilds the whole model from scratch

The outer loop tries `T = 1, 2, ...` and constructs a fresh solver each time:

- [simplification_smt.py](/Users/anan/git-projects/quantum/phoenix/phoenix/primitive/simplification_smt.py#L150)

That is simple and reasonable for research code, but it leaves performance on the table:

- no incremental reuse
- no assumptions-based depth extension
- no symmetry breaking on equivalent gate choices

## Overall judgment

The current code does **not** change my earlier solver recommendation. If anything, it strengthens it:

- the main issue is still model semantics and objective consistency,
- not whether the backend is Z3 or OR-Tools.

As the code stands today, Z3 is still the better fit, because the model is naturally expressed as:

- Boolean tableau variables
- XOR-heavy GF(2) updates
- symbolic conditionals
- exact satisfiability across bounded depths

That is closer to SMT than to the style of pure Boolean / pseudo-Boolean model that typically makes CP-SAT shine.

## Z3 vs OR-Tools

### Why Z3 still makes sense here

Z3 remains a good match for the current implementation because:

- the transition relation is naturally symbolic
- XOR and conditional structure are first-class citizens in the model
- UNSAT-at-depth is meaningful as a lower-bound certificate
- the code is still evolving at the modeling level, where Z3 is usually easier to iterate on

### When OR-Tools would become more attractive

OR-Tools CP-SAT becomes more compelling after a substantial reformulation, for example:

- Booleanizing the CNOT choice
- Booleanizing the 1Q Clifford choice
- replacing selector-style `If` expressions with explicit support literals
- encoding the true cost as a pseudo-Boolean objective

At that point, CP-SAT's parallel multi-worker search may become a real advantage. But that is not a fair comparison to the current code, because the current code is not yet in CP-SAT's preferred shape.

## Recommendation

I would prioritize changes in this order:

1. Fix the semantic mismatch between peeling and circuit construction.
2. Make the cost model consistent with the emitted circuit.
3. Replace the surrogate objective with the exact 2Q-cost objective you really care about.
4. Strengthen the encoding with one-hot action variables and symmetry breaking.
5. Only then benchmark Z3 against CP-SAT.

## Bottom line

The present implementation is still better viewed as:

- a promising research prototype in Z3,

not yet as:

- a solver formulation ready for a meaningful “Z3 vs OR-Tools” bake-off.

Until the semantic mismatch is fixed, any apparent optimization win from the current `optimize_total_cx` path should be treated with caution.
