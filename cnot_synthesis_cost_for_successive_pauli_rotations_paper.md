# Analytic Criteria for the CNOT Cost of Two-Qubit Pauli-Rotation Blocks

  > We specialize the known canonical theory of two-qubit Hamiltonians to Pauli-rotation blocks and derive explicit symbolic criteria, based on interaction-matrix rank and support-graph matching, for deciding whether a block belongs to the 2-CNOT or generic 3-CNOT class without numerical KAK decomposition.

## Abstract

We consider two-qubit unitaries of the form

$$
U = e^{-iH},
\qquad
H = \sum_{t=1}^{m} a_t P_t,
\qquad
P_t \in \{X,Y,Z\}^{\otimes 2},
$$

where each $P_t$ is a weight-2 Pauli operator. The goal is to determine whether $U$ belongs to the 2-CNOT class or the generic 3-CNOT class without invoking a numerical KAK decomposition. We show that this question admits a purely analytic characterization in terms of a $3\times 3$ real interaction matrix. In particular, if the interaction matrix has rank at most two, then the third Weyl-coordinate vanishes and the unitary requires at most two CNOT gates. If the interaction matrix has full rank, then the unitary is generically in the 3-CNOT class. This yields several useful structural lemmas and a simple graph-theoretic criterion.

## 1. Motivation

For a product of successive two-qubit Pauli rotations, it is tempting to classify the CNOT cost by explicitly computing the final unitary and then applying a numerical Cartan or KAK decomposition. While this is always possible, it obscures the underlying Pauli structure and is inconvenient for compiler-side reasoning. We therefore seek symbolic criteria, derived directly from the Pauli formalism, that decide whether a block of two-qubit Pauli rotations can be synthesized with at most two CNOT gates.

## 2. Commutator Structure of Weight-2 Pauli Terms

Let

$$
A = P_a \otimes P_b,
\qquad
B = P_c \otimes P_d,
$$

with $P_a, P_b, P_c, P_d \in \{X,Y,Z\}$. Their commutator is

$$
[P_a \otimes P_b,\; P_c \otimes P_d]
=
2i\,\delta_{ac}\,\varepsilon_{bdf}\,(I \otimes P_f)
+
2i\,\varepsilon_{ace}\,\delta_{bd}\,(P_e \otimes I).
$$

Hence the commutator of two weight-2 Pauli terms is either zero or a local term. In particular, it never generates a new nonlocal two-qubit Pauli interaction. This immediately implies that any pair of successive two-qubit Pauli rotations spans at most a two-dimensional nonlocal Cartan sector, and therefore cannot belong to the generic 3-CNOT class.

## 3. Interaction-Matrix Representation

Let

$$
\vec{\sigma} = (X,Y,Z)^T.
$$

Any purely nonlocal two-qubit Hamiltonian can be written as

$$
H = \sum_{\mu,\nu \in \{X,Y,Z\}} J_{\mu\nu}\,\sigma_\mu \otimes \sigma_\nu,
$$

where $J \in \mathbb{R}^{3\times 3}$ is the interaction matrix. For example,

$$
H = \alpha ZX + \beta YY + \gamma ZZ
$$

corresponds to

$$
J =
\begin{pmatrix}
0 & 0 & 0 \\
0 & \beta & 0 \\
\gamma & 0 & \alpha
\end{pmatrix},
$$

with rows and columns ordered as $(X,Y,Z)$.

## 4. Local Equivalence as an $SO(3)\times SO(3)$ Action

For every $u \in SU(2)$, there exists a corresponding rotation $R_u \in SO(3)$ such that

$$
u\,(\vec r \cdot \vec \sigma)\,u^\dagger
=
(R_u \vec r)\cdot \vec \sigma.
$$

Therefore a local transformation $(u\otimes v)$ acts on the interaction matrix as

$$
J \mapsto R_u J R_v^T.
$$

This reduces the local-equivalence classification of $H$ to a real matrix problem.

## 5. Canonical Form via Singular Value Decomposition

Since every real matrix admits a singular value decomposition, there exist $R_1, R_2 \in SO(3)$ such that

$$
R_1 J R_2^T = \operatorname{diag}(s_1,s_2,s_3),
$$

where $s_1 \ge s_2 \ge s_3 \ge 0$ are the singular values of $J$. Consequently:

### Theorem 1

For every nonlocal two-qubit Hamiltonian

$$
H = \sum_{\mu,\nu} J_{\mu\nu}\,\sigma_\mu \otimes \sigma_\nu,
$$

there exist local unitaries $u,v \in SU(2)$ such that

$$
(u\otimes v)\,H\,(u^\dagger\otimes v^\dagger)
=
s_1 XX + s_2 YY + s_3 ZZ.
$$

Equivalently,

$$
e^{-iH}
\sim_{\mathrm{local}}
e^{-i(s_1 XX + s_2 YY + s_3 ZZ)}.
$$

### Proof sketch

The adjoint action of single-qubit unitaries induces left and right multiplication of $J$ by special orthogonal matrices. Applying the real singular value decomposition to $J$ therefore yields a local basis in which the Hamiltonian is diagonal in the Pauli-bilinear basis $\{XX,YY,ZZ\}$. Since these three canonical generators commute, the exponential inherits the same local canonical form.

## 6. CNOT-Cost Consequences

The theorem immediately yields a symbolic criterion for the CNOT cost class.

### Corollary 1

If

$$
\operatorname{rank}(J) \le 2,
$$

then $s_3 = 0$, and therefore the corresponding unitary lies on the Weyl-chamber boundary $c_3 = 0$. Hence the unitary requires at most two CNOT gates.

### Corollary 2

If

$$
\operatorname{rank}(J) = 3,
$$

then generically $s_3 \neq 0$, so the unitary belongs to the generic 3-CNOT class.

The qualifier “generically” is necessary because special parameter choices may reduce the rank or land on lower-dimensional Clifford subvarieties.

## 7. Structural Lemmas

The interaction-matrix formulation gives several compiler-friendly lemmas.

### Lemma 1

Let

$$
L = \mathrm{span}\{P_t^{(L)}\} \subset \mathrm{span}\{X,Y,Z\},
\qquad
R = \mathrm{span}\{P_t^{(R)}\} \subset \mathrm{span}\{X,Y,Z\},
$$

where $P_t^{(L)}$ and $P_t^{(R)}$ denote the left and right Pauli axes of $P_t$, respectively. If

$$
\dim L \le 2
\qquad\text{or}\qquad
\dim R \le 2,
$$

then

$$
\operatorname{rank}(J) \le 2,
$$

and therefore $e^{-iH}$ requires at most two CNOT gates.

### Proof sketch

If the left axes span at most a two-dimensional subspace, then the row space of $J$ has dimension at most two. Likewise, if the right axes span at most a two-dimensional subspace, then the column space of $J$ has dimension at most two. Either condition implies $\operatorname{rank}(J)\le 2$.

### Lemma 2

If $\det J \neq 0$, then $\operatorname{rank}(J)=3$, and the unitary is generically in the 3-CNOT class.

### Example

For

$$
H = \alpha XX + \beta YY + \gamma ZZ,
$$

we have

$$
J = \operatorname{diag}(\alpha,\beta,\gamma).
$$

Thus, whenever $\alpha\beta\gamma \neq 0$, the unitary is generically a 3-CNOT gate.

## 8. A Graph-Theoretic Generic-Rank Criterion

For symbolic coefficients, one can determine the generic rank of $J$ from the support pattern alone.

Construct a bipartite graph whose left vertex set is $\{X,Y,Z\}$ and whose right vertex set is also $\{X,Y,Z\}$. For each Pauli term $\sigma_\mu\otimes\sigma_\nu$ appearing in $H$, add an edge $(\mu,\nu)$.

### Proposition 1

For a $3\times 3$ interaction matrix with algebraically independent symbolic coefficients on its nonzero entries, the generic rank of $J$ equals the maximum matching number of the support bipartite graph.

### Consequences

- If the maximum matching number is at most two, then generically $\operatorname{rank}(J)\le 2$, so the block is generically in the 2-CNOT class.
- If there exists a perfect matching of size three, then generically $\operatorname{rank}(J)=3$, so the block is generically in the 3-CNOT class.

This yields a purely combinatorial criterion for generic two-versus-three-CNOT classification.

## 9. Examples

### Example 1: A block that remains in the 2-CNOT class

Consider

$$
H = \alpha ZX + \beta YY + \gamma ZZ.
$$

Its interaction matrix is

$$
J =
\begin{pmatrix}
0 & 0 & 0 \\
0 & \beta & 0 \\
\gamma & 0 & \alpha
\end{pmatrix}.
$$

Since the first row is zero, we have $\operatorname{rank}(J)\le 2$. Therefore

$$
e^{-i(\alpha ZX + \beta YY + \gamma ZZ)}
$$

lies on the $c_3=0$ boundary and requires at most two CNOT gates.

### Example 2: A generic 3-CNOT block

Consider

$$
H = \alpha XX + \beta YY + \gamma ZZ.
$$

As above,

$$
J = \operatorname{diag}(\alpha,\beta,\gamma).
$$

For generic nonzero coefficients, this matrix has full rank, so the unitary belongs to the generic 3-CNOT class.

## 10. Practical Decision Procedure

For

$$
U = e^{-iH},
\qquad
H = \sum_t a_t P_t,
$$

one can decide the CNOT cost class analytically as follows:

1. Build the interaction matrix $J$.
2. Apply a fast structural test:
   - if $\dim L \le 2$ or $\dim R \le 2$, conclude “at most 2 CNOTs”;
   - if the support graph has a perfect matching, conclude “generic 3-CNOT class”.
3. If the coefficients are explicit numbers, compute $\operatorname{rank}(J)$ exactly:
   - $\operatorname{rank}(J)\le 2$ implies $c_3=0$ exactly;
   - $\operatorname{rank}(J)=3$ implies generic 3-CNOT behavior for that parameter point.

This replaces a numerical KAK decomposition by a symbolic linear-algebraic analysis of a $3\times 3$ matrix.

## 11. Main Takeaway

The CNOT cost of a two-qubit Pauli-rotation block can often be inferred directly from the Pauli support of its generator Hamiltonian. The decisive object is the interaction matrix $J$, not the raw number of Pauli terms. In particular:

- any pair of weight-2 Pauli rotations necessarily lies in the $c_3=0$ sector and never requires three CNOTs;
- blocks with interaction rank at most two remain in the 2-CNOT class;
- full-rank interaction matrices generically produce genuine 3-CNOT unitaries.

This provides a symbolic alternative to brute-force KAK decomposition and suggests compiler-level heuristics based on support rank and bipartite matching.
