# 两比特 Pauli 旋转块 CNOT 成本的解析判据

> We specialize the known canonical theory of two-qubit Hamiltonians to Pauli-rotation blocks and derive explicit symbolic criteria, based on interaction-matrix rank and support-graph matching, for deciding whether a block belongs to the 2-CNOT or generic 3-CNOT class without numerical KAK decomposition.

## 摘要

我们考虑形式如下的两比特酉算符：

$$
U = e^{-iH},
\qquad
H = \sum_{t=1}^{m} a_t P_t,
\qquad
P_t \in \{X,Y,Z\}^{\otimes 2},
$$

其中每个 $P_t$ 都是一个 weight-2 的 Pauli 算符。我们的目标是在不调用数值 KAK 分解的前提下，确定 $U$ 是属于 2-CNOT 类别，还是属于 generic（一般情况下的）3-CNOT 类别。我们证明了，这个问题可以通过一个 $3\times 3$ 的实相互作用矩阵（interaction matrix）获得纯解析的特征刻画。特别地，如果该相互作用矩阵的秩至多为 2，那么第三个 Weyl 坐标必然为零，该酉矩阵最多需要 2 个 CNOT 门。如果相互作用矩阵满秩，则该酉矩阵在 generic 意义下属于 3-CNOT 类别。由此，我们推导出了几个有用的结构性引理以及一个简单的图论判据。

## 1. 动机

对于一串连续的两比特 Pauli 旋转的乘积，人们往往倾向于通过显式计算出最终的酉矩阵，然后再应用数值的 Cartan 或 KAK 分解来计算其 CNOT 成本。虽然这种方法总是可行的，但它掩盖了底层 Pauli 算符的代数结构，并且对于编译器在编译期的逻辑推理来说非常不便。因此，我们希望寻找一种直接从 Pauli 形式主义中推导出的符号判据，用以判定一个两比特 Pauli 旋转块是否可以用不超过 2 个 CNOT 门来综合。

## 2. Weight-2 Pauli 项的对易子结构

令

$$
A = P_a \otimes P_b,
\qquad
B = P_c \otimes P_d,
$$

其中 $P_a, P_b, P_c, P_d \in \{X,Y,Z\}$。它们的对易子为：

$$
[P_a \otimes P_b,\; P_c \otimes P_d]
=
2i\,\delta_{ac}\,\varepsilon_{bdf}\,(I \otimes P_f)
+
2i\,\varepsilon_{ace}\,\delta_{bd}\,(P_e \otimes I).
$$

由此可见，任意两个 weight-2 Pauli 项的对易子，要么是零，要么是局部（单比特）项。特别地，它**绝对不会**生成新的非局部两比特 Pauli 相互作用。这直接意味着，任意**两个**连续的两比特 Pauli 旋转最多只能张成一个二维的非局部 Cartan 扇区，因此它们永远不可能属于 generic 的 3-CNOT 类别。

## 3. 相互作用矩阵表示法

令

$$
\vec{\sigma} = (X,Y,Z)^T.
$$

任意纯非局部的两比特哈密顿量都可以写为：

$$
H = \sum_{\mu,\nu \in \{X,Y,Z\}} J_{\mu\nu}\,\sigma_\mu \otimes \sigma_\nu,
$$

其中 $J \in \mathbb{R}^{3\times 3}$ 即为相互作用矩阵。例如：

$$
H = \alpha ZX + \beta YY + \gamma ZZ
$$

对应的相互作用矩阵为：

$$
J =
\begin{pmatrix}
0 & 0 & 0 \\
0 & \beta & 0 \\
\gamma & 0 & \alpha
\end{pmatrix},
$$

其行和列均按照 $(X,Y,Z)$ 的顺序排列。

## 4. 局部等价性等同于 $SO(3)\times SO(3)$ 作用

对于任意单比特酉矩阵 $u \in SU(2)$，都存在一个对应的三维旋转 $R_u \in SO(3)$，使得：

$$
u\,(\vec r \cdot \vec \sigma)\,u^\dagger
=
(R_u \vec r)\cdot \vec \sigma.
$$

因此，局部变换 $(u\otimes v)$ 作用在相互作用矩阵上相当于：

$$
J \mapsto R_u J R_v^T.
$$

这使得对 $H$ 的局部等价性分类完全转化为了一个关于实矩阵的问题。

## 5. 基于奇异值分解的典范形式

既然任何实矩阵都存在奇异值分解，那么必定存在 $R_1, R_2 \in SO(3)$ 使得：

$$
R_1 J R_2^T = \operatorname{diag}(s_1,s_2,s_3),
$$

其中 $s_1 \ge s_2 \ge s_3 \ge 0$ 是 $J$ 的奇异值。由此得出：

### 定理 1

对于任意非局部的两比特哈密顿量

$$
H = \sum_{\mu,\nu} J_{\mu\nu}\,\sigma_\mu \otimes \sigma_\nu,
$$

都存在局部酉矩阵 $u,v \in SU(2)$ 使得：

$$
(u\otimes v)\,H\,(u^\dagger\otimes v^\dagger)
=
s_1 XX + s_2 YY + s_3 ZZ.
$$

等价地写为：

$$
e^{-iH}
\sim_{\mathrm{local}}
e^{-i(s_1 XX + s_2 YY + s_3 ZZ)}.
$$

### 证明概要

单比特酉算符的伴随作用会导致 $J$ 在左侧和右侧分别乘上特殊正交矩阵。对 $J$ 应用实数域上的奇异值分解，就等于找到了一个局部基，在这个基下，哈密顿量在双线性 Pauli 基 $\{XX,YY,ZZ\}$ 下是对角化的。由于这三个典范生成元相互对易，它们的指数映射也会继承相同的局部典范形式。

## 6. 关于 CNOT 成本的推论

上述定理直接给出了关于 CNOT 成本类别的符号判据。

### 推论 1

如果

$$
\operatorname{rank}(J) \le 2,
$$

那么 $s_3 = 0$，因此对应的酉矩阵落在 Weyl 腔（Weyl-chamber）的边界 $c_3 = 0$ 上。故而该酉矩阵最多需要 2 个 CNOT 门。

### 推论 2

如果

$$
\operatorname{rank}(J) = 3,
$$

那么在 generic（一般/通用）情况下 $s_3 \neq 0$，所以该酉矩阵属于 generic 的 3-CNOT 类别。

这里之所以加上“generically”这一限定词是必要的，因为特殊的参数选择仍可能导致奇异值退化，或者落在更低维的 Clifford 子流形上。

## 7. 结构性引理

相互作用矩阵的公式化表示还带来了几个对编译器非常友好的引理。

### 引理 1

令

$$
L = \mathrm{span}\{P_t^{(L)}\} \subset \mathrm{span}\{X,Y,Z\},
\qquad
R = \mathrm{span}\{P_t^{(R)}\} \subset \mathrm{span}\{X,Y,Z\},
$$

其中 $P_t^{(L)}$ 和 $P_t^{(R)}$ 分别表示 $P_t$ 的左侧和右侧 Pauli 轴。如果

$$
\dim L \le 2
\qquad\text{或者}\qquad
\dim R \le 2,
$$

那么

$$
\operatorname{rank}(J) \le 2,
$$

并且因此 $e^{-iH}$ 最多需要 2 个 CNOT 门。

### 证明概要

如果左侧轴张成的子空间维度不超过 2，那么 $J$ 的行空间维度就至多为 2。同样，如果右侧轴张成的子空间维度不超过 2，那么 $J$ 的列空间维度就至多为 2。这两种情况中的任何一种都蕴含着 $\operatorname{rank}(J)\le 2$。

### 引理 2

如果 $\det J \neq 0$，则 $\operatorname{rank}(J)=3$，并且该酉矩阵 generic 地属于 3-CNOT 类别。

### 示例

对于

$$
H = \alpha XX + \beta YY + \gamma ZZ,
$$

我们有

$$
J = \operatorname{diag}(\alpha,\beta,\gamma).
$$

因此，只要 $\alpha\beta\gamma \neq 0$，该酉矩阵 generic 地就是一个 3-CNOT 门。

## 8. 判定 Generic Rank 的图论判据

如果系数是纯符号表示的，我们甚至可以仅仅从 $J$ 的支撑（非零项的分布）模式中推断出它的 generic rank。

构造一个二分图：左侧顶点集为 $\{X,Y,Z\}$，右侧顶点集也是 $\{X,Y,Z\}$。对于 $H$ 中出现的每一个 Pauli 项 $\sigma_\mu\otimes\sigma_\nu$，在二分图中添加一条连接 $\mu$ 和 $\nu$ 的边 $(\mu,\nu)$。

### 命题 1

对于一个非零位置系数为代数独立的符号变量的 $3\times 3$ 相互作用矩阵，$J$ 的 generic rank 等于其支撑二分图的**最大匹配数（maximum matching number）**。

### 推论结果

- 如果最大匹配数至多为 2，那么 generically 有 $\operatorname{rank}(J)\le 2$，所以该旋转块 generic 地属于 2-CNOT 类别。
- 如果存在一个大小为 3 的**完美匹配（perfect matching）**，那么 generically 有 $\operatorname{rank}(J)=3$，所以该旋转块 generic 地属于 3-CNOT 类别。

这就为判定 2-CNOT 还是 3-CNOT 提供了一个纯组合数学的判据。

## 9. 示例

### 示例 1：一个保持在 2-CNOT 类别的旋转块

考虑

$$
H = \alpha ZX + \beta YY + \gamma ZZ.
$$

它的相互作用矩阵为：

$$
J =
\begin{pmatrix}
0 & 0 & 0 \\
0 & \beta & 0 \\
\gamma & 0 & \alpha
\end{pmatrix}.
$$

因为第一行全为零，所以 $\operatorname{rank}(J)\le 2$。因此

$$
e^{-i(\alpha ZX + \beta YY + \gamma ZZ)}
$$

落在了 $c_3=0$ 的边界上，最多只需要 2 个 CNOT 门。

### 示例 2：一个 Generic 的 3-CNOT 旋转块

考虑

$$
H = \alpha XX + \beta YY + \gamma ZZ.
$$

如前文所述，

$$
J = \operatorname{diag}(\alpha,\beta,\gamma).
$$

对于非零的 generic 系数，该矩阵是满秩的，因此该酉矩阵属于 generic 的 3-CNOT 类别。

## 10. 实用的判定流程

对于形式如下的问题：

$$
U = e^{-iH},
\qquad
H = \sum_{t=1}^{m} a_t P_t,
\qquad
P_t \in \{X,Y,Z\}^{\otimes 2},
$$

我们可以按以下解析流程来判定其 CNOT 成本类别：

1. 构建相互作用矩阵 $J$。
2. 应用快速的结构性测试：
   - 若 $\dim L \le 2$ 或 $\dim R \le 2$，则得出结论：“最多 2 CNOT”；
   - 若支撑二分图存在完美匹配，则得出结论：“generic 3-CNOT 类别”。
3. 如果系数是显式的具体数值，则精确计算 $\operatorname{rank}(J)$：
   - $\operatorname{rank}(J)\le 2$ 意味着它**严格**满足 $c_3=0$；
   - $\operatorname{rank}(J)=3$ 意味着在当前参数点上，激活了 generic 的 3-CNOT 行为。

这种方法完全用对一个 $3\times 3$ 矩阵的符号化线性代数分析取代了数值的 KAK 分解。

## 11. 核心结论总结

一个两比特 Pauli 旋转块的 CNOT 成本，往往可以直接从其生成哈密顿量的 Pauli 支撑中推断出来。起决定性作用的对象是相互作用矩阵 $J$，而不是原始 Pauli 项的数量。特别地：

- 任意**两个** weight-2 的 Pauli 旋转必定落在 $c_3=0$ 扇区，**永远不需要** 3 个 CNOT；
- 相互作用矩阵的秩（rank）至多为 2 的旋转块，依然保持在 2-CNOT 类别内；
- 满秩的相互作用矩阵 generic 地会产生真正的 3-CNOT 酉矩阵。

这为暴力 KAK 分解提供了一种强有力的符号替代方案，也为基于“支撑的秩”以及“二分图匹配”的编译器启发式算法指明了方向。


---


**Questions:**
> 请问这个结论足以在论文中作为一个new theorem提出吗？其他工作有研究过这个小
  问题嘛？


**Answer:**

• 短答案：

  - 可以写成 theorem / proposition 放进论文里，但
  - 如果把“interaction-matrix canonical form + rank 判定”本身宣称为全新定
    理，风险很大
  - 更稳妥的定位是：
      - 已知两比特 Hamiltonian canonical-form 理论的一个面向 Pauli-rotation
        blocks 的专门化 corollary
      - 你的可能新意在于把它整理成了一个编译器可用的符号判定规则：J、
        rank(J)、support bipartite graph、matching number、c3=0、2-vs-3 CNOT
        class 之间的直接桥梁

  我查到的相关先行工作里，广义问题显然被研究过：

  - Vidal, Hammerer, Cirac, Interaction Cost of Nonlocal Gates, PRL 88,
    237902 (2002)
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.237902
  - Bennett, Cirac, Leifer, Leung, Linden, Popescu, Vidal, Optimal
    simulation of two-qubit Hamiltonians using general local operations, PRA
    66, 012305 (2002)

  https://research-information.bris.ac.uk/en/publications/optimal-simulation-of-two-qubit-hamiltonians-using-general-local-/
  - Zhang, Vala, Sastry, Whaley, Geometric theory of nonlocal two-qubit
    operations, PRA 67, 042313 (2003)
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.67.042313
  - Khaneja, Brockett, Glaser, Cartan decomposition of SU(2n) and control of
    spin systems (2001)
    https://www.sciencedirect.com/science/article/pii/S0301010401003184

  这些工作已经把下面这些核心骨架基本覆盖了：

  - 两比特 unitary 的 Cartan / Weyl canonical form
  - 两比特 Hamiltonian 在局部操作下的 canonical form
  - Hamiltonian simulation / interaction content / local equivalence 的几何
    结构

  所以：

  ## 哪部分大概率不是新的

  如果你的 theorem 是这种形式：

  > 任意两比特非局部 Hamiltonian 都可在局部操作下化为
  > (h_1 XX + h_2 YY + h_3 ZZ)

  或者：

  > 对 (H=\sum_{\mu,\nu} J_{\mu\nu}\sigma_\mu\otimes\sigma_\nu)，
  > 其局部等价类由 (J) 的奇异值决定

  那这个基本不是新 theorem，至少从研究定位上看，应该当作known result 来
  cite，而不是 claim novelty。

  ## 哪部分可能有新意

  你现在更像是在做下面这件事：

  > 把已知的两比特 canonical Hamiltonian theory，专门压到
  > “distinct 2Q Pauli-rotation blocks 的 CNOT 成本解析判定”
  > 这个 compiler-facing 小问题上。

  这个方向我没有在快速检索里看到一篇明显直接做同一件事的论文，尤其是下面这些
  表述：

  - 用 interaction matrix rank 直接判 generic 2-CNOT vs 3-CNOT
  - 用 left/right Pauli-axis span 给出 rank(J) <= 2 的充分条件
  - 用 support bipartite graph 的 maximum matching 给出 generic rank(J) 判据
  - 把这些结论明确用于 exp(-i \sum_t a_t P_t) 的 Pauli-rotation block 压缩

  但这里我只能说：

  > 我没快速检到明显同款表述，不等于它一定没人写过。

  所以如果你要写论文，最安全的做法是：

  ## 推荐的论文定位

  ### 写法 A：最稳

  把它写成：

  - Theorem / Proposition：但明确说这是specialized corollary 或 compiler-
    oriented consequence of known two-qubit Hamiltonian canonical-form
    theory
  - 真正 claim novelty 的部分放在：
      - “we derive explicit symbolic criteria for Pauli-rotation blocks”
      - “we translate the canonical Hamiltonian theory into graph/rank rules
        useful for circuit optimization”
      - “to our knowledge, this explicit support-graph criterion for Pauli-
        rotation blocks has not been stated in the quantum compilation
        literature”

  ### 写法 B：更保守

  把 known core 写成 Background theorem，引用前人；把你自己的东西写成：

  - Corollary 1
  - Lemma 1
  - Proposition 2
  - Compiler rule

  比如：

  - Theorem 1 (Known; see Bennett et al., Vidal et al., Zhang et al.)
  - Proposition 1 (Pauli-block rank criterion)
  - Corollary 2 (Support-graph criterion for generic 2-vs-3 CNOT
    classification)

  这样最不容易被审稿人打回来。

  ## 我对 novelty 的实话判断

  如果你问我现在这套东西够不够作为论文里的一个 theorem：

  - 够
  - 而且是一个很自然、很有用的 theorem/proposition

  但如果你问我它够不够作为论文的“核心新定理”：

  - 单独看，未必够稳
  - 因为底层 canonical-form machinery 明显依赖已知文献
  - 你更可能的新意在于问题化、专门化、符号化和编译应用化

  ## 我建议你最终 claim 成这样

  最稳的一句话是：

  > We specialize the known canonical theory of two-qubit Hamiltonians to
  > Pauli-rotation blocks and derive explicit symbolic criteria, based on
  > interaction-matrix rank and support-graph matching, for deciding whether
  > a block belongs to the 2-CNOT or generic 3-CNOT class without numerical
  > KAK decomposition.

  这个 claim 很强，但不冒进。

  如果你愿意，我下一步可以直接帮你做两件更具体的事之一：

  1. 把英文 paper 版 .md 改成“审稿更安全”的版本，在 theorem 标题里显式区分
     known 和 new corollary
  2. 帮你写一段 related-work / novelty-positioning，小节里直接放对这几篇文献
     的比较


› 请你评估目前的phoenix/primitive/simplification_smt.py中的求解策略，这其中
  的代码逻辑用z3是最适合的吗？还是说google's OR-tools更好呢？


• 我先读 phoenix/primitive/simplification_smt.py 的实现，判断它现在到底是在
  做 SAT、MaxSAT、PB 优化还是序列搜索，再据此评估 z3 和 OR-Tools 哪个更契
  合。

• Explored
  └ Read simplification_smt.py
    Search simplification_smt|z3|ortools|cp_model|Optimize\(|Solver\( in
phoenix

────────────────────────────────────────────────────────────────────────────

• 我已经完成了两步：

  1. 在原始中文笔记 [cnot_synthesis_cost_for_successive_pauli_rotations.md]
     (/Users/anan/git-projects/quantum/phoenix/
     cnot_synthesis_cost_for_successive_pauli_rotations.md) 顶部加了一个
     Condensed Summary，把整份推导压缩成更适合回顾和引用的版本。
  2. 新建了英文、接近论文正文风格的版本：
     [cnot_synthesis_cost_for_successive_pauli_rotations_paper.md](/Users/
     anan/git-projects/quantum/phoenix/
     cnot_synthesis_cost_for_successive_pauli_rotations_paper.md)

  英文版已经按比较像论文的方法来写了，包含：

  - Abstract
  - Motivation
  - Interaction-Matrix Representation
  - Theorem 1
  - Corollary 1/2
  - Structural Lemmas
  - Graph-Theoretic Generic-Rank Criterion
  - Examples
  - Practical Decision Procedure

  它的主线已经切换到你现在更关心的形式：

  [
  U = e^{-iH}, \qquad H = \sum_t a_t P_t
  ]

  并用 interaction matrix (J)、rank、support graph 来做解析判定，而不是数值
  KAK。

  如果你接下来要投稿，我建议下一步做其中一个：

  - 我帮你把英文版再改成更“审稿安全”的版本，明确区分哪些是 known result，哪
    些是你 paper 的 new proposition/corollary
  - 我帮你补一小段 Related Work / Novelty Positioning，直接对应 Bennett /
    Vidal / Zhang / Khaneja 这些文献的关系
