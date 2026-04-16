# CNOT Synthesis Cost for Successive Pauli Rotations

## Condensed Summary

这份笔记的核心结论可以提炼为下面几条。

### 1. 两个 2Q Pauli generators 的对易子不会产生新的非局部项

对

$$
A = P_a \otimes P_b,\qquad B = P_c \otimes P_d,
$$

有

$$
[P_a \otimes P_b,\; P_c \otimes P_d]
=
2i\,\delta_{ac}\,\varepsilon_{bdf}\,(I \otimes P_f)
+
2i\,\varepsilon_{ace}\,\delta_{bd}\,(P_e \otimes I).
$$

因此两个 weight-2 Pauli generators 的对易子要么是 0，要么是局部项，绝不会生成新的两比特交互项。

### 2. 任意两个 successive 2Q Pauli rotations 都满足 `c3 = 0`

无论这两项是：

- 共享一个 qubit 位置，还是
- 在两个位置都不同但彼此对易，

最终的非局部内容都至多落在二维 Cartan 子空间中。因此任意两个 successive 2Q Pauli rotations 都满足

$$
c_3 = 0,
$$

所以它们至多需要 2 个 CNOT，而不会需要 3 个 CNOT。

### 3. `只有第一个 Weyl 分量非零` 不等于 `1 CNOT`

这是最容易混淆的一点。  
像单个 $R_{ZZ}(\theta)$ 的 Weyl 坐标就是 $(\theta/2,0,0)$，但对 generic $\theta$，最小 CNOT 数仍然是 2。只有在 Clifford 特殊角度时才会降到 1 或 0。

因此真正有用的判别不是“只有一个 Weyl 分量非零”，而是：

- $c_3 = 0$ -> 至多 2 CNOT
- $c_3 \neq 0$ -> generic 地需要 3 CNOT

### 4. 对多个 Pauli rotations，最自然的解析工具是 interaction matrix

对

$$
H = \sum_t a_t P_t,
\qquad P_t \in \{X,Y,Z\}^{\otimes 2},
$$

把它写成

$$
H = \sum_{\mu,\nu} J_{\mu\nu}\,\sigma_\mu \otimes \sigma_\nu,
$$

其中 $J \in \mathbb{R}^{3\times 3}$ 是 interaction matrix。

局部门的作用对应于

$$
J \mapsto R_u J R_v^T,
\qquad R_u,R_v \in SO(3),
$$

所以对 $J$ 做 SVD 就得到

$$
H \sim_{\mathrm{local}} s_1 XX + s_2 YY + s_3 ZZ,
$$

其中 $(s_1,s_2,s_3)$ 是 $J$ 的奇异值。

### 5. 由 `rank(J)` 得到 generic 的 CNOT class

- $\operatorname{rank}(J) \le 2$  
  $\Rightarrow s_3 = 0$  
  $\Rightarrow c_3 = 0$  
  $\Rightarrow$ 至多 2 CNOT

- $\operatorname{rank}(J) = 3$  
  $\Rightarrow$ generic 地 $s_3 \neq 0$  
  $\Rightarrow$ generic 地属于 3-CNOT class

### 6. 一个很实用的充分条件

若所有项在左边 Pauli 轴的张成维数不超过 2，或右边 Pauli 轴的张成维数不超过 2，即

$$
\dim L \le 2
\quad\text{or}\quad
\dim R \le 2,
$$

则一定有

$$
\operatorname{rank}(J) \le 2,
$$

从而整个 block 一定满足 `c3 = 0`，所以至多只要 2 个 CNOT。

### 7. support graph 的组合判据

构造一个二分图：

- 左边顶点：$\{X,Y,Z\}$
- 右边顶点：$\{X,Y,Z\}$
- Hamiltonian 中若含有 $\sigma_\mu \otimes \sigma_\nu$，则连边 $(\mu,\nu)$

则：

- 若最大匹配数 $\le 2$，则 generic 地 rank$(J)\le 2$
- 若存在大小为 3 的 perfect matching，则 generic 地 rank$(J)=3$

这给出了一个完全解析、组合化的 `generic 2-vs-3 CNOT` 判据。

### 8. 最终结论

- 单个 arbitrary-angle 2Q Pauli rotation：generic 需要 2 个 CNOT
- 任意两个 successive 2Q Pauli rotations：一定满足 `c3 = 0`，所以 generic 需要 2 个 CNOT
- 三个及以上的 2Q Pauli rotations：不一定需要 3 个 CNOT  
  例如 `ZX + YY + ZZ` 对应 rank$(J)=2$，仍是 2-CNOT class；  
  而 `XX + YY + ZZ` generic 地 rank$(J)=3$，属于 3-CNOT class

下面把这个问题整理成一个更精确的版本。结论先说在前面：

- 你的对易子公式是对的。
- 你的核心 conjecture 也基本对：**任意两个 successive 2Q Pauli rotations 的第三个 Cartan / Weyl 分量恒为 0**，因此它们**至多**需要 2 个 CNOT，**不会**需要 3 个 CNOT。
- 但文档里有一个关键推论需要修正：  
  **“只有第一个 KAK 分量非零”并不意味着“只要 1 个 CNOT”。**  
  对 generic angle，单个 $R_{PP}(\theta)$ 或两门合成后落在 $(c_1,0,0)$ 边上的门，通常仍然需要 **2 个 CNOT**。  
  只有落在 Clifford 特殊点时才会降到 1 个甚至 0 个 CNOT。

你最后几行的 specific question 的答案是：

- `successive two 2Q Pauli rotations`：总是 `c3 = 0`，所以 generically 需要 2 个 CNOT。
- `successive three or more 2Q Pauli rotations`：**不一定**要 3 个 CNOT。  
  例如 `Rzx -- Ryy -- Rzz` 仍然有 `c3 = 0`，所以只要 2 个 CNOT。  
  而 `Rxx -- Ryy -- Rzz` generically 有 `c3 != 0`，所以要 3 个 CNOT。
- 对多个 successive 2Q Pauli rotations，**最稳妥的精确判定**就是：先算最终两比特 unitary 的 Weyl coordinates $(c_1,c_2,c_3)$，再看 `c3` 是否为 0。

---

## 1. 基本对易子公式

令

$$
A = P_a \otimes P_b,\qquad B = P_c \otimes P_d,
$$

其中 $P_a, P_b, P_c, P_d \in \{X,Y,Z\}$。

则

$$
[P_a \otimes P_b,\; P_c \otimes P_d]
=
2i\,\delta_{ac}\,\varepsilon_{bdf}\,(I \otimes P_f)
+
2i\,\varepsilon_{ace}\,\delta_{bd}\,(P_e \otimes I).
$$

这个公式是正确的。

它的直接推论也对：

- 如果 $a=c$ 且 $b\neq d$，对易子是局部项 $I \otimes P_f$。
- 如果 $a\neq c$ 且 $b=d$，对易子是局部项 $P_e \otimes I$。
- 如果 $a\neq c$ 且 $b\neq d$，则对易子为 0，也就是这两项对易。

所以：

> 任意两个 weight-2 Pauli 张量积的对易子，要么是 0，要么是局部项，绝不会产生新的非局部两比特项。

这点是后面所有结论的核心。

---

## 2. 两个 successive 2Q Pauli rotations 的结论

考虑

$$
U = e^{-i\theta_1 P_a\otimes P_b}\, e^{-i\theta_2 P_c\otimes P_d}.
$$

### 情况 A：共享一个位置

即

- $a=c,\; b\neq d$，或
- $a\neq c,\; b=d$。

此时两项的对易子是局部项，因此它们生成的非局部 Lie 部分不会逃出一条“单边固定”的子空间。

例如 $a=c$ 时，

$$
\mathrm{span}\{P_a\otimes P_b,\; P_a\otimes P_d\}
=
P_a \otimes \mathrm{span}\{P_b,P_d\}.
$$

对第二个 qubit 做局部旋转，可以把这个二维方向压到单一轴上，所以整个非局部内容都局限在一条 Weyl 边上：

$$
(c_1,c_2,c_3) = (c_1,0,0).
$$

### 情况 B：两个位置都不同

即

$$
a\neq c,\qquad b\neq d.
$$

这时两项对易，所以

$$
e^{-i\theta_1 P_a\otimes P_b}\, e^{-i\theta_2 P_c\otimes P_d}
=
e^{-i(\theta_1 P_a\otimes P_b + \theta_2 P_c\otimes P_d)}.
$$

并且通过同一组局部 Clifford，可以把这两个 commuting generators 同时化到两条 Cartan 方向上，例如局部等价于

$$
e^{-i(\alpha XX + \beta YY)}.
$$

因此仍然有

$$
(c_1,c_2,c_3) = (c_1,c_2,0).
$$

### 小结

所以任意两个 successive 2Q Pauli rotations 都满足：

$$
c_3 = 0.
$$

也就是说：

> 任意两个 successive 2Q Pauli rotations，最多只需要 2 个 CNOT，绝不会需要 3 个 CNOT。

---

## 3. 需要修正的一点：`(c1,0,0)` 不等于 `1 CNOT`

这是你文档里唯一真正需要改的地方。

你原文把

> “KAK 只有第一个非零分量”

直接解释成

> “1 CNOT”

这在 generic angle 下是不对的。

例如：

- 单个 $R_{ZZ}(\theta)$ 的 Weyl 坐标就是 $(\theta/2,0,0)$；
- 但对 generic $\theta$，它的最小 CNOT 数是 2，不是 1。

只有在特殊 Clifford 点，例如 $\theta=\pi/2$（局部等价于 CNOT）时，才会降成 1 CNOT。

因此更准确的说法是：

- $(0,0,0)$：0 CNOT
- $(c_1,0,0)$：generic 情况下 2 CNOT；只有离散特殊点才是 1 或 0
- $(c_1,c_2,0)$ 且 $c_2\neq 0$：generic 情况下 2 CNOT
- $c_3\neq 0$：generic 情况下 3 CNOT

换句话说：

> 对这个问题，真正决定“要不要 3 个 CNOT”的是 **第三个 Weyl 分量是否为 0**，不是“只有第一个分量非零”。

---

## 4. 三个及以上 successive 2Q Pauli rotations

这里答案是：

> **不一定要 3 个 CNOT。**

你的例子正是一个反例：

$$
R_{ZX}(\alpha)\, R_{YY}(\beta)\, R_{ZZ}(\gamma)
$$

其中 `ZX` 和 `ZZ` 共享第一个 qubit 上的 `Z`。这两项的非局部部分先压成一条边，再和 `YY` 合起来最多给出二维 Cartan 内容，因此仍然满足

$$
c_3 = 0.
$$

所以它 generically 仍然只需要 2 个 CNOT。

相反，

$$
R_{XX}(\alpha)\, R_{YY}(\beta)\, R_{ZZ}(\gamma)
$$

generically 会得到三维 Cartan 内容：

$$
(c_1,c_2,c_3) = (\ast,\ast,\ast),\qquad c_3\neq 0,
$$

于是 generically 需要 3 个 CNOT。

---

## 5. 一个很有用的结构性判据

把每个 generator $P_{a_t}\otimes P_{b_t}$ 分别看成“左边 axis”和“右边 axis”。

定义

$$
L = \mathrm{span}\{P_{a_t}\} \subset \mathrm{span}\{X,Y,Z\},
\qquad
R = \mathrm{span}\{P_{b_t}\} \subset \mathrm{span}\{X,Y,Z\}.
$$

则有一个非常有用的充分条件：

> 如果 $\dim L \le 2$ 或 $\dim R \le 2$，那么整个 successive block 一定满足 $c_3 = 0$，因此最多只要 2 个 CNOT。

这解释了很多例子：

- `XX, XY, XZ`：$\dim L = 1$，所以至多 2 CNOT
- `ZX, YY, ZZ`：$\dim L = 2$，所以至多 2 CNOT
- `XX, XY, YZ`：$\dim L = 2$，所以至多 2 CNOT
- `XX, YY, ZZ`：左右两边都张成 3 维，generic 情况下会有 $c_3 \neq 0$

这个判据是**充分条件**，而不是最强的必要条件：

- 若两边都张成 3 维，generic 情况下通常会落到 3-CNOT 区域；
- 但特殊角度下仍可能因为精细 cancellation 恰好落回 $c_3 = 0$ 的边界。

所以：

> `dim L = 3` 且 `dim R = 3` 并不意味着“必然 3 CNOT”，而是“generic 上会是 3 CNOT；要精确判断还得算最终 Weyl coordinates”。

---

## 6. 对你的具体问题的回答

你最后的总结里，应该改成下面这版。

### 6.1 一个 2Q Pauli rotation

- 单个 arbitrary-angle 2Q Pauli rotation generically 需要 **2 个 CNOT**
- 不是 1 个
- 只有在 Clifford 特殊角度时，才会降到 1 或 0

### 6.2 两个 successive 2Q Pauli rotations

- 总是满足 $c_3 = 0$
- 因此 generically 需要 **2 个 CNOT**
- 不会需要 3 个 CNOT

### 6.3 三个或更多 successive 2Q Pauli rotations

- 不一定需要 3 个 CNOT
- 某些时候仍然只要 2 个，例如 `Rzx -- Ryy -- Rzz`
- generically 什么时候要 3 个？答案是：当最终 unitary 的第三个 Weyl 分量 $c_3$ 非零

---

## 7. Interaction-Matrix Theorem

上面讨论的是“successive product”视角。  
如果你真正关心的是

$$
U = e^{-iH},
\qquad
H = \sum_{t=1}^m a_t P_t,
\qquad
P_t \in \{X,Y,Z\}^{\otimes 2},
$$

那么最自然的解析工具不是 BCH，而是把 $H$ 写成一个 $3\times 3$ 的实系数 interaction matrix。

### 定义

记

$$
\vec{\sigma} = (X,Y,Z)^T.
$$

任意纯两比特非局部 Hamiltonian 都可以写成

$$
H = \sum_{\mu,\nu\in\{X,Y,Z\}} J_{\mu\nu}\,\sigma_\mu\otimes\sigma_\nu,
$$

其中 $J\in\mathbb{R}^{3\times 3}$。

例如

$$
H = \alpha ZX + \beta YY + \gamma ZZ
$$

对应

$$
J =
\begin{pmatrix}
0 & 0 & 0 \\
0 & \beta & 0 \\
\gamma & 0 & \alpha
\end{pmatrix},
$$

其中行和列都按 $(X,Y,Z)$ 排序。

### Theorem

对任意

$$
H = \sum_{\mu,\nu} J_{\mu\nu}\,\sigma_\mu\otimes\sigma_\nu,
$$

存在单比特门 $u,v\in SU(2)$，使得

$$
(u\otimes v)\,H\,(u^\dagger\otimes v^\dagger)
=
s_1 XX + s_2 YY + s_3 ZZ,
$$

其中 $(s_1,s_2,s_3)$ 是矩阵 $J$ 的奇异值。

因此

$$
e^{-iH}
\sim_{\mathrm{local}}
e^{-i(s_1 XX + s_2 YY + s_3 ZZ)}.
$$

特别地：

- 若 $\operatorname{rank}(J)\le 2$，则 $s_3=0$，所以 $e^{-iH}$ 必定落在 Weyl chamber 边界 $c_3=0$ 上，从而至多需要 2 个 CNOT。
- 若 $\operatorname{rank}(J)=3$，则 generic 情况下 $s_3\neq 0$，因此 generic 地属于 3-CNOT class。

### Proof sketch

单比特门对 Pauli 向量的伴随作用给出一个 $SO(3)$ 旋转。  
也就是说，对任意 $u\in SU(2)$，存在 $R_u\in SO(3)$ 满足

$$
u(\vec r\cdot \vec \sigma)u^\dagger = (R_u \vec r)\cdot \vec \sigma.
$$

于是局部变换 $(u\otimes v)$ 在 interaction matrix 上的作用正是

$$
J \mapsto R_u J R_v^T.
$$

但任意实矩阵都存在奇异值分解

$$
J = R_1^T \operatorname{diag}(s_1,s_2,s_3) R_2,
\qquad
R_1,R_2\in SO(3),
$$

故可选择相应的 $u,v$，把 Hamiltonian 化到

$$
s_1 XX + s_2 YY + s_3 ZZ.
$$

由于 $XX,YY,ZZ$ 两两对易，

$$
e^{-iH}
\sim_{\mathrm{local}}
e^{-i(s_1 XX + s_2 YY + s_3 ZZ)}.
$$

因此第三个 Cartan 分量是否为 0，完全由 $J$ 的秩是否不超过 2 决定。

---

## 8. 可直接使用的 Lemmas / Rules

### Lemma 1

若所有项在左边 Pauli 轴的张成维数不超过 2，或右边 Pauli 轴的张成维数不超过 2，即

$$
\dim \mathrm{span}\{P_t^{(L)}\}\le 2
\quad\text{or}\quad
\dim \mathrm{span}\{P_t^{(R)}\}\le 2,
$$

则 $\operatorname{rank}(J)\le 2$，因此

$$
e^{-iH}
$$

至多需要 2 个 CNOT。

#### Proof sketch

若左边只用了不超过两种 Pauli 轴，则 $J$ 的行空间维数至多为 2；  
若右边只用了不超过两种 Pauli 轴，则 $J$ 的列空间维数至多为 2。  
两种情况都推出 $\operatorname{rank}(J)\le 2$。

### Lemma 2

若 $\det J \neq 0$，则 $\operatorname{rank}(J)=3$，因此 generic 地属于 3-CNOT class。

#### Example

$$
H = \alpha XX + \beta YY + \gamma ZZ
$$

对应

$$
J = \operatorname{diag}(\alpha,\beta,\gamma).
$$

只要 $\alpha\beta\gamma \neq 0$，就有 $\det J \neq 0$，所以 generic 上需要 3 个 CNOT。

### Lemma 3

对“distinct 2Q Pauli terms，且系数视为相互独立符号变量”的情形，`generic rank(J)` 等于 support bipartite graph 的最大匹配数。

#### 定义

构造一个二分图：

- 左边顶点集：$\{X,Y,Z\}$
- 右边顶点集：$\{X,Y,Z\}$
- 若 Hamiltonian 中存在项 $\sigma_\mu\otimes\sigma_\nu$，则连一条边 $(\mu,\nu)$

则：

- 若最大匹配数 $\le 2$，则 generic rank$(J)\le 2$，所以 generic 地至多 2 CNOT
- 若存在大小为 3 的 perfect matching，则 generic rank$(J)=3$，所以 generic 地是 3-CNOT class

#### Why this works

对一个条目由独立符号变量支撑的矩阵，generic rank 等于其支撑图的最大匹配数。这在 $3\times 3$ 情形尤其容易检查：是否存在一个非零的置换积，也就是 determinant 里是否存在不会被符号独立性消掉的项。

---

## 9. 对多个 2Q Pauli rotations 的解析判定

所以，对于

$$
U = e^{-iH},
\qquad
H = \sum_t a_t P_t,
$$

一个完全解析、避免数值 KAK 的判定流程就是：

1. 构造 interaction matrix $J$
2. 先看结构性快速规则：
   - 若 $\dim L\le 2$ 或 $\dim R\le 2$，则直接判为 `at most 2 CNOT`
   - 若 support graph 有 perfect matching，则 generic 地判为 `3-CNOT class`
3. 若系数是具体数值，还可直接算 $\operatorname{rank}(J)$：
   - $\operatorname{rank}(J)\le 2$ -> exact 地有 $c_3=0$
   - $\operatorname{rank}(J)=3$ -> generic 地 $c_3\neq 0$

这就把问题从“对最终 unitary 做数值 KAK”完全转成了“对 Hamiltonian 的 $3\times 3$ 矩阵做线性代数”。

---

## 10. 最终的形式化判定方法

对于任意一个 successive 2Q Pauli rotation block，

$$
U = \prod_t e^{-i\theta_t P_{a_t}\otimes P_{b_t}},
$$

精确判定它需要多少个 CNOT 的方法是：

1. 先构造最终两比特 unitary $U$
2. 计算它的 Weyl / Cartan coordinates
   $$
   (c_1,c_2,c_3),\qquad \frac{\pi}{4}\ge c_1\ge c_2\ge |c_3|\ge 0
   $$
3. 然后看：
   - 若 $c_3 \neq 0$：generic 上需要 3 个 CNOT
   - 若 $c_3 = 0$：最多 2 个 CNOT
   - 若还落在离散 Clifford 特殊点：可能进一步降到 1 或 0

所以对你这个研究问题来说，最重要的不是“有几个 Pauli rotations”，而是：

> **最终 unitary 是否落在 Weyl chamber 的边界 $c_3=0$ 上。**

---

## 11. 一个可直接使用的实践版结论

如果你的目标只是做编译期快速筛选，那么可以用这套三层策略：

### Level 1: 立即可判

- 1 项：generic 2 CNOT
- 2 项：一定 `c3=0`，generic 2 CNOT

### Level 2: 结构上可判“至多 2 CNOT”

如果所有项在某一边只用了不超过两种独立 Pauli 轴，也就是

$$
\dim L \le 2 \quad \text{or} \quad \dim R \le 2,
$$

则直接判为 `c3=0`，所以至多 2 CNOT。

### Level 3: 精确判定

如果两边都张成 3 维，则不要猜，直接算最终 unitary 的 Weyl coordinates：

- `c3 = 0` -> 2 CNOT class
- `c3 != 0` -> 3 CNOT class

---

## 12. 结论一句话版

> 任意两个 successive 2Q Pauli rotations 都不会触发第三个 Cartan 分量，因此最多 2 CNOT；三个及以上 rotations 则不一定，有些 block 仍然满足 $c_3=0$（例如 `Rzx-Ryy-Rzz`），而 generic 的 3-CNOT 情况出现在最终 unitary 的 Weyl coordinate 满足 $c_3\neq 0$ 时。

如果你愿意，下一步我可以继续补一段：

1. 一个简洁的 **证明草稿**，把“若 $\dim L \le 2$ 或 $\dim R \le 2$，则 $c_3=0$”写成论文里能用的 lemma；
2. 或者直接给你一段 **Qiskit / NumPy 代码**，输入一串 Pauli rotations，自动输出它的 Weyl coordinates 和最小 CNOT 数。  
