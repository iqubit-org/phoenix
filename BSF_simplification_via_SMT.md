下面给出一个用 SMT / OMT（Optimization Modulo Theories）来做 4-10 qubit 规模 BSF simplification 的论文式问题表述。内容按可直接落到 Z3 / cvc5 / OptiMathSAT 一类求解器的形式组织。为了清晰起见，先抽象 PHOENIX Alg.1 的 BSF 任务，再给 SMT 变量、约束和目标函数。

# 1. 任务抽象

## BSF Simplification as Discrete Symplectic Rewrite

输入是一个 Pauli exponentials 集合的 BSF 表示：

- 有 $m$ 个 Pauli string 或 Pauli blocks。
- 每个 Pauli 由 BSF 向量
  $$
  \mathbf{p}_k = (\mathbf{z}_k \mid \mathbf{x}_k) \in \mathbb{F}_2^{2n}
  $$
  表示。
- 将它们堆成一个 $2n \times m$ 的二进制矩阵：
  $$
  P \in \mathbb{F}_2^{2n \times m}, \qquad
  P = \big[\mathbf{p}_1^\top, \dots, \mathbf{p}_m^\top\big].
  $$

一个 2Q Clifford 操作等价于对某对量子比特 $(i, j)$ 施加一个局部辛变换：

$$
P \mapsto S_{i,j,g} P,
$$

其中 $S_{i,j,g} \in Sp(2n, \mathbb{F}_2)$ 只在 $(i, j)$ 对应的 $4 \times 4$ 子空间上非平凡，其余位置是单位阵。这里 $g$ 表示两比特辛群 $Sp(4, 2)$ 中的某个元素，也即某个 2Q Clifford。

目标是找到一个长度不超过 $L$ 的 2Q Clifford 序列

$$
(g_1, \dots, g_L), \qquad g_t \in \mathcal{G}_{2Q},
$$

使得最终矩阵

$$
P^{(L)} = S_{g_L} \cdots S_{g_1} P
$$

被简化到“可直接综合”的形态。PHOENIX 中常用的简化指标就是列权重（column weight）或其某种组合。

# 2. 决策变量

## 2.1 操作选择变量

对每一步 $t = 1, \dots, L$，引入离散选择变量：

- 选择哪一对 qubit：
  $$
  a_t, b_t \in \{0, \dots, n - 1\}, \qquad a_t < b_t
  $$
- 选择哪一个 2Q Clifford 类型：
  $$
  r_t \in \{1, \dots, |\mathcal{G}_{2Q}|\}
  $$

这里 $\mathcal{G}_{2Q}$ 是你允许的 2Q Clifford 候选集合。它可以是 PHOENIX 当前使用的 option 集，也可以扩展为全部两比特 Cliffords。

## 2.2 中间 BSF 状态变量

引入每一步后的 BSF 矩阵变量：

$$
P^{(t)} \in \mathbb{F}_2^{2n \times m}, \qquad t = 0, \dots, L,
$$

其中 $P^{(0)} = P$ 是常量输入。

在 SMT 实现中，所有矩阵元素都编码成布尔变量：

$$
p^{(t)}_{u,k} \in \{0,1\},
\qquad
u = 1, \dots, 2n,\;
k = 1, \dots, m.
$$

# 3. 约束

## 3.1 初值约束

$$
P^{(0)} = P.
$$

即

$$
p^{(0)}_{u,k} = P_{u,k}
\qquad \forall u, k.
$$

## 3.2 辛变换更新约束

对每一步 $t$，要求

$$
P^{(t)} = S_{a_t,b_t,r_t} P^{(t-1)}.
$$

因为是在 $\mathbb{F}_2$ 上的线性更新，所以每个输出 bit 都是输入 bit 的 XOR 线性组合：

$$
p^{(t)}_{u,k}
=
\bigoplus_{v=1}^{2n}
\left(S_{a_t,b_t,r_t}\right)_{u,v}
\wedge
p^{(t-1)}_{v,k}.
$$

实现技巧：

- 由于 $S_{a_t,b_t,r_t}$ 只有 $(a_t, b_t)$ 对应的 $4 \times 4$ 子块不是单位阵，所以只需要写局部 $4 \times 4$ 更新。
- 对应两比特的 4 行，即 $(z_i, z_j, x_i, x_j)$，按一个 $4 \times 4$ 二进制矩阵做线性变换。
- 其余行直接等于上一时刻。

形式化地，设

$$
I = \{z_i, z_j, x_i, x_j\}
$$

是这 4 行索引，则

$$
P^{(t)}_{I,\cdot} = M_{r_t} P^{(t-1)}_{I,\cdot},
\qquad
M_{r_t} \in Sp(4,2),
$$

且

$$
P^{(t)}_{\overline{I},\cdot} = P^{(t-1)}_{\overline{I},\cdot}.
$$

对 SMT 来说，这意味着每一步只增加 $4m$ 量级的 XOR 约束，规模相对可控。

## 3.3 选择变量合法性

$$
0 \le a_t < b_t < n,
\qquad
r_t \in [1, |\mathcal{G}_{2Q}|].
$$

如果你要避免重复或无效操作，还可以加：

- 禁止同一步选相同 qubit：
  $$
  a_t \ne b_t
  $$
- 如果要满足硬件拓扑，则要求
  $$
  (a_t, b_t) \in E_{\text{coupling}}.
  $$

## 3.4 可选的步长与动作使用约束

例如：

- 限制某类 2Q Clifford 的使用次数。
- 给某类动作赋更高 penalty。

这些都可以作为 soft constraint，或直接进入目标函数。

# 4. 目标函数

这里可以选与 PHOENIX 最接近的指标，也可以换成 FT / QEC 更关心的指标。下面给出三个常用目标，它们都可以写成 OMT / Max-SMT / pseudo-Boolean optimization。

## 4.1 目标 A：最小化最终总列权重

定义第 $k$ 列在终态的权重为

$$
w_k^{(L)} = \sum_{u=1}^{2n} p^{(L)}_{u,k}.
$$

目标函数为

$$
\min \sum_{k=1}^m w_k^{(L)}.
$$

这是 PHOENIX 中 greedy 代价函数的直接全局化版本。

## 4.2 目标 B：最小化“坏列”数量

设“可直接综合”的阈值为 $\tau$。例如 $\tau = 2$ 表示每个 Pauli 最终只作用在不超过 2 个 qubit 上。

引入布尔变量 $bad_k$：

$$
bad_k = 1 \;\Longleftrightarrow\; w_k^{(L)} > \tau.
$$

目标函数为

$$
\min \sum_{k=1}^m bad_k.
$$

直觉上，这是优先把尽量多的列清理到“可综合”状态。

## 4.3 目标 C：词典序分层优化

先最小化坏列数，再最小化总权重：

$$
\min_{\text{lex}}
\left(
\sum_k bad_k,\;
\sum_k w_k^{(L)}
\right).
$$

这种 OMT 目标更稳，因为它先保证宏观结构最优，再做细调。

# 5. 可行性与最优性的语义

- 可行解对应一条长度不超过 $L$ 的 2Q Clifford 序列，使 BSF 在每一步都按合法辛变换更新。
- 全局最优表示：在固定 $L$ 和候选集合 $\mathcal{G}_{2Q}$ 下，对所选目标函数取得最小值。

注意：

- 如果 $L$ 太小，可能不存在能达到阈值要求的可行解。
- 实践中可以逐步增大 $L$。
- 也可以允许“某一步不使用动作”，等价于选择一个 identity Clifford。

# 6. 规模估计

为什么在 4-10 qubit 范围内 SMT 还有希望跑得动：

变量量级：

- 状态变量约为
  $$
  (L+1)\cdot 2n \cdot m
  $$
  个布尔变量。
- 动作变量约为
  $$
  L \cdot (\log n + \log |\mathcal{G}_{2Q}|)
  $$
  个离散编码变量。

约束量级：

- 每步局部辛更新约为 $O(4m)$ 个 XOR 约束，总体约为 $O(Lm)$。
- 再加上少量选择变量合法性约束。

因此，对 $n = 4, \dots, 10$ 且 $m$ 在几十到几百的情况下，如果服务器资源足够，取 $L = 5, \dots, 20$ 进行窗口型全局最优搜索是有希望的；但如果想让 $L$ 上百并做端到端全局优化，通常仍会指数爆炸。

# 7. 可直接放进论文方法段的总结

> 我们将 BSF simplification 表述为一个有界两比特辛变换序列的离散优化问题。给定输入 BSF 矩阵 $P$，在候选 2Q Clifford 集 $\mathcal{G}_{2Q}$ 上搜索长度不超过 $L$ 的操作序列，使最终 BSF 状态 $P^{(L)}$ 的列权重或坏列数最小。我们在 SMT / OMT 中引入每一步的动作选择变量与中间 BSF 状态变量，并用 $\mathbb{F}_2$ 上的局部 $4 \times 4$ 辛线性更新来约束状态演化，从而得到一个可由现代 Max-SMT / OMT 求解器全局求优的 formulation。

# 8. 后续可扩展内容

如果需要，接下来还可以继续补：

1. 一个具体的 Z3 / PySMT 伪代码骨架，例如 XOR 约束怎么写、词典序目标怎么设。
2. 按 PHOENIX 当前 2Q Clifford option 列表给出对应的 $Sp(4,2)$ 的 $4 \times 4$ 矩阵表，方便直接编码。
