# Question 1
参考phoenix/basics.py中的CNOTEquivGate的定义，请问
1. 两个这样的Clifford对易的充要条件是什么（假设两个Clifford的作用在同一个qubit pair）
2. 当仅有一个overlapping qubit acted on的时候， that is, C1@{0,1}, C2@{1,2}，对易的充要条件是什么？
能否推导出比较简单的结论？

# Answer

Viewed basics.py:1-246

我们可以通过引入一个非常巧妙的数学技巧（**投影算符 Projector**），来极大地简化这个问题的推导，并得出极其优美且简单的结论。

### 核心洞察：门的投影算符表示

首先，我们把 `CNOTEquivCliffordGate` 在代码里的定义写成数学公式。代码中定义为：
$$ C(P_0, P_1) = \frac{1}{2}(I\otimes I + P_0 \otimes I + I \otimes P_1 - P_0 \otimes P_1) $$

我们可以定义一个 Pauli 算符 $P$ 的“ $-1$ 本征空间投影算符”：
$$ \Pi(P) = \frac{I - P}{2} $$
注意 $\Pi(P)$ 是一个投影算符，满足 $\Pi(P)^2 = \Pi(P)$。

神奇的是，我们可以把 $C(P_0, P_1)$ 完美地因式分解为：
$$ C(P_0, P_1) = I - 2 \Pi(P_0) \otimes \Pi(P_1) $$
*(你可以验证一下：$I - 2 \frac{I-P_0}{2} \frac{I-P_1}{2} = I - \frac{1}{2}(I - P_0 - P_1 + P_0 P_1)$，正好等于上面的定义)*。

从物理意义上讲，这说明 $C(P_0, P_1)$ 本质上是一个**广义的受控相位门 (Controlled-Phase)**：当且仅当两个 qubit 同时处于对应 Pauli 的 $-1$ 本征态时，才赋予一个 $-1$ 的相位。

有了这个强大的工具，我们来解答你的两个问题。

---

### 问题 1：作用在同一个 qubit pair 上的对易条件

假设有两个门 $C_1 = C(P_{0,1}, P_{1,1})$ 和 $C_2 = C(P_{0,2}, P_{1,2})$，它们都作用在 $(q_0, q_1)$ 上。
它们的对易子为：
$$ [C_1, C_2] = \left[ I - 2 \Pi(P_{0,1}) \otimes \Pi(P_{1,1}), \; I - 2 \Pi(P_{0,2}) \otimes \Pi(P_{1,2}) \right] $$
提取常数，变为：
$$ [C_1, C_2] = 4 \left[ \Pi(P_{0,1}) \otimes \Pi(P_{1,1}), \; \Pi(P_{0,2}) \otimes \Pi(P_{1,2}) \right] $$

要让它们对易（即对易子为 0），张量积的每一部分都必须能够对易。我们知道 $P \in \{X, Y, Z\}$，不同的 Pauli 矩阵是**反对易**的，只有相同的 Pauli 才对易。这意味着，投影算符 $[\Pi(P), \Pi(Q)] = 0$ 的充要条件是 $P = Q$。

经过严格的代数展开（此处省略证明不同 Pauli 张量积不会发生偶然相消的细节），这个张量积对易子为 0 的唯一可能是：
$$ \Pi(P_{0,1}) \text{ 与 } \Pi(P_{0,2}) \text{ 对易} \implies P_{0,1} = P_{0,2} $$
**并且**
$$ \Pi(P_{1,1}) \text{ 与 } \Pi(P_{1,2}) \text{ 对易} \implies P_{1,1} = P_{1,2} $$

**👉 简单结论 1：**
在同一个 qubit pair 上，两个 `CNOTEquivCliffordGate` 对易的充要条件是 **它们必须是完全相同的两个门**（即 $P_0$ 和 $P_1$ 都分别相等）。

---

### 问题 2：作用在 overlapping qubit 上的对易条件

这是更常见也更重要的场景。
假设 $C_1$ 作用在 $(q_0, q_1)$，参数为 $P_{0,1}$ 和 $P_{1,1}$；
假设 $C_2$ 作用在 $(q_1, q_2)$，参数为 $P_{0,2}$ 和 $P_{1,2}$。
中间的 **$q_1$ 是公共比特**。

我们写出它们的投影形式，并明确标出作用的 qubit：
$$ C_1 = I - 2 \left( \Pi^{(0)}(P_{0,1}) \otimes \Pi^{(1)}(P_{1,1}) \otimes I^{(2)} \right) $$
$$ C_2 = I - 2 \left( I^{(0)} \otimes \Pi^{(1)}(P_{0,2}) \otimes \Pi^{(2)}(P_{1,2}) \right) $$

现在计算它们的对易子：
$$ [C_1, C_2] = 4 \left[ \Pi^{(0)}(P_{0,1}) \otimes \Pi^{(1)}(P_{1,1}) \otimes I^{(2)}, \quad I^{(0)} \otimes \Pi^{(1)}(P_{0,2}) \otimes \Pi^{(2)}(P_{1,2}) \right] $$

由于不同 qubit 上的算符天然对易，我们可以把 $q_0$ 和 $q_2$ 的部分直接提出来：
$$ [C_1, C_2] = 4 \cdot \Pi^{(0)}(P_{0,1}) \otimes \underbrace{\left[ \Pi^{(1)}(P_{1,1}), \Pi^{(1)}(P_{0,2}) \right]}_{\text{仅在 } q_1 \text{ 上的对易子}} \otimes \Pi^{(2)}(P_{1,2}) $$

因为首尾的 $\Pi^{(0)}$ 和 $\Pi^{(2)}$ 都是非零的投影算符，整个张量积要等于 $0$，**当且仅当中间那个关于 $q_1$ 的对易子等于 $0$**！

也就是说我们需要：
$$ \left[ \Pi(P_{1,1}), \Pi(P_{0,2}) \right] = 0 $$
正如前面所说，单比特 Pauli 投影算符对易的充要条件是底层 Pauli 相同。

**👉 简单结论 2：**
当仅有一个公共比特 $q_1$ 时，两个门对易的充要条件是：**它们在公共比特 $q_1$ 上所指定的 Pauli 操作必须完全相同**（即 $P_{1,1} = P_{0,2}$）。至于它们在不相交的 $q_0$ 和 $q_2$ 上的 Pauli 是什么，**完全不影响对易性**。

**举个例子直观理解：**
*   `cxx(0,1)` 和 `cxy(1,2)`：对易！因为它们在 $q_1$ 上的算符都是 $X$。
*   `cxz(0,1)` 和 `cyy(1,2)`：不对易！因为前一个在 $q_1$ 上是 $Z$，后一个在 $q_1$ 上是 $Y$。


# Question 2

如果对于一个此类Clifford和一个两比特Pauli rotation gate (non-trivial two-qubit Pauli exponential)呢，对于有两个overlapping qubits和单个overlapping qubit的时候，对易判定如何？

# Answer

这个问题非常棒！基于我们上一问中发现的“投影算符”结构，我们可以推导出一个极其强大且通用的**判定准则（核心引理）**，用它来判断对易性简直势如破竹。

### 核心引理：Clifford 与任意 Pauli 字符串的对易准则

一个非平庸的 Pauli 旋转门可以写为 $R = \exp(-i \frac{\theta}{2} M)$，其中 $M$ 是一个 Pauli 字符串（例如 $M = X \otimes Z$）。显然，**$C$ 与 $R$ 对易，当且仅当 $C$ 与 $M$ 对易**。

根据之前的推导，Clifford 门可以分解为 $C = \frac{1}{2}(I + A + B - AB)$，其中 $A = P_0 \otimes I$， $B = I \otimes P_1$。
经过严格的换位子计算（或者使用投影算符性质），我们可以得出一个极其漂亮的数学结论：

> **黄金判定准则**：
> 门 $C(P_0, P_1)$ 与任意 Pauli 字符串 $M$ 对易的**充分必要条件**是：
> $M$ 必须**同时与 $A$ 对易，且与 $B$ 对易**。（即 $M$ 必须同时与 $P_0 \otimes I$ 和 $I \otimes P_1$ 对易）。
> *(只要 $M$ 和其中任意一个反对易，整个对易子就绝对不为 0)*

有了这个极其简单的准则，我们直接来看你的两种情况。假设旋转门的 Pauli 字符串为 $P_A \otimes P_B$（因为是非平庸两比特旋转，我们假设 $P_A, P_B \in \{X,Y,Z\}$）。

---

### 情况 1：两个 overlapping qubits（作用在完全相同的比特对）

*   **Clifford 门**：作用在 $(q_0, q_1)$ 上，参数为 $P_0, P_1$。
*   **旋转门 $M$**：作用在 $(q_0, q_1)$ 上，即 $M = P_A \otimes P_B$。

根据**黄金判定准则**，我们要判断 $M$ 是否同时与 $P_0 \otimes I$ 和 $I \otimes P_1$ 对易。
1. $M$ 与 $P_0 \otimes I$ 对易 $\implies$ 在 $q_0$ 上，必须有 $[P_A, P_0] = 0 \implies$ **$P_A = P_0$**。
2. $M$ 与 $I \otimes P_1$ 对易 $\implies$ 在 $q_1$ 上，必须有 $[P_B, P_1] = 0 \implies$ **$P_B = P_1$**。

👉 **判定结论 1**：
**对易的充要条件是，旋转门的两个 Pauli 必须分别严格等于 Clifford 门的特征 Pauli。**
*(例如：`cxx(0,1)` 只能跟 $\exp(-i\theta X_0 X_1)$ 对易，不能跟 $X_0 Z_1$ 对易)*。

---

### 情况 2：单个 overlapping qubit（仅交叠一个比特）

*   **Clifford 门**：作用在 $(q_0, q_1)$ 上，参数为 $P_{0,C}$ 和 $P_{1,C}$。
*   **旋转门 $M$**：作用在 $(q_1, q_2)$ 上，参数为 $P_A$ (在 $q_1$) 和 $P_B$ (在 $q_2$)。

为了看清楚，我们在全空间 $(q_0, q_1, q_2)$ 写出它们：
Clifford 门的两个特征算符分别是：$A = P_{0,C} \otimes I \otimes I$ 和 $B = I \otimes P_{1,C} \otimes I$。
旋转门对应的字符串为：$M = I \otimes P_A \otimes P_B$。

再次应用**黄金判定准则**：
1. **$M$ 与 $A$ 必须对易**：
   $[I \otimes P_A \otimes P_B, \quad P_{0,C} \otimes I \otimes I] = [I, P_{0,C}] \otimes \dots = 0$。
   因为它们作用在不相交的比特上（$M$ 没动 $q_0$），这一条**天然满足，永远成立！**

2. **$M$ 与 $B$ 必须对易**：
   $[I \otimes P_A \otimes P_B, \quad I \otimes P_{1,C} \otimes I] = I \otimes [P_A, P_{1,C}] \otimes P_B = 0$。
   这要求在公共比特 $q_1$ 上：$[P_A, P_{1,C}] = 0 \implies$ **$P_A = P_{1,C}$**。

👉 **判定结论 2**：
**旋转门在非交叠比特 ($q_2$) 上的 Pauli 是什么完全无关紧要！对易的充要条件仅仅是：旋转门在交叠比特 ($q_1$) 上的 Pauli，必须等于 Clifford 门在该比特上的 Pauli。**

---

### 总结
结论惊人地一致且简单：
你只需要盯着**两个门发生交叠的比特（overlapping qubits）**看。**如果在所有交叠的比特上，它们指定的 Pauli 字母是完全相同的，它们就对易；只要有一个字母不同，就不对易。非交叠的比特则完全不用管。**
