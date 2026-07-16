# Holistic peel-forward vs. BSF-based Hamiltonian-simulation synthesis — novelty & design insight

> **用途 (how to use this doc).** 这是给你写 paper 的 *related work / novelty / design insight* 小节用的
> 素材库。每个对比方法都有：核心数据结构、算法策略、优化目标、是否 topology-aware、
> 报告的量级改进、以及**可引用出处**。最后两节给出**可直接搬进论文的 novelty 句式**。
> 技术内容用英文写(便于直接粘贴),关键定位用中文标注。
>
> **Verification status.** PCOAST 和 Rustiq 的技术论断经过 3-vote 对抗式验证(3-0 confirmed)。
> PHOENIX(DAC'25)/Paulihedral/Tetris/QuCLEAR/ZX-lineage 的论断是从**一手来源**(arXiv 正文/摘要/官方
> repo)抽取,但未在本轮跑满 3-vote 复核 —— 引用前建议对着原文再核一遍页码/数字。
> 所有 benchmark 数字均为**各作者自报**,基线不同,横向不可直接比较(见 §7 Caveats)。
>
> **我们的方法 = `phoenix.primitive.holistic.holistic_compile`**(forward-frame two-qubit peeling engine)。
> 代码事实见 `phoenix/primitive/holistic.py` 与 `phoenix/compiler.py`(`grouping="support"` 即 DAC'25 旧
> 流水线,被显式保留为 ablation baseline)。

---

## 1. 一句话 novelty (elevator pitch)

> 现有所有基于 BSF/tableau 的 Pauli-exponential 综合器 —— 包括我们自己的 DAC'25 PHOENIX —— 本质都是
> **启发式贪心搜索**(带 cost 函数、tabu、stall/patience、随机重启、或全局 phase-polynomial 优化),没有
> 终止界、依赖数值超参、并且把 *grouping → simplification → ordering* 拆成显式的多个阶段。
> **我们的 holistic peel-forward 引擎把 BSF 降权综合变成一个"有下降证书"的确定性单遍过程**:
> (i) 对最小权重目标行,在其 support 内**总存在**一个严格降权的 2-qubit Clifford(穷举验证的 lemma),
> 于是搜索退化为**保证下降、有界(≤ m(n−2) 步)、零数值超参**的循环;
> (ii) **没有 a-priori grouping/ordering** —— 分组与排序从"前向剥离顺序 + 精确对易偏序上的 ASAP 调度"中
> **自然涌现**;(iii) Clifford 只**前向累积**,最后用**单个 terminal Clifford**(replay / 一次性 greedy
> resynthesis / QuCLEAR 式 absorb)收帧。结果:在 logical(all-to-all)regime 对 count 和 depth **同时**
> Pareto 占优(HamLib-100 上 Depth2Q opt-rate 0.096,优于 DAC'25 PHOENIX 自身的 0.134)。

---

## 2. Common ground: 大家都用 BSF —— 所以 novelty 不在"用不用 BSF"

所有对标工作都把 Pauli operator 编码进 **binary symplectic form / stabilizer tableau**(每个 n-qubit
Pauli string 是 `[X|Z]` 里的一行/一列,Clifford 共轭 = tableau 上的行加/交换,F₂ 线性代数)。这是本领域
的公共基础设施,**不是**任何一方的创新点。真正区分各方法的是**在 BSF 上跑什么算法**:

| 维度 | 说明 |
|---|---|
| **搜索类型** | 贪心 / 全局搜索 / 有证书的确定性下降 |
| **终止性** | 有无可证明的步数上界 |
| **超参** | 是否依赖 cost 权重、tabu 长度、patience、nshuffles 等数值旋钮 |
| **grouping/ordering** | 是显式独立阶段,还是 emergent |
| **Clifford 处理** | 图节点重综合 / 交错 / 前向单帧 |
| **目标** | 2Q count / 2Q depth / 二者 |
| **topology-aware** | 是否把 routing 纳入 |

下面逐个方法拆解,然后是我们的引擎和对照表。

---

## 3. 对标方法逐个拆解 (method cards)

### 3.1 PCOAST — Pauli-based Circuit Optimization, Analysis & Synthesis (Intel) ✅ *3-0 verified*
- **数据结构**: **PCOAST graph** —— 一个 DAG,节点是 Pauli-rotation 节点 + 广义**非酉**的
  preparation/measurement 节点(全部用 Pauli string 参数化),边编码(反)对易;Clifford 用 **Pauli frame /
  Pauli tableau**(k×2 的 Hermitian Pauli 数组,即标准 BSF)紧凑存储。
- **算法**: 把输入**电路**转成 Pauli graph → 用"把 Clifford 对易穿过 Pauli rotation"暴露优化机会 →
  用**可调 greedy 综合**(改编自 Schmitz et al. arXiv:2103.08602)重综合回目标门集。节点代价按 **TQE
  (two-qubit entangling) 门**度量,`NODECOST(n(P)) = supp(P)−1`,`REDUCE_NODE` 单调把节点代价降到 0。
- **目标**: 总门数 / 2Q 门数 / depth。**self-reported**: 相对 Qiskit/t|ket⟩ 平均降 32.53%/43.33%(总门)、
  29.22%/20.58%(2Q)、42.02%/51.27%(depth)。
- **独特点**: 关注**酉/非酉界面** —— 测量代价缩减 + 经典 outcome remapping(companion arXiv:2305.09843
  报告最高 7.91× 测量压缩)。是**通用电路优化框架**(吃任意输入电路),不是只吃一组 Pauli rotations。
- **局限(相对我们)**: graph→greedy-resynthesis 是启发式、无终止界;通用性来自复杂 IR;是"给电路做优化"
  而非"给定 Pauli 指数集直接综合"。
- **出处**: Paykin, Schmitz, Ibrahim, Wu, Matsuura, *PCOAST*, IEEE QCE 2023, pp.715–726;
  arXiv:2305.10966;界面篇 arXiv:2305.09843;greedy 基础 arXiv:2103.08602。

### 3.2 Rustiq — greedy single Pauli-network synthesis ✅ *3-0 verified*
- **数据结构**: 标准 BSF tableau(`I=(0,0),X=(0,1),Z=(1,0),Y=(1,1)`,一组 m 个算子存成 2n×m 的 F₂ 表),
  Clifford 共轭 = 行加/交换,向量化。
- **算法**: **刻意放弃**传统三段式(把 rotation 分成两两对易组 → 用一个大 Clifford 联合对角化 → phase-
  polynomial/parity-network 综合),改为从空电路**贪心生长一个"Pauli network"**:每步在 leading rotation 的
  support 上枚举小 Clifford chunk(CNOT+H+√X),按"新产生多少 leading identity"打分(CLI = tableau 两行的
  leading-zero 数),选把 support 最快降到 1 的那个。
- **两个变体**: `rcount`(Alg.1,18 个候选 chunk 里选最省 CNOT 的)优化 **count**;`rdepth`(Alg.2,用
  **blossom 最大权匹配**选每层互不重叠的多个 chunk 得到 depth-1 层)优化 **depth**;还有 order-preserving
  DAG 变体。Qiskit 插件里用 `optimize_count` / `preserve_order` / `nshuffles=400` 暴露。
- **目标**: 2Q count **或** 2Q depth(两个独立算法)。**self-reported**: UCCSD 上 depth 比 TKet 最多 4×、
  比 Paulihedral 最多 5.7×("up to",单行 best-case);通用重综合在 39 个 benchmark 里 19 个打平/超过
  Amy/Nam 的**全局** phase-polynomial 优化器。
- **局限(相对我们)**: 贪心无终止界;`nshuffles` 是随机重启超参;count 与 depth **要两套算法**;每步作用于
  单个 rotation 的 support,而非全表整数收益。
- **出处**: Goubault de Brugière & Martiel, *Faster and shorter synthesis of Hamiltonian simulation
  circuits*, arXiv:2404.03280 (2024);repo github.com/smartiel/rustiq;Qiskit `PauliEvolutionSynthesisRustiq`。

### 3.3 PHOENIX (DAC'25) — 我们自己的前作,也是最强对照 ⚠️ *primary-source extracted*
- **数据结构**: **BSF**;把 Pauli-IR 综合**重述为"通过 Clifford 变换降低 BSF 的列权重(column weights)"**
  的问题。
- **算法**: **启发式 greedy BSF 简化** —— 逐步搜索"最合适的 2Q Clifford"最大化每步简化,把 total weight
  `w_tot` 降到 ≤2、再**剥离(peel)权重-1 的 Pauli string**,直到能用基础 1Q/2Q 门直接综合。
- **流水线(显式多段)**: `grouping`(按 non-trivial support 把 Pauli 项分组) → `group-wise
  simplification`(每组在 BSF 上做上面那个贪心搜索) → `IR group ordering`(**Tetris-like**,TSP 或 greedy,
  把 routing overhead / depth / gate-cancellation 一起考虑) → circuit construction → Qiskit 后端 O3。
- **目标 & 属性**: 2Q(CNOT)count **和** 2Q depth;**topology/routing-aware**;**ISA-independent**(CNOT / B-gate /
  SU(4))。**self-reported**: 相对原始 logical 电路降 CNOT 80.47% / 2Q-depth 82.7%;heavy-hex 上相对
  Paulihedral(Tetris)降 CNOT 36.17%(22.62%)、depth 43.85%(28.12%);对标 Qiskit/TKet/Paulihedral/Tetris/
  PauliOpt/QuCLEAR,suite = UCCSD(18 分子×3 拓扑)+ HamLib(100)。HamLib-100 Depth2Q opt-rate **0.134**。
- **和我们的关系(关键)**: 我们的 `holistic_compile` **就是 PHOENIX 的后继**。DAC'25 的
  "group → greedy-BSF-search(带 tabu/stall/patience)→ Tetris-order" 流水线在本仓库里被完整保留为
  `compiler.py` 的 `grouping="support"` **ablation baseline**;新引擎把 grouping/ordering/search-heuristics
  **全部换成**一个有证书的确定性 peel。**同一 HamLib-100 suite,新引擎 Depth2Q opt-rate 0.096 vs 前作
  0.134**(见本仓库 `make -f Makefile-Hamlib disp_result`;metric 定义需与论文再核一遍)。
- **出处**: Yang, Ding, Zhu, Chen, Xie, *PHOENIX*, DAC 2025, pp.1–7;arXiv:2504.03529;
  DOI 10.1109/DAC63849.2025.11133028;repo github.com/iqubit-org/phoenix。

### 3.4 Paulihedral (ASPLOS'22) ⚠️ *primary-source extracted*
- **数据结构**: **block-wise Pauli IR**(保留仿真 kernel 的高层语义/约束的中间表示;是 Pauli-string block IR,
  不是显式 BSF tableau)。
- **算法**: block-wise 编译框架 = 2 个 **technology-independent** 指令调度 pass + 2 个 **technology-dependent**
  代码优化 pass,联合协调 circuit synthesis / gate cancellation / qubit mapping。
- **目标 & 属性**: 侧重降 2Q,**架构/topology-aware**(含 qubit mapping 阶段),面向近期超导 + 容错。
- **局限(相对我们)**: 优化局限在 subcircuit / 局部 IR pattern(PHOENIX 论文语),默认 CNOT-based ISA。
- **出处**: Gui et al., *Paulihedral*, ASPLOS 2022;arXiv:2109.03371。

### 3.5 Tetris (ISCA'24) ⚠️ *primary-source extracted*
- **数据结构**: refined Pauli-string IR(专门表达 synthesis 阶段的 2Q-gate 优化机会)。
- **算法**: 在 **synthesis 阶段**挖被以往 VQA 编译器忽略的 2Q 门缩减机会;配 **fast bridging** 降低映射到
  硬件连通性的代价(topology-aware)。
- **目标**: 主打降 **2Q/CNOT count**(也报 depth/duration)。**self-reported**: 相对 SOTA 最多降 CNOT
  41.3% / depth 37.9% / duration 42.6%。
- **出处**: Jin et al., *Tetris*, ISCA 2024;arXiv:2309.01905。

### 3.6 QuCLEAR ⚠️ *primary-source extracted*
- **数据结构**: **stabilizer tableau**(存 Clifford 只需 4n²+O(n) 经典 bit)。
- **算法**: 两步 —— **Clifford Extraction**(把 Clifford 子电路移到电路末尾,同时变换/优化后续 Pauli string) +
  **Clifford Absorption**(把抽出来的 Clifford **经典地**吸收进 observable / 输出分布);用 greedy
  `find_next_pauli`(选提取后非平凡分量最少的 Pauli)。
- **目标 & 属性**: 2Q(CNOT)count 与 entangling depth;**非** topology-aware(但自称能超过 hardware-aware
  方法)。**self-reported**: 相对 T|ket⟩ 最多降 CNOT 77.7%(均 49.3%)/ depth 84.1%(均 59.1%),19 个
  benchmark 里 16 个 CNOT-count 最优。
- **和我们的关系**: 我们的 **`terminal="absorb"`** 就是 QuCLEAR 式 observable 吸收 —— 但在我们这里它只是
  terminal Clifford 的**四选一**收帧方式之一,不是整个方法。
- **出处**: QuCLEAR, arXiv:2408.13316。

### 3.7 ZX / phase-gadget / simultaneous-diagonalization lineage(TKet 等) ⚠️ *primary-source extracted*
- **数据结构/思想**: 把 Pauli 指数表示成 **phase gadget**,在 **ZX-calculus** 里推理;或把 Pauli 项分成两两
  对易组、用 Clifford **联合对角化**、再用 **phase-polynomial / parity-network** 综合。这是 Rustiq 明确"抛弃"的
  传统三段式的理论血统,也是 TKet 的 Pauli-exp 编译路线。
- **出处**: Cowtan, Dilkes, Duncan, Simmons, Sivarajah, *Phase Gadget Synthesis for Shallow Circuits*,
  QPL 2019, arXiv:1906.01734;simultaneous-diagonalization: van den Berg & Temme (Pauli-cluster
  diagonalization), arXiv:2003.13599。

---

## 4. 我们的方法: holistic peel-forward engine (`holistic_compile`)

**代码事实(可自证,`holistic.py` 顶部 docstring + `compiler.py`)。** 引擎恒等式(frame='s',9 个 option 全自逆):

```
U = E₀ · C₁ · E₁ · C₂ ··· C_T · E_T · [ C_T···C₁ | greedy-synth | absorb ]
```

- **数据结构**: BSF(`x,z` uint8 矩阵 = m 个 Pauli 行 × n qubit + 实系数);9 个 2-qubit Clifford option 的
  行为预计算成 16 项查表 `_NEWCODE16 / _DELTA16 / _SIGN16`(4-bit code → 新 code / 权重变化 / 相位)。
- **核心循环(guaranteed-descent peel)**:
  1. **Emit**: 权重 ≤2 的活动行**立即**发射成 1Q/2Q 旋转门并移出活动表 → 活动表**单调收缩**;
  2. **Target**: 锁定最小权重行(平局按 pattern popularity);
  3. **Certified descent**: 在 target 行 support 内的 2-qubit pair 上,**总存在**一个把它严格降权的 Clifford
     (穷举验证的 lemma,`_force_reduce_min_row`)—— 所以**不需要搜索/tabu/退火/visited-set/patience**;
  4. **Whole-table tie-break**: 在保证下降的候选里,用**全表整数代价**(Δ总权重、#改善行、#恶化行,查表 O(1)/pair)
     选最优,平局按枚举序(一个 rank-J tie-break 经 46/46 program 消融证明无效,已删)。
- **终止性**: 势函数 `(#活动行, target 权重)` 每步字典序严格下降 → **≤ m(n−2) 步**,无 visited set、无 patience、
  无 fission。**零数值超参**。
- **Grouping / ordering = emergent**: **没有** a-priori 分组阶段;subcircuit 的分组由 peel 顺序产生,排序由
  **精确对易偏序上的 ASAP list scheduling**(`SCHEDULE_ASAP`,利用 Trotter 自由度压 depth)产生。
- **Terminal Clifford(四选一收帧)**: `replay`(T 个自逆门反向重放,相位精确)/ `synth`(把 replay 尾巴收成
  **一个** Clifford 再 greedy 重综合,≤O(n²/log n) 2Q)/ `auto`(取二者更省 2Q 的)/ `absorb`(QuCLEAR 式
  observable 吸收)。含 π/4-格点**相位精确**恢复(`phase_exact`)。
- **属性**: logical(all-to-all)regime 优化 2Q count **和** depth;**当前不做 native topology-aware routing**
  (routing 下仍 competitive,拓扑感知列为 future work)。
- **self-measured(本仓库)**: HamLib-100 all-to-all,Num2Q opt-rate 0.447、Depth2Q opt-rate 0.096
  (`make -f Makefile-Hamlib disp_result`),对 5 个 SOTA baseline 在 count 与 depth 上同时 Pareto 占优。

---

## 5. 对照表 (comparison table)

对齐到最能区分的几个维度(✅ 3-0 verified / ⚠️ primary-source extracted):

| 方法 | Pauli 数据结构 | 核心算法 | 搜索类型 | 终止界 | 数值超参 | grouping/ordering | Clifford 处理 | 目标 | topo-aware |
|---|---|---|---|---|---|---|---|---|---|
| **Ours (holistic)** | BSF + 16 项查表 | certified peel + 全表整数代价 | **保证下降的确定性下降** | **✅ ≤ m(n−2)** | **✅ 零** | **emergent**(peel 序 + ASAP) | 前向单帧 + 单 terminal(replay/synth/absorb) | count **&** depth | ✳️ future work |
| PHOENIX DAC'25 ⚠️ | BSF(列权重) | greedy BSF 简化 + peel w=1 | 启发式贪心(+tabu/stall/patience) | ✗ | ✗(patience 等) | **显式** group→simplify→Tetris-order | 分组内 Clifford 共轭 | count & depth | ✅ |
| PCOAST ✅ | PCOAST graph + Pauli frame | graph 化 + 可调 greedy 重综合(TQE) | 启发式贪心 | ✗ | ✗(可调) | 图结构隐含 | 图节点重综合 | count & depth(+非酉界面) | 未证实 |
| Rustiq ✅ | BSF tableau | greedy 生长单 Pauli network(CLI 打分) | 启发式贪心(+nshuffles 重启) | ✗ | ✗(nshuffles) | 无显式分组;order-preserving DAG 变体 | 交错累积 | count **或** depth(**两套算法**) | 未证实(近似 all-to-all) |
| Paulihedral ⚠️ | block-wise Pauli IR | 2 调度 pass + 2 优化 pass | 启发式 | ✗ | ✗ | **显式** block 调度 | 与 mapping 协同 | 主 count | ✅ |
| Tetris ⚠️ | refined Pauli IR | synthesis 阶段 2Q 缩减 + fast bridging | 启发式 | ✗ | ✗ | **显式** IR 调度 | — | 主 count(+depth) | ✅ |
| QuCLEAR ⚠️ | stabilizer tableau | Clifford Extraction + Absorption | greedy | ✗ | ✗ | 顺序扫描 | **抽到末尾 + 经典吸收** | count & depth | ✗ |
| ZX/TKet ⚠️ | phase gadget / ZX | 对易分组 + 联合对角化 + phase-poly | 启发式/结构化 | ✗ | ✗ | **显式**对易分组 | 大 Clifford 对角化 | 主 depth(shallow) | 部分 |

> **读法**: 我们**唯一**在"终止界 ✅ / 零超参 ✅ / grouping-ordering **emergent**"三格同时成立的方法;这三格
> 正是把"启发式搜索"变成"确定性有证书综合"的技术标志。

---

## 6. 我们的 novelty & design insight —— 可直接搬进论文的句式

每条:**英文 paper-ready 句** + 中文定位 + 对应的**证据/对比锚点**。

### I1 — Certified guaranteed-descent 取代启发式搜索(核心贡献)
> *We replace the heuristic BSF-reduction search common to prior Pauli-IR compilers with a **certified
> guaranteed-descent engine**: for the minimum-weight target row there **provably always exists** a
> single 2-qubit Clifford, on a pair inside its own support, that strictly lowers its weight (an
> exhaustively verified lemma). This collapses an open-ended search into a **deterministic loop with a
> proven bound of at most m(n−2) moves and zero numeric hyperparameters** — no tabu list, no stall
> patience, no visited set, no random restarts.*

- **中文**: 把"贪心搜 + tabu/patience/nshuffles"变成"有下降证书的确定性有界过程"。这是相对 **PHOENIX-DAC
  (tabu/stall/patience)/ Rustiq(nshuffles=400 随机重启)/ PCOAST(可调 greedy)** 的第一层差异。
- **锚点**: `holistic.py` docstring "exhaustively verified lemma … at most m(n−2) moves … zero numeric
  hyperparameters";对照 Rustiq abstract 的 greedy + IBM 插件 `nshuffles`。

### I2 — 取消 a-priori grouping/ordering:结构 emergent
> *Unlike PHOENIX (DAC'25), Paulihedral, Tetris and diagonalization-based pipelines, which run **explicit
> grouping and ordering stages** (group Paulis by support / commuting sets, then schedule blocks with a
> TSP/Tetris-like pass), our engine performs **no a-priori grouping**: subcircuit grouping falls out of
> the forward peeling order, and scheduling falls out of **ASAP list scheduling over the exact commutation
> partial order**. Grouping and ordering are emergent, not separate heuristics to tune.*

- **中文**: 别人是 group→simplify→order 三段式;我们是一遍 peel,分组/排序自然涌现。相对 **PHOENIX-DAC 的
  group+Tetris-order / Paulihedral 的 block 调度** 的第二层差异。
- **锚点**: `compiler.py` 注释 "grouping fully emergent";`_asap_order` 精确对易偏序调度。

### I3 — Forward-only 单 Clifford 帧 + 灵活 terminal
> *Cliffords accumulate **forward only** and the whole frame is closed by **exactly one terminal Clifford**,
> realized in whichever way is cheapest: replayed (phase-exact), resynthesized once at full width
> (greedy, ≤ O(n²/log n) two-qubit gates), or **absorbed into the observables** (QuCLEAR-style) for
> expectation-value workloads. This unifies, as a single tunable knob, strategies that other tools treat
> as distinct methods.*

- **中文**: Clifford 只前向累积、单帧收尾;replay/synth/absorb 四选一 —— 把 QuCLEAR 的"吸收"降格成我们
  terminal 的一个选项。相对 **PCOAST(图节点重综合)/ QuCLEAR(整方法就是抽+吸)** 的差异。
- **锚点**: `holistic.py` 恒等式 + `_synth_terminal` + `terminal="absorb"`。

### I4 — 全表整数代价 + 预计算查表(廉价且确定)
> *Among guaranteed-descent candidates, the move is chosen by an **exact-integer whole-table objective**
> (total weight change, #rows improved, #rows hurt), evaluated in O(1) per pair via precomputed 16-entry
> tables over the 9 Clifford options. The tie-break is deterministic enumeration order — an additional
> rank-based tie-break was **ablated to be provably inert (46/46 programs gate-identical) and removed**.*

- **中文**: 每步看的是**全表**收益(不是单个 rotation 的 support),而且是精确整数、查表 O(1),外加"消融证明
  无效即删"的极简主义。相对 **Rustiq 的单 rotation CLI 打分** 的差异。
- **锚点**: `_NEWCODE16/_DELTA16/_SIGN16`;docstring §3.2.1 的 rank-J 消融。

### I5 — 单引擎同时管 count 与 depth
> *A single engine optimizes **both** two-qubit count and depth: peeling minimizes gate count while the
> ASAP commutation scheduler recovers brickwork depth — whereas e.g. Rustiq needs **two separate
> algorithms** (rcount vs. rdepth with blossom matching) for the two objectives.*

- **中文**: 我们一套引擎兼顾 count/depth;Rustiq 要 rcount / rdepth 两套。
- **锚点**: 本仓库 HamLib-100 同时 Pareto 占优 count(0.447)与 depth(0.096)。

### I6 — 相对我们自己 DAC'25 PHOENIX 的定位(最强的自证)
> *This work is the successor to our DAC'25 PHOENIX. There, IR synthesis was framed as **greedy** column-
> weight reduction over support-grouped BSF blocks, followed by a Tetris-like ordering pass. We keep that
> exact pipeline as an ablation baseline (`grouping="support"`) and show that replacing it with the
> certified grouping-free peel engine **improves the published result on the same HamLib-100 suite from a
> Depth2Q optimization rate of 0.134 to 0.096**, while removing every numeric hyperparameter.*

- **中文**: 这是 PHOENIX 的**期刊/进阶**式差异化 —— 用新引擎替掉自己旧的 group+greedy+Tetris,并在**同一
  suite** 上把 depth opt-rate 从 0.134 提到 0.096。**这条是最有说服力的 novelty 证据**(自我超越 + 保留旧法作
  ablation)。⚠️ 数字口径:0.134 来自 DAC'25 论文(journal 抽取),0.096 来自本仓库 `disp_result`;发表前请
  确认两者 metric 定义一致(all-to-all / geomean-of-ratios)。

---

## 7. Caveats(写论文时必须诚实交代)

1. **所有 benchmark 数字都是各作者自报**,且基线不同(PCOAST vs Qiskit/tket 通用电路;Rustiq vs
   TKet/Paulihedral 的 UCCSD;QuCLEAR vs TKet;PHOENIX vs 6 baseline),**横向不可直接比较**。跨方法的定量
   排名需要在**统一 suite + 统一基线**下自己重跑。
2. **"up to N×" 是单行 best-case**(Rustiq 的 4×/5.7×),不是平均。
3. **Topology-awareness 是我们目前的短板 / future work**,而 PHOENIX-DAC / Paulihedral / Tetris / PauliOpt
   **有** routing-aware ordering。因此 novelty 要**限定在 logical(all-to-all)regime**("Pareto-dominate in
   the logical regime; competitive under routing"),不要外推到硬件映射后。
4. **验证分级**: PCOAST / Rustiq 的论断经 3-vote 对抗验证;PHOENIX / Paulihedral / Tetris / QuCLEAR /
   ZX-lineage 为一手来源抽取但未跑满复核 —— 引用前对原文核页码/数字。
5. **0.134 vs 0.096 的口径**需在发表前对齐(见 I6)。

---

## 8. References

- **PCOAST**: J. Paykin, A. T. Schmitz, M. Ibrahim, X.-C. Wu, A. Y. Matsuura, *PCOAST: A Pauli-based Quantum
  Circuit Optimization Framework*, IEEE QCE 2023, vol.1, pp.715–726. arXiv:2305.10966. 界面篇:
  A. T. Schmitz et al., *Optimization at the Interface of Unitary and Non-unitary Quantum Operations in
  PCOAST*, arXiv:2305.09843. Greedy 基础: Schmitz et al., arXiv:2103.08602.
- **Rustiq**: T. Goubault de Brugière, S. Martiel, *Faster and shorter synthesis of Hamiltonian simulation
  circuits*, arXiv:2404.03280 (2024). Repo: github.com/smartiel/rustiq. Qiskit `PauliEvolutionSynthesisRustiq`.
  相关: Goubault de Brugière, Martiel, Vuillot, *A graph-state based synthesis framework for Clifford
  isometries*, arXiv:2212.06928.
- **PHOENIX (DAC'25)**: Z. Yang, D. Ding, C. Zhu, J. Chen, Y. Xie, *PHOENIX: Pauli-Based High-Level
  Optimization Engine for Instruction Execution on NISQ Devices*, DAC 2025, pp.1–7. arXiv:2504.03529.
  DOI 10.1109/DAC63849.2025.11133028. Repo: github.com/iqubit-org/phoenix.
- **Paulihedral**: G. Li (Gui) et al., *Paulihedral: A Generalized Block-Wise Compiler Optimization
  Framework for Quantum Simulation Kernels*, ASPLOS 2022. arXiv:2109.03371.
- **Tetris**: Y. Jin et al., *Tetris: A Compilation Framework for VQA Applications*, ISCA 2024.
  arXiv:2309.01905.
- **QuCLEAR**: *QuCLEAR: Clifford Extraction and Absorption for Quantum Circuit Optimization*,
  arXiv:2408.13316.
- **ZX / phase-gadget**: A. Cowtan, S. Dilkes, R. Duncan, W. Simmons, S. Sivarajah, *Phase Gadget Synthesis
  for Shallow Circuits*, QPL 2019. arXiv:1906.01734. Simultaneous diagonalization: E. van den Berg,
  K. Temme, arXiv:2003.13599.
- **Our method**: `phoenix/primitive/holistic.py` (`holistic_compile`, `peel_forward`, `peel_circuit`),
  `phoenix/compiler.py`, `docs/peel_forward_design.md`.
