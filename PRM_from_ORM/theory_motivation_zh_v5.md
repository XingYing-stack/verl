# Inducing Process Supervision from Outcome-Only Reinforcement Learning

# 理论动机与实验验证

## 核心直觉（五步链条）

1. 在多步推理任务里，final correctness 往往依赖 step correctness。
2. 因而，想稳定地判断正确 outcome，模型通常需要隐式地建模"哪些步骤可靠、哪些步骤会导致最终失败"。
3. GRPO 在多个 trajectory 之间做相对比较，会偏好这种低方差、可复现的正确策略，而非高方差 shortcut。
4. 因此，用 ORM + GRPO 训练时，模型内部 supporting outcome prediction 的 step-level discrimination ability 会一起增强。
5. 当我们再要求模型显式输出 step labels 时，这种隐式能力就更容易被"外化"为 PRM 行为。

下面将这五步逐一对应到数学表述，并以实验结果加以验证。

**关于 rigor 的说明。** 本文的理论框架包含三个环节，数学严格性递减。环节一是精确的信息论不等式（定理级别）；环节二包含一个精确的 MI 下界（Eq.4'）和一个依赖结构性假设的定性论证（Argument 1）；环节三提供优化动力学层面的机制假说，核心假设通过实验验证而非数学证明。整体而言，本文提出一个 **带可验证预测的机制假说（mechanistic hypothesis with falsifiable predictions）**，不是一个自足的理论证明。

**与相关理论工作的关系。** Jia, Rakhlin & Xie (ICML 2025) 从 sample complexity 的角度证明了 outcome 和 process supervision 的统计等价性——在标准 data coverage 假设下，从 outcome data 学习在统计上不比 process data 更难。特别地，其 Theorem 2 证明了任意策略的 advantage function 可以作为最优的 process reward model，从 RL 理论角度独立支持了"outcome 信息包含 process 信息"的论点。本文从互补的角度出发，研究这种理论可能性在 GRPO 的训练动态中如何具体实现、在什么条件下有效、以及何时失效。AutoPSV (Lu et al., NeurIPS 2024) 从工程角度验证了"从 ORM 可以提取 step-level labels"，本文为该经验观察提供了信息论层面的解释，并进一步指出了适用边界（$\delta$）和更优的提取机制（GRPO + structured output vs. 简单的 confidence 差分）。

### 符号约定与概率空间

以下所有概率、熵、互信息均在同一联合分布下定义。令 $\theta$ 为训练后的固定策略参数，联合分布为：

$$p(X, D, Z) \;=\; p_{\text{data}}(X, D)\;\cdot\;p_\theta(Z \mid X) \tag{0}$$

其中 $X = (x, s)$ 为完整输入（问题 $x$ + 候选解题步骤 $s$），$D \in \{0,1\}$ 为最终正确性（由 ground truth 决定），$Z$ 为模型在策略 $\theta$ 下生成的 **emitted token sequence**（thinking chain 的文本输出，而非 hidden states）。

**注意：** 由于 $Z$ 的分布依赖于 $\theta$，所有信息论量（如 $I(Z; D \mid X)$）都是 policy-dependent 的。当我们讨论"训练使 $I(Z; D \mid X)$ 增大"时，指的是 $\theta$ 的更新改变了 $p_\theta(Z \mid X)$，从而改变了联合分布下的互信息。环节一刻画的是训练后某一时刻的快照性质，而非训练动态本身。

此外，令 $E$ 为候选解题步骤 $s$ 中第一个错误步骤的位置（$E = 0$ 表示全对）。$E$ 的定义依赖于固定的步骤切分方案和正确性标注协议；以下所有涉及 $E$ 的量均在该协议下定义。

---

## 环节一：ORM准 → Z里必须有outcome信息

**对应直觉第1-2步。** 模型判得准，说明thinking chain里编码了有用信息。

**Rigor level: 定理（精确信息论不等式，无近似）。**

**本环节的性质与局限。** Eq.1 是一个关于 **表征存在性** 的静态刻画：它说明对于训练后的固定策略 $\theta$，任何达到高ORM准确率的模型，其thinking chain $Z$ 必然编码了关于 $D$ 的信息。它 **不** 直接解释为什么GRPO能引导模型找到这样的 $Z$（这一动态问题由环节三处理），也 **不** 说明这些信息是否 process-structured 或 step-localizable（这由环节二处理）。如果模型通过死记硬背达到高 $\text{Acc}_{\text{ORM}}$，Eq.1 同样成立，但 $Z$ 中的信息可能与 process supervision 毫无关系。

### 严格推导

模型的预测 $\hat{D}$ 是整个上下文的函数：$\hat{D} = f(X, Z)$。

**Step 1（Fano不等式）**：既然 $\hat{D}$ 是 $(X, Z)$ 的确定性函数，且 $P(\hat{D} \neq D) = 1 - \text{Acc}_{\text{ORM}}$，由Fano不等式：

$$H(D \mid X, Z) \;\leq\; H_b(1 - \text{Acc}_{\text{ORM}}) \tag{1a}$$

**Step 2（互信息分解）**：由互信息定义展开：

$$I(Z;\, D \mid X) = H(D \mid X) - H(D \mid X, Z) \tag{1b}$$

**Step 3（代入）**：将 (1a) 代入 (1b)：

$$\boxed{I(Z;\, D \mid X) \;\geq\; H(D \mid X) - H_b(1 - \text{Acc}_{\text{ORM}})} \tag{1}$$

### 适用条件

Eq.1 的 bound strength 完全取决于 $H(D \mid X)$——即 **在不经过思考链分析的情况下，仅凭输入 $X$ 的原文，答案对错有多少不确定性**。

- 若 $H(D \mid X) \approx 0$（一眼能看出对错），则 bound vacuous。
- 若 $H(D \mid X)$ 接近 $\log 2$（需要深度推理才能判断对错），则 bound strong。

| 任务难度 | $H(D \mid X)$ | Eq.1 bound | 例子 |
|:---|:---|:---|:---|
| 简单 | $\approx 0$ | vacuous | 明显计算错误 |
| 中等 | 中等 | moderate | GSM8K、MATH |
| 困难 | $\approx \log 2$ | strong | OlympiadBench、OmniMath |

**$H(D \mid X)$ 的可操作估计。** 定义不使用 thinking chain 的 baseline accuracy：

$$\text{Acc}_0 := \max_f\; P\bigl(f(X) = D\bigr) \tag{1c}$$

其中 $f$ 遍历所有可测函数。由 Fano 不等式的同一逻辑，$H(D \mid X) \leq H_b(1 - \text{Acc}_0)$。当 $\text{Acc}_0$ 远低于 $\text{Acc}_{\text{ORM}}$ 时，$H(D \mid X)$ non-trivial，$Z$ 确实贡献了实质信息。这对应预测 **P4**。实践中 $\text{Acc}_0$ 可通过让同一模型不经思考直接判断对错来近似估计。

**一句话：** Eq.1 是严格的信息论不等式，刻画高准确率模型的表征必然性质（而非优化动态）。它在 $\text{Acc}_0 \ll \text{Acc}_{\text{ORM}}$ 的场景下非平凡——这恰好是 PRM 最有价值的场景。

---

## 环节二：outcome信息与process信息的关联（程度由 $\delta$ 调节）

**对应直觉第1-2步的深化。** 模型要稳定地判对 outcome，通常需要隐式建模 step correctness——但这个"通常"有多强，取决于任务本身 outcome 和 process 的耦合程度。

**Rigor level: Eq.4' 为精确的 MI 下界；Argument 1 为依赖结构性假设的定性论证。**

### 关键量：$\delta$

为避免条件层级混乱，本节所有概率和熵均在 **固定任务族 $\mathcal{T}$ 的条件分布** $p(\cdot \mid X \in \mathcal{T})$ 下计算。记号中省略条件 $X \in \mathcal{T}$ 以简洁，但所有出现的 $P(\cdot)$、$H(\cdot)$ 均隐含此条件。

定义任务族内的脱钩系数：

$$\delta := P(D = 1 \mid E \neq 0) \tag{2}$$

即在任务族 $\mathcal{T}$ 内，"过程有错但答案仍然对"的概率。$\delta$ 衡量 outcome-process 之间的 **脱钩程度**。

类似地，定义 $\alpha := P(D = 0 \mid E = 0)$（过程全对但答案错的概率，通常很小——对应抄写错误、格式问题等）。

### $\delta$ 不需要趋近于零

我们 **不** 假设 $\delta \approx 0$。ProcessBench 等工作已表明，真实数据中存在大量"答案正确但过程有误"的 case。

我们的论点更温和：只要知道"过程是否有错"能改变对 outcome 的预测，二者之间就有信息关联。形式化地，考虑 binary error indicator $\mathbb{1}[E \neq 0]$ 与 $D$ 之间的条件互信息：

$$I(D;\, \mathbb{1}[E \neq 0] \mid X) = H(D \mid X) - H(D \mid \mathbb{1}[E \neq 0],\, X) \tag{3}$$

对右边第二项，按 $\mathbb{1}[E \neq 0]$ 的取值展开：

$$H(D \mid \mathbb{1}[E \neq 0],\, X) = P(E=0 \mid X)\cdot H_b\!\bigl(\alpha(X)\bigr) + P(E \neq 0 \mid X)\cdot H_b\!\bigl(\delta(X)\bigr) \tag{4}$$

其中 $\alpha(X) = P(D=0 \mid E=0, X)$，$\delta(X) = P(D=1 \mid E \neq 0, X)$，$H_b$ 为二元熵。在任务族条件分布下对 $X$ 取期望，即得到族级别的平均量。

**关键观察：** 只要在 $X$ 的正测集合上 $\delta(X) \neq P(D=1 \mid X)$——即知道"过程有错"确实改变了答案正确的概率——那么 $I(D; \mathbb{1}[E \neq 0] \mid X) > 0$。这个条件在几乎所有多步推理任务中都成立。

### $\delta$ 是调节变量，不是常数

不同任务族的 $\delta$ 差异很大：

| 场景 | $\delta$ 的预期大小 | 原因 |
|:---|:---|:---|
| 多步代数推导 | 小 | 错误逐步传播，几乎不可能碰巧对 |
| 证明题 | 中等 | 局部错误可能不影响整体结论 |
| 选择题/答案空间小 | 大 | 即使推理全错，蒙对的概率不低 |

我们的框架预测（P2）：$\delta$ 越小的任务族，outcome 训练带来的 process signal 越强。这不是 bug 而是 feature——它给出了方法适用范围的清晰边界。

### 从 $I(Z;D)$ 到 $I(Z;E)$ 的 MI 下界

将 Eq.1 和 Eq.3-4 结合。由 chain rule：

$$I(Z;\, D,\, E \mid X) = I(Z;\, D \mid X) + I(Z;\, E \mid D,\, X) = I(Z;\, E \mid X) + I(Z;\, D \mid E,\, X)$$

由于 $I(Z;\, E \mid D,\, X) \geq 0$ 且 $I(Z;\, D \mid E,\, X) \leq H(D \mid E,\, X) \leq H_b(\delta')$（其中 $\delta' = \max(\alpha, \delta)$，取任务族内的上界），得：

$$I(Z;\, E \mid X) \;\geq\; I(Z;\, D \mid X) - H_b(\delta') \tag{4'}$$

**Eq.4' 的解读。** 这是一个精确的 MI 下界：$Z$ 中关于 error position $E$ 的信息量，至少有 $I(Z; D \mid X) - H_b(\delta')$ 这么大。当 ORM 准确率高（$I(Z; D \mid X)$ 大，由 Eq.1）且 $\delta'$ 小时，该下界非平凡。需要强调的是，这是 **信息量级上的下界，不是语义成分分解**——我们不能将 $I(Z; D \mid X)$ 中的各个 "bit" 逐一归类为 "process bit" 或 "non-process bit"。Eq.4' 只说 $Z$ 中关于 $E$ 的信息量有一个下界，不说这些信息以什么形式编码、是否可被解码。

**与 Jia et al. (2025) 的联系。** Jia et al. Theorem 2 证明了任意策略 $\pi$ 的 advantage function $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 就是一个最优的 PRM。这从 RL 理论角度独立支持了 Eq.4' 的方向：outcome 优化产生的 advantage function 天然包含 step-level reward signal。我们的 $\delta$-based 分析是对这一一般性结论的细化——它告诉我们在什么任务条件下这个理论保证的信息量是大是小。

### 从 binary detection 到 error localization

Eq.4' bound 了 $I(Z; E \mid X)$（categorical），但 $E$ 是第一个错误步骤的位置——知道"有错"和知道"第几步错"是不同层次的信息。$I(Z; E \mid X)$ 可以分解为：

$$I(Z;\, E \mid X) = I(Z;\, \mathbb{1}[E \neq 0] \mid X) + I(Z;\, E \mid \mathbb{1}[E \neq 0],\, X) \tag{4''}$$

第一项（binary detection）由 Eq.4' 和 Eq.3-4 的逻辑保证有非平凡下界。第二项（error localization）需要额外论证。

**Argument 1（Thinking Chain 的 Sequential Localization；定性论证）。** 当 thinking chain $Z$ 对 solution 的 $K$ 个段落逐一分析时，$Z$ 可分解为 $Z = (Z_1, \dots, Z_K)$，其中 $Z_k$ 是对第 $k$ 段的分析。对于段落 $k$，定义局部正确性 $Y_k = \mathbb{1}[\text{step } k \text{ correct given previous steps}]$。

> **Structural Assumption (Sequential Analysis).** 自回归语言模型在逐段分析模式下，由于 causal mask 的约束，$Z_k$ 的生成只能依赖 $(X, Z_{<k})$。我们假设：当模型的 outcome 预测准确率高时，每个 $Z_k$ 中编码了关于第 $k$ 步可靠性的信息，即 $I(Z_k; Y_k \mid Z_{<k}, X) > 0$。

我们不把这个假设写成形式化不等式，因为它在严格意义上不可证。我们坦诚承认以下局限：

(a) **假设与结论的距离。** 假设 $I(Z_k; Y_k \mid Z_{<k}, X) > 0$ 已经相当接近我们想要论证的结论（step-local information exists）。这不是一个从远处出发的推导，而是一个对观察到的现象的结构性假设。

(b) **替代编码的可能。** $Z_k$ 完全可以通过与 $Y_k$ 不直接对应的方式（如编码该步的"风险等级"或"常见错误模式"）来贡献 outcome 预测信息，而非直接分类正确性。

该假设的合理性来自两个方面：(1) causal mask 和 sequential decoding 使得"评估当前步"成为最自然的信息编码方式——$Z_k$ 在生成时只能看到 $Z_{<k}$，要对 outcome 做出有用贡献，最直接的途径是评估当前步的状况；(2) 预测 P3（去掉 structured output 降低 probing 精度）和 P4（去掉 thinking chain 同时降低 outcome 和 process 表现）直接测试此假设的经验有效性。

**一句话：** outcome 和 process 不需要完美等价，只要正相关就够。$\delta$ 控制 binary detection 的带宽（Eq.4'，精确 MI 下界）；thinking chain 的 sequential nature 提供从 detection 到 localization 的升级（Argument 1，结构性假设，由 P3/P4 实验支撑）。

---

## 环节三：隐式能力的外化——从"Z里有信息"到"模型能输出step labels"

**对应直觉第2-5步。** 即使 $Z$ 里有 process 信息（环节一、二的结论），也不代表模型能把它变成显式的 step-level 预测。环节一、二回答了"信息在哪里"和"信息有多少"；本节回答"优化过程如何提取和利用这些信息"——这是静态信息论界限与动态学习过程的桥接。

**Rigor level: 机制假说（mechanistic hypothesis）。本节不追求信息论级别的精确 bound，而是提供优化动力学层面的机制解释，核心假设（Assumption 1）需要通过实验验证。**

### Decodability gap

令 $\mathcal{F}$ 为解码器类（如 linear probe），定义可提取精度：

$$\text{Acc}_{\text{PRM}}(\mathcal{F}) := \max_{f \in \mathcal{F}}\; P\bigl(f(Z) = \mathbb{1}[E \neq 0]\bigr) \tag{5}$$

**Decodability gap** $\Delta(\mathcal{F})$ 定义为由 Fano 不等式诱导的精度代理上界与 $\mathcal{F}$ 实际提取精度之间的差：

$$\Delta(\mathcal{F}) := \bigl[1 - H_b^{-1}(H(\mathbb{1}[E \neq 0] \mid Z, X))\bigr] - \text{Acc}_{\text{PRM}}(\mathcal{F}) \;\geq\; 0 \tag{6}$$

**关于 $H_b^{-1}$：** 二元熵函数 $H_b(p)$ 在 $[0,1]$ 上非单射（关于 $p=0.5$ 对称），此处 $H_b^{-1}$ 取 $[0, 0.5]$ 上的分支，对应 error probability $\leq 0.5$，与 Fano 不等式的标准用法一致。

**关于"上界"的说明：** 严格来说，由条件熵经 Fano 不等式诱导的精度界限并非 Bayes-optimal accuracy 的 tight upper bound，而是一个 entropy-based surrogate。因此 $\Delta(\mathcal{F})$ 度量的是 surrogate ceiling 与实际 probe 精度的差距，是信息可提取程度的近似指标。

$\Delta$ 越大，信息越"在但拿不出来"。我们的设计目标是缩小 $\Delta$。

### 机制 A：Structured Output = 用模型自身当解码器（对应直觉第5步）

强制输出 `step_labels`，等于让模型自己的 autoregressive 前向传播充当 $\mathcal{F}$。这比任何 post-hoc probe 都强——模型自身就是一个巨大的非线性函数族。

更关键的是 generation 顺序带来的 **一致性约束（consistency pressure）**。JSON 中 `step_labels` 在 `final_label` 前面生成：

$$\hat{d} = h(z,\; \hat{e}_1, \dots, \hat{e}_K) \tag{7}$$

模型写 `final_label` 时已经 conditioned on 自己的 `step_labels`。如果 `step_labels` 与 `final_label` 不一致（例如声称每步都对但最终判错），这种内部矛盾在部分样本上会导致 outcome 预测出错 → 拿不到 reward。因此 outcome reward 通过 generation 的自回归依赖，**对 step_labels 施加一致性压力**，即使我们从未直接奖励 step_labels。

**重要区分：这是 consistency pressure，不是 accuracy pressure。** 一致性压力确保 step_labels 与 final_label 之间逻辑自洽，但不直接保证 step_labels 反映真实的步骤正确性。模型原则上可以学会一种 "cheap consistent fiction"——编造一套与 final_label 一致但不真实的 step_labels。该机制能在多大程度上促进 step_labels 的 **准确性**（而非仅仅一致性），取决于一个经验性条件：在 outcome reward 压力下，与真实 process assessment 一致的 step_labels 是否比虚构的 step_labels 更容易支持准确的 outcome 预测。我们通过预测 P3 来实验检验这一条件。

### 机制 B：GRPO 的 Outcome Stability Selection（对应直觉第3-4步）

#### B.1 GRPO 梯度的正确/错误分解

同一道题采样 $G$ 个回答，$r_j \in \{0,1\}$。令 $G_1 = \{j: r_j = 1\}$（正确集）、$G_0 = \{j: r_j = 0\}$（错误集），代入 $\bar{r} = |G_1|/G$：

$$\nabla_\theta J(x) = \frac{|G_0|}{G^2}\sum_{j \in G_1}\nabla_\theta\log\pi_\theta(o_j \mid x) \;-\; \frac{|G_1|}{G^2}\sum_{j \in G_0}\nabla_\theta\log\pi_\theta(o_j \mid x) \tag{8}$$

**解读**：GRPO 做两件事——(a) 强化正确 responses 的生成方式；(b) 抑制错误 responses 的生成方式。但 Eq.8 本身只是 group-relative binary reward 的代数展开；从"强化正确 responses"到"偏好 process-correct strategy"之间还需要额外的机制论证。

**关键观察：** 当 ORM 高度饱和时（$|G_0| \to 0$），Eq.8 中强化项的系数 $|G_0|/G^2 \to 0$，抑制项的系数 $|G_1|/G^2 \to 1/G$——两项同时趋近于零。此时 GRPO 的有效梯度信号消失。我们将在 §实验验证·两阶段动态 中用实验数据验证这一机制。

#### B.2 Outcome Stability：为什么 process-correct 策略被偏好

$G_1$（正确集）中可能混有两类 responses：
- **过程正确型**：通过 genuine 逐步分析得到正确 outcome
- **碰巧正确型**：通过 shortcut 或运气得到正确 outcome

ORM reward 无法在单个 prompt 内区分这两类。我们提出以下假设来解释 GRPO 的跨 prompt 偏好机制：

> **Assumption 1（Outcome Stability）。** 在任务族 $\mathcal{T}$ 内，过程正确型策略的 outcome reward 跨 prompt 方差小于碰巧正确型策略。形式化地，令 $\sigma^2_P(x)$ 和 $\sigma^2_C(x)$ 分别为策略类型 $P$（过程正确）和 $C$（碰巧正确）在 prompt $x$ 上的 reward 方差（variance over response sampling $o \sim \pi_\theta(\cdot \mid x)$ within each strategy type），则：
>
> $$\mathbb{E}_{x \sim \mathcal{T}}[\sigma^2_P(x)] \;<\; \mathbb{E}_{x \sim \mathcal{T}}[\sigma^2_C(x)] \tag{9}$$

**理由。** 过程正确型策略基于对解题步骤的逐步验证，其成功与否主要取决于步骤本身的正确性——这是一个相对稳定的信号。碰巧正确型策略的成功高度依赖于具体题目是否恰好匹配该 shortcut，不同题目匹配不同 shortcut 的概率变化大，导致高方差。

**Assumption 1 的局限与适用边界。** 我们不假设 Assumption 1 在所有场景下成立。以下情形中它可能失效：(a) 数据集存在系统性 shortcut，使得 shortcut reward 在该分布上也很稳定；(b) 任务本身高噪声或 ORM 自身 noisy，导致 process-correct 策略的 observed reward variance 也很高；(c) $\delta$ 很大的任务族中，process-correct 策略的优势本身就不明显。Assumption 1 的适用范围与 $\delta$ 的大小高度相关——这进一步支持了 P2 的预测。

#### B.3 跨 prompt 的梯度积累

在 Assumption 1 成立的条件下，GRPO 的 advantage normalization 赋予 process-correct 策略更高的 effective signal-to-noise ratio：

- **过程正确型策略**：outcome 在不同题目上稳定地高于均值 → advantage-weighted 梯度方向稳定 → 有效梯度信号 coherently accumulate。
- **碰巧正确型策略**：outcome 高方差 → advantage 在不同题目上正负交替 → 有效梯度信号因正负抵消而 accumulate more slowly。

**我们不对积累速率做具体的 scaling claim**，因为这需要额外的 concentration / sign-coherence 条件，超出 Assumption 1 的范围。

#### B.4 GRPO 的失效模式：ORM 饱和与 effective signal collapse

环节三的机制假说有一个重要的 **适用边界**：它依赖于 GRPO 组内存在充分的 reward contrast。当训练进行到 ORM 高度饱和阶段时：

$$|G_0| \to 0 \;\Rightarrow\; \text{Eq.8 中两项系数均} \to 0 \;\Rightarrow\; \|\nabla_\theta J(x)\| \to 0 \tag{10}$$

此时 Assumption 1 的前提条件失效——不是因为两类策略的 reward variance 差距消失，而是因为 **所有策略的 reward 都趋近于 1**，组内不再有对比信号。我们将此称为 **effective signal collapse**。

**预测（P1'）：** Process signal 的涌现呈 **倒 U 型**——在 ORM accuracy 的上升区间与 Process F1 正相关，但在 ORM 饱和后，由于 effective signal collapse，Process F1 下降。

**与 DAPO 的联系。** DAPO (Yu et al., 2025) 的 Dynamic Sampling 技术通过过滤 zero-variance prompts（所有 response 获得相同 reward 的 prompt）来维持有效训练信号。如果 PRM 退化的原因确实是 effective signal collapse（而非其他因素如模型容量限制），那么 DAPO 应该能延缓或避免 PRM 的 late decline。这提供了一个独立的因果验证（预测 P6）。

#### B.5 与 rejection sampling 的比较

Rejection sampling（best-of-N）只保留高 reward 样本，也能选出正确 responses。它与 GRPO 的关键差异在于：(1) 无抑制项——不会主动降低错误推理模式的概率；(2) 无跨 prompt 的 advantage normalization——不利用跨题的 reward 稳定性差异来区分策略类型。预测 P5 的具体内容是 GRPO **更快** 出现 process signal（在相同计算预算下），而非 rejection sampling 完全无效。

---

## 完整链条

```
            环节一                    环节二                        环节三
     (表征存在的必然性)         (信息与process的关联)          (优化如何提取信息)
      [定理·精确bound]       [精确MI下界 + 定性论证]         [机制假说·实验验证]

  ORM训练, Acc↑           δ < P(D=1|X)                  机制A: Consistency Pressure
       │                  (outcome-process正相关)         机制B: Outcome Stability
       ▼                       │                                  │
  Z 编码 outcome信息  ───→  Z中关于E的MI有  ───→  process信息被外化为 step_labels
       (Eq.1)               非平凡下界            (Eq.5-7: 一致性压力)
                            (Eq.4')               (Eq.8-9: 优化动力学)
                                │                  (Eq.10: 失效模式)
                         δ 越小，下界越大
                         δ 越大，下界越小（方法适用边界）
```

---

## 可验证预测

| 预测 | 内容 | 对应环节 | 验证状态 |
|:---|:---|:---|:---|
| **P1'** | ORM accuracy 与 Process F1 呈**倒 U 型关系**：上升期正相关，饱和后因 effective signal collapse 而负相关 | 环节一 + 环节三·B.4 | ✅ 已验证（§两阶段动态） |
| **P2** | 上述相关性在 $\hat{\delta}$ 小的任务族上更强，$\hat{\delta}$ 大的任务族上更弱 | 环节二 (Eq.4') | ◐ 方向性一致（§Per-subset 分析） |
| **P3** | 去掉 `step_labels` 输出，probing 精度显著下降（$\Delta$ 增大） | 环节三·机制A (Eq.7) | 待验证 |
| **P4** | 去掉 thinking chain，outcome 和 process 都变差 | 环节一前提 + Argument 1 | ✅ 已验证（§Instruct 无 CoT 对比） |
| **P5** | GRPO 比 rejection sampling 更快出现 process signal | 环节三·机制B (Assumption 1) | 待验证 |
| **P6** | DAPO（Dynamic Sampling）相比 GRPO 能延缓 ORM 饱和后的 Process F1 下降 | 环节三·B.4 (Eq.10) | 待验证 |

**注：** P1' 是对原始 P1 的修正——原始 P1 预测单调正相关，实验发现为倒 U 型，理论上对应 effective signal collapse（B.4 节）。P2 同时也是方法的 **适用性边界声明**。P6 提供了对 saturation 机制的因果验证。

---

## 实验验证

### 实验设置

**模型。** Qwen3-4B-2507-Thinking（thinking model）和 Qwen3-4B-2507-Instruct（instruct model，无 thinking chain）。

**训练。** 使用 GRPO 算法，在 MATH 数据集上进行 ORM 训练，约 3000 条轨迹。训练 reward 为 binary（ORM 判断最终答案正确性）。

**评估。** 在 ProcessBench 上评估 step-level error detection 的 F1 score（micro-averaged first-error F1）和 ORM accuracy（trajectory-level reward mean@1）。ProcessBench 包含四个子集：GSM8K、MATH、OlympiadBench、OmniMath，难度递增。评估在多个训练 checkpoint（每 20 steps）进行。

### 主要结果：Process Signal 从 Outcome Training 中涌现

在 Qwen3-4B-Thinking + GRPO 训练中，ProcessBench step-level F1 提升约 20 个百分点，接近 o1-mini 水平：

| 模型 | GSM8K | MATH | Olympiad | OmniMath |
|:---|:---|:---|:---|:---|
| Qwen3-4B-Thinking (baseline) | 59.3 | 69.8 | 56.8 | 54.7 |
| + ORM RL | **80.0** | **87.3** | **83.3** | **79.4** |
| o1-mini (reference) | 93.2 | 88.9 | 87.2 | 82.4 |

这一提升**不完全归因于格式对齐**——控制格式后仍有显著提升。

### P4 验证：Thinking Chain 是 Process Signal 涌现的必要条件

在 Qwen3-4B-Instruct（不输出 thinking chain）上进行相同的 ORM RL 训练，**不观察到上述 PRM 提升现象**。这直接验证了 P4：没有 thinking chain $Z$，环节一的 Eq.1 中 $Z$ 退化为空序列，outcome 信息无处编码，process signal 无法涌现。

此外，在 AgentProcessBench 上观察到类似结果，表明 process signal 涌现不限于数学推理任务。

### P1' 验证：两阶段动态——涌现与饱和

**这是本文的核心实验发现。** 我们在 Qwen3-4B-Instruct + GRPO 训练中观察到 process signal 涌现呈现清晰的两阶段动态：

<!--
[插入 fig1_orm_vs_prm_dynamics.png]
Figure: Training Dynamics: ORM Accuracy vs Process F1 (Qwen3-4B-Instruct + GRPO)
-->
![fig1_orm_vs_prm_dynamics](/Users/dada/Downloads/files/fig1_orm_vs_prm_dynamics.png)



**涌现期（Step 0–40）。** ORM accuracy 从 74.4% 上升至 88.2%（+13.8pp），Process F1 从 64.8% 同步上升至 77.9%（+13.1pp）。两者在此阶段的 Pearson 相关系数为 $r = 1.00$（$p = 0.007$）。这与理论预测一致：GRPO 组内存在充分的 reward contrast（mean training reward 0.905），Assumption 1 的 outcome stability selection 有效运作。

**饱和期（Step 40–160）。** ORM accuracy 继续微升至 90.3%（+2.1pp），但 Process F1 反而下降至 74.7%（**-3.2pp from peak**）。两者在此阶段的 Pearson 相关系数反转为 $r = -0.84$（$p = 0.019$）。

<!--
[插入 fig3_scatter_trajectory.png]
Figure: ORM Accuracy vs Process F1 (Training Trajectory). 训练轨迹呈"钩形"：先沿对角线上升（涌现期），在 ORM ~88% 处到达拐点，然后 ORM 继续微升但 Process F1 掉头向下（饱和期）。
-->
![fig3_scatter_trajectory](/Users/dada/Downloads/files/fig3_scatter_trajectory.png)


**机制解释：Effective Signal Collapse。** 训练 reward 的饱和速度极快：77.8% 的 batch reward ≥ 0.95，15.9% 直接打满 1.0。

<!--
[插入 fig2_training_saturation.png]
Figure: Training Reward Saturation and Effective Gradient Signal Collapse.
-->
![fig2_training_saturation](/Users/dada/Downloads/files/fig2_training_saturation.png)


训练后期的 effective gradient signal（$1 - \text{reward}$）接近零，对应 Eq.10 的分析：当 $|G_0| \to 0$ 时，GRPO 的强化项和抑制项系数同时趋近零，有效梯度消失。残余梯度主要来自 ORM 自身的判断噪声而非真实的推理质量差异，导致模型逐渐遗忘 process 表征。

**Instruct vs Thinking 模型的饱和对比：**

| | Instruct | Thinking |
|:---|:---|:---|
| 训练 batches | 176 | 92 |
| Mean training reward | 0.968 | 0.951 |
| Batch reward ≥ 0.95 | 77.8% | 63.0% |
| Batch reward = 1.0 | 15.9% | 8.7% |

Thinking 模型的饱和程度更低（63% vs 78% ≥ 0.95），保留了更多 effective gradient signal。这与 P4 一致：thinking chain 提供了更丰富的 $Z$，使得 ORM 任务对模型仍有挑战性。

**实践指导：** 这一发现表明 GRPO + ORM 训练应在 reward saturation 之前停止或切换策略。Process signal 的涌现窗口可能短暂（本实验中约 40 steps）。监控组内 reward variance 可作为 early stopping 的指标。

### P2 的方向性证据：Per-Subset 分析

Thinking 模型在 ProcessBench 四个子集上的 ORM accuracy 提升幅度不同：

| 子集 | Baseline | Peak | 提升 | 拐点后下降 |
|:---|:---|:---|:---|:---|
| GSM8K | 71.0% | 95.8% (s80) | +24.8pp | 0.0pp |
| MATH | 85.2% | 97.1% (s60) | +11.9pp | -0.1pp |
| OlympiadBench | 80.2% | 94.4% (s40) | +14.2pp | -0.7pp |
| OmniMath | 74.4% | 86.0% (s60) | +11.0pp | -0.6pp |

OmniMath（最难子集，预期 $\delta$ 最大）的提升最小（+11.0pp），最终绝对水平最低（86%），这与 P2 的预测方向一致。

**但我们谨慎地注意到**：所有子集使用相同的 MATH 训练数据，per-subset 差异可能部分反映评估难度差异（ceiling/floor effects）而非 $\delta$ 的调节效应。P2 的严格验证需要在不同 $\delta$ 的训练数据上分别训练模型，或对 per-sample $\hat{\delta}$ 进行分层分析，留作 future work。

### Case Study：Process Signal 的具体表现

以下展示一个 ORM RL 训练后的模型如何在 thinking chain 中进行逐步验证。

**输入 prompt**：要求模型判断一道竞赛数学题的解答是否正确（是否存在 4004 个正整数使得任意 2003 个的和不被 2003 整除）。

**模型行为**：模型在 thinking chain 中逐段分析 solution，在第 3 段（paragraph_3）发现了一个关键计算错误——"4004 ≡ 1 (mod 2003)"实际应为 $4004 = 2 \times 2003 - 2$，故 $4004 \equiv -2 \pmod{2003}$。模型据此将 paragraph_3 标记为错误，后续依赖此错误的段落也被标记为错误，最终给出正确的 step labels 和 final label。

这个案例说明模型确实在 thinking chain 中编码了 **step-local correctness information**——它不是简单地给出整体判断，而是逐步检查每段推理的逻辑和计算，与 Argument 1 的 Sequential Analysis 假设一致。

---

## 讨论

### 与 AutoPSV 的关系

AutoPSV 和本文利用了同一个底层原理——**准确预测 outcome 需要编码 step-level 信息**——但通过不同的技术路径实现。

AutoPSV 训练一个 BCE/MSE regression 模型，直接预测  
$$
P(D=1 \mid q, S^{(1:t)})
$$

当该条件概率随 step prefix 变化时（例如某一步引入错误导致 confidence 骤降），confidence 差分  
$$
\Delta_{\text{conf}}^t
$$
就携带了 step-level error signal。这里的 process 信息编码在模型的 **hidden states** 中，由 BCE loss 的梯度直接雕刻，不涉及 thinking chain 或 token-level generation。

相比之下，本文的框架中，process 信息编码在 emitted thinking chain $Z$ 中（Eq. 1），通过 GRPO 的 outcome reward 间接诱导，并通过 structured output 外化为显式 step labels。

两种路径的关键差异在于：

1. **AutoPSV 需要单独训练一个 verifier 模型并做 confidence 差分计算**，而我们的方法在 GRPO 训练过程中**同时**获得 ORM 和 PRM 能力；

2. **我们的 $\delta$-based 分析（Eq. 4'）为两种方法都提供了适用边界预测**——无论通过 hidden states 还是 thinking chain 编码，$\delta$ 大的任务族中 process signal 都会更弱，因为 outcome-process 脱钩本身是任务的固有属性。

### 与 Jia et al. (2025) 的关系

Jia et al. 从统计等价性的角度证明了"outcome supervision 原则上不比 process supervision 更难"。我们的工作提供了互补视角：(1) 具体的优化机制（GRPO + structured output）如何实现这种理论可能性；(2) 实践中的适用边界（$\delta$ 和 ORM 饱和）；(3) 方法失效的条件（effective signal collapse）。Jia et al. 的结论可以概括为"信息在那里"，我们的贡献是"怎么拿到、什么时候拿得到、什么时候拿不到"。

### 局限性

(1) **实验规模有限。** 目前仅在 Qwen3-4B 上验证，generalizability 需要在更多模型族和尺寸上检验。

(2) **P2 未严格验证。** Per-subset 分析受 ceiling/floor effects 的 confound，严格验证需要不同 $\delta$ 的训练数据或 per-sample 分层。

(3) **P3、P5 待验证。** Structured output 的 ablation 和 GRPO vs rejection sampling 的对比实验尚未完成。

(4) **理论环节三为机制假说。** Assumption 1 和 Argument 1 未经形式化证明，其有效性依赖于实验支持。

(5) **因果链尚需进一步验证。** 两阶段动态的机制解释（effective signal collapse）目前基于 Eq.10 的分析和训练 reward 的饱和数据。计划中的 DAPO 对比实验（P6）将提供因果验证：如果 DAPO 的 Dynamic Sampling 能延缓 PRM 退化，则确认 effective signal collapse 是 PRM 退化的直接原因。
