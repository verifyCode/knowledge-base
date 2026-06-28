# Reasoning Model 笔记

Inference,Reasoning Model,Reasoning Behavior 是个过渡态 PE 工程。

- **Wei et al., 2022** — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (NeurIPS 2022, arXiv:2201.11903)。few-shot CoT 起源，PaLM 540B + 8 exemplars 在 GSM8K 上 SOTA。
- **Kojima et al., 2022** — *Large Language Models are Zero-Shot Reasoners* (arXiv:2205.11916)。"Let's think step by step" 的原始出处。

---

## 为什么 Reasoning 能提升模型表现？

Answer 阶段生成的答案，它所能 attend 到的上下文里多了什么。

开了 Reasoning 后，think 段就被摆在 kv-cache 里面，answer 阶段每生成一个 token 都能 attend 到这段「中间结论」。

> **什么是 kv-cache** → 填坑

### 旁证：

- **Anthropic** — *Extended thinking* 官方文档。Adaptive thinking、interleaved thinking、prompt caching 与 thinking 块的交互规则、signature 加密。文档明确说，在带工具调用的多轮场景里，thinking 块必须原样回传才能保持推理连续性——signature 字段被加密用于验证，任何篡改都会破坏链路。这是一个很强的信号：thinking 段并不是给人看的解释，而是模型自己依赖的中间状态。
- **DeepSeek-AI, 2025** — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (arXiv:2501.12948)。R1-Zero 纯 RL 路径，GRPO，rule-based reward，AIME 2024 15.6%→77.9%，Aha Moment。模型在 RL 训练过程中，response 平均长度从几千 tokens 一路涨到上万——「想得更长」和「答得更准」耦合上升（§2.3 与 Figure 1）。
- **Snell et al., 2024** — *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters* (arXiv:2408.03314, UC Berkeley × Google DeepMind)。test-time scaling 的系统化研究，compute-optimal 策略可比 best-of-N 高效 4 倍。在固定 base 模型不变的前提下，只增加 test-time compute（无论是 best-of-N 还是 sequential revision）就能持续提高准确率。
- 在多轮对话里，旧轮次的 thinking 段通常会被丢弃。

---

## 模型是怎么学会 Reasoning 的？

### PE 工程（COT、Let's think step by step）

- **局限** → 这种思考本质是模仿，一旦走错就将错就错写到底，prompt 只能让模型"分步"，不能让它"想深"。
- **亮点** → PaLM 540B 仅用 8 个 CoT 示范就在 GSM8K 上把准确率拉到当时的 SOTA。

### SFT on COT

数据贵，泛化差，模型只能模仿出现过的推理套路，换个模式提醒就傻眼。

### RLHF（基于人类反馈的强化学习）

优化的是答案听起来好不好，不是推理过程对不对。

- **Lightman et al., 2023** — *Let's Verify Step by Step* (arXiv:2305.20050, OpenAI)。PRM vs ORM，MATH 78.2%，PRM800K 数据集开源。

解决方式：

- **ORM**（只评最终答案）
- **PRM**（对推理过程的每一步都打分）

PRM 的优势在题目难度上来了以后才显现。

### RL：让模型通过试错和奖励信号不断改进决策策略的方法

**DeepSeek-R1**：算法用的是 GRPO。Reward 完全是 rule-based，没有用 ORM 和 PRM。

- R1-Zero 是纯 RL，完全跳过 SFT。
- **Aha Moment**
- **缺陷**：强烈依赖任务能不能被自动验证。

---

## Reasoning 时，模型在 thinking 段里到底在做什么？

1. 问题理解
2. 规划
3. 执行
4. 验证
5. 元认知

---

## 几家主流 Reasoning 模型的不同选择

- **Google** — Gemini 2.5 Deep Think 技术报告。Parallel thinking 路线。
- **DeepSeek** — R1/R1-Zero，纯 RL + GRPO，thinking 完全可见。
- **Anthropic** — Extended thinking，Adaptive thinking，interleaved thinking，signature 加密。
- **OpenAI** — o1/o3，thinking 不可见（仅摘要），test-time compute scaling 工业化。

| 维度               | Google            | DeepSeek   | Anthropic                  | OpenAI              |
| ------------------ | ----------------- | ---------- | -------------------------- | ------------------- |
| thinking 可见性    |                   | 完全可见   | 可见                       | 不可见（仅摘要）    |
| think 长度控制     |                   |            | Adaptive thinking          |                     |
| thinking 计费      |                   |            | thinking tokens 单独计费   |                     |
| 历史 thinking 保留 |                   |            | signature 加密，需原样回传 | 多轮丢弃            |
| 特色能力           | Parallel thinking | Aha Moment | Interleaved thinking       | Cipher / STRAWBERRY |

---

## Test-Time Compute Scaling：思考越久越聪明吗

从更大的模型、更多的数据、更长的训练时间，Reasoning 则是：同一个模型，你只要在推理时多花算力（让它想得更久），就能持续提升正确率，而且这种关系还服从某种 scaling law。

**OpenAI, 2024** — *Learning to Reason with LLMs*（o1 公告）。test-time compute scaling 的工业化实证，Cipher 与 STRAWBERRY 示例。公告直接说「performance smoothly improves with both train-time and test-time compute」。

同一个 o1，只是 test-time 投入从 1 个样本扩到 1000 个、再加个排序器，准确率就从 74% 拉到 93%——一脚踩进了美国数学奥赛 USAMO 的入围线。

要更系统地理解这条曲线，推荐去看 **Snell et al., 2024**。

- budget 不是越大越好，而是要找到「拐点」——而拐点的位置和题目难度强相关。
- **parallel** 适合「需要在解空间里探索多种高阶方法」的难题。
- **sequential revision** 适合「方向大致对、需要逐步精炼」的题。

---

## Reasoning 也会翻车

[图片]

---

## 什么时候开？什么时候关？

### 开启 Reasoning，设合理 budget：

- 涉及多步推理、规划
- 有明确正确答案且 wrong answer 代价高
- Agent 工具编排
- 代码 / 数学 / 科学计算

### 关闭 Reasoning 或用低 effort：

- 简单问答 / 翻译 / 格式化
- 流式延迟敏感
- 高吞吐量批处理
- 创意 / 风格类任务

---

## Budget 怎么定

---

## 和 Prompt Cache 的关系
