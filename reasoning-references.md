# Reasoning Model 参考文献

## 核心论文

1. **Wei et al., 2022** — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (NeurIPS 2022, arXiv:2201.11903)。few-shot CoT 起源，PaLM 540B + 8 exemplars 在 GSM8K 上 SOTA。

2. **Kojima et al., 2022** — *Large Language Models are Zero-Shot Reasoners* (arXiv:2205.11916)。"Let's think step by step" 的原始出处。

3. **Lightman et al., 2023** — *Let's Verify Step by Step* (arXiv:2305.20050, OpenAI)。PRM vs ORM，MATH 78.2%，PRM800K 数据集开源。

4. **Snell et al., 2024** — *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters* (arXiv:2408.03314, UC Berkeley × Google DeepMind)。test-time scaling 的系统化研究，compute-optimal 策略可比 best-of-N 高效 4 倍。

5. **DeepSeek-AI, 2025** — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (arXiv:2501.12948)。R1-Zero 纯 RL 路径，GRPO，rule-based reward，AIME 2024 15.6%→77.9%，Aha Moment。

6. **OpenAI, 2024** — *Learning to Reason with LLMs*（o1 公告）。test-time compute scaling 的工业化实证，Cipher 与 STRAWBERRY 示例。

## 官方文档

7. **Anthropic** — Extended thinking 官方文档。Adaptive thinking、interleaved thinking、prompt caching 与 thinking 块的交互规则、signature 加密。

8. **Google** — Gemini 2.5 Deep Think 技术报告。Parallel thinking 路线。
