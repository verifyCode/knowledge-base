"""
Transformer 模型完整实现（"Attention Is All You Need" — Vaswani et al., 2017）
================================================================================

本文件按照从底向上的顺序组织代码（数字编号对应论文架构的解析顺序），
以帮助逐步理解 Transformer 的每个组件：

  编号    组件                        论文中的位置
  ───────────────────────────────────────────────────────────────
  1.     Transformer                 顶层：Encoder + Decoder + 输出投影层
  2.     Encoder                     编码器：Embedding + PositionalEncoding + N×EncoderLayer
  3.     PositionalEncoding          位置编码（固定正弦/余弦，不可学习）
  4.     get_attn_pad_mask           工具函数：生成 Padding 掩码，屏蔽 <PAD> token
  5.     EncoderLayer                编码器单层：Multi-Head Self-Attention + FFN
  6.     MultiHeadAttention          多头注意力：拆分 Head → 分别计算注意力 → 拼接
  7.     ScaledDotProductAttention   注意力核心运算：Q·K^T / √d_k → Softmax → 加权求和 V
  8.     PoswiseFeedForwardNet       逐位置前馈网络：两层 1D 卷积（等价于 Linear）+ ReLU
  9.     Decoder                     解码器：Embedding + PositionalEncoding + N×DecoderLayer
  10.    DecoderLayer                解码器单层：Masked Self-Attn + Cross-Attn + FFN
  10.    get_attn_subsequent_mask    工具函数：生成因果掩码（上三角矩阵），防止"看到未来"

数据流概览（训练时，Teacher Forcing）：
  src: "ich mochte ein bier P"
      → Embedding → +PositionalEncoding → Encoder×6 → enc_outputs [B, src_len, d_model]
                                                                        │
                              ┌─ Cross-Attention: K, V ────────────────┘
                              │
  tgt: "S i want a beer"     │
      → Embedding → +PositionalEncoding → Decoder×6 → Linear Projection → logits

出处：https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


# ================================================================================
# 数据准备：将原始句子转换为模型输入张量
# ================================================================================

def make_batch(sentences):
    """
    将句子三元组转换为模型所需的张量。

    参数:
        sentences: 长度为 3 的列表
            sentences[0] — 源语言句子（德语），以 <PAD> 结尾，如 "ich mochte ein bier P"
            sentences[1] — 目标语言解码器输入，以 <S> (Start) 开头，如 "S i want a beer"
            sentences[2] — 目标语言真实标签（Ground Truth），以 <E> (End) 结尾，如 "i want a beer E"

    返回:
        input_batch:  [1, src_len] — 编码器输入（batch_size=1 仅用于演示）
        output_batch: [1, tgt_len] — 解码器输入（Teacher Forcing）
        target_batch: [1, tgt_len] — 损失函数的目标序列

    Teacher Forcing 说明:
        训练时，解码器每一步的输入使用"真实的上一个 token"，
        而不是模型自己预测的上一个 token。这样做的好处是：
        - 训练更稳定：避免了错误预测的累积（error propagation）
        - 收敛更快：梯度的传递路径更直接
        - 可以并行计算所有位置的输出（配合 subsequent mask 防止信息泄漏）

    举例（德语→英语翻译）:
        src:   "ich  mochte  ein  bier  P"   → [1, 2, 3, 4, 0]  编码器输入
        tgt:   "S     i     want  a   beer"  → [5, 1, 2, 3, 4]  解码器输入（右移一位，S 开头）
        label: "i    want   a   beer  E"     → [1, 2, 3, 4, 6]  训练目标（E 结尾）
    """
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]   # [1, src_len]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]  # [1, tgt_len]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]  # [1, tgt_len]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


# ================================================================================
# 10. 因果掩码（Subsequent Mask / Look-ahead Mask）
# ================================================================================

def get_attn_subsequent_mask(seq):
    """
    生成"因果掩码"——一个上三角矩阵，用于防止解码器自注意力"看到"未来的 token。

    ╔══════════════════════════════════════════════════════════════╗
    ║  问题：为什么需要这个 mask？                                   ║
    ║  训练时解码器一次性接收整个目标序列，如果不加限制，             ║
    ║  位置 i 的 self-attention 会看到位置 i+1, i+2, ... 的词。      ║
    ║  这等于"作弊"——模型在预测第 3 个词时已经看到了正确答案。        ║
    ║  推理时不存在这个问题（因为只能一个一个生成），                ║
    ║  但训练必须模拟推理时的因果约束。                              ║
    ╚══════════════════════════════════════════════════════════════╝

    实现方式:
        用 np.triu(k=1) 生成上三角矩阵（主对角线以上全为 1），
        这些位置在后续注意力计算中会被填充为 -1e9，
        经过 Softmax 后趋近于 0，即"看不见"。

    参数:
        seq: [batch_size, tgt_len] — 解码器输入序列（仅用于获取形状）

    返回:
        mask: [batch_size, tgt_len, tgt_len] — 布尔/byte 张量
              True  = 需要遮蔽（不可见）
              False = 可见

    可视化（tgt_len=4）:
               key 位置
             ┌─────────────┐
             │ 0  1  2  3  │
        q=0  │ F  T  T  T  │  ← 位置 0 只能看到自己
        q=1  │ F  F  T  T  │  ← 位置 1 能看到 [0,1]
        q=2  │ F  F  F  T  │  ← 位置 2 能看到 [0,1,2]
        q=3  │ F  F  F  F  │  ← 位置 3 能看到所有
             └─────────────┘
              T = True  = 遮蔽
              F = False = 可见
    """
    # attn_shape: [batch_size, tgt_len, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    # np.triu: 返回矩阵的上三角部分，k=1 表示从主对角线以上第 1 条开始
    # 例如 np.triu([[1,1,1],[1,1,1],[1,1,1]], k=1) = [[0,1,1],[0,0,1],[0,0,0]]
    # np.ones(attn_shape): 每个样本独立生成上三角矩阵，np.triu 对最后两维操作
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 转为 uint8 张量
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ================================================================================
# 7. 缩放点积注意力（Scaled Dot-Product Attention）—— 注意力机制的核心计算
# ================================================================================

class ScaledDotProductAttention(nn.Module):
    """
    注意力机制的原子操作，论文第 3.2.1 节。

    公式:
        Attention(Q, K, V) = Softmax( Q·K^T / √d_k ) · V

    步骤拆解:
        ① Q·K^T        → 计算 Query 与所有 Key 的相似度（点积越大越相关）
        ② ÷ √d_k       → 缩放（scale），防止 d_k 过大时点积值进入 Softmax 饱和区
        ③ mask fill     → 将需要遮蔽的位置置为 -1e9（接近 -∞），Softmax 后 ≈ 0
        ④ Softmax       → 沿 Key 维度归一化为概率分布（权重之和 = 1）
        ⑤ · V           → 用注意力权重对 Value 做加权平均，得到上下文表示

    为什么除以 √d_k（缩放因子的由来）？
        假设 Q 和 K 的每个分量独立且均值为 0、方差为 1，则点积 Q·K^T
        的方差为 d_k。当 d_k 很大时，点积的绝对值也很大，Softmax 会进入
        梯度极小的区域（输出接近 one-hot），导致训练困难。
        除以 √d_k 将方差拉回 1，保持 Softmax 梯度在健康范围。

    每个张量的形状:
        Q:          [batch_size, n_heads, len_q, d_k]
        K:          [batch_size, n_heads, len_k, d_k]
        V:          [batch_size, n_heads, len_k, d_v]
        attn_mask:  [batch_size, n_heads, len_q, len_k]  (True = 需要遮蔽)
        ─────────────────────────────────────────────────
        scores:     [batch_size, n_heads, len_q, len_k]
        attn:       [batch_size, n_heads, len_q, len_k]  (注意力权重矩阵)
        context:    [batch_size, n_heads, len_q, d_v]    (加权求和后的输出)
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # ── 步骤 ① + ②：计算缩放点积 ──
        # K.transpose(-1, -2): 交换最后两维 → K^T
        #   [batch, n_heads, len_k, d_k] → [batch, n_heads, d_k, len_k]
        # matmul(Q, K^T): [batch, n_heads, len_q, d_k] × [batch, n_heads, d_k, len_k]
        #               → [batch, n_heads, len_q, len_k]
        # 每个元素 scores[b,h,i,j] = 第 b 个样本、第 h 个头中，
        #   第 i 个 Query 与第 j 个 Key 的相似度
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # ── 步骤 ③：应用注意力掩码 ──
        # masked_fill_(mask, value): 将 mask 中为 True 的位置替换为 value
        # -1e9 ≈ 负无穷，exp(-1e9) ≈ 0，Softmax 后该位置的权重趋近于 0
        # 典型被遮蔽的场景：
        #   - <PAD> token: 没有实际语义，不应被任何 Query 关注
        #   - 未来 token:  解码时不应看到还未生成的词（训练时作弊）
        scores.masked_fill_(attn_mask, -1e9)

        # ── 步骤 ④：Softmax 归一化 ──
        # dim=-1: 沿最后一维（len_k 方向）做 Softmax
        # 含义：对于每个 Query，对它在所有 Key 上的相似度做概率归一化
        # 结果：每个 Query 对所有 Key 的注意力权重之和 = 1
        attn = nn.Softmax(dim=-1)(scores)  # [batch, n_heads, len_q, len_k]

        # ── 步骤 ⑤：加权求和 ──
        # attn × V: [batch, n_heads, len_q, len_k] × [batch, n_heads, len_k, d_v]
        #          → [batch, n_heads, len_q, d_v]
        # 对于每个 Query，将所有 Value 按注意力权重加权求和
        # 如果 Query 对某个 Key 的注意力权重接近 0，该 Key 对应的 Value 几乎被忽略
        context = torch.matmul(attn, V)

        return context, attn  # (输出, 注意力权重矩阵——可用于可视化分析)


# ================================================================================
# 6. 多头注意力（Multi-Head Attention）—— Transformer 的核心创新
# ================================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（论文第 3.2.2 节）。

    ╔═══════════════════════════════════════════════════════════════╗
    ║  核心直觉：                                                    ║
    ║  与其用一组 (Q,K,V) 做一个"大"注意力（d_model 维），           ║
    ║  不如将 d_model 拆成 h 个"小头"（每个 d_k = d_model/h 维），  ║
    ║  每个头在低维子空间独立计算注意力，最后拼起来。                 ║
    ║                                                               ║
    ║  类比：                                                        ║
    ║  多头注意力 ≈ 多个"专家"各自从不同角度理解句子，               ║
    ║  有的头关注语法结构（主语-谓语），有的关注语义相关（同义词），   ║
    ║  有的关注位置相邻（局部短语），最后综合所有专家的意见。         ║
    ╚═══════════════════════════════════════════════════════════════╝

    计算流程:
        输入 Q, K, V: [B, S, d_model]
           │
           ├─ ① 线性投影 d_model → h·d_k (or h·d_v)
           │     W_Q: d_model → n_heads × d_k
           │     W_K: d_model → n_heads × d_k
           │     W_V: d_model → n_heads × d_v
           │
           ├─ ② 分头 (Split heads): 将 d_model 拆成 n_heads 个 d_k/d_v
           │     reshape:  [B, S, h·d_k] → [B, S, h, d_k]
           │     transpose: [B, S, h, d_k] → [B, h, S, d_k]
           │
           ├─ ③ ScaledDotProductAttention（每个头独立计算）
           │
           ├─ ④ 合头 (Concat heads): 将 h 个头拼回 d_model 维
           │
           ├─ ⑤ 输出投影: Linear(n_heads×d_v → d_model)
           │
           └─ ⑥ 残差连接 + LayerNorm
                output = LayerNorm(x + MultiHeadAttention(x))

    三种使用场景:
        ┌─────────────────────┬───────────┬───────────┬────────────────────┐
        │ 场景                 │ Q 来源     │ K,V 来源   │ 说明               │
        ├─────────────────────┼───────────┼───────────┼────────────────────┤
        │ Encoder Self-Attn   │ Encoder   │ Encoder     │ 源语言的全局上下文 │
        │ Decoder Self-Attn   │ Decoder   │ Decoder     │ 目标语言的因果上下文│
        │ Decoder Cross-Attn  │ Decoder   │ Encoder     │ 翻译的核心：对齐    │
        └─────────────────────┴───────────┴───────────┴────────────────────┘
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # ── 线性投影矩阵 ──
        # 将输入的 d_model 维投影到 h·d_k 维（为分头做准备）
        # 一次性对所有 head 做投影：d_model → n_heads × d_k，比分别投影更高效
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # Query 投影矩阵
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # Key   投影矩阵
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # Value 投影矩阵

        # ── 输出投影矩阵 ──
        # 将所有 head 合并后的结果 (h·d_v，通常等于 d_model) 再做一次线性变换
        # 让模型能学习不同 head 输出之间的交互关系
        self.linear = nn.Linear(n_heads * d_v, d_model)

        # ── 层归一化（Layer Normalization）──
        # 对每个样本、每个位置的特征向量（d_model 维）做归一化
        # 与 BatchNorm 的区别：
        #   - BN:  对同一特征在 batch 内归一化 → 依赖 batch 内的其他样本，不适合变长序列
        #   - LN:  对同一样本内的所有特征归一化 → 独立于 batch，适合 NLP 的变长序列
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        参数:
            Q: [batch_size, len_q, d_model] — Query 序列
            K: [batch_size, len_k, d_model] — Key   序列
            V: [batch_size, len_k, d_model] — Value 序列
            attn_mask: [batch_size, len_q, len_k] — 注意力掩码（True = 需要遮蔽的位置）

        注意 Q, K, V 的序列长度:
            - Self-Attention:   Q=K=V → len_q = len_k（同一序列）
            - Cross-Attention:  Q 来自 Decoder, K=V 来自 Encoder → len_q ≠ len_k

        返回:
            output: [batch_size, len_q, d_model] — 多头注意力输出（含残差+LN）
            attn:   [batch_size, n_heads, len_q, len_k] — 所有头的注意力权重
        """
        # ── 保存残差 ──
        # residual 保存了输入的原始值，用于后面的残差连接 (skip connection)
        # 残差连接让梯度可以"绕过"注意力层直接回传，是 Transformer 能堆叠
        # 6+ 层而不出现梯度消失的关键
        residual, batch_size = Q, Q.size(0)

        # ── 步骤 ① + ②：线性投影 + 分头 ──
        # 数据流示意（以 Q 为例）:
        #   Q: [B, len_q, d_model]
        #     → W_Q(Q): [B, len_q, n_heads × d_k]
        #     → .view(B, -1, n_heads, d_k): [B, len_q, n_heads, d_k]
        #       (-1 自动推断为 len_q)
        #     → .transpose(1, 2): [B, n_heads, len_q, d_k]
        #       将 head 维度移到 batch 后面，方便各 head 并行计算

        # q_s: [batch_size, n_heads, len_q, d_k] — 分头后的 Query
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # k_s: [batch_size, n_heads, len_k, d_k] — 分头后的 Key
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # v_s: [batch_size, n_heads, len_k, d_v] — 分头后的 Value
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # ── 将 mask 扩展到头维度 ──
        # 原始 mask: [batch_size, len_q, len_k]
        # .unsqueeze(1): 在第 1 维插入一个维度 → [batch_size, 1, len_q, len_k]
        # .repeat(1, n_heads, 1, 1): 沿第 1 维复制 n_heads 次 → [batch_size, n_heads, len_q, len_k]
        # 所有 head 共享同一份 mask（PAD 对所有头都应该被屏蔽）
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # ── 步骤 ③：缩放点积注意力 ──
        # 对每个 head 独立计算 Attention(Q_h, K_h, V_h)
        # context: [batch_size, n_heads, len_q, d_v] — 各头输出
        # attn:    [batch_size, n_heads, len_q, len_k] — 各头注意力权重
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # ── 步骤 ④：合并所有头 ──
        # transpose(1,2): [B, n_heads, len_q, d_v] → [B, len_q, n_heads, d_v]
        # contiguous(): 确保张量在内存中连续排列
        #   transpose 操作只改变了张量的"视图"（stride 变化），
        #   数据在内存中的实际排列并未改变。如果不调 contiguous()，
        #   后续 view() 会因为内存不连续而报错。
        # view(B, -1, n_heads*d_v): [B, len_q, n_heads, d_v] → [B, len_q, h·d_v]
        #   h·d_v 通常 = d_model（论文设置 d_v = d_k = d_model/h）
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # ── 步骤 ⑤ + ⑥：输出投影 + 残差连接 + LayerNorm ──
        output = self.linear(context)  # [B, len_q, d_model]

        # 残差连接的意义：
        #   假设注意力层什么都没学到（权重接近 0），output ≈ 0
        #   → output + residual ≈ residual，信息不会丢失
        #   这确保了"增加层数不会让模型变差"，深层网络至少能达到浅层的效果
        return self.layer_norm(output + residual), attn


# ================================================================================
# 8. 逐位置前馈网络（Position-wise Feed-Forward Network）
# ================================================================================

class PoswiseFeedForwardNet(nn.Module):
    """
    逐位置前馈网络（论文第 3.3 节）。

    公式:
        FFN(x) = ReLU( x · W₁ + b₁ ) · W₂ + b₂

    结构:
        d_model ──Linear(→d_ff)──→ ReLU ──→ Linear(→d_model) ──→ 残差+LayerNorm

    ╔══════════════════════════════════════════════════════════════╗
    ║  为什么需要 FFN？                                            ║
    ║  注意力层本质上是"加权求和"，虽然能灵活捕获 token 间的关系，  ║
    ║  但它是一个线性操作（对 Value 的加权组合）。                  ║
    ║  FFN 为每个位置引入了非线性变换（ReLU），让模型能学习         ║
    ║  更复杂的特征表示。                                          ║
    ║                                                              ║
    ║  为什么先升维再降维？                                        ║
    ║  d_ff = 2048 是 d_model = 512 的 4 倍。                     ║
    ║  先升到高维→ReLU→再降维，类似 SVM 的核技巧：                ║
    ║  高维空间让特征更容易线性可分，降维后保留关键信息。           ║
    ╚══════════════════════════════════════════════════════════════╝

    这里用 Conv1d(kernel_size=1) 实现而非 Linear：
        - 数学上等价：kernel_size=1 的 Conv1d 就是对每个位置做矩阵乘法
        - Conv1d 天然在 seq 维度上独立操作，不需要 reshape 来 reshape 去
        - 这是原作者的个人偏好，使用 nn.Linear + transpose 也是完全等效的
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # kernel_size=1: 对每个位置独立做线性变换（position-wise）
        # in_channels=d_model, out_channels=d_ff: 第一层升维，512 → 2048
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # 第二层降维，2048 → 512，恢复原始维度以进行残差连接
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        参数:
            inputs: [batch_size, len_q, d_model] — 注意力层的输出

        返回:
            output: [batch_size, len_q, d_model] — FFN 输出（含残差+LayerNorm）

        维度变换过程:
            inputs:        [B, len_q, d_model]
            transpose(1,2): [B, d_model, len_q]   ← Conv1d 期望 (B, C, L) 格式
            conv1:         [B, d_ff,   len_q]     ← ReLU 非线性激活
            conv2:         [B, d_model, len_q]    ← 降维
            transpose(1,2): [B, len_q, d_model]   ← 恢复原始形状
            + residual +
            LayerNorm:     [B, len_q, d_model]    ← 最终输出
        """
        residual = inputs  # 保存用于残差连接

        # Conv1d 的输入格式为 (batch_size, channels, sequence_length)
        # transpose(1,2): 将 channel(d_model) 维换到第二个位置
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # 经过 conv1: [B, d_ff, len_q]，ReLU 提供非线性

        output = self.conv2(output).transpose(1, 2)
        # 经过 conv2 + transpose: [B, len_q, d_model]，恢复形状

        # 残差连接 + 层归一化
        return self.layer_norm(output + residual)


# ================================================================================
# 4. Padding 掩码（Pad Mask）
# ================================================================================

def get_attn_pad_mask(seq_q, seq_k):
    """
    生成 Padding 掩码——标记 Key 序列中哪些位置是 <PAD>。

    ╔══════════════════════════════════════════════════════════════╗
    ║  问题：一个 batch 中的句子长度不同，需要 padding 到统一长度。  ║
    ║  <PAD> 只是占位符，没有实际语义，在注意力中不应被关注。       ║
    ║                                                              ║
    ║  这个函数生成一个 mask，标记出 Key 中的 PAD 位置，            ║
    ║  后续会被填充为 -∞，Softmax 后权重 ≈ 0。                     ║
    ╚══════════════════════════════════════════════════════════════╝

    关键设计：只 mask Key 的 PAD，不 mask Query 的 PAD。为什么？

        注意力权重由 Q·K^T 得到，然后沿 Key 维度做 Softmax：
           - Key 是 PAD：它的 Value 会被注入到上下文表示中，必须屏蔽
           - Query 是 PAD：它只产生一个权重分布，不影响其他 Query，伤害较小
        所以只需要标记 Key 中哪些是 PAD 即可。这也是论文的默认做法。

    参数:
        seq_q: [batch_size, len_q] — Query 序列（仅用于确定 mask 形状）
        seq_k: [batch_size, len_k] — Key   序列（实际检查其中哪些是 PAD）

    返回:
        pad_attn_mask: [batch_size, len_q, len_k]
            True  = 该位置对应 Key 中的 PAD，需要遮蔽
            False = 可以正常关注

    可视化:
        Key = [2, 5, 0, 0]  （0 = PAD）
        → mask 为 [F, F, T, T]（最后两列为 True，因为它们是 PAD）
        → expand 到所有 Query 行：每一行都是 [F, F, T, T]
        → 无论哪个 Query，都不会关注最后两个位置
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # seq_k.eq(0): 标记 Key 中值为 0 的位置（约定 0 = PAD）
    # .unsqueeze(1): [batch_size, len_k] → [batch_size, 1, len_k]
    #   升维是为了后面的 .expand：让每一行（每个 Query）共享同一份 mask
    # 注意：这里使用了 .data（旧式写法），直接 seq_k.eq(0) 即可
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]

    # .expand(batch_size, len_q, len_k):
    #   将第 1 维从 1 扩展到 len_q，所有 len_q 行共享相同的 PAD 标记
    #   这是一个 view 操作，不复制数据，零额外内存开销
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


# ================================================================================
# 3. 位置编码（Positional Encoding）—— 给 Transformer 注入"顺序"概念
# ================================================================================

class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码（论文第 3.5 节）。

    ╔══════════════════════════════════════════════════════════════════╗
    ║  核心问题：Transformer 没有 RNN 的循环结构，也没有 CNN 的滑动窗口， ║
    ║  自注意力是"排列不变"的（Permutation-Invariant）——               ║
    ║  如果把句子打乱，注意力计算出的结果只是对应位置交换了而已。         ║
    ║  因此必须显式地告诉模型每个 token 在序列中的位置。                 ║
    ╚══════════════════════════════════════════════════════════════════╝

    编码公式（对每个位置 pos 和维度 i）：
        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )    ← 偶数维用 sin
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )    ← 奇数维用 cos

    为什么用正弦和余弦？
        1. 相对位置可学习: 对于任意固定偏移 k，
           PE(pos+k) 可以表示为 PE(pos) 的线性函数，模型可能学会"相对位置"
        2. 值域稳定: [-1, 1]，与 Embedding 的量级匹配
        3. 多尺度: 不同维度的正弦波频率不同（波长从 2π 到 ~10000·2π），
           低维度（高频）捕捉局部位置差异，高维度（低频）捕捉全局位置
        4. 可外推: 不需要训练参数，可以对训练时未见过的位置长度进行编码
           但在实践中，外推效果有限，所以后来出现了 RoPE、ALiBi 等改进方案

    参数:
        d_model: 词嵌入维度（位置编码维度与之相同）
        dropout: 位置编码后的 Dropout 概率（正则化，防止过拟合位置信息）
        max_len: 预计算的最大序列长度（默认 5000，对应论文中 max_len=512 是保守的）
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ── 预计算位置编码矩阵 ──
        # pe 最终形状: [max_len, 1, d_model]
        #   维度 0: 位置索引 (0, 1, 2, ..., max_len-1)
        #   维度 1: 大小为 1（用于广播到 batch_size 维度）
        #   维度 2: 编码维度 (0, 1, ..., d_model-1)

        # 初始化全零矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1]
        #   列向量 [[0], [1], [2], ..., [max_len-1]]
        #   每个位置的 pos 值
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: [d_model/2]  → 公式中的 1 / 10000^(2i/d_model)
        #   使用 exp 和 log 代替直接幂运算：
        #     exp( -2i * log(10000) / d_model ) = exp( 2i * (-log(10000)/d_model) )
        #   这样写更简洁，且避免了中间结果的数值溢出风险
        #   torch.arange(0, d_model, 2): 只计算偶数索引对应的除数 (0, 2, 4, ..., d_model-2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维度（0, 2, 4, ...）填充 sin
        # pe[:, 0::2]: 所有行、从列 0 开始、步长 2 → 即第 0, 2, 4, ..., d_model-2 列
        # position * div_term: [max_len, 1] * [d_model/2] → [max_len, d_model/2]（广播乘法）
        #   → 第 i 列包含 pos / 10000^(2i/d_model) 的值
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度（1, 3, 5, ...）填充 cos
        # pe[:, 1::2]: 所有行、从列 1 开始、步长 2 → 即第 1, 3, 5, ..., d_model-1 列
        pe[:, 1::2] = torch.cos(position * div_term)

        # ── 调整形状 ──
        # unsqueeze(0): [max_len, d_model] → [1, max_len, d_model]
        # transpose(0,1): [1, max_len, d_model] → [max_len, 1, d_model]
        # 第 1 维为 1 是为了在 forward 中与 [seq_len, batch_size, d_model]
        # 的 x 做广播加法（batch 维自动扩展）
        pe = pe.unsqueeze(0).transpose(0, 1)

        # register_buffer: 注册为"缓冲区"（Buffer）
        #   - 会随模型一起保存和加载（在 state_dict 中）
        #   - 会随 model.to(device) 自动迁移到 GPU
        #   - 但不是可训练参数，optimizer.step() 不会更新它
        #   - 本质上是一个不可训练的持久化张量
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: [seq_len, batch_size, d_model] — 词嵌入的输出
               注意：seq_len 在第一维！这与 Transformer 其他地方
               的 [batch_size, seq_len, d_model] 不同，是原作者
               的风格选择，可能受当时 Seq2Seq 代码习惯的影响。

        返回:
            output: [seq_len, batch_size, d_model] — 加了位置编码的表示

        广播加法:
            self.pe[:seq_len, :] → [seq_len, 1, d_model]
            x                     → [seq_len, batch_size, d_model]
            加法时第 1 维自动广播，每个 batch 加上相同的位置编码
        """
        # self.pe[:x.size(0), :]: 截取前 seq_len 个位置的编码 → [seq_len, 1, d_model]
        # 与 x [seq_len, batch_size, d_model] 相加时自动广播
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ================================================================================
# 5. 编码器层（Encoder Layer）—— 编码器的一层
# ================================================================================

class EncoderLayer(nn.Module):
    """
    编码器的单层（论文第 3.1 节），包含两个子层：

        子层 1: 多头自注意力（Multi-Head Self-Attention）
            Q = K = V = 输入序列本身
            → 让序列中每个 token 都能看到序列中所有其他 token
            → 建立全局的上下文依赖关系

        子层 2: 逐位置前馈网络（Position-wise FFN）
            对每个 token 的表示独立做非线性变换
            → 增强单个 token 的特征表达能力

    每个子层后都有残差连接 + LayerNorm（Post-LN 风格）：
        output = LayerNorm( x + Sublayer(x) )

    后来研究发现 Pre-LN（先 LN 再 Sublayer）训练更稳定：
        output = x + Sublayer( LayerNorm(x) )
    但此代码遵循原始论文的 Post-LN 设计。
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # 自注意力：Q, K, V 都来自同一个输入序列
        self.enc_self_attn = MultiHeadAttention()
        # 前馈网络：对每个位置做非线性变换
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        参数:
            enc_inputs: [batch_size, src_len, d_model]
                当前层的输入表示（第一层是 Embedding+PE 的输出）

            enc_self_attn_mask: [batch_size, src_len, src_len]
                Padding 掩码，标记 Key 中的 PAD 位置

        返回:
            enc_outputs: [batch_size, src_len, d_model] — 编码后的序列表示
            attn: [batch_size, n_heads, src_len, src_len] — 自注意力权重（所有头）

        数据流:
            enc_inputs
              │
              ├─→ MultiHeadAttention(Q=enc_inputs, K=enc_inputs, V=enc_inputs)
              │     + residual + LayerNorm  → 每个 token 融合了全局上下文信息
              │
              └─→ PoswiseFeedForwardNet
                      + residual + LayerNorm  → 每个 token 的非线性特征变换
        """
        # ── 子层 1：多头自注意力 ──
        # 自注意力的核心：Q=K=V，模型自己决定关注序列中的哪些位置
        # 例如在 "The cat sat on the mat" 中，"sat" 可能会高度关注 "cat"（主语）和 "mat"（宾语）
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        # ── 子层 2：前馈网络 ──
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


# ================================================================================
# 2. 编码器（Encoder）—— 完整编码器
# ================================================================================

class Encoder(nn.Module):
    """
    Transformer 编码器（论文第 3.1 节），将源语言序列编码为上下文表示。

    结构（自底向上）:
        Input: [batch_size, src_len] — 源语言 token ID 序列
            │
            ▼
        ┌──────────────────────────┐
        │ 1. Token Embedding       │  将 token ID 映射为稠密向量 [src_vocab_size → d_model]
        │    src_emb (nn.Embedding)│  查表操作，每个 token 对应一个 d_model 维的可学习向量
        └──────────────────────────┘
            │  [batch_size, src_len, d_model]
            ▼
        ┌──────────────────────────┐
        │ 2. Positional Encoding   │  加上正弦/余弦位置编码
        │    pos_emb               │  让模型知道 token 的位置信息
        └──────────────────────────┘
            │  [batch_size, src_len, d_model]
            ▼
        ┌──────────────────────────┐
        │ 3. EncoderLayer × 6      │  堆叠 6 层编码器
        │    Self-Attn + FFN       │  逐层提取越来越抽象的语义特征
        └──────────────────────────┘
            │
            ▼
        Output: [batch_size, src_len, d_model]
        编码器输出会被解码器的 Cross-Attention 用做 K 和 V

    为什么堆叠多层？
        低层 EncoderLayer 捕获局部语法特征（词性、短语结构），
        高层 EncoderLayer 捕获全局语义特征（指代、语义角色）。
        多层堆叠使模型能逐步建立从浅到深的语言理解。
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # ── 源语言词嵌入矩阵 ──
        # 大小: [src_vocab_size, d_model]
        # 本质是一个查找表：给定 token ID，返回对应的 d_model 维可学习向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model)

        # ── 位置编码 ──
        # 固定的正弦/余弦编码（不可学习），与可学习的 Embedding 相加
        self.pos_emb = PositionalEncoding(d_model)

        # ── 堆叠 n_layers 个 EncoderLayer ──
        # ModuleList 是 PyTorch 的"模块列表容器"，确保其中的子模块
        # 被正确追踪（parameters、to(device)、state_dict 等都能正常工作）。
        # 不能用普通的 Python list，那样 PyTorch 不会追踪子模块的参数。
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        参数:
            enc_inputs: [batch_size, src_len] — token ID 序列，如 [1, 2, 3, 4, 0]

        返回:
            enc_outputs:  [batch_size, src_len, d_model] — 最后一层的编码输出
            enc_self_attns: list of [batch_size, n_heads, src_len, src_len]
                            每层编码器的自注意力权重（用于可视化分析注意力模式）
        """
        # ── 步骤 1：词嵌入 ──
        # [batch_size, src_len] → [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)

        # ── 步骤 2：位置编码 ──
        # PositionalEncoding.forward 期望输入形状为 [seq_len, batch_size, d_model]
        #   ① transpose(0,1): [batch_size, src_len, d_model] → [src_len, batch_size, d_model]
        #   ② pos_emb 内部:   加上位置编码 PE [src_len, 1, d_model]
        #   ③ transpose(0,1): [src_len, batch_size, d_model] → [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # ── 步骤 3：生成 Padding 掩码 ──
        # 对编码器自注意力使用：标记源语言序列中哪些位置是 PAD
        # seq_k = enc_inputs → 标记 Key 中的 PAD，防止自注意力关注 PAD token
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # [batch_size, src_len, src_len]

        # ── 步骤 4：逐层传递 ──
        # 每一层的输出作为下一层的输入（标准的"前馈堆叠"）
        enc_self_attns = []  # 收集每层的注意力权重用于分析
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


# ================================================================================
# 10. 解码器层（Decoder Layer）—— 解码器的一层
# ================================================================================

class DecoderLayer(nn.Module):
    """
    解码器的单层（论文第 3.1 节），包含三个子层（比编码器多一个交叉注意力）：

        子层 1: 带掩码的多头自注意力（Masked Multi-Head Self-Attention）
            Q = K = V = 解码器输入
            使用 causal mask 防止看到未来 token
            → 建立目标语言的因果上下文依赖

        子层 2: 多头交叉注意力（Multi-Head Cross-Attention）
            Q = 解码器当前表示 (dec_outputs)
            K = V = 编码器输出 (enc_outputs)
            → 让解码器"查阅"源语言信息，实现翻译对齐

        子层 3: 逐位置前馈网络（Position-wise FFN）
            → 非线性特征变换

    ╔══════════════════════════════════════════════════════════════════╗
    ║  Self-Attention vs Cross-Attention 的直觉理解：                 ║
    ║                                                                  ║
    ║  Self-Attn:  "我在生成什么？已经生成了哪些词？                   ║
    ║               当前应该关注已生成序列的哪个部分？"                 ║
    ║                                                                  ║
    ║  Cross-Attn: "源语言说了什么？翻译到当前位置时，                 ║
    ║               源语言中哪些词最相关？"                             ║
    ║                                                                  ║
    ║  举例：翻译 "ich mochte ein bier" → "i want a beer"              ║
    ║  生成 "beer" 时，Cross-Attn 会高度关注源语言的 "bier"            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        # 带掩码的自注意力（解码器内部）：Q=K=V=解码器输入
        self.dec_self_attn = MultiHeadAttention()
        # 交叉注意力（解码器→编码器）：Q=解码器,K=V=编码器
        self.dec_enc_attn = MultiHeadAttention()
        # 前馈网络
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        参数:
            dec_inputs:  [batch_size, tgt_len, d_model] — 解码器输入表示
            enc_outputs: [batch_size, src_len, d_model] — 编码器输出（K, V 的来源）
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                — 自注意力掩码（PAD mask + subsequent mask 的组合）
            dec_enc_attn_mask:  [batch_size, tgt_len, src_len]
                — 交叉注意力掩码（标记编码器输出中的 PAD）

        返回:
            dec_outputs:   [batch_size, tgt_len, d_model] — 解码后表示
            dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len] — 自注意力权重
            dec_enc_attn:  [batch_size, n_heads, tgt_len, src_len] — 交叉注意力权重
        """
        # ── 子层 1：带掩码的自注意力 ──
        # 解码器看自己的输入，但由于 causal mask 的存在，
        # 位置 i 只能看到位置 0..i，看不到 i+1..len-1
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)

        # ── 子层 2：交叉注意力 ──
        # Q = 解码器当前的表示（"我现在在生成什么"）
        # K = V = 编码器输出（"源语言说了什么"）
        # 这是"翻译对齐"发生的核心位置：模型学习将目标语言词与源语言词对应起来
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                       dec_enc_attn_mask)

        # ── 子层 3：前馈网络 ──
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


# ================================================================================
# 9. 解码器（Decoder）—— 完整解码器
# ================================================================================

class Decoder(nn.Module):
    """
    Transformer 解码器（论文第 3.1 节），根据编码器输出和已生成的前缀，
    逐个预测目标语言的下一个 token。

    结构（自底向上）:
        Input: [batch_size, tgt_len] — 目标语言 token ID（右移一位，S 开头）
            │
            ▼
        ┌──────────────────────────┐
        │ 1. Token Embedding       │  目标语言词嵌入
        │    tgt_emb (nn.Embedding)│  [tgt_vocab_size → d_model]
        └──────────────────────────┘
            │  [batch_size, tgt_len, d_model]
            ▼
        ┌──────────────────────────┐
        │ 2. Positional Encoding   │  位置编码
        │    pos_emb               │
        └──────────────────────────┘
            │  [batch_size, tgt_len, d_model]
            ▼
        ┌──────────────────────────┐
        │ 3. DecoderLayer × 6      │  堆叠 6 层解码器
        │    Self-Attn (masked)    │  + 交叉注意力 + FFN
        │    Cross-Attn            │
        │    FFN                   │
        └──────────────────────────┘
            │
            ▼
        Output: [batch_size, tgt_len, d_model]
        输出会传给 Linear Projection 层映射到词表大小

    Teacher Forcing 训练 vs 自回归推理:
        ┌──────────────┬──────────────────────────────────────┐
        │ 训练          │ 一次性输入整个目标序列（右移），       │
        │ (Teacher      │ 用 subsequent mask 防止看到未来。    │
        │  Forcing)     │ 一次 forward 并行计算所有位置的 loss。│
        ├──────────────┼──────────────────────────────────────┤
        │ 推理          │ 自回归生成：起始 token=<S>，          │
        │ (Auto-        │ 每次 forward 预测一个 token，        │
        │  regressive)  │ 拼接到序列末尾，重复直到输出 <E>     │
        │               │ 或达到最大长度。                     │
        │               │ 需要 O(n) 次 forward，比训练慢得多。  │
        └──────────────┴──────────────────────────────────────┘
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # 目标语言词嵌入矩阵 [tgt_vocab_size, d_model]
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码
        self.pos_emb = PositionalEncoding(d_model)
        # 堆叠 n_layers 个 DecoderLayer
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        参数:
            dec_inputs:  [batch_size, tgt_len] — 解码器输入 token ID（以 S 开头）
            enc_inputs:  [batch_size, src_len] — 编码器输入 token ID（用于生成 PAD mask）
            enc_outputs: [batch_size, src_len, d_model] — 编码器输出（Cross-Attn 的 K,V）

        返回:
            dec_outputs:   [batch_size, tgt_len, d_model] — 解码器最终输出
            dec_self_attns: list — 每层自注意力权重 [B, n_heads, tgt_len, tgt_len]
            dec_enc_attns:  list — 每层交叉注意力权重 [B, n_heads, tgt_len, src_len]

        掩码构造细节:
            ┌─────────────────────────┬───────────────┬──────────────────────┐
            │ 掩码                    │ 来源           │ 作用                 │
            ├─────────────────────────┼───────────────┼──────────────────────┤
            │ dec_self_attn_pad_mask  │ get_attn_pad  │ 屏蔽 dec_input 的PAD │
            │ dec_self_attn_sub_mask  │ get_subsequent│ 防止看到未来 token   │
            │ dec_self_attn_mask      │ 上面两者相加   │ 最终 SA 掩码         │
            │ dec_enc_attn_mask       │ get_attn_pad  │ 屏蔽 enc 输出的 PAD  │
            └─────────────────────────┴───────────────┴──────────────────────┘
        """
        # ── 步骤 1：词嵌入 ──
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]

        # ── 步骤 2：位置编码 ──
        # 同样需要 transpose(0,1) 适配 PositionalEncoding 的输入格式
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        # ── 步骤 3：构造自注意力掩码 ──
        # 3a. PAD 掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # 3b. Causal 掩码（防止看到未来）
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # 3c. 合并两个掩码
        # pad_mask + sub_mask: 每个元素为 0(都不可见), 1(被一个mask遮蔽), 2(被两个mask遮蔽)
        # .gt(0): >0 即 True → "只要被任一 mask 标记，就遮蔽该位置"
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # [batch_size, tgt_len, tgt_len]

        # ── 步骤 4：构造交叉注意力掩码 ──
        # 只遮蔽编码器输出中的 PAD 位置
        # Q=dec_inputs, K=enc_inputs → mask 形状 [batch_size, tgt_len, src_len]
        # 解码器的 PAD 不需要在交叉注意力中被 mask（前面解释过）
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        # ── 步骤 5：逐层传递 ──
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


# ================================================================================
# 1. Transformer（顶层完整模型）
# ================================================================================

class Transformer(nn.Module):
    """
    完整的 Transformer 模型（论文图 1 的完整实现）。

    架构概览:
        ┌──────────────────────────────────────────────────┐
        │                   Transformer                      │
        │                                                    │
        │  Encoder                    Decoder                │
        │  ┌──────────┐             ┌──────────┐            │
        │  │ Embedding │             │ Embedding │            │
        │  │   + PE    │             │   + PE    │            │
        │  │     │     │             │     │     │            │
        │  │  N×Layer  │             │  N×Layer  │            │
        │  │  Self-Attn│             │ Self-Attn │            │
        │  │    FFN    │             │Cross-Attn │◄───────┐  │
        │  └────┬─────┘             │    FFN    │        │  │
        │       │                   └────┬─────┘        │  │
        │       │        K, V            │              │  │
        │       └────────────────────────┘              │  │
        │                                               │  │
        │                          ┌──────────┐         │  │
        │                          │  Linear   │         │  │
        │                          │ Projection│         │  │
        │                          └────┬─────┘         │  │
        │                               │               │  │
        │                          Softmax (in loss)    │  │
        └──────────────────────────────────────────────────┘

    训练目标:
        给定源语言句子 (enc_inputs) 和目标语言前缀 (dec_inputs)，
        最小化模型预测与真实标签 (target_batch) 之间的交叉熵损失。

    参数量估算（论文 Base 配置）:
        d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab~37k
        → 总参数量约 65M
    """
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()                                          # 编码器
        self.decoder = Decoder()                                          # 解码器
        # 输出投影层：将 d_model 维的表示映射到目标词表大小
        # bias=False: 输出层无偏置（与 Embedding 共享权重的常见预处理步骤）
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        参数:
            enc_inputs: [batch_size, src_len] — 源语言 token ID
            dec_inputs: [batch_size, tgt_len] — 目标语言 token ID（shifted right）

        返回:
            dec_logits: [batch_size * tgt_len, tgt_vocab_size]
                每个位置对词表的原始分数（未归一化 logits）
                形状被展平以匹配 CrossEntropyLoss 的输入格式 (N, C)
            enc_self_attns: list — 编码器各层自注意力权重
            dec_self_attns: list — 解码器各层自注意力权重
            dec_enc_attns:  list — 解码器各层交叉注意力权重
                (可用于可视化注意力模式，分析"谁在关注谁")
        """
        # ── 编码阶段 ──
        # enc_outputs:  [batch_size, src_len, d_model]
        # enc_self_attns: list of [batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # ── 解码阶段 ──
        # dec_outputs:    [batch_size, tgt_len, d_model]
        # dec_self_attns: list of [batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attns:  list of [batch_size, n_heads, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # ── 输出投影 ──
        # 将 d_model 维的密集表示映射到 tgt_vocab_size 维的 logits
        # [batch_size, tgt_len, d_model] → [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)

        # ── 展平用于 CrossEntropyLoss ──
        # CrossEntropyLoss 期望:
        #   input:  (N, C) 其中 N=batch×seq, C=num_classes
        #   target: (N,)   每个样本的类别标签
        # .view(-1, tgt_vocab_size): 合并 batch 和 seq 维度 → [batch*tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


# ================================================================================
# 训练演示：德语 → 英语（迷你数据集）
# ================================================================================

if __name__ == '__main__':
    """
    在一个迷你句子对上训练 Transformer，完整演示数据准备、前向传播、损失计算和反向传播。

    翻译任务: 德语 "ich mochte ein bier" → 英语 "i want a beer"

    特殊 Token 约定:
        P (PAD=0):   填充符，将不等长句子填充到统一长度
        S (Start=5): 序列起始符，解码器输入的第一个 token
        E (End=6):   序列结束符，告诉解码器何时停止生成

    关键约定: <PAD> 的索引必须为 0！
        get_attn_pad_mask 函数通过 seq_k.eq(0) 来检测 PAD 位置，
        如果 PAD 的索引不为 0，该检测会失败。
    """

    # ── 训练数据 ──
    # 三元组: (德语源句, 英语解码器输入, 英语目标标签)
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # ── 构建词表 ──
    # 源语言词表（德语）：PAD 必须索引为 0
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)  # 5

    # 目标语言词表（英语）：PAD 必须索引为 0
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)  # 7

    # 序列长度
    src_len = 5  # "ich mochte ein bier P" → 5 个 token
    tgt_len = 5  # "S i want a beer" → 5 个 token

    # ── 模型超参数（与论文 Transformer Base 一致）──
    d_model = 512   # 词嵌入维度 & 所有子层输出维度
    d_ff = 2048     # FFN 隐藏层维度（d_model 的 4 倍）
    d_k = d_v = 64  # 每个注意力头的 Key/Query/Value 维度
                     # d_k = d_model / n_heads = 512 / 8 = 64
    n_layers = 6    # Encoder 和 Decoder 各自的堆叠层数
    n_heads = 8     # 多头注意力的头数

    # ── 实例化模型 ──
    model = Transformer()

    # ── 损失函数 ──
    # CrossEntropyLoss 内部先对 logits 做 Softmax，再计算负对数似然
    # 等价于: loss = -log(softmax(logits)[target_class])
    criterion = nn.CrossEntropyLoss()

    # ── 优化器 ──
    # Adam: 自适应学习率的优化器，结合了 Momentum 和 RMSprop 的优点
    # lr=0.001 是演示用的学习率（论文中使用的是带 warmup 的动态学习率）
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ── 准备数据 ──
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # ── 训练循环 ──
    for epoch in range(20):
        # 清空上一轮的梯度（PyTorch 默认累积梯度）
        optimizer.zero_grad()

        # 前向传播
        # outputs: [batch_size * tgt_len, tgt_vocab_size] = [5, 7]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

        # 计算损失
        # target_batch: [1, 5] → .view(-1) → [5]
        # 每个位置的预测与真实标签比较
        loss = criterion(outputs, target_batch.contiguous().view(-1))

        # 打印训练进度
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        # 反向传播：计算所有参数的梯度
        loss.backward()

        # 参数更新：根据梯度调整模型参数
        optimizer.step()

    # ── 训练结果说明 ──
    # 在单样本上训练 20 epoch 后，loss 会很快下降到接近 0。
    # 这证明 Transformer 有能力"记住"训练样本，但在真实翻译任务中
    # 需要大规模的平行语料（如 WMT 数据集）和更长的训练时间才能泛化。

    print("\n训练完成！")
    print(f"最终 Loss: {loss.item():.6f}")
