
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model  # 模型维度（如512）
        self.num_heads = num_heads  # 注意力头数（如8）
        self.d_k = d_model // num_heads  # 每个头的维度（如64）

        # 定义线性变换层（无需偏置）
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model)  # 键变换
        self.W_v = nn.Linear(d_model, d_model)  # 值变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        输入形状：
            Q: (batch_size, num_heads, seq_length, d_k)
            K, V: 同Q
        输出形状： (batch_size, num_heads, seq_length, d_k)
        """
        # 计算注意力分数（Q和K的点积）
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（如填充掩码或未来信息掩码）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重（softmax归一化）
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 对值向量加权求和
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        将输入张量分割为多个头
        输入形状: (batch_size, seq_length, d_model)
        输出形状: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多个头的输出合并回原始形状
        输入形状: (batch_size, num_heads, seq_length, d_k)
        输出形状: (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        输入形状: Q/K/V: (batch_size, seq_length, d_model)
        输出形状: (batch_size, seq_length, d_model)
        """
        # 线性变换并分割多头
        Q = self.split_heads(self.W_q(Q))  # (batch, heads, seq_len, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头并输出变换
        output = self.W_o(self.combine_heads(attn_output))
        return output

if __name__ == '__main__':
    d_model = 512
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    W_q = nn.Linear(d_model,d_model)
    # 随便造一个 Q：batch=1, 3个token, 每个512维
    Q = torch.randn(1, 3, 512)
    print(Q)
    output = W_q(Q)
    print("输入形状:", Q.shape)  # torch.Size([1, 3, 512])
    print("输出形状:", output.shape)  # torch.Size([1, 3, 512])
    print("权重形状:", W_q.weight.shape)  # torch.Size([512, 512])
    print("偏置形状:", W_q.bias.shape)  # torch.Size([512])