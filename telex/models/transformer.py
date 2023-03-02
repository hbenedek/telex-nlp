from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel

from params import BLOCK_SIZE


class Head(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k

        self.Q = nn.Linear(d_model, d_k, bias=False)
        self.K = nn.Linear(d_model, d_k, bias=False)
        self.V = nn.Linear(d_model, d_v, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (batch_size, seq_len, d_model)
        Q = self.Q(x)  # (batch_size, seq_len, d_k)
        K = self.K(x)  # (batch_size, seq_len, d_k)
        V = self.V(x)  # (batch_size, seq_len, d_v)

        QK = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.d_k))  # (batch_size, seq_len, seq_len)
        QK = QK.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        QK = F.softmax(QK, dim=2)  # (batch_size, seq_len, seq_len)
        return torch.matmul(QK, V)  # (batch_size, seq_len, d_v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.heads = nn.ModuleList([Head(self.d_model, self.d_k, self.d_k) for _ in range(self.heads)])
        self.W = nn.Linear(self.heads * self.d_v, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([att(x) for att in self.heads], dim=2)
        return self.W(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)  # (batch_size, seq_len, d_ff)
        x = F.relu(x)  # (batch_size, seq_len, d_ff)
        x = self.fc2(x)  # (batch_size, seq_len, d_model)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        self.attention = MultiHeadAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.attention(x)
        x = x + x2
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + x2
        x = self.norm2(x)
        return x


class GPT(BaseModel):
    """ """

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float, num_layers: int, block_size: int):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_layers = num_layers
        self.block_size = block_size
        self.positional_encoding = nn.Embedding(self.block_size, d_model)
        self.token_embedding = nn.Embedding(self.block_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, self.block_size, bias=False)

    def forward(self, idx: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        x = self.token_embedding(idx) + self.positional_encoding(pos)
        for block in self.blocks:
            x = block(x)

        logits = self.fc(x)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        return logits, loss


if __name__ == "__main__":
    model = Head()
    input = torch.randn(1, 2, 3)
    output = model(input)
    print(output.shape)
