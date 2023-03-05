from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from params import TrainingParams
from telex.models.base_model import BaseModel


class BigramLM(BaseModel):
    """Bigram language model."""

    def __init__(self, save_dir: Path, vocab: set, block_size: int):
        super().__init__(save_dir, block_size)
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.vocab_size + 1)
        self.optimizer = AdamW(self.parameters(), lr=TrainingParams.LR)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.embedding(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape  # B = batch size, T = block size, C = vocab size
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
