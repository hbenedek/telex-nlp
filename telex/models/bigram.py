from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from params import LR
from telex.models.base_model import BaseModel


class BigramLM(BaseModel):
    """Bigram language model."""

    def __init__(self, save_dir: Path, vocab: set, block_size: int):
        super().__init__(save_dir, block_size)
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.vocab_size + 1)
        self.optimizer = AdamW(self.parameters(), lr=LR)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.embedding(idx)
        B, T, C = logits.shape  # B = batch size, T = block size, C = vocab size
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


if __name__ == "__main__":
    vocab = {"a", "b", "c"}
    model = BigramLM.create_for_training(save_dir=Path("test"), model_kwargs={"vocab": vocab, "block_size": 10})
    idx = torch.tensor([[0]])
    generate = model.generate(idx, max_new_tokens=10)
    print(generate)
