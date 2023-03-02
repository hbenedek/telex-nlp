from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel
from torch.utils.data import DataLoader


class BigramLM(BaseModel):
    """Bigram language model."""

    def __init__(self, save_dir: Path, vocab: set, block_size: int):
        super().__init__(save_dir)
        self.vocab_size = len(vocab)
        self.block_size = block_size
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)
        self.losses = []

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.embedding(idx)

        loss = None
        if targets:
            loss = F.cross_entropy(logits, idx)

        return logits, loss


if __name__ == "__main__":
    vocab = {"a", "b", "c"}
    model = BigramLM.create_for_training(save_dir=Path("test"), model_kwargs={"vocab": vocab, "block_size": 10})
    idx = torch.tensor([[0]])
    generate = model.generate(idx, max_new_tokens=10)
    print(generate)
