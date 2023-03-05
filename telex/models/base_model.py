from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from params import TrainingParams


class BaseModel(ABC, nn.Module):
    """Base class for all models."""

    def __init__(self, save_dir: Path, block_size: int) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.block_size = block_size
        self.optimizer = None  # initialize in subclass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_train_losses = []
        self.batch_val_losses = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample: bool = False
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.forward(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def save(self) -> None:
        """Save the model to `save_path` directory."""
        output_file = self.save_dir / "model.pt"
        torch.save(self.state_dict(), output_file)

    @classmethod
    def create_for_training(cls, save_dir: Path, model_kwargs: Optional[dict] = None) -> BaseModel:
        return cls(save_dir, **model_kwargs)

    @classmethod
    def load_for_prediction(cls, model_dir: Path, model_kwargs: Optional[dict] = None) -> BaseModel:
        model_kwargs = model_kwargs or {}
        model = cls(save_dir=model_dir, **model_kwargs)
        model.load_state_dict(torch.load(model_dir))
        return model

    def trainer(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.batch_train_losses = []
            self.epoch_val_losses = []

            self.train()
            for i, batch in enumerate(train_loader):
                batch = [char.to(self.device) for char in batch]
                X, Y = batch
                logits, loss = self.forward(X, Y)
                self.batch_train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if i % 100 == 0:
                    print(i, "batch", loss.item())
                if i > TrainingParams.STEPS:
                    break

            epoch_train_loss = sum(self.batch_train_losses) / len(self.batch_train_losses)
            self.epoch_train_losses.append(epoch_train_loss)

            self.eval()
            for j, batch in enumerate(val_loader):
                batch = [char.to(self.device) for char in batch]
                X, Y = batch
                logits, loss = self.forward(X, Y)
                self.batch_val_losses.append(loss.item())
                if j % 100 == 0:
                    print(j, "batch", loss.item())
                if j > TrainingParams.STEPS:
                    break

            epoch_val_loss = sum(self.batch_val_losses) / len(self.batch_val_losses)
            self.epoch_val_losses.append(epoch_val_loss)
            print(f"Epoch {epoch + 1} train_loss: {epoch_train_loss:.4f} val_loss: {epoch_val_loss:.4f}")

    @abstractmethod
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
