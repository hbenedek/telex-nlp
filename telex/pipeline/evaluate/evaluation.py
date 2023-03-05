"""Module for model evalutation."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.functional import F
from torch.utils.data import DataLoader

from params import VOCAB, TrainingParams
from telex.models.base_model import BaseModel
from telex.models.bigram import BigramLM
from telex.utils.dataset import CharDataset


@torch.no_grad()
def perplexity(model: BaseModel, loader: DataLoader, subset: Optional[int] = None) -> float:
    """Calculate character-wise perplexity of the model."""
    total_log_likelihood = 0.0
    total_num_chars = 0.0

    for k, (idx, targets) in enumerate(loader):
        logits, log_probs = model(idx, targets)
        targets = targets.view(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        log_likelihood = torch.sum(log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1))
        total_log_likelihood += log_likelihood.item()
        total_num_chars += targets.numel()
        if subset is not None:
            if k > subset:
                break

    perplexity = np.exp(-total_log_likelihood / total_num_chars)
    return perplexity


def plot_learning_curve(model: BaseModel, output_folder: Path) -> None:
    """Plot learning curve."""
    fig, ax = plt.subplots()
    ax.set(ylabel="Loss", title="Learning curve")
    if TrainingParams.EPOCHS > 1:
        ax.plot(model.epoch_train_losses, label="train")
        ax.plot(model.epoch_val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.legend(["train", "val"], loc="upper left")
    else:
        ax.plot(model.batch_train_losses, label="train")
        ax.set_xlabel("Batch")
        ax.legend(loc="upper left")
    fig.savefig(output_folder / "learning_curve.png")


def main(model_file: Path = typer.Option(...), input_file: Path = typer.Option(...)) -> None:
    dataset = CharDataset(input_file, VOCAB, TrainingParams.BLOCK_SIZE, type_="test", split=TrainingParams.SPLIT)
    test_loader = DataLoader(dataset, batch_size=TrainingParams.BATCH_SIZE, shuffle=True)
    model_kwargs = {"vocab": VOCAB, "block_size": TrainingParams.BLOCK_SIZE}
    model = BigramLM.load_for_prediction(model_file, model_kwargs)
    # model = BigramLM.create_for_training(model_file, model_kwargs)
    plot_learning_curve(model, model_file.parent)
    p = perplexity(model, test_loader, subset=1000)
    print(f"Perplexity: {p}")


if __name__ == "__main__":
    typer.run(main)
