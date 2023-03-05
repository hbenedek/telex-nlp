"""Module for training the model."""


import os
import random
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from params import MODEL, VOCAB, BigramParams, TrainingParams, TransformerParams
from telex.models.bigram import BigramLM
from telex.models.transformer import GPT
from telex.utils.dataset import CharDataset, get_loaders


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(input_file: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function, print hello message."""
    seed_everything(TrainingParams.SEED)
    output_folder.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_loaders(input_file)
    assert MODEL in ["bigram", "transformer"]
    if MODEL == "bigram":
        model_kwargs = {"vocab": VOCAB, "block_size": BigramParams.BLOCK_SIZE}
        model = BigramLM.create_for_training(output_folder, model_kwargs)
    elif MODEL == "transformer":
        model_kwargs = {
            "vocab": VOCAB,
            "block_size": TransformerParams.BLOCK_SIZE,
            "num_layers": TransformerParams.NUM_LAYERS,
            "heads": TransformerParams.HEADS,
            "d_model": TransformerParams.D_MODEL,
            "d_ff": TransformerParams.D_FF,
            "dropout": TransformerParams.DROPOUT,
        }
        model = GPT.create_for_training(output_folder, model_kwargs)
    model.trainer(train_loader, val_loader, num_epochs=TrainingParams.EPOCHS)
    model.save()


if __name__ == "__main__":
    typer.run(main)
