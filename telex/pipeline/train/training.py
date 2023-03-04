"""Module for training the model."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from params import BATCH_SIZE, BLOCK_SIZE, SEED, VOCAB
from telex.models.bigram import BigramLM
from telex.utils.dataset import CharDataset


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(input_file: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Main function, print hello message."""
    seed_everything(SEED)
    output_folder.mkdir(parents=True, exist_ok=True)
    dataset = CharDataset(input_file, VOCAB, BLOCK_SIZE, type_="train", split=0.9)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model_kwargs = {"vocab": VOCAB, "block_size": BLOCK_SIZE}
    model = BigramLM.create_for_training(output_folder, model_kwargs)
    model.train(train_loader, num_epochs=1)
    model.save()


if __name__ == "__main__":
    typer.run(main)
