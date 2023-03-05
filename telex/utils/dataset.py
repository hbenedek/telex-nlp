"Module for datasets"

import string
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from params import VOCAB, TrainingParams
from telex.utils.io import load_json


# TODO deal with weird characters, creat splitter
class CharDataset(Dataset):
    """Dataset for character-level language modeling."""

    def __init__(self, file_name: Path, vocab: str, block_size: int, type_: str, split: list):
        self.data_dictionary = load_json(file_name)
        self.block_size = block_size
        self.set = self.data_dictionary["content"]
        train_split, val_split, test_split = split
        val_split = train_split + val_split
        test_split = val_split + test_split
        assert type_ in ["train", "val", "test"]
        # assert train_split + val_split + test_split == 1
        if type_ == "train":
            self.text = "".join(self.set[: int(len(self.set) * train_split)])
        elif type_ == "val":
            self.text = "".join(self.set[int(len(self.set) * train_split) : int(len(self.set) * val_split)])
        elif type_ == "test":
            self.text = "".join(self.set[int(len(self.set) * val_split) :])
        self.vocab = vocab
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}  # inverse mapping

        self.encoded = self.encode(self.text)

    def __len__(self):
        return len(self.text) - self.block_size

    def get_vocab_size(self):
        return len(self.vocab) + 1  # all the possible characters and special 0 token

    def decode(self, idx: torch.Tensor) -> str:
        """Decode indices into text."""
        return "".join(self.itos.get(i, "[UNK]") for i in idx.tolist())

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into indices."""
        return torch.tensor([self.stoi.get(ch, 0) for ch in text], dtype=torch.long)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.encoded[idx : idx + self.block_size]
        y = self.encoded[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_loaders(input_file: Path):
    train_dataset = CharDataset(input_file, VOCAB, TrainingParams.BLOCK_SIZE, type_="train", split=TrainingParams.SPLIT)
    val_dataset = CharDataset(input_file, VOCAB, TrainingParams.BLOCK_SIZE, type_="val", split=TrainingParams.SPLIT)
    train_loader = DataLoader(train_dataset, batch_size=TrainingParams.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingParams.BATCH_SIZE, shuffle=True)
    return train_loader, val_loader


# if __name__ == "__main__":
#    dataset = CharDataset(Path("data/preprocessed/telex.json"), 10)
#    for x, y in DataLoader(dataset, batch_size=2):
#        print(x)
#        print(y)
#        print(x.shape, y.shape)
#        break
#    print(dataset.decode(dataset.encoded[:10]))
