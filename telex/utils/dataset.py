"Module for datasets"

import string
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from telex.utils.io import load_json


# TODO deal with weird characters, creat splitter
class CharDataset(Dataset):
    """Dataset for character-level language modeling."""

    def __init__(self, file_name: Path, vocab: str, block_size: int, type_: str, split: float):
        self.data_dictionary = load_json(file_name)
        self.block_size = block_size
        self.set = self.data_dictionary["content"]
        if type_ == "train":
            self.text = "".join(self.set[: int(len(self.set) * split)])
        if type_ == "test":
            self.text = "".join(self.set[int(len(self.set) * split) :])
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
        return "".join(self.itos[i] for i in idx.tolist())

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into indices."""
        return torch.tensor([self.stoi.get(ch, 0) for ch in text], dtype=torch.long)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.encoded[idx : idx + self.block_size]
        y = self.encoded[idx + 1 : idx + self.block_size + 1]
        return x, y


# if __name__ == "__main__":
#    dataset = CharDataset(Path("data/preprocessed/telex.json"), 10)
#    for x, y in DataLoader(dataset, batch_size=2):
#        print(x)
#        print(y)
#        print(x.shape, y.shape)
#        break
#    print(dataset.decode(dataset.encoded[:10]))
