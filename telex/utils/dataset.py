"Module for datasets"

import torch
from torch.utils.data import Dataset


# TODO review this part
class CharDataset(Dataset):
    """Dataset for character-level language modeling."""

    def __init__(self, text: str, vocab: set, max_word_length: int):
        self.text = text
        self.max_word_length = max_word_length
        self.vocab = vocab
        self.stoi = {ch: i + 1 for i, ch in enumerate(vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}  # inverse mapping

    def __len__(self):
        return len(self.text)

    def get_vocab_size(self):
        return len(self.chars) + 1  # all the possible characters and special 0 token

    def decode(self, idx: torch.Tensor) -> str:
        """Decode indices into text."""
        return "".join(self.itos[i] for i in idx)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into indices."""
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __getitem__(self, idx: int) -> torch.Tensor:
        word = self.text[idx]
        idx = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1 : 1 + len(idx)] = idx
        y[: len(idx)] = idx
        y[len(idx) + 1 :] = -1  # index -1 will mask the loss at the inactive locations
        return x, y
