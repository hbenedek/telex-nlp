"""Params file."""
from dataclasses import dataclass

MODEL = "transformer"
VOCAB = (
    "AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZaábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz0123456789.,;:?!-–_()[]{}'\"/\\@#$%^&*+=|~` "
)


@dataclass
class ScrapeParams:
    MAX_PAGES = 187
    PER_PAGE = 400
    SLEEP_TIME = 0.5


@dataclass
class TrainingParams:
    SEED = 42
    LR = 10e-4
    SPLIT = [0.9, 0.05, 0.05]
    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    EPOCHS = 1
    STEPS = 5000


@dataclass
class TransformerParams:
    MODEL = "transformer"
    D_MODEL: int = 120
    VOCAB: set = VOCAB
    HEADS: int = 6
    D_FF: int = 512
    DROPOUT: float = 0.1
    NUM_LAYERS: int = 2
    BLOCK_SIZE: int = TrainingParams.BLOCK_SIZE


@dataclass
class BigramParams:
    MODEL = "bigram"
    VOCAB: set = VOCAB
    BLOCK_SIZE: int = TrainingParams.BLOCK_SIZE
