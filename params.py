"""Params file."""

SEED = 42


# params for scraping

MAX_PAGES = 187
PER_PAGE = 400
SLEEP_TIME = 0.5

# params for bigram model
LR = 10e-3
BLOCK_SIZE = 20
BATCH_SIZE = 32
STEPS = 50000
SPLIT = 0.9

# vocab
VOCAB = (
    "AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZaábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz0123456789.,;:?!-–_()[]{}'\"/\\@#$%^&*+=|~` "
)
