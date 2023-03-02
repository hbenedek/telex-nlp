import json
from pathlib import Path


def save_json(article: dict, output_folder: Path, name: str) -> None:
    """Save json file."""
    with open(output_folder / f"{name}.json", "w", encoding="utf-8") as outfile:
        json.dump(article, outfile, default=str)


def load_json(file_path: Path) -> dict:
    """Load json file."""
    with open(file_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def save_text(text: str, output_folder: Path, name: str) -> None:
    """Save text file."""
    with open(output_folder / f"{name}.txt", "w", encoding="utf-8") as outfile:
        outfile.write(text)


def load_text(file_path: Path) -> str:
    """Load text file."""
    with open(file_path, "r", encoding="utf-8") as infile:
        return infile.read()
