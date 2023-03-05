"""Module for data preprocessing."""

import re
from pathlib import Path

import tqdm
import typer

from telex.utils.io import load_json, save_json


def get_attributes(article: dict) -> str:
    """Get text from article."""
    return article["id"], article["type"], article["title"], article["content"], article["slug"], article["pubDate"]


def remove_tags(text: str) -> str:
    """Remove html tags from text."""
    tag = re.compile(r"<[^>]+>")
    return tag.sub("", text)


def get_all_articles(input_folder: Path) -> list:
    """Get all article paths from input folder."""
    return [path for path in input_folder.rglob("*.json") if not path.name.endswith("slug.json")]


def main(input_folder: Path = typer.Option(...), output_folder: Path = typer.Option(...)) -> None:
    """Preprocess articles."""
    dataset = {
        "id": [],
        "type": [],
        "title": [],
        "content": [],
        "slug": [],
        "date": [],
    }
    paths_to_articles = get_all_articles(input_folder)

    for idx, path in tqdm.tqdm(enumerate(paths_to_articles)):
        article = load_json(path)
        try:
            id, type, title, content, slug, date = get_attributes(article)

            title = remove_tags(title)
            content = remove_tags(content)

            dataset["id"].append(id)
            dataset["type"].append(type)
            dataset["title"].append(title)
            dataset["content"].append(content)
            dataset["slug"].append(slug)
            dataset["date"].append(date)

        except KeyError:
            print(f"KeyError: {path}")
            continue

    output_folder.mkdir(parents=True, exist_ok=True)
    save_json(dataset, output_folder, "telex")


if __name__ == "__main__":
    typer.run(main)

    # typer.run(main)
