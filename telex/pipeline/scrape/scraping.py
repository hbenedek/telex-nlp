"""Module for scraping articles from the web."""
import json
import time
from pathlib import Path

import requests
import typer

from params import MAX_PAGES, PER_PAGE, SLEEP_TIME


def get_json(url: str) -> dict:
    """Get json content from web api."""
    response = requests.request("GET", url, timeout=10)
    return response.json()


def save_json(article: dict, output_folder: Path, slug: str) -> None:
    """Save json file."""
    with open(output_folder / f"{slug}.json", "w", encoding="utf-8") as outfile:
        json.dump(article, outfile, default=str)


def scrape_slugs() -> dict:
    """Scrape slugs from the web."""
    slugs_by_page = {}
    for page in range(MAX_PAGES):
        try:
            api = f"https://telex.hu/api/search?oldal={page}&perPage={PER_PAGE}"
            time.sleep(SLEEP_TIME)
            soup = get_json(api)
            slugs = [item["slug"] for item in soup["items"]]
            slugs_by_page[page] = slugs
            print(f"page {page} --- slugs {len(slugs)}")
        except ValueError:
            pass
    return slugs_by_page


def scrape_articles(output_folder: Path) -> None:
    """Scrape articles from the web."""
    slugs_by_page = scrape_slugs()
    i = 0
    for page in range(MAX_PAGES):
        output = output_folder / str(page)
        output.mkdir(parents=True, exist_ok=True)
        for slug in slugs_by_page[page]:
            try:
                time.sleep(SLEEP_TIME)
                article = get_json(f"https://telex.hu/api/articles/{slug}")
                save_json(article, output, slug)
                print(f"page {page} --- article {i} --- slug {slug}")
                i += 1
            except ValueError:
                pass


# TODO: refactor, first scrape all slugs, then scrape all articles
def main(output_folder: Path = typer.Option(...)) -> None:
    """Scrape articles from the web."""

    slugs_by_page = scrape_slugs()
    output = output_folder / "slugs"
    output.mkdir(parents=True, exist_ok=True)
    save_json(slugs_by_page, output, "slugs")

    # scrape_articles(output_folder)


if __name__ == "__main__":
    typer.run(main)
