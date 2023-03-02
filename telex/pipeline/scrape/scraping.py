"""Module for scraping articles from telex."""
import json
import time
from pathlib import Path

import requests
import typer

from params import MAX_PAGES, PER_PAGE, SLEEP_TIME
from telex.utils.io import save_json


def get_request_as_json(url: str) -> dict:
    """Get json content from web api."""
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    }
    response = requests.request("GET", url, timeout=10, headers=headers)
    return response.json()


def scrape_slugs() -> dict:
    """Scrape slugs from the web."""
    slugs_by_page = {}
    for page in range(MAX_PAGES):
        try:
            api = f"https://telex.hu/api/search?oldal={page}&perPage={PER_PAGE}"
            time.sleep(SLEEP_TIME)
            soup = get_request_as_json(api)
            slugs = [item["slug"] for item in soup["items"]]
            slugs_by_page[page] = slugs
            print(f"page {page} --- slugs {len(slugs)}")
        except ConnectionError:
            pass
    return slugs_by_page


def scrape_articles(slugs_by_page: dict, output_folder: Path) -> None:
    """Scrape articles from the web."""
    i = 0
    for page in range(MAX_PAGES):
        output = output_folder / str(page)
        output.mkdir(parents=True, exist_ok=True)
        for slug in slugs_by_page[page]:
            try:
                time.sleep(SLEEP_TIME)
                article = get_request_as_json(f"https://telex.hu/api/articles/{slug}")
                save_json(article, output, slug)
                print(f"page {page} --- article {i} --- slug {slug}")
                i += 1
            except ConnectionError:
                pass


def main(output_folder: Path = typer.Option(...)) -> None:
    """Scrape articles from telex and save them as json files."""

    slugs_by_page = scrape_slugs()
    output = output_folder / "slugs"
    output.mkdir(parents=True, exist_ok=True)
    save_json(slugs_by_page, output, "slugs")

    scrape_articles(slugs_by_page, output_folder)


if __name__ == "__main__":
    typer.run(main)
