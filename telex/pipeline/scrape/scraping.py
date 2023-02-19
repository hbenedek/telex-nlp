"""Module for scraping articles from the web."""
import json
from datetime import datetime
from pathlib import Path

import requests
import tqdm
import typer
from bs4 import BeautifulSoup as BS
from bs4.element import ResultSet

from params import MAX_PAGES


def get_soup(url: str) -> BS:
    """Get soup from the web."""
    response = requests.request("GET", url, timeout=5)
    soup = BS(response.content, "html.parser")
    return soup


def parse_date(raw: str) -> datetime:
    """Parse date from raw string."""
    months = {
        "január": "01",
        "február": "02",
        "március": "03",
        "április": "04",
        "május": "05",
        "június": "06",
        "július": "07",
        "augusztus": "08",
        "szeptember": "09",
        "október": "10",
        "november": "11",
        "december": "12",
    }
    splitted = raw.split("\n")[1].lstrip().split(" ")
    splitted[1] = months[splitted[1]]
    date = " ".join(splitted)
    return datetime.strptime(date, "%Y. %m %d. – %H:%M")


def parse_author(raw: str) -> str:
    """Parse author from raw string."""
    return raw.split("\n")[4].lstrip()


def scrape_article(article: ResultSet, data: dict) -> None:
    """Scrape article text, metadata and append dictionary."""
    href = article.find("a", href=True)["href"]
    if not href[1:8] == "english":  # for now skipping english articles
        data["href"].append(href)
        data["language"].append("hu")
        raw = article.find(class_="article_date").text
        try:
            data["author"].append(parse_author(raw))
        except ValueError:
            data["author"].append("")
        try:
            data["date"].append(parse_date(raw))
        except ValueError:
            data["date"].append(raw)
    data["lead"].append(article.find("p", class_="list__item__lead hasHighlight").text)

    text = scrape_text(href)
    data["text"].append(text)


def scrape_text(href: str) -> str:
    """Scrape article text."""
    soup2 = get_soup("https://telex.hu" + href)
    article = soup2.find("div", {"class": "article-html-content"})
    return article.text


def scrape_archive_page(page: int) -> dict:
    """Scrape archive page and return a dictionary with the scraped data."""
    url = f"https://telex.hu/archivum?oldal={page}"
    data = {"date": [], "author": [], "lead": [], "href": [], "language": [], "text": []}

    soup = get_soup(url)
    list_group = soup.find_all("div", {"class": "list__item__info"})
    for article in list_group:
        scrape_article(article, data)
    return data


def save_json(data: dict, output_folder: Path, page: int) -> None:
    """Save json file."""
    with open(output_folder / f"raw_archive_{page}.json", "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, default=str)


def main(output_folder: Path = typer.Option(...)) -> None:
    """Main function, scrape archive pages and save them as json files."""
    for page in tqdm.tqdm(range(MAX_PAGES)):
        archive_page = scrape_archive_page(page)
        save_json(archive_page, output_folder, page)


if __name__ == "__main__":
    typer.run(main)
