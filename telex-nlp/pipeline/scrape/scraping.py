"""Module for scraping articles from the web."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import tqdm
import typer
from bs4 import BeautifulSoup as BS

from params import MAX_PAGES


def get_soup(url) -> BS:
    """Get soup from the web."""
    response = requests.request("GET", url)
    soup = BS(response.content, "html.parser")
    return soup


def parse_date(raw):
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


def parse_author(raw):
    return raw.split("\n")[4].lstrip()


def scrape_article_metadata(article, data):
    href = article.find("a", href=True)["href"]
    if not href[1:8] == "english":  # for now skipping english articles
        data["href"].append(href)
        data["language"].append("hu")
        raw = article.find(class_="article_date").text
        try:
            data["author"].append(parse_author(raw))
        except:
            data["author"].append("")
        try:
            data["date"].append(parse_date(raw))
        except:
            data["date"].append(raw)
    data["lead"].append(article.find("p", class_="list__item__lead hasHighlight").text)

    text = scrape_text(href)
    data["text"].append(text)


def scrape_text(href):
    soup2 = get_soup("https://telex.hu" + href)
    article = soup2.find("div", {"class": "article-html-content"})
    return article.text


def scrape_archive_page(page: int) -> dict:
    url = f"https://telex.hu/archivum?oldal={page}"
    data = {"date": [], "author": [], "lead": [], "href": [], "language": [], "text": []}

    soup = get_soup(url)
    list_group = soup.find_all("div", {"class": "list__item__info"})
    for i, article in enumerate(list_group):
        scrape_article_metadata(article, data)
    return data


def save_json(data, output_folder: Path, page: int) -> None:
    with open(output_folder / f"raw_archive_{page}.json", "w") as outfile:
        json.dump(data, outfile, default=str)


def main(output_folder: Path = typer.Option(...)) -> None:
    for page in tqdm.tqdm(range(MAX_PAGES)):
        archive_page = scrape_archive_page(page)
        save_json(archive_page, output_folder, page)


if __name__ == "__main__":
    typer.run(main)
