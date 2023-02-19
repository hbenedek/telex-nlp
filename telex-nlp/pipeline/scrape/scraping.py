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


def get_soup(url) -> None:
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


def main(output_folder: Path = typer.Option(...)) -> None:
    for page in tqdm.tqdm(range(MAX_PAGES)):
        url = "https://telex.hu/archivum?oldal={i}"
        soup = get_soup(url)
        list_group = soup.find_all("div", {"class": "list__item__info"})
        data = {"date": [], "author": [], "lead": [], "href": [], "language": [], "text": []}
        for i, article in enumerate(list_group):
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

        soup2 = get_soup("https://telex.hu" + href)
        article = soup2.find_all("div", {"class": "article-html-content"})[0]
        data["text"].append(article.text)

        # Using a JSON string
        with open(output_folder / f"raw_archive_{page}.json", "w") as outfile:
            json.dump(data, outfile, default=str)


if __name__ == "__main__":
    typer.run(main)
