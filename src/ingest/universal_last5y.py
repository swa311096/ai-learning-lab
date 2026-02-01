#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


NUMBERS_URL = "https://www.the-numbers.com/movies/production-company/Universal-Pictures"
WIKI_API = "https://en.wikipedia.org/w/api.php"
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "").strip()


def _today() -> dt.date:
    return dt.date.today()


def _subtract_years(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        # Handle Feb 29 -> Feb 28
        return d.replace(month=2, day=28, year=d.year - years)


def _parse_date(text: str) -> Optional[dt.date]:
    text = text.strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _clean_text(node) -> str:
    if not node:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def _parse_money(value: str) -> Optional[int]:
    if not value:
        return None
    value = value.replace(",", "").replace("$", "").strip()
    if value in ("-", "N/A", "n/a"):
        return None
    # Handle values like "150 million" or "150m"
    m = re.match(r"^(\d+(\.\d+)?)(\s*(million|billion|m|bn))?$", value, re.I)
    if m:
        num = float(m.group(1))
        unit = (m.group(4) or "").lower()
        if unit in ("million", "m"):
            return int(num * 1_000_000)
        if unit in ("billion", "bn"):
            return int(num * 1_000_000_000)
        return int(num)
    # Fallback: strip non-digits
    digits = re.sub(r"[^\d]", "", value)
    return int(digits) if digits else None


def _request(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    headers = {"User-Agent": "will-this-movie-make-money/1.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_numbers_universal() -> List[Dict[str, Any]]:
    html = _request(NUMBERS_URL)
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    target = None
    for table in tables:
        headers = [_clean_text(h) for h in table.find_all("th")]
        if "Release Date" in headers and "Worldwide Box Office" in headers:
            target = table
            break
    if target is None:
        raise RuntimeError("Could not find Universal Pictures table on The Numbers page.")

    rows = []
    for tr in target.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        cols = [_clean_text(td) for td in tds]
        release_date = _parse_date(cols[0]) if cols else None
        title_cell = tds[1] if len(tds) > 1 else None
        title = _clean_text(title_cell)
        link = None
        if title_cell:
            a = title_cell.find("a")
            if a and a.get("href"):
                link = "https://www.the-numbers.com" + a.get("href")
        row = {
            "title": title,
            "release_date": release_date,
            "numbers_url": link,
            "budget_numbers_usd": _parse_money(cols[2]) if len(cols) > 2 else None,
            "opening_weekend_numbers_usd": _parse_money(cols[3]) if len(cols) > 3 else None,
            "domestic_numbers_usd": _parse_money(cols[4]) if len(cols) > 4 else None,
            "worldwide_numbers_usd": _parse_money(cols[5]) if len(cols) > 5 else None,
        }
        rows.append(row)
    return rows


def _wiki_get(params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"User-Agent": "will-this-movie-make-money/1.0"}
    resp = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
    try:
        return resp.json() if resp.text else {}
    except json.JSONDecodeError:
        return {}


def _wiki_search(title: str, year: Optional[int]) -> Optional[str]:
    query = f"{title} {year or ''} film".strip()
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
        "format": "json",
    }
    data = _wiki_get(params)
    for result in data.get("query", {}).get("search", []):
        return result.get("title")
    return None


def _wiki_extract(page_title: str) -> Tuple[str, Optional[str]]:
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": page_title,
        "format": "json",
    }
    data = _wiki_get(params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "").strip(), page.get("fullurl")
    return "", None


def _wiki_parse_infobox(page_title: str) -> Dict[str, str]:
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "format": "json",
    }
    data = _wiki_get(params)
    html = data.get("parse", {}).get("text", {}).get("*", "")
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.find("table", class_=lambda x: x and "infobox" in x)
    result = {}
    if not infobox:
        return result
    for tr in infobox.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        key = _clean_text(th)
        val = _clean_text(td)
        if key and val:
            result[key] = val
    return result


def _split_list(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;,]| and ", text) if p.strip()]
    return parts


def _verify_match(a: Optional[int], b: Optional[int], tolerance: float = 0.1) -> str:
    if a is None or b is None:
        return "missing"
    if a == 0 or b == 0:
        return "missing"
    diff = abs(a - b) / max(a, b)
    return "match" if diff <= tolerance else "conflict"


def enrich_with_wikipedia(movie: Dict[str, Any], sleep_s: float = 0.2) -> None:
    try:
        year = movie.get("release_date").year if movie.get("release_date") else None
        page_title = _wiki_search(movie["title"], year)
        if not page_title:
            return
        extract, _ = _wiki_extract(page_title)
        infobox = _wiki_parse_infobox(page_title)
    except Exception:
        return

    wiki_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    movie["wiki_url"] = wiki_url
    movie["description"] = extract

    def take(key: str) -> str:
        return infobox.get(key, "")

    movie["budget_wiki_usd"] = _parse_money(take("Budget"))
    movie["box_office_wiki_usd"] = _parse_money(take("Box office"))
    movie["genres"] = _split_list(take("Genre"))
    movie["directors"] = _split_list(take("Directed by"))
    movie["producers"] = _split_list(take("Produced by"))
    movie["actors"] = _split_list(take("Starring"))

    movie.setdefault("sources", {})
    if movie.get("budget_wiki_usd") is not None:
        movie["sources"].setdefault("budget_wiki_usd", []).append(wiki_url)
    if movie.get("box_office_wiki_usd") is not None:
        movie["sources"].setdefault("box_office_wiki_usd", []).append(wiki_url)
    for field in ("genres", "directors", "producers", "actors", "description"):
        if movie.get(field):
            movie["sources"].setdefault(field, []).append(wiki_url)

    time.sleep(sleep_s)


def enrich_with_omdb(movie: Dict[str, Any], sleep_s: float = 0.2) -> None:
    if not OMDB_API_KEY:
        return
    year = movie.get("release_date").year if movie.get("release_date") else None
    params = {
        "t": movie.get("title"),
        "y": str(year) if year else None,
        "apikey": OMDB_API_KEY,
    }
    resp = requests.get("https://www.omdbapi.com/", params=params, timeout=30)
    data = resp.json() if resp.ok else {}
    if data.get("Response") != "True":
        return

    source = "https://www.omdbapi.com/"
    movie.setdefault("sources", {})

    if not movie.get("genres") and data.get("Genre"):
        movie["genres"] = _split_list(data.get("Genre"))
        movie["sources"].setdefault("genres", []).append(source)
    if not movie.get("directors") and data.get("Director"):
        movie["directors"] = _split_list(data.get("Director"))
        movie["sources"].setdefault("directors", []).append(source)
    if not movie.get("actors") and data.get("Actors"):
        movie["actors"] = _split_list(data.get("Actors"))
        movie["sources"].setdefault("actors", []).append(source)
    if not movie.get("description") and data.get("Plot"):
        movie["description"] = data.get("Plot").strip()
        movie["sources"].setdefault("description", []).append(source)

    time.sleep(sleep_s)


def build_rows(movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for movie in movies:
        budget_ver = _verify_match(movie.get("budget_numbers_usd"), movie.get("budget_wiki_usd"))
        box_ver = _verify_match(movie.get("worldwide_numbers_usd"), movie.get("box_office_wiki_usd"))
        row = {
            "title": movie.get("title"),
            "release_date": movie.get("release_date").isoformat() if movie.get("release_date") else None,
            "numbers_url": movie.get("numbers_url"),
            "wiki_url": movie.get("wiki_url"),
            "budget_numbers_usd": movie.get("budget_numbers_usd"),
            "budget_wiki_usd": movie.get("budget_wiki_usd"),
            "opening_weekend_numbers_usd": movie.get("opening_weekend_numbers_usd"),
            "domestic_numbers_usd": movie.get("domestic_numbers_usd"),
            "worldwide_numbers_usd": movie.get("worldwide_numbers_usd"),
            "box_office_wiki_usd": movie.get("box_office_wiki_usd"),
            "budget_verification": budget_ver,
            "box_office_verification": box_ver,
            "genres": "; ".join(movie.get("genres", [])),
            "directors": "; ".join(movie.get("directors", [])),
            "producers": "; ".join(movie.get("producers", [])),
            "actors": "; ".join(movie.get("actors", [])),
            "description": movie.get("description"),
            "sources_json": json.dumps(movie.get("sources", {}), ensure_ascii=True),
        }
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(movies: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for movie in movies:
            movie_copy = dict(movie)
            if isinstance(movie_copy.get("release_date"), dt.date):
                movie_copy["release_date"] = movie_copy["release_date"].isoformat()
            f.write(json.dumps(movie_copy, ensure_ascii=True) + "\n")


def write_sqlite(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS universal_last5y")
    columns = list(rows[0].keys())
    col_defs = ", ".join(f"{c} TEXT" for c in columns)
    cur.execute(f"CREATE TABLE universal_last5y ({col_defs})")
    placeholders = ", ".join(["?"] * len(columns))
    for row in rows:
        cur.execute(
            f"INSERT INTO universal_last5y ({', '.join(columns)}) VALUES ({placeholders})",
            [row.get(c) for c in columns],
        )
    conn.commit()
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", default="data/universal_last5y.csv")
    parser.add_argument("--output-jsonl", default="data/universal_last5y.jsonl")
    parser.add_argument("--output-sqlite", default="data/universal_last5y.sqlite")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    today = _today()
    start_date = _subtract_years(today, 5)
    if args.start_date:
        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()

    movies = fetch_numbers_universal()
    movies = [
        m for m in movies if m.get("release_date") and m["release_date"] >= start_date
    ]

    for movie in movies:
        movie.setdefault("sources", {})
        if movie.get("budget_numbers_usd") is not None:
            movie["sources"].setdefault("budget_numbers_usd", []).append(NUMBERS_URL)
        if movie.get("opening_weekend_numbers_usd") is not None:
            movie["sources"].setdefault("opening_weekend_numbers_usd", []).append(NUMBERS_URL)
        if movie.get("domestic_numbers_usd") is not None:
            movie["sources"].setdefault("domestic_numbers_usd", []).append(NUMBERS_URL)
        if movie.get("worldwide_numbers_usd") is not None:
            movie["sources"].setdefault("worldwide_numbers_usd", []).append(NUMBERS_URL)

        enrich_with_wikipedia(movie, sleep_s=args.sleep)
        enrich_with_omdb(movie, sleep_s=args.sleep)

    rows = build_rows(movies)
    write_csv(rows, args.output_csv)
    write_jsonl(movies, args.output_jsonl)
    write_sqlite(rows, args.output_sqlite)
    print(f"Wrote {len(rows)} movies from {start_date} to {today}.")
    print()
    # Print output to console
    for row in rows:
        title = (row.get("title") or "")[:40]
        date = row.get("release_date") or ""
        budget = row.get("budget_numbers_usd") or ""
        worldwide = row.get("worldwide_numbers_usd") or ""
        print(f"  {date}  {title:<40}  budget={budget}  worldwide={worldwide}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
