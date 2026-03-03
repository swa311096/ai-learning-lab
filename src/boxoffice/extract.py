import datetime as dt
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .utils import parse_money, parse_percent


BOM_BASE_URL = "https://www.boxofficemojo.com"


@dataclass
class MovieSeed:
    movie_id: str
    title: str
    studio: str
    release_date: Optional[dt.date]
    numbers_url: str
    budget_numbers_usd: Optional[float]
    opening_weekend_numbers_usd: Optional[float]
    domestic_numbers_usd: Optional[float]
    worldwide_numbers_usd: Optional[float]


@dataclass
class DailyGrossPoint:
    movie_id: str
    gross_date: dt.date
    day_number: Optional[int]
    rank_text: str
    gross_usd: Optional[float]
    percent_change: Optional[float]
    theaters: Optional[int]
    per_theater_usd: Optional[float]
    total_gross_usd: Optional[float]


class NumbersClient:
    """V1 extractor that seeds from Box Office Mojo year pages and parses daily domestic rows."""

    def __init__(self, timeout_s: int = 30, use_playwright: bool = False):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.timeout_s = timeout_s
        self.use_playwright = use_playwright

    def _get(self, url: str) -> str:
        resp = self._session.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.text

    def _get_page(self, url: str) -> str:
        html = self._get(url)
        if self.use_playwright and ("Access denied" in html[:800] or "<title></title>" in html):
            rendered = self._get_playwright(url)
            if rendered:
                return rendered
        return html

    def _get_playwright(self, url: str) -> Optional[str]:
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=self.timeout_s * 1000)
            html = page.content()
            browser.close()
            return html

    @staticmethod
    def _movie_id(title: str, release_date: Optional[dt.date]) -> str:
        basis = f"{title}|{release_date.isoformat() if release_date else 'na'}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_title(title: str) -> str:
        t = (title or "").lower().strip()
        t = re.sub(r"[^a-z0-9]+", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    @staticmethod
    def _parse_month_day_with_year(text: str, year: int) -> Optional[dt.date]:
        text = (text or "").strip()
        m = re.match(r"^([A-Za-z]{3,9})\s+(\d{1,2})", text)
        if not m:
            return None
        month_txt, day_txt = m.group(1), m.group(2)
        for fmt in ("%b %d %Y", "%B %d %Y"):
            try:
                return dt.datetime.strptime(f"{month_txt} {day_txt} {year}", fmt).date()
            except ValueError:
                continue
        return None

    def _fetch_world_totals(self, year: int) -> Dict[str, float]:
        url = f"{BOM_BASE_URL}/year/world/{year}/"
        html = self._get_page(url)
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return {}

        out: Dict[str, float] = {}
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            title = tds[1].get_text(" ", strip=True)
            worldwide = parse_money(tds[2].get_text(" ", strip=True))
            if title and worldwide is not None:
                out[self._normalize_title(title)] = float(worldwide)
        return out

    def fetch_market_titles(self, start_date: dt.date, end_date: dt.date) -> List[MovieSeed]:
        seeds: Dict[str, MovieSeed] = {}

        for year in range(start_date.year, end_date.year + 1):
            world_map = self._fetch_world_totals(year)
            year_url = f"{BOM_BASE_URL}/year/{year}/"
            html = self._get_page(year_url)
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")
            if not table:
                continue

            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 10:
                    continue

                release_anchor = tds[1].find("a", href=True)
                if not release_anchor:
                    continue

                title = tds[1].get_text(" ", strip=True)
                if not title:
                    continue

                release_date = self._parse_month_day_with_year(tds[8].get_text(" ", strip=True), year)
                if not release_date or release_date < start_date or release_date > end_date:
                    continue

                release_url = urljoin(BOM_BASE_URL, release_anchor["href"].split("?")[0])
                studio = tds[9].get_text(" ", strip=True) or "Unknown"
                domestic_total = parse_money(tds[7].get_text(" ", strip=True))
                worldwide_total = world_map.get(self._normalize_title(title))

                movie_id = self._movie_id(title, release_date)
                seeds[movie_id] = MovieSeed(
                    movie_id=movie_id,
                    title=title,
                    studio=studio,
                    release_date=release_date,
                    numbers_url=release_url,
                    budget_numbers_usd=parse_money(tds[3].get_text(" ", strip=True)),
                    opening_weekend_numbers_usd=None,
                    domestic_numbers_usd=domestic_total,
                    worldwide_numbers_usd=worldwide_total,
                )

        return sorted(seeds.values(), key=lambda m: (m.release_date or dt.date.min, m.title))

    def _parse_daily_date(self, text: str, release_date: dt.date) -> Optional[dt.date]:
        m = re.match(r"^([A-Za-z]{3,9})\s+(\d{1,2})", (text or "").strip())
        if not m:
            return None
        month_txt, day_txt = m.group(1), m.group(2)

        candidates: List[dt.date] = []
        for year in [release_date.year - 1, release_date.year, release_date.year + 1]:
            for fmt in ("%b %d %Y", "%B %d %Y"):
                try:
                    candidates.append(dt.datetime.strptime(f"{month_txt} {day_txt} {year}", fmt).date())
                except ValueError:
                    continue

        if not candidates:
            return None

        return min(candidates, key=lambda d: abs((d - release_date).days))

    def fetch_daily_domestic(self, movie: MovieSeed) -> List[DailyGrossPoint]:
        if not movie.release_date:
            return []

        html = self._get_page(movie.numbers_url)
        soup = BeautifulSoup(html, "html.parser")

        target = None
        for table in soup.find_all("table"):
            headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
            if {"Date", "DOW", "Daily", "To Date", "Day"}.issubset(set(headers)):
                target = table
                break

        if target is None:
            return []

        rows: List[DailyGrossPoint] = []
        for tr in target.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 10:
                continue

            gross_date = self._parse_daily_date(tds[0].get_text(" ", strip=True), movie.release_date)
            if not gross_date:
                continue

            day_raw = tds[9].get_text(" ", strip=True)
            day_number = int(day_raw) if day_raw.isdigit() else None

            theaters_txt = tds[6].get_text(" ", strip=True).replace(",", "")
            theaters = int(theaters_txt) if theaters_txt.isdigit() else None

            rows.append(
                DailyGrossPoint(
                    movie_id=movie.movie_id,
                    gross_date=gross_date,
                    day_number=day_number,
                    rank_text=tds[2].get_text(" ", strip=True),
                    gross_usd=parse_money(tds[3].get_text(" ", strip=True)),
                    percent_change=parse_percent(tds[4].get_text(" ", strip=True)),
                    theaters=theaters,
                    per_theater_usd=parse_money(tds[7].get_text(" ", strip=True)),
                    total_gross_usd=parse_money(tds[8].get_text(" ", strip=True)),
                )
            )

        rows.sort(key=lambda r: ((r.day_number or 10_000), r.gross_date))
        return rows

    def fetch_daily_for_many(self, movies: Iterable[MovieSeed], sleep_s: float = 0.25) -> List[DailyGrossPoint]:
        all_rows: List[DailyGrossPoint] = []
        for movie in movies:
            rows = self.fetch_daily_domestic(movie)
            all_rows.extend(rows)
            time.sleep(sleep_s)
        return all_rows
