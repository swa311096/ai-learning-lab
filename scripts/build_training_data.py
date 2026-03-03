#!/usr/bin/env python3
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxoffice.db import connect, init_schema
from boxoffice.extract import NumbersClient
from boxoffice.transform import rebuild_training_examples, upsert_daily, upsert_movies


def parse_args():
    p = argparse.ArgumentParser(description="Build training dataset from Box Office Mojo")
    p.add_argument("--db", default="data/processed/boxoffice.sqlite")
    p.add_argument("--years", type=int, default=3, help="How many years back from today")
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds between movie page requests")
    p.add_argument("--use-playwright", action="store_true", help="Use Playwright fallback for dynamic/blocked pages")
    p.add_argument("--all-studios", action="store_true", help="Include all studios; default is US distributors only")
    return p.parse_args()


def subtract_years(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year - years)


def main() -> int:
    args = parse_args()

    today = dt.date.today()
    start_date = subtract_years(today, args.years)

    client = NumbersClient(use_playwright=args.use_playwright)
    movies = client.fetch_market_titles(start_date=start_date, end_date=today)
    daily_rows = client.fetch_daily_for_many(movies, sleep_s=args.sleep)

    conn = connect(args.db)
    init_schema(conn)
    upsert_movies(conn, movies)
    upsert_daily(conn, daily_rows)
    training_count = rebuild_training_examples(conn, as_of_day=7, us_only=not args.all_studios)

    print(f"Seed movies: {len(movies)}")
    print(f"Daily rows: {len(daily_rows)}")
    print(f"Training rows: {training_count}")
    print(f"DB: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
