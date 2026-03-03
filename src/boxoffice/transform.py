import datetime as dt
import sqlite3
from typing import Dict, List, Optional, Set

from .extract import DailyGrossPoint, MovieSeed

# US-based studios/distributors (case-insensitive substring match).
# Box Office Mojo uses distributor names; this filters to Hollywood/US releases.
US_STUDIO_SUBSTRINGS: Set[str] = {
    "disney",
    "universal",
    "warner",
    "paramount",
    "sony",
    "columbia",
    "lionsgate",
    "mgm",
    "20th century",
    "fox ",
    "dreamworks",
    "focus features",
    "a24",
    "amazon",
    "netflix",
    "neon",
    "searchlight",
    "orion",
    "new line",
    "blumhouse",
    "touchstone",
    "roadside",
    "bleecker",
    "magnolia",
    "vertical",
    "gravitas",
    "stx",
    "saban",
    "entertainment studios",
    "briarcliff",
    "broad green",
    "annapurna",
    "legendary",
}


def _is_us_studio(studio: Optional[str]) -> bool:
    if not studio or not studio.strip():
        return False
    s = studio.lower().strip()
    return any(sub in s for sub in US_STUDIO_SUBSTRINGS)


def upsert_movies(conn: sqlite3.Connection, movies: List[MovieSeed]) -> None:
    now = dt.datetime.utcnow().isoformat()
    conn.executemany(
        """
        INSERT INTO movies_raw (
            movie_id, title, studio, release_date, numbers_url,
            budget_numbers_usd, opening_weekend_numbers_usd, domestic_numbers_usd,
            worldwide_numbers_usd, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id) DO UPDATE SET
            title=excluded.title,
            studio=excluded.studio,
            release_date=excluded.release_date,
            numbers_url=excluded.numbers_url,
            budget_numbers_usd=excluded.budget_numbers_usd,
            opening_weekend_numbers_usd=excluded.opening_weekend_numbers_usd,
            domestic_numbers_usd=excluded.domestic_numbers_usd,
            worldwide_numbers_usd=excluded.worldwide_numbers_usd,
            ingested_at=excluded.ingested_at
        """,
        [
            (
                m.movie_id,
                m.title,
                m.studio,
                m.release_date.isoformat() if m.release_date else None,
                m.numbers_url,
                m.budget_numbers_usd,
                m.opening_weekend_numbers_usd,
                m.domestic_numbers_usd,
                m.worldwide_numbers_usd,
                now,
            )
            for m in movies
        ],
    )
    conn.commit()


def upsert_daily(conn: sqlite3.Connection, rows: List[DailyGrossPoint]) -> None:
    conn.executemany(
        """
        INSERT INTO daily_gross (
            movie_id, gross_date, day_number, rank_text, gross_usd,
            percent_change, theaters, per_theater_usd, total_gross_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id, gross_date) DO UPDATE SET
            day_number=excluded.day_number,
            rank_text=excluded.rank_text,
            gross_usd=excluded.gross_usd,
            percent_change=excluded.percent_change,
            theaters=excluded.theaters,
            per_theater_usd=excluded.per_theater_usd,
            total_gross_usd=excluded.total_gross_usd
        """,
        [
            (
                r.movie_id,
                r.gross_date.isoformat(),
                r.day_number,
                r.rank_text,
                r.gross_usd,
                r.percent_change,
                r.theaters,
                r.per_theater_usd,
                r.total_gross_usd,
            )
            for r in rows
        ],
    )
    conn.commit()


def _opening_weekend_from_dailies(dailies: list) -> Optional[float]:
    """Sum first Fri+Sat+Sun from daily gross. Returns None if not found."""
    by_date: Dict[dt.date, float] = {}
    for row in dailies:
        d_raw, _, gross = row[0], row[1], row[2]
        if gross is None:
            continue
        d = d_raw if isinstance(d_raw, dt.date) else dt.date.fromisoformat(str(d_raw))
        by_date[d] = float(gross)
    for d in sorted(by_date.keys()):
        if d.weekday() == 4:  # Friday
            sat = d + dt.timedelta(days=1)
            sun = d + dt.timedelta(days=2)
            if sat in by_date and sun in by_date:
                return by_date[d] + by_date[sat] + by_date[sun]
    return None


def _holiday_flag(release_date: Optional[dt.date]) -> int:
    if not release_date:
        return 0
    # Simple V1 holiday windows: Thanksgiving week, Christmas/New Year, Memorial Day weekend, Independence Day week.
    m = release_date.month
    d = release_date.day
    if (m == 11 and 20 <= d <= 30) or (m == 12 and d >= 20) or (m == 1 and d <= 3):
        return 1
    if (m == 5 and 20 <= d <= 31) or (m == 7 and d <= 7):
        return 1
    return 0


def rebuild_training_examples(
    conn: sqlite3.Connection,
    as_of_day: int = 7,
    us_only: bool = True,
) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM training_examples")

    movies = cur.execute(
        """
        SELECT movie_id, title, studio, release_date, opening_weekend_numbers_usd,
               domestic_numbers_usd, worldwide_numbers_usd
        FROM movies_raw
        """
    ).fetchall()

    now = dt.datetime.utcnow().isoformat()
    inserted = 0

    for movie_id, title, studio, release_date_raw, opening_weekend, domestic_total, worldwide_total in movies:
        if us_only and not _is_us_studio(studio):
            continue

        dailies = cur.execute(
            """
            SELECT gross_date, day_number, gross_usd, percent_change, theaters, total_gross_usd
            FROM daily_gross
            WHERE movie_id = ?
            ORDER BY gross_date ASC
            """,
            (movie_id,),
        ).fetchall()
        if len(dailies) < 4:
            continue

        # Use first 7 calendar rows for V1. We keep this deterministic for backtesting.
        first = dailies[:as_of_day]
        if len(first) < min(as_of_day, 4):
            continue

        gross_values = [r[2] for r in first]
        if any(v is None for v in gross_values[:4]):
            continue

        if not opening_weekend:
            opening_weekend = _opening_weekend_from_dailies(dailies)
            if not opening_weekend:
                first3 = [v for v in gross_values[:3] if v is not None]
                if len(first3) == 3:
                    opening_weekend = sum(first3)

        if not domestic_total:
            last_total = dailies[-1][5]
            domestic_total = last_total if last_total else None

        if not opening_weekend or not domestic_total:
            continue

        day3_total = sum(v for v in gross_values[:3] if v)
        day7_total = sum(v for v in gross_values if v)

        fri_sat = None
        sat_sun = None
        sun_mon = None
        if gross_values[0] and gross_values[1]:
            fri_sat = (gross_values[1] - gross_values[0]) / gross_values[0]
        if gross_values[1] and gross_values[2]:
            sat_sun = (gross_values[2] - gross_values[1]) / gross_values[1]
        if len(gross_values) >= 4 and gross_values[2] and gross_values[3]:
            sun_mon = (gross_values[3] - gross_values[2]) / gross_values[2]

        theaters_day1 = first[0][4]
        theaters_day7 = first[-1][4]

        worldwide = worldwide_total or None
        international_total = max((worldwide - domestic_total), 0.0) if worldwide else None
        domestic_multiplier = domestic_total / opening_weekend if opening_weekend else None
        intl_dom_ratio = (international_total / domestic_total) if (international_total is not None and domestic_total) else None

        release_date = dt.date.fromisoformat(release_date_raw) if release_date_raw else None

        cur.execute(
            """
            INSERT INTO training_examples (
                movie_id, title, release_date, as_of_day, opening_weekend_usd,
                day3_total_usd, day7_total_usd, fri_sat_change_pct, sat_sun_change_pct,
                sun_mon_change_pct, theaters_day1, theaters_day7, release_month,
                is_holiday_window, domestic_total_usd, international_total_usd,
                worldwide_total_usd, domestic_multiplier, intl_dom_ratio, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                movie_id,
                title,
                release_date_raw,
                as_of_day,
                opening_weekend,
                day3_total,
                day7_total,
                fri_sat,
                sat_sun,
                sun_mon,
                theaters_day1,
                theaters_day7,
                release_date.month if release_date else None,
                _holiday_flag(release_date),
                domestic_total,
                international_total,
                worldwide_total,
                domestic_multiplier,
                intl_dom_ratio,
                now,
            ),
        )
        inserted += 1

    conn.commit()
    return inserted


def fetch_training_frame(conn: sqlite3.Connection):
    import pandas as pd

    return pd.read_sql_query(
        """
        SELECT
            movie_id,
            title,
            release_date,
            opening_weekend_usd,
            day3_total_usd,
            day7_total_usd,
            fri_sat_change_pct,
            sat_sun_change_pct,
            sun_mon_change_pct,
            theaters_day1,
            theaters_day7,
            release_month,
            is_holiday_window,
            domestic_multiplier,
            intl_dom_ratio
        FROM training_examples
        """,
        conn,
    )
