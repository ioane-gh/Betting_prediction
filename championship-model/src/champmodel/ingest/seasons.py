"""Season codes.

football-data.co.uk names its files by a four-digit season code ("2526" for
2025-26), which doubles as the ``champ.dim_season.season_id``.
"""

from __future__ import annotations

import datetime as dt

# English seasons start in early August; anything from July onward belongs to
# the season named for that calendar year.
SEASON_START_MONTH = 7


def season_start_year(day: dt.date) -> int:
    return day.year if day.month >= SEASON_START_MONTH else day.year - 1


def season_code(start_year: int) -> str:
    """2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_id(start_year: int) -> int:
    return int(season_code(start_year))


def season_label(start_year: int) -> str:
    """2025 -> '2025-26'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_id_for_date(day: dt.date) -> int:
    return season_id(season_start_year(day))


def season_bounds(start_year: int) -> tuple[dt.date, dt.date]:
    return dt.date(start_year, 7, 1), dt.date(start_year + 1, 6, 30)


def recent_start_years(count: int, today: dt.date | None = None) -> list[int]:
    """The last ``count`` season start years, oldest first, including the current one."""
    today = today or dt.date.today()
    current = season_start_year(today)
    return list(range(current - count + 1, current + 1))


def season_row(start_year: int) -> dict[str, object]:
    start, end = season_bounds(start_year)
    return {
        "season_id": season_id(start_year),
        "label": season_label(start_year),
        "start_date": start,
        "end_date": end,
    }
