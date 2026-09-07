"""GitHub-hosted mirror of football-data.co.uk -- the fallback historical
source for a network that cannot reach football-data.co.uk directly.

football-data.co.uk itself is unreachable from some networks entirely --
not flaky, not rate-limited, just unroutable, which no retry or timeout
fix can do anything about. ``xgabora/Club-Football-Match-Data-2000-2025``
(MIT licensed) is a GitHub-hosted, actively updated consolidation of the
same underlying football-data.co.uk match data across many leagues, one
file, refreshed regularly. Since it lives on ``raw.githubusercontent.com``
rather than football-data.co.uk's own host, a network that already reaches
GitHub for anything else (as this project's own source does) reaches this
too.

What it has: goals, half-time goals, shots, shots on target, corners,
cards, 1X2 odds, and Over/Under 2.5 odds -- everything the model and the
Phase 7 Over 2.5 benchmark need. What it does not have: BTTS-specific
odds. That one column family comes back NULL from this source, which the
rest of the pipeline already treats as "no market price for this match" --
the BTTS backtest runs without a market comparison; the Over 2.5 one does
not lose anything.

This is opt-in (``CHAMP_HISTORICAL_SOURCE=github-mirror``), not a silent
substitute: the primary source remains football-data.co.uk's own per-season
files, which additionally carry BTTS odds and several individual
bookmakers rather than one blended price, for the network that can reach
them.
"""

from __future__ import annotations

import datetime as dt
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests
import sqlalchemy as sa

from ..db import fact_match, fact_match_stats, dim_season, market_odds, upsert
from ..logging_setup import get_logger
from .footballdata_uk import (DEFAULT_TIMEOUT, DOWNLOAD_USER_AGENT,
                              SourceUnavailable, _RETRYABLE_STATUS,
                              _bounded_get, active_proxies)
from .seasons import season_bounds, season_id_for_date, season_row, season_start_year
from .teams import TeamRegistry, UnknownTeamError, resolve_name

log = get_logger(__name__)

SOURCE = "github-mirror(xgabora)"
MIRROR_URL = ("https://raw.githubusercontent.com/xgabora/"
              "Club-Football-Match-Data-2000-2025/main/data/Matches.csv")
DIVISION_CODE = "E1"          # football-data.co.uk's own Championship code,
                              # preserved verbatim by the mirror.
CACHE_FILENAME = "mirror_matches.csv"
MATCH_KEYS = ["match_date", "home_team_id", "away_team_id"]
BOOK = "Mirror"                # not an average across many books like the
                               # primary source's "Avg" family -- named
                               # distinctly so the two are never confused.

STAT_COLUMNS: dict[str, str] = {
    "HomeShots": "home_shots", "AwayShots": "away_shots",
    "HomeTarget": "home_sot", "AwayTarget": "away_sot",
    "HomeCorners": "home_corners", "AwayCorners": "away_corners",
    "HomeYellow": "home_yellow", "AwayYellow": "away_yellow",
    "HomeRed": "home_red", "AwayRed": "away_red",
}
REQUIRED_COLUMNS = ("Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway")


class MirrorFileUnusable(SourceUnavailable):
    """The mirror responded but not with the CSV this module expects."""


@dataclass
class MirrorReport:
    """What loading the mirror actually produced."""

    matches: int = 0
    stats_rows: int = 0
    odds_rows: int = 0
    seasons_loaded: list[int] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        if self.error:
            return f"unavailable: {self.error}"
        return (f"seasons={len(self.seasons_loaded)} matches={self.matches} "
                f"stats={self.stats_rows} odds={self.odds_rows}")


def cache_path(cache_dir: Path) -> Path:
    return cache_dir / CACHE_FILENAME


def _is_stale(path: Path, max_age_days: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    if max_age_days < 0:
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age > dt.timedelta(days=max_age_days)


def download_mirror(
    cache_dir: Path,
    *,
    max_age_days: int = 3,
    timeout: Any = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    max_retries: int = 4,
    sleeper: Any = time.sleep,
    hard_timeout: float | None = None,
) -> Path:
    """Fetch the one consolidated CSV, reusing the cached copy when fresh.

    One file covers every league since 2000 (~45MB), so a 3-day cache is
    plenty fresh for a source that is itself only updated periodically,
    while still picking up new results well within a week.

    Uses the exact same retry-with-backoff and hard-timeout machinery as
    the primary source's ``download_season`` -- imported, not reimplemented,
    because getting the hard-timeout right (a genuine daemon thread, not a
    ``ThreadPoolExecutor``) was subtle enough once already.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir)
    if not _is_stale(path, max_age_days):
        log.debug("using cached mirror file", extra={"path": str(path)})
        return path

    http = session or requests
    last_error: Exception | None = None
    budget = hard_timeout if hard_timeout is not None else (
        sum(timeout) + 30.0 if isinstance(timeout, tuple) else float(timeout) + 30.0
    )  # a ~45MB file needs more read headroom than a single season CSV

    for attempt in range(max_retries):
        try:
            response = _bounded_get(http, MIRROR_URL, timeout=timeout,
                                    headers={"User-Agent": DOWNLOAD_USER_AGENT},
                                    hard_timeout=budget)
            if response.status_code in _RETRYABLE_STATUS and attempt < max_retries - 1:
                backoff = float(response.headers.get("Retry-After", 2 ** attempt * 3))
                log.warning("transient error fetching mirror file, retrying",
                            extra={"status": response.status_code, "attempt": attempt + 1,
                                   "seconds": backoff})
                sleeper(backoff)
                continue
            response.raise_for_status()
            if not response.content.strip():
                raise SourceUnavailable(f"{MIRROR_URL} returned an empty body")
            if b"Division" not in response.content[:200]:
                raise MirrorFileUnusable(
                    f"{MIRROR_URL} did not return the expected CSV "
                    f"(no Division column in the first 200 bytes)"
                )
            path.write_bytes(response.content)
            log.info("downloaded mirror file", extra={"bytes": len(response.content),
                                                       "attempt": attempt + 1})
            return path
        except MirrorFileUnusable:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                log.warning("mirror download failed, retrying",
                            extra={"attempt": attempt + 1, "error": str(exc)})
                sleeper(backoff)

    if path.exists() and path.stat().st_size > 0:
        log.warning("mirror download failed after retries, falling back to stale cache",
                    extra={"error": str(last_error)})
        return path
    proxies = active_proxies()
    log.warning("mirror unreachable after all retries",
                extra={"error": str(last_error), "system_proxy": proxies or "none detected"})
    if isinstance(last_error, SourceUnavailable):
        raise last_error
    raise SourceUnavailable(f"could not fetch {MIRROR_URL} after {max_retries} attempts: {last_error}")


# --------------------------------------------------------------------------
# Parsing and load
# --------------------------------------------------------------------------
def read_mirror_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MirrorFileUnusable(f"{path.name} is missing column(s) {missing}")
    return df


def _championship_rows(df: pd.DataFrame, start_years: Sequence[int]) -> pd.DataFrame:
    frame = df[df["Division"] == DIVISION_CODE].copy()
    frame["match_date"] = pd.to_datetime(frame["MatchDate"], errors="coerce").dt.date
    frame = frame.dropna(subset=["match_date"])

    bounds = [season_bounds(y) for y in start_years]
    lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
    return frame[(frame["match_date"] >= lo) & (frame["match_date"] <= hi)].reset_index(drop=True)


def check_names(df: pd.DataFrame) -> list[str]:
    names = set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str))
    bad = []
    for name in sorted(names):
        try:
            resolve_name(name, SOURCE)
        except UnknownTeamError:
            bad.append(name)
    return bad


def _to_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 1.0 else None


def _kickoff(day: dt.date, clock: Any) -> dt.datetime | None:
    """MatchTime is only populated for part of the mirror's history; blank
    is normal, not an error, and NULL kickoff on a finished match costs the
    model nothing -- only match_date feeds the fit."""
    from zoneinfo import ZoneInfo

    if not isinstance(clock, str) or ":" not in clock:
        return None
    try:
        hour, minute = (int(part) for part in clock.split(":")[:2])
        local = dt.datetime(day.year, day.month, day.day, hour, minute,
                            tzinfo=ZoneInfo("Europe/London"))
        return local.astimezone(dt.timezone.utc).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def load_mirror(
    conn: sa.Connection,
    registry: TeamRegistry,
    path: Path,
    start_years: Sequence[int],
) -> MirrorReport:
    """Load the requested seasons' Championship rows from the mirror file."""
    report = MirrorReport()
    df = read_mirror_csv(path)
    frame = _championship_rows(df, start_years)
    if frame.empty:
        report.error = "no Championship rows in the requested date range"
        return report

    unresolved = check_names(frame)
    if unresolved:
        # Same hard-failure policy as the primary source: an unmapped club
        # is never guessed at, whichever source found it.
        raise UnknownTeamError(", ".join(unresolved), SOURCE)

    seasons: dict[int, dict[str, Any]] = {}
    match_rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        day = row.match_date
        sid = season_id_for_date(day)
        seasons.setdefault(sid, season_row(season_start_year(day)))
        home_goals = _to_int(row.FTHome)
        away_goals = _to_int(row.FTAway)
        match_rows.append({
            "season_id": sid,
            "match_date": day,
            "kickoff_utc": _kickoff(day, getattr(row, "MatchTime", None)),
            "home_team_id": registry.team_id(row.HomeTeam, SOURCE),
            "away_team_id": registry.team_id(row.AwayTeam, SOURCE),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_ht": _to_int(getattr(row, "HTHome", None)),
            "away_ht": _to_int(getattr(row, "HTAway", None)),
            "status": "FINISHED" if home_goals is not None and away_goals is not None
                      else "SCHEDULED",
            "source": SOURCE,
        })

    upsert(conn, dim_season, list(seasons.values()), ["season_id"])
    upsert(conn, fact_match, match_rows, MATCH_KEYS)
    report.matches = len(match_rows)
    report.seasons_loaded = sorted(seasons.keys())

    match_ids = {
        (d, h, a): mid
        for mid, d, h, a in conn.execute(
            sa.select(fact_match.c.match_id, fact_match.c.match_date,
                     fact_match.c.home_team_id, fact_match.c.away_team_id)
            .where(fact_match.c.match_date.between(
                min(r["match_date"] for r in match_rows),
                max(r["match_date"] for r in match_rows)))
        ).all()
    }

    stats_rows: list[dict[str, Any]] = []
    odds_rows: list[dict[str, Any]] = []
    for row, match in zip(frame.itertuples(index=False), match_rows):
        mid = match_ids.get((match["match_date"], match["home_team_id"], match["away_team_id"]))
        if mid is None:
            continue
        stats = {dest: _to_int(getattr(row, src, None)) for src, dest in STAT_COLUMNS.items()}
        if any(v is not None for v in stats.values()):
            stats_rows.append({"match_id": mid, "home_xg": None, "away_xg": None, **stats})

        odds = {
            "odds_over25": _to_float(getattr(row, "Over25", None)),
            "odds_under25": _to_float(getattr(row, "Under25", None)),
            "odds_btts_yes": None,     # not available from this source
            "odds_btts_no": None,
            "odds_home": _to_float(getattr(row, "OddHome", None)),
            "odds_draw": _to_float(getattr(row, "OddDraw", None)),
            "odds_away": _to_float(getattr(row, "OddAway", None)),
        }
        if any(v is not None for v in odds.values()):
            odds_rows.append({"match_id": mid, "book": BOOK, **odds})

    if stats_rows:
        upsert(conn, fact_match_stats, stats_rows, ["match_id"],
               update_cols=list(STAT_COLUMNS.values()))
        report.stats_rows = len(stats_rows)
    if odds_rows:
        upsert(conn, market_odds, odds_rows, ["match_id", "book"])
        report.odds_rows = len(odds_rows)

    log.info("mirror seasons loaded", extra={"summary": report.summary()})
    return report


def backfill(
    conn: sa.Connection,
    registry: TeamRegistry,
    start_years: Sequence[int],
    cache_dir: Path,
    *,
    max_age_days: int = 3,
    session: requests.Session | None = None,
    sleeper: Any = time.sleep,
    hard_timeout: float | None = None,
) -> MirrorReport:
    """Download (if needed) and load every requested season from the mirror.

    One request for every season requested, unlike the primary source's
    ten -- the mirror is a single file, so there is nothing to burst against
    and nothing to space out.
    """
    try:
        path = download_mirror(cache_dir, max_age_days=max_age_days, session=session,
                               sleeper=sleeper, hard_timeout=hard_timeout)
    except SourceUnavailable as exc:
        log.warning("mirror unavailable", extra={"error": str(exc)})
        return MirrorReport(error=str(exc))
    # UnknownTeamError is not caught here on purpose -- an unmapped club is a
    # hard failure, exactly as it is for the primary source.
    return load_mirror(conn, registry, path, start_years)
