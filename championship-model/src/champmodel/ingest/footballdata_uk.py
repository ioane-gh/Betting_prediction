"""football-data.co.uk -- primary historical source (E1.csv = Championship).

One CSV per season, 1993 to date: goals, half-time goals, shots, shots on
target, corners, cards, and closing bookmaker odds. The odds columns are not
decoration: the de-vigged closing Over/Under 2.5 line is the benchmark the
model is scored against in Phase 7.

Files are cached under ``data/raw`` and only re-downloaded when stale, because
completed seasons never change and the current one changes weekly.
"""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import sqlalchemy as sa

from ..db import fact_match, fact_match_stats, dim_season, market_odds, upsert
from ..logging_setup import get_logger
from .seasons import season_code, season_id, season_row
from .teams import TeamRegistry, UnknownTeamError, resolve_name

log = get_logger(__name__)

SOURCE = "football-data.co.uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E1.csv"
UK_TZ = ZoneInfo("Europe/London")
MATCH_KEYS = ["match_date", "home_team_id", "away_team_id"]


class SourceUnavailable(RuntimeError):
    """The source could not be reached and no usable cache exists."""


@dataclass
class IngestReport:
    """What actually made it in, and what did not."""

    seasons_loaded: list[int] = field(default_factory=list)
    seasons_failed: dict[int, str] = field(default_factory=dict)
    matches: int = 0
    stats_rows: int = 0
    odds_rows: int = 0
    unresolved_names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.seasons_failed and not self.unresolved_names

    def summary(self) -> str:
        parts = [
            f"seasons={len(self.seasons_loaded)}",
            f"matches={self.matches}",
            f"stats={self.stats_rows}",
            f"odds={self.odds_rows}",
        ]
        if self.seasons_failed:
            parts.append(f"failed={sorted(self.seasons_failed)}")
        return " ".join(parts)


# --------------------------------------------------------------------------
# Download / cache
# --------------------------------------------------------------------------
def cache_path(raw_dir: Path, start_year: int) -> Path:
    return raw_dir / f"E1_{season_code(start_year)}.csv"


def _is_stale(path: Path, max_age_days: int) -> bool:
    """Whether the cached file should be re-fetched.

    ``max_age_days`` of 0 means always refresh; a negative value means never
    refresh, which is how an offline run pins itself to whatever is on disk.
    """
    if not path.exists() or path.stat().st_size == 0:
        return True
    if max_age_days < 0:
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age > dt.timedelta(days=max_age_days)


def download_season(
    start_year: int,
    raw_dir: Path,
    *,
    max_age_days: int = 7,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> Path:
    """Fetch one season CSV, reusing the cached copy when it is fresh enough.

    A finished season is immutable, so only the current season is refreshed on
    the weekly cadence. If the network fails but a cached copy exists, the
    cached copy is used and a warning is logged -- the run degrades, it does
    not die.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(raw_dir, start_year)
    if not _is_stale(path, max_age_days):
        log.debug("using cached season file", extra={"season": start_year, "path": str(path)})
        return path

    url = BASE_URL.format(season=season_code(start_year))
    http = session or requests
    try:
        response = http.get(url, timeout=timeout, headers={"User-Agent": "champmodel/0.1"})
        response.raise_for_status()
        if not response.content.strip():
            raise SourceUnavailable(f"{url} returned an empty body")
        path.write_bytes(response.content)
        log.info("downloaded season file",
                 extra={"season": start_year, "bytes": len(response.content)})
        return path
    except Exception as exc:  # network, HTTP, proxy policy -- all the same here
        if path.exists() and path.stat().st_size > 0:
            log.warning("download failed, falling back to stale cache",
                        extra={"season": start_year, "error": str(exc)})
            return path
        raise SourceUnavailable(f"could not fetch {url}: {exc}") from exc


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def _coalesce(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    """First non-null value per row across ``candidates``, best source first.

    Column layouts changed over the years: the closing-odds family (the 'C'
    columns, e.g. AvgC>2.5) arrived in 2019-20, before which the Betbrain
    averages (BbAv...) were the market summary. Coalescing row by row rather
    than picking one column keeps fixtures whose closing price is blank -- an
    unplayed match in the current season -- from losing their opening price.
    """
    present = [name for name in candidates if name in df.columns]
    if not present:
        return None
    series = pd.to_numeric(df[present[0]], errors="coerce")
    for name in present[1:]:
        series = series.fillna(pd.to_numeric(df[name], errors="coerce"))
    return series

# Ordered best-first: closing average, then pre-match average, then Betbrain.
ODDS_COLUMNS: dict[str, dict[str, Sequence[str]]] = {
    "Avg": {
        "odds_over25": ("AvgC>2.5", "Avg>2.5", "BbAv>2.5"),
        "odds_under25": ("AvgC<2.5", "Avg<2.5", "BbAv<2.5"),
        "odds_btts_yes": ("AvgCBTSY", "AvgBTSY", "BbAvBTSY"),
        "odds_btts_no": ("AvgCBTSN", "AvgBTSN", "BbAvBTSN"),
        "odds_home": ("AvgCH", "AvgH", "BbAvH"),
        "odds_draw": ("AvgCD", "AvgD", "BbAvD"),
        "odds_away": ("AvgCA", "AvgA", "BbAvA"),
    },
    "B365": {
        "odds_over25": ("B365C>2.5", "B365>2.5"),
        "odds_under25": ("B365C<2.5", "B365<2.5"),
        "odds_btts_yes": ("B365CBTSY", "B365BTSY"),
        "odds_btts_no": ("B365CBTSN", "B365BTSN"),
        "odds_home": ("B365CH", "B365H"),
        "odds_draw": ("B365CD", "B365D"),
        "odds_away": ("B365CA", "B365A"),
    },
}

STAT_COLUMNS: dict[str, str] = {
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
}


def parse_dates(raw: pd.Series) -> pd.Series:
    """'09/08/2025' and '09/08/25' both appear, sometimes in the same decade."""
    parsed = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(raw, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(fallback)


def _kickoff_utc(dates: pd.Series, times: pd.Series | None) -> list[dt.datetime | None]:
    if times is None:
        return [None] * len(dates)
    out: list[dt.datetime | None] = []
    for day, clock in zip(dates, times):
        if pd.isna(day) or not isinstance(clock, str) or ":" not in clock:
            out.append(None)
            continue
        try:
            hour, minute = (int(part) for part in clock.split(":")[:2])
            local = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=UK_TZ)
            out.append(local.astimezone(dt.timezone.utc).replace(tzinfo=None))
        except (ValueError, TypeError):
            out.append(None)
    return out


def _to_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # Odds below 1.01 are data errors, not prices.
    return number if number > 1.0 else None


def read_season_csv(path: Path) -> pd.DataFrame:
    """Load one E1.csv, dropping the blank trailing rows the files carry."""
    raw = path.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"], how="any")
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]
    return df.reset_index(drop=True)


def check_names(df: pd.DataFrame) -> list[str]:
    """Every name in the file must resolve. Returns the ones that do not."""
    names = set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str))
    bad = []
    for name in sorted(names):
        try:
            resolve_name(name, SOURCE)
        except UnknownTeamError:
            bad.append(name)
    return bad


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_season(
    conn: sa.Connection,
    registry: TeamRegistry,
    start_year: int,
    path: Path,
    report: IngestReport,
) -> None:
    df = read_season_csv(path)
    if df.empty:
        report.seasons_failed[start_year] = "empty file"
        return

    unresolved = check_names(df)
    if unresolved:
        # Hard failure by design (Phase 3c): a silently mis-mapped club splits
        # one team's history in two and quietly halves its strength.
        report.unresolved_names.extend(unresolved)
        raise UnknownTeamError(", ".join(unresolved), SOURCE)

    sid = season_id(start_year)
    upsert(conn, dim_season, [season_row(start_year)], ["season_id"])

    dates = parse_dates(df["Date"])
    kickoffs = _kickoff_utc(dates, df["Time"] if "Time" in df.columns else None)

    match_rows: list[dict[str, Any]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        day = dates.iloc[i]
        if pd.isna(day):
            continue
        home_goals = _to_int(getattr(row, "FTHG", None))
        away_goals = _to_int(getattr(row, "FTAG", None))
        match_rows.append({
            "season_id": sid,
            "match_date": day.date(),
            "kickoff_utc": kickoffs[i],
            "home_team_id": registry.team_id(row.HomeTeam, SOURCE),
            "away_team_id": registry.team_id(row.AwayTeam, SOURCE),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_ht": _to_int(getattr(row, "HTHG", None)),
            "away_ht": _to_int(getattr(row, "HTAG", None)),
            "status": "FINISHED" if home_goals is not None and away_goals is not None
                      else "SCHEDULED",
            "source": SOURCE,
        })

    if not match_rows:
        report.seasons_failed[start_year] = "no parseable rows"
        return

    upsert(conn, fact_match, match_rows, MATCH_KEYS)
    report.matches += len(match_rows)

    match_ids = _match_id_map(conn, match_rows)
    report.stats_rows += _load_stats(conn, df, match_rows, match_ids)
    report.odds_rows += _load_odds(conn, df, match_rows, match_ids)
    report.seasons_loaded.append(start_year)


def _match_id_map(conn: sa.Connection, match_rows: Sequence[dict[str, Any]]) -> dict[tuple, int]:
    """Natural key -> match_id, for the rows just written."""
    dates = sorted({r["match_date"] for r in match_rows})
    stmt = sa.select(
        fact_match.c.match_id, fact_match.c.match_date,
        fact_match.c.home_team_id, fact_match.c.away_team_id,
    ).where(fact_match.c.match_date.between(dates[0], dates[-1]))
    return {
        (d, h, a): mid
        for mid, d, h, a in conn.execute(stmt).all()
    }


def _load_stats(conn: sa.Connection, df: pd.DataFrame, match_rows: Sequence[dict[str, Any]],
                match_ids: dict[tuple, int]) -> int:
    present = {src: dest for src, dest in STAT_COLUMNS.items() if src in df.columns}
    if not present:
        return 0
    rows: list[dict[str, Any]] = []
    for i, match in enumerate(match_rows):
        mid = match_ids.get((match["match_date"], match["home_team_id"], match["away_team_id"]))
        if mid is None:
            continue
        values = {dest: _to_int(df.iloc[i][src]) for src, dest in present.items()}
        if all(v is None for v in values.values()):
            continue
        row = {"match_id": mid}
        row.update({dest: values.get(dest) for dest in STAT_COLUMNS.values()})
        row["home_xg"] = None
        row["away_xg"] = None
        rows.append(row)
    if rows:
        # xG comes from FBref, if enabled; never clobber it with NULLs here.
        upsert(conn, fact_match_stats, rows, ["match_id"],
               update_cols=list(STAT_COLUMNS.values()))
    return len(rows)


def _load_odds(conn: sa.Connection, df: pd.DataFrame, match_rows: Sequence[dict[str, Any]],
               match_ids: dict[tuple, int]) -> int:
    rows: list[dict[str, Any]] = []
    for book, mapping in ODDS_COLUMNS.items():
        series = {field: _coalesce(df, names) for field, names in mapping.items()}
        if all(s is None for s in series.values()):
            continue
        for i, match in enumerate(match_rows):
            mid = match_ids.get((match["match_date"], match["home_team_id"], match["away_team_id"]))
            if mid is None:
                continue
            values = {f: (_to_float(s.iloc[i]) if s is not None else None)
                      for f, s in series.items()}
            if all(v is None for v in values.values()):
                continue
            rows.append({"match_id": mid, "book": book, **values})
    if rows:
        upsert(conn, market_odds, rows, ["match_id", "book"])
    return len(rows)


def backfill(
    conn: sa.Connection,
    registry: TeamRegistry,
    start_years: Iterable[int],
    raw_dir: Path,
    *,
    max_age_days: int = 7,
    session: requests.Session | None = None,
) -> IngestReport:
    """Download and load a run of seasons. Unreachable seasons are recorded."""
    report = IngestReport()
    for start_year in start_years:
        try:
            path = download_season(start_year, raw_dir, max_age_days=max_age_days,
                                   session=session)
        except SourceUnavailable as exc:
            log.warning("season unavailable", extra={"season": start_year, "error": str(exc)})
            report.seasons_failed[start_year] = str(exc)
            continue
        load_season(conn, registry, start_year, path, report)
    log.info("historical backfill complete", extra={"summary": report.summary()})
    return report
