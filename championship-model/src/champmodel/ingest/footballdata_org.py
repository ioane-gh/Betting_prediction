"""football-data.org v4 -- today's fixture list for the Championship (ELC).

The free tier covers ELC but allows only 10 requests per minute, so this
client does three things carefully: it rate-limits itself, it caches every
response to disk keyed by the date range, and it never asks for the same range
twice in one run.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
import sqlalchemy as sa

from ..db import dim_season, dim_team, fact_match, upsert
from ..logging_setup import get_logger
from .footballdata_uk import MATCH_KEYS, SourceUnavailable
from .seasons import season_id_for_date, season_row, season_start_year
from .teams import TeamRegistry, UnknownTeamError, resolve_name

log = get_logger(__name__)

SOURCE = "football-data.org"
BASE_URL = "https://api.football-data.org/v4"
COMPETITION = "ELC"

# football-data.org statuses -> champ.fact_match.status
STATUS_MAP = {
    "SCHEDULED": "SCHEDULED",
    "TIMED": "SCHEDULED",
    "IN_PLAY": "IN_PLAY",
    "PAUSED": "IN_PLAY",
    "FINISHED": "FINISHED",
    "POSTPONED": "POSTPONED",
    "SUSPENDED": "POSTPONED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
    "AWARDED": "FINISHED",
}


class RateLimiter:
    """A simple sliding-window limiter: at most ``limit`` calls per ``period``."""

    def __init__(self, limit: int = 10, period: float = 60.0) -> None:
        self.limit = limit
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, *, sleeper=time.sleep, clock=time.monotonic) -> float:
        """Block until a slot is free. Returns how long it waited."""
        with self._lock:
            now = clock()
            while self._calls and now - self._calls[0] >= self.period:
                self._calls.popleft()
            waited = 0.0
            if len(self._calls) >= self.limit:
                waited = self.period - (now - self._calls[0]) + 0.01
                if waited > 0:
                    log.info("rate limit reached, sleeping", extra={"seconds": round(waited, 2)})
                    sleeper(waited)
                    now = clock()
                    while self._calls and now - self._calls[0] >= self.period:
                        self._calls.popleft()
            self._calls.append(clock())
            return waited


@dataclass
class FixtureReport:
    fetched: int = 0
    written: int = 0
    from_cache: bool = False
    error: str | None = None
    unresolved_names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.unresolved_names


class FootballDataOrgClient:
    """Rate-limited, disk-cached client for the free football-data.org tier."""

    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        *,
        limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.limiter = limiter or RateLimiter()
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self._seen: set[str] = set()   # ranges already fetched this run

    # -- caching -----------------------------------------------------------
    def _cache_path(self, date_from: dt.date, date_to: dt.date) -> Path:
        return self.cache_dir / f"fdorg_{COMPETITION}_{date_from:%Y%m%d}_{date_to:%Y%m%d}.json"

    def matches(
        self,
        date_from: dt.date,
        date_to: dt.date,
        *,
        use_cache: bool = True,
        max_age_hours: float = 6.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return (matches, came_from_cache) for a date range."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(date_from, date_to)
        key = path.name

        if use_cache and path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            # Within one run a range is fetched at most once, regardless of age.
            if key in self._seen or age_h <= max_age_hours:
                payload = json.loads(path.read_text(encoding="utf-8"))
                log.debug("fixture cache hit", extra={"range": key, "age_hours": round(age_h, 2)})
                return payload.get("matches", []), True

        if not self.api_key:
            if path.exists():
                log.warning("no API key; serving stale fixture cache", extra={"range": key})
                return json.loads(path.read_text(encoding="utf-8")).get("matches", []), True
            raise SourceUnavailable(
                "FOOTBALL_DATA_ORG_KEY is not set and no cached fixtures exist. "
                "Register for a free key at https://www.football-data.org/client/register"
            )

        payload = self._get(
            f"/competitions/{COMPETITION}/matches",
            {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()},
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._seen.add(key)
        return payload.get("matches", []), False

    # -- transport ---------------------------------------------------------
    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = BASE_URL + path
        headers = {"X-Auth-Token": self.api_key, "User-Agent": "champmodel/0.1"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                response = self.session.get(url, params=params, headers=headers,
                                            timeout=self.timeout)
                if response.status_code == 429:
                    backoff = float(response.headers.get("Retry-After", 2 ** attempt * 5))
                    log.warning("429 from football-data.org, backing off",
                                extra={"seconds": backoff})
                    time.sleep(backoff)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                sleep_for = 2 ** attempt
                log.warning("football-data.org request failed",
                            extra={"attempt": attempt + 1, "error": str(exc)})
                if attempt < self.max_retries - 1:
                    time.sleep(sleep_for)
        raise SourceUnavailable(f"could not fetch {url}: {last_error}")


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def parse_matches(payload: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the API's match objects into the fields fact_match needs."""
    out: list[dict[str, Any]] = []
    for match in payload:
        home = (match.get("homeTeam") or {}).get("name")
        away = (match.get("awayTeam") or {}).get("name")
        if not home or not away:
            continue
        utc_date = match.get("utcDate")
        kickoff = None
        if utc_date:
            kickoff = dt.datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
            kickoff = kickoff.astimezone(dt.timezone.utc).replace(tzinfo=None)
        score = ((match.get("score") or {}).get("fullTime") or {})
        half = ((match.get("score") or {}).get("halfTime") or {})
        out.append({
            "home_name": home,
            "away_name": away,
            "home_fd_org_id": (match.get("homeTeam") or {}).get("id"),
            "away_fd_org_id": (match.get("awayTeam") or {}).get("id"),
            "kickoff_utc": kickoff,
            "match_date": kickoff.date() if kickoff else None,
            "status": STATUS_MAP.get(str(match.get("status", "")).upper(), "SCHEDULED"),
            "home_goals": score.get("home"),
            "away_goals": score.get("away"),
            "home_ht": half.get("home"),
            "away_ht": half.get("away"),
        })
    return out


def load_fixtures(
    conn: sa.Connection,
    registry: TeamRegistry,
    matches: list[dict[str, Any]],
) -> FixtureReport:
    """Upsert fixtures. Results already in the database are never overwritten
    with NULLs -- football-data.co.uk stays authoritative for finished matches."""
    report = FixtureReport(fetched=len(matches))
    parsed = parse_matches(matches)

    bad: list[str] = []
    for row in parsed:
        for name in (row["home_name"], row["away_name"]):
            try:
                resolve_name(name, SOURCE)
            except UnknownTeamError:
                bad.append(name)
    if bad:
        report.unresolved_names = sorted(set(bad))
        raise UnknownTeamError(", ".join(report.unresolved_names), SOURCE)

    seasons: dict[int, dict[str, Any]] = {}
    match_rows: list[dict[str, Any]] = []
    for row in parsed:
        if row["match_date"] is None:
            continue
        sid = season_id_for_date(row["match_date"])
        seasons.setdefault(sid, season_row(season_start_year(row["match_date"])))
        match_rows.append({
            "season_id": sid,
            "match_date": row["match_date"],
            "kickoff_utc": row["kickoff_utc"],
            "home_team_id": registry.team_id(row["home_name"], SOURCE),
            "away_team_id": registry.team_id(row["away_name"], SOURCE),
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            "home_ht": row["home_ht"],
            "away_ht": row["away_ht"],
            "status": row["status"],
            "source": SOURCE,
        })

    if seasons:
        upsert(conn, dim_season, list(seasons.values()), ["season_id"])
    if match_rows:
        # A scored match from this feed may overwrite everything. A scoreless
        # one may refresh the kickoff time, but must not blank a result that
        # football-data.co.uk already supplied -- nor demote its status back to
        # SCHEDULED, which would drop a real result out of the training set.
        finished = [r for r in match_rows if r["home_goals"] is not None]
        pending = [r for r in match_rows if r["home_goals"] is None]
        if finished:
            upsert(conn, fact_match, finished, MATCH_KEYS)
        if pending:
            upsert(conn, fact_match, pending, MATCH_KEYS,
                   update_cols=["kickoff_utc", "season_id"])
            _refresh_pending_status(conn, pending)
        report.written = len(match_rows)

    _store_fd_org_ids(conn, registry, parsed)
    log.info("fixtures loaded", extra={"fetched": report.fetched, "written": report.written})
    return report


def _refresh_pending_status(conn: sa.Connection, pending: list[dict[str, Any]]) -> None:
    """Update status for unplayed matches only -- a postponement is news, a
    finished match reported without a score is not."""
    stmt = (
        sa.update(fact_match)
        .where(
            fact_match.c.match_date == sa.bindparam("b_date"),
            fact_match.c.home_team_id == sa.bindparam("b_home"),
            fact_match.c.away_team_id == sa.bindparam("b_away"),
            fact_match.c.status != "FINISHED",
        )
        .values(status=sa.bindparam("b_status"))
    )
    conn.execute(stmt, [
        {"b_date": r["match_date"], "b_home": r["home_team_id"],
         "b_away": r["away_team_id"], "b_status": r["status"]}
        for r in pending
    ])


def _store_fd_org_ids(conn: sa.Connection, registry: TeamRegistry,
                      parsed: list[dict[str, Any]]) -> None:
    """Keep the API's team ids on dim_team so future calls can use them."""
    ids: dict[str, int] = {}
    for row in parsed:
        for side in ("home", "away"):
            fd_id = row[f"{side}_fd_org_id"]
            if fd_id:
                ids[resolve_name(row[f"{side}_name"], SOURCE)] = int(fd_id)
    if not ids:
        return
    for canonical, fd_id in ids.items():
        conn.execute(
            sa.update(dim_team)
            .where(dim_team.c.canonical_name == canonical)
            .values(fd_org_id=fd_id)
        )


def ingest_day(
    conn: sa.Connection,
    registry: TeamRegistry,
    client: FootballDataOrgClient,
    day: dt.date,
    *,
    window_days: int = 0,
) -> FixtureReport:
    """Fetch and load fixtures for ``day`` (optionally a window around it)."""
    date_from = day - dt.timedelta(days=window_days)
    date_to = day + dt.timedelta(days=window_days)
    try:
        matches, cached = client.matches(date_from, date_to)
    except SourceUnavailable as exc:
        log.warning("fixture source unavailable", extra={"error": str(exc)})
        return FixtureReport(error=str(exc))
    report = load_fixtures(conn, registry, matches)
    report.from_cache = cached
    return report
