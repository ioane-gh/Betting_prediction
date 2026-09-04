"""FBref enrichment via ``soccerdata`` -- optional.

FBref covers the Championship (Understat does not, so do not reach for it).
This module supplies two things the core model can live without:

* per-match team xG, stored on ``fact_match_stats``;
* per-player minutes and yellow/red cards, which Phase 5 turns into derived
  suspension risk.

Everything here is behind ``CHAMP_ENABLE_FBREF``. If the flag is off, or
``soccerdata`` is not installed, or the crawl fails, the caller records the
missing input and carries on -- it never substitutes a default.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa

from ..db import fact_match, fact_match_stats, upsert
from ..logging_setup import get_logger
from .seasons import season_code
from .teams import TeamRegistry, UnknownTeamError, resolve_name

log = get_logger(__name__)

SOURCE = "fbref"
LEAGUE = "ENG-Championship"
# Championship yellow-card accumulation: 5 in the first 19 matches earns a
# one-match ban, 10 by match 37, 15 by the end of the season.
YELLOW_THRESHOLDS = (5, 10, 15)


class FbrefUnavailable(RuntimeError):
    """soccerdata is missing, disabled, or the crawl failed."""


@dataclass
class FbrefReport:
    xg_rows: int = 0
    player_rows: int = 0
    error: str | None = None
    unresolved_names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def _make_reader(seasons: list[int], data_dir: Path, crawl_delay: float) -> Any:
    try:
        import soccerdata  # noqa: PLC0415  -- optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FbrefUnavailable(
            "soccerdata is not installed. `pip install 'champmodel[fbref]'` "
            "or leave CHAMP_ENABLE_FBREF=0."
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        return soccerdata.FBref(
            leagues=LEAGUE,
            seasons=[season_code(year) for year in seasons],
            data_dir=data_dir,
            # FBref asks for a courteous crawl rate; honour it.
            **({"delay": crawl_delay} if crawl_delay else {}),
        )
    except TypeError:
        # Older soccerdata releases take the delay via a module-level setting.
        return soccerdata.FBref(
            leagues=LEAGUE,
            seasons=[season_code(year) for year in seasons],
            data_dir=data_dir,
        )


def fetch_schedule(seasons: list[int], data_dir: Path, crawl_delay: float = 3.0) -> pd.DataFrame:
    """Match-level schedule including home_xg / away_xg."""
    reader = _make_reader(seasons, data_dir, crawl_delay)
    try:
        return reader.read_schedule().reset_index()
    except Exception as exc:  # network, parse, layout change
        raise FbrefUnavailable(f"FBref schedule crawl failed: {exc}") from exc


def fetch_player_season_stats(seasons: list[int], data_dir: Path,
                              crawl_delay: float = 3.0) -> pd.DataFrame:
    """Per-player season totals: minutes, yellows, reds."""
    reader = _make_reader(seasons, data_dir, crawl_delay)
    try:
        return reader.read_player_season_stats(stat_type="standard").reset_index()
    except Exception as exc:
        raise FbrefUnavailable(f"FBref player crawl failed: {exc}") from exc


def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    """FBref column names move around between soccerdata versions."""
    flat = {str(c).lower().replace(" ", "_"): c for c in df.columns}
    for name in candidates:
        key = name.lower().replace(" ", "_")
        if key in flat:
            return df[flat[key]]
    return None


def load_xg(conn: sa.Connection, registry: TeamRegistry, schedule: pd.DataFrame) -> FbrefReport:
    """Attach FBref xG to matches already in fact_match. Unmatched rows are skipped."""
    report = FbrefReport()
    home = _pick(schedule, "home_team", "home")
    away = _pick(schedule, "away_team", "away")
    date = _pick(schedule, "date", "match_date")
    home_xg = _pick(schedule, "home_xg", "hxg")
    away_xg = _pick(schedule, "away_xg", "axg")
    if home is None or away is None or date is None or home_xg is None or away_xg is None:
        report.error = f"unexpected FBref schedule columns: {list(schedule.columns)[:12]}"
        log.warning("fbref schedule unusable", extra={"error": report.error})
        return report

    dates = pd.to_datetime(date, errors="coerce")
    keyed: dict[tuple, tuple[float | None, float | None]] = {}
    bad: set[str] = set()
    for i in range(len(schedule)):
        if pd.isna(dates.iloc[i]):
            continue
        try:
            h_id = registry.team_id(str(home.iloc[i]), SOURCE)
            a_id = registry.team_id(str(away.iloc[i]), SOURCE)
        except UnknownTeamError:
            bad.add(str(home.iloc[i]))
            bad.add(str(away.iloc[i]))
            continue
        hx, ax = home_xg.iloc[i], away_xg.iloc[i]
        keyed[(dates.iloc[i].date(), h_id, a_id)] = (
            None if pd.isna(hx) else round(float(hx), 2),
            None if pd.isna(ax) else round(float(ax), 2),
        )

    if bad:
        # Only the names FBref uses that we could not map; xG is optional, so
        # this is reported rather than fatal.
        report.unresolved_names = sorted(n for n in bad if n and n != "nan")

    if not keyed:
        return report

    rows: list[dict[str, Any]] = []
    stmt = sa.select(fact_match.c.match_id, fact_match.c.match_date,
                     fact_match.c.home_team_id, fact_match.c.away_team_id)
    for match_id, day, h_id, a_id in conn.execute(stmt).all():
        found = keyed.get((day, h_id, a_id))
        if found and (found[0] is not None or found[1] is not None):
            rows.append({"match_id": match_id, "home_xg": found[0], "away_xg": found[1]})

    if rows:
        upsert(conn, fact_match_stats, rows, ["match_id"], update_cols=["home_xg", "away_xg"])
    report.xg_rows = len(rows)
    log.info("fbref xg loaded", extra={"rows": report.xg_rows})
    return report


def derive_suspension_risk(
    players: pd.DataFrame,
    matches_played: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Flag players at a yellow-card accumulation threshold or freshly sent off.

    Layer 2 of Phase 5: automatic, advisory, and always subordinate to the
    manual override table. Returns one row per at-risk player with the
    ``minutes_share`` the availability adjustment needs.
    """
    if players is None or players.empty:
        return pd.DataFrame(columns=["team", "player", "reason", "minutes_share", "detail"])

    team = _pick(players, "team", "squad")
    player = _pick(players, "player")
    minutes = _pick(players, "playing_time_min", "min", "minutes", "playing_time_minutes")
    yellows = _pick(players, "performance_crdy", "crdy", "cards_yellow", "yellow_cards")
    reds = _pick(players, "performance_crdr", "crdr", "cards_red", "red_cards")
    if team is None or player is None or minutes is None:
        log.warning("fbref player columns unusable",
                    extra={"columns": [str(c) for c in players.columns][:12]})
        return pd.DataFrame(columns=["team", "player", "reason", "minutes_share", "detail"])

    frame = pd.DataFrame({
        "team": team.astype(str),
        "player": player.astype(str),
        "minutes": pd.to_numeric(minutes, errors="coerce").fillna(0.0),
        "yellows": pd.to_numeric(yellows, errors="coerce").fillna(0.0) if yellows is not None else 0.0,
        "reds": pd.to_numeric(reds, errors="coerce").fillna(0.0) if reds is not None else 0.0,
    })

    # minutes_share is share of the team's outfield minutes (10 outfield + 1 GK
    # = 11 players * 90 minutes per match), capped at 1.
    team_minutes = frame.groupby("team")["minutes"].transform("sum") / 11.0
    frame["minutes_share"] = (frame["minutes"] / team_minutes.replace(0, pd.NA)).fillna(0.0).clip(0, 1)

    at_risk = []
    for row in frame.itertuples(index=False):
        if row.reds >= 1:
            at_risk.append((row.team, row.player, "SUSPENSION", row.minutes_share,
                            "red card in the current season"))
            continue
        for threshold in YELLOW_THRESHOLDS:
            if row.yellows == threshold - 1:
                at_risk.append((row.team, row.player, "DOUBT", row.minutes_share,
                                f"one yellow from the {threshold}-card ban"))
                break
            if row.yellows == threshold:
                at_risk.append((row.team, row.player, "SUSPENSION", row.minutes_share,
                                f"reached the {threshold}-card threshold"))
                break

    return pd.DataFrame(at_risk, columns=["team", "player", "reason", "minutes_share", "detail"])


def ingest(
    conn: sa.Connection,
    registry: TeamRegistry,
    seasons: list[int],
    data_dir: Path,
    *,
    crawl_delay: float = 3.0,
) -> FbrefReport:
    """Best-effort xG enrichment. Never raises; reports instead."""
    try:
        schedule = fetch_schedule(seasons, data_dir, crawl_delay)
    except FbrefUnavailable as exc:
        log.warning("fbref unavailable", extra={"error": str(exc)})
        return FbrefReport(error=str(exc))
    return load_xg(conn, registry, schedule)
