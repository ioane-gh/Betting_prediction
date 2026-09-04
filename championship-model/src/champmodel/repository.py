"""Reads and writes between the database and the frames the model works in."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Sequence

import pandas as pd
import sqlalchemy as sa

from . import MODEL_VERSION
from .config import ModelParams
from .db import (dim_team, fact_match, market_odds, model_run, prediction, upsert)
from .logging_setup import get_logger

log = get_logger(__name__)

BENCHMARK_BOOK = "Avg"


def _match_select() -> sa.Select:
    home = dim_team.alias("home")
    away = dim_team.alias("away")
    return sa.select(
        fact_match.c.match_id,
        fact_match.c.match_date,
        fact_match.c.kickoff_utc,
        fact_match.c.season_id,
        fact_match.c.status,
        fact_match.c.home_team_id,
        fact_match.c.away_team_id,
        home.c.canonical_name.label("home_team"),
        away.c.canonical_name.label("away_team"),
        fact_match.c.home_goals,
        fact_match.c.away_goals,
    ).join_from(
        fact_match, home, fact_match.c.home_team_id == home.c.team_id
    ).join(
        away, fact_match.c.away_team_id == away.c.team_id
    )


def _attach_odds(conn: sa.Connection, frame: pd.DataFrame,
                 book: str = BENCHMARK_BOOK) -> pd.DataFrame:
    """Left-join the benchmark book's prices. Missing prices stay NaN."""
    if frame.empty:
        for col in ("odds_over25", "odds_under25", "odds_btts_yes", "odds_btts_no",
                    "odds_home", "odds_draw", "odds_away"):
            frame[col] = pd.Series(dtype=float)
        return frame

    stmt = sa.select(
        market_odds.c.match_id, market_odds.c.odds_over25, market_odds.c.odds_under25,
        market_odds.c.odds_btts_yes, market_odds.c.odds_btts_no,
        market_odds.c.odds_home, market_odds.c.odds_draw, market_odds.c.odds_away,
    ).where(market_odds.c.book == book)
    odds = pd.DataFrame(conn.execute(stmt).mappings().all())
    if odds.empty:
        for col in ("odds_over25", "odds_under25", "odds_btts_yes", "odds_btts_no",
                    "odds_home", "odds_draw", "odds_away"):
            frame[col] = float("nan")
        return frame
    for col in odds.columns:
        if col != "match_id":
            odds[col] = pd.to_numeric(odds[col], errors="coerce")
    return frame.merge(odds, on="match_id", how="left")


def load_matches(
    conn: sa.Connection,
    *,
    finished_only: bool = True,
    until: dt.date | None = None,
    since: dt.date | None = None,
    with_odds: bool = True,
    book: str = BENCHMARK_BOOK,
) -> pd.DataFrame:
    """Results (and optionally fixtures) as a model-ready frame."""
    stmt = _match_select()
    if finished_only:
        stmt = stmt.where(
            sa.and_(fact_match.c.status == "FINISHED",
                    fact_match.c.home_goals.isnot(None),
                    fact_match.c.away_goals.isnot(None))
        )
    if until is not None:
        stmt = stmt.where(fact_match.c.match_date <= until)
    if since is not None:
        stmt = stmt.where(fact_match.c.match_date >= since)
    stmt = stmt.order_by(fact_match.c.match_date)

    frame = pd.DataFrame(conn.execute(stmt).mappings().all())
    if frame.empty:
        frame = pd.DataFrame(columns=[
            "match_id", "match_date", "kickoff_utc", "season_id", "status",
            "home_team_id", "away_team_id", "home_team", "away_team",
            "home_goals", "away_goals",
        ])
    else:
        frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date
        for col in ("home_goals", "away_goals"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return _attach_odds(conn, frame, book) if with_odds else frame


def load_fixtures(conn: sa.Connection, day: dt.date,
                  book: str = BENCHMARK_BOOK) -> pd.DataFrame:
    """Every fixture scheduled on ``day``, played or not."""
    stmt = _match_select().where(fact_match.c.match_date == day).order_by(
        fact_match.c.kickoff_utc.nulls_last(), fact_match.c.match_id
    )
    frame = pd.DataFrame(conn.execute(stmt).mappings().all())
    if frame.empty:
        return pd.DataFrame(columns=[
            "match_id", "match_date", "kickoff_utc", "season_id", "status",
            "home_team_id", "away_team_id", "home_team", "away_team",
            "home_goals", "away_goals", "odds_over25", "odds_under25",
            "odds_btts_yes", "odds_btts_no", "odds_home", "odds_draw", "odds_away",
        ])
    frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date
    return _attach_odds(conn, frame, book)


def create_model_run(
    conn: sa.Connection,
    params: ModelParams,
    *,
    train_rows: int,
    train_end_date: dt.date | None,
    extra: dict[str, Any] | None = None,
    model_version: str = MODEL_VERSION,
) -> int:
    """Insert a model_run row and return its run_id."""
    payload: dict[str, Any] = {"model": params.to_dict()}
    if extra:
        payload.update(extra)
    result = conn.execute(
        sa.insert(model_run).values(
            run_utc=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            model_version=model_version,
            params=json.dumps(payload, default=str),
            train_rows=int(train_rows),
            train_end_date=train_end_date,
        )
    )
    run_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
    if run_id is None:  # pragma: no cover - driver dependent
        run_id = conn.execute(sa.select(sa.func.max(model_run.c.run_id))).scalar_one()
    log.info("model run created", extra={"run_id": int(run_id), "train_rows": train_rows})
    return int(run_id)


def write_predictions(conn: sa.Connection, run_id: int,
                      rows: Sequence[dict[str, Any]]) -> int:
    """Upsert prediction rows. Re-running a day replaces, never duplicates."""
    usable = [row for row in rows if row.get("match_id") is not None]
    if not usable:
        return 0
    upsert(conn, prediction, usable, ["run_id", "match_id"])
    return len(usable)


def latest_run(conn: sa.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        sa.select(model_run).order_by(model_run.c.run_utc.desc(),
                                      model_run.c.run_id.desc()).limit(1)
    ).mappings().first()
    return dict(row) if row else None


def unresolved_alias_count(conn: sa.Connection) -> int:
    """The Phase 3 gate: alias rows that point at no team."""
    from .db import team_alias
    stmt = (
        sa.select(sa.func.count())
        .select_from(team_alias.outerjoin(dim_team, team_alias.c.team_id == dim_team.c.team_id))
        .where(dim_team.c.team_id.is_(None))
    )
    return int(conn.execute(stmt).scalar_one())


def counts(conn: sa.Connection) -> dict[str, int]:
    """Row counts for the status command."""
    from .db import availability_override, dim_season, team_alias
    out: dict[str, int] = {}
    for name, table in (
        ("teams", dim_team), ("aliases", team_alias), ("seasons", dim_season),
        ("matches", fact_match), ("odds", market_odds),
        ("availability", availability_override), ("runs", model_run),
        ("predictions", prediction),
    ):
        out[name] = int(conn.execute(sa.select(sa.func.count()).select_from(table)).scalar_one())
    finished = conn.execute(
        sa.select(sa.func.count()).select_from(fact_match)
        .where(fact_match.c.status == "FINISHED")
    ).scalar_one()
    out["finished_matches"] = int(finished)
    return out
