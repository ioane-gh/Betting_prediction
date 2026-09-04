"""Shared fixtures.

The tests run against SQLite (with a ``champ`` schema attached) so the suite
needs no SQL Server instance. Point ``CHAMP_TEST_DB_URL`` at a real SQL Server
to exercise the MERGE path instead:

    CHAMP_TEST_DB_URL='mssql+pyodbc://...' pytest
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from champmodel.config import Config, DbConfig, ModelParams  # noqa: E402
from champmodel import db as dbm  # noqa: E402


@pytest.fixture()
def engine(tmp_path):
    url = os.environ.get("CHAMP_TEST_DB_URL") or f"sqlite:///{tmp_path / 'champ_test.db'}"
    eng = dbm.make_engine(DbConfig(url=url))
    dbm.create_schema(eng)
    yield eng
    if not url.startswith("sqlite"):
        dbm.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def cfg(tmp_path) -> Config:
    return Config(
        db=DbConfig(url=f"sqlite:///{tmp_path / 'champ_test.db'}"),
        model=ModelParams(),
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
    )


@pytest.fixture()
def seeded(engine):
    """A tiny league: 4 teams, one season, ready for match inserts."""
    teams = ["Leeds United", "Norwich City", "Sheffield Wednesday", "Millwall"]
    with engine.begin() as conn:
        dbm.upsert(conn, dbm.dim_team,
                   [{"canonical_name": t, "fbref_id": None, "fd_org_id": None} for t in teams],
                   ["canonical_name"])
        dbm.upsert(conn, dbm.dim_season,
                   [{"season_id": 2526, "label": "2025-26",
                     "start_date": dt.date(2025, 8, 1), "end_date": dt.date(2026, 5, 31)}],
                   ["season_id"])
        ids = dict(conn.execute(sa.select(dbm.dim_team.c.canonical_name,
                                          dbm.dim_team.c.team_id)).all())
    return {"engine": engine, "team_ids": ids, "season_id": 2526}
