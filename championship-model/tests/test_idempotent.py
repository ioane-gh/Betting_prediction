"""Phase 9: running the same ingest twice must leave row counts unchanged.

Every load in the project goes through db.upsert on a natural key, so a rerun
updates in place. A daily job that duplicated rows would inflate the training
set with copies of the same matches and quietly distort every fit after it.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from champmodel import db as dbm
from champmodel.ingest import availability as availability_mod
from champmodel.ingest.footballdata_org import FootballDataOrgClient, ingest_day
from champmodel.ingest.footballdata_uk import IngestReport, load_season
from champmodel.ingest.teams import TeamRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "E1_sample.csv"

TRACKED = ("dim_team", "team_alias", "dim_season", "fact_match",
           "fact_match_stats", "market_odds", "availability_override")


def _counts(engine) -> dict[str, int]:
    with engine.connect() as conn:
        return {name: dbm.table_count(conn, dbm.metadata.tables[f"{dbm.SCHEMA}.{name}"])
                for name in TRACKED}


def _load_historical(engine) -> IngestReport:
    report = IngestReport()
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        load_season(conn, registry, 2025, FIXTURE, report)
    return report


def test_historical_ingest_is_idempotent(engine):
    _load_historical(engine)
    first = _counts(engine)
    _load_historical(engine)
    second = _counts(engine)
    assert first == second
    assert first["fact_match"] == 4          # the fixture holds four matches
    assert first["market_odds"] == 7         # Avg for 3 played, B365 for all 4


def test_reingest_fills_in_a_result_without_adding_a_row(engine):
    """The scheduled fourth fixture gets its score later; the row must update."""
    import sqlalchemy as sa

    _load_historical(engine)
    with engine.connect() as conn:
        before = dbm.table_count(conn, dbm.fact_match)
        row = conn.execute(
            sa.select(dbm.fact_match).where(dbm.fact_match.c.status == "SCHEDULED")
        ).mappings().one()

    with engine.begin() as conn:
        dbm.upsert(conn, dbm.fact_match, [{
            "season_id": row["season_id"], "match_date": row["match_date"],
            "kickoff_utc": row["kickoff_utc"], "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"], "home_goals": 2, "away_goals": 2,
            "home_ht": 1, "away_ht": 1, "status": "FINISHED", "source": "test",
        }], ["match_date", "home_team_id", "away_team_id"])

    with engine.connect() as conn:
        assert dbm.table_count(conn, dbm.fact_match) == before
        updated = conn.execute(
            sa.select(dbm.fact_match).where(dbm.fact_match.c.match_id == row["match_id"])
        ).mappings().one()
    assert updated["home_goals"] == 2
    assert updated["status"] == "FINISHED"


def test_fixture_ingest_is_idempotent(engine, tmp_path):
    payload = {"matches": [{
        "id": 1, "utcDate": "2026-09-04T18:45:00Z", "status": "TIMED",
        "homeTeam": {"id": 345, "name": "Sheffield Wednesday FC"},
        "awayTeam": {"id": 351, "name": "Queens Park Rangers FC"},
        "score": {"fullTime": {"home": None, "away": None},
                  "halfTime": {"home": None, "away": None}},
    }]}
    day = dt.date(2026, 9, 4)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"fdorg_ELC_{day:%Y%m%d}_{day:%Y%m%d}.json").write_text(json.dumps(payload))
    client = FootballDataOrgClient(api_key="", cache_dir=cache)

    for _ in range(2):
        with engine.begin() as conn:
            registry = TeamRegistry.sync(conn)
            report = ingest_day(conn, registry, client, day)
        assert report.ok
    with engine.connect() as conn:
        assert dbm.table_count(conn, dbm.fact_match) == 1


def test_fixture_ingest_never_blanks_an_existing_result(engine, tmp_path):
    """football-data.co.uk owns finished matches; a NULL score from the fixture
    feed must not overwrite one."""
    import sqlalchemy as sa

    _load_historical(engine)
    with engine.connect() as conn:
        played = conn.execute(
            sa.select(dbm.fact_match).where(dbm.fact_match.c.status == "FINISHED")
            .order_by(dbm.fact_match.c.match_id)
        ).mappings().first()
        home = conn.execute(sa.select(dbm.dim_team.c.canonical_name)
                            .where(dbm.dim_team.c.team_id == played["home_team_id"])).scalar_one()
        away = conn.execute(sa.select(dbm.dim_team.c.canonical_name)
                            .where(dbm.dim_team.c.team_id == played["away_team_id"])).scalar_one()

    day = played["match_date"]
    payload = {"matches": [{
        "id": 99, "utcDate": f"{day.isoformat()}T14:00:00Z", "status": "TIMED",
        "homeTeam": {"id": 1, "name": home}, "awayTeam": {"id": 2, "name": away},
        "score": {"fullTime": {"home": None, "away": None},
                  "halfTime": {"home": None, "away": None}},
    }]}
    cache = tmp_path / "cache2"
    cache.mkdir()
    (cache / f"fdorg_ELC_{day:%Y%m%d}_{day:%Y%m%d}.json").write_text(json.dumps(payload))

    with engine.begin() as conn:
        ingest_day(conn, TeamRegistry.load(conn),
                   FootballDataOrgClient(api_key="", cache_dir=cache), day)

    with engine.connect() as conn:
        after = conn.execute(
            sa.select(dbm.fact_match).where(dbm.fact_match.c.match_id == played["match_id"])
        ).mappings().one()
    assert after["home_goals"] == played["home_goals"]
    assert after["status"] == "FINISHED"


def test_availability_load_is_idempotent(engine, tmp_path):
    path = tmp_path / "availability.csv"
    path.write_text(
        "team,player_name,reason,minutes_share,is_attacker,is_defender,"
        "valid_from,valid_to,note,source_url\n"
        "Sheffield Weds,A Player,INJURY,0.4,1,0,2026-09-01,,knock,\n"
        "QPR,B Player,SUSPENSION,0.3,0,1,2026-09-01,2026-09-08,red,\n",
        encoding="utf-8",
    )
    for _ in range(2):
        with engine.begin() as conn:
            registry = TeamRegistry.sync(conn)
            report = availability_mod.load_csv(conn, registry, path)
        assert report.ok and report.rows_loaded == 2
    with engine.connect() as conn:
        assert dbm.table_count(conn, dbm.availability_override) == 2


def test_predictions_upsert_rather_than_duplicate(engine, seeded):
    import sqlalchemy as sa

    from champmodel.config import ModelParams
    from champmodel.repository import create_model_run, write_predictions

    with engine.begin() as conn:
        dbm.upsert(conn, dbm.fact_match, [{
            "season_id": seeded["season_id"], "match_date": dt.date(2026, 9, 4),
            "kickoff_utc": None, "home_team_id": seeded["team_ids"]["Leeds United"],
            "away_team_id": seeded["team_ids"]["Millwall"], "home_goals": None,
            "away_goals": None, "home_ht": None, "away_ht": None,
            "status": "SCHEDULED", "source": "test",
        }], ["match_date", "home_team_id", "away_team_id"])
        match_id = conn.execute(sa.select(dbm.fact_match.c.match_id)).scalar_one()
        run_id = create_model_run(conn, ModelParams(), train_rows=100,
                                  train_end_date=dt.date(2026, 9, 3))

    row = {"run_id": run_id, "match_id": match_id, "lambda_home": 1.5,
           "lambda_away": 1.1, "p_btts": 0.51, "p_over25": 0.49, "p_home": 0.44,
           "p_draw": 0.27, "p_away": 0.29, "availability_applied": False,
           "missing_inputs": None}
    with engine.begin() as conn:
        write_predictions(conn, run_id, [row])
    with engine.begin() as conn:
        write_predictions(conn, run_id, [dict(row, p_over25=0.55)])

    with engine.connect() as conn:
        assert dbm.table_count(conn, dbm.prediction) == 1
        stored = conn.execute(sa.select(dbm.prediction.c.p_over25)).scalar_one()
    assert float(stored) == pytest.approx(0.55)
