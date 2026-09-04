"""Phase 2 gate: the schema applies to an empty database and a match round-trips."""

from __future__ import annotations

import datetime as dt

import sqlalchemy as sa

from champmodel import db as dbm


def _match_row(seeded, **overrides):
    row = {
        "season_id": seeded["season_id"],
        "match_date": dt.date(2025, 8, 9),
        "kickoff_utc": dt.datetime(2025, 8, 9, 14, 0),
        "home_team_id": seeded["team_ids"]["Leeds United"],
        "away_team_id": seeded["team_ids"]["Norwich City"],
        "home_goals": 2,
        "away_goals": 1,
        "home_ht": 1,
        "away_ht": 0,
        "status": "FINISHED",
        "source": "test",
    }
    row.update(overrides)
    return row


def test_schema_creates_every_table(engine):
    inspector = sa.inspect(engine)
    tables = set(inspector.get_table_names(schema=dbm.SCHEMA))
    expected = {t.name for t in dbm.metadata.sorted_tables}
    assert expected <= tables, f"missing: {expected - tables}"


def test_match_round_trip(engine, seeded):
    keys = ["match_date", "home_team_id", "away_team_id"]
    with engine.begin() as conn:
        dbm.upsert(conn, dbm.fact_match, [_match_row(seeded)], keys)

    with engine.connect() as conn:
        row = conn.execute(sa.select(dbm.fact_match)).mappings().one()

    assert row["home_goals"] == 2
    assert row["away_goals"] == 1
    assert row["status"] == "FINISHED"
    assert row["match_date"] == dt.date(2025, 8, 9)
    assert row["match_id"] is not None


def test_upsert_updates_rather_than_duplicates(engine, seeded):
    keys = ["match_date", "home_team_id", "away_team_id"]
    with engine.begin() as conn:
        dbm.upsert(conn, dbm.fact_match, [_match_row(seeded, home_goals=None,
                                                     away_goals=None, status="SCHEDULED")], keys)
    with engine.begin() as conn:
        dbm.upsert(conn, dbm.fact_match, [_match_row(seeded)], keys)

    with engine.connect() as conn:
        assert dbm.table_count(conn, dbm.fact_match) == 1
        row = conn.execute(sa.select(dbm.fact_match)).mappings().one()
    assert row["status"] == "FINISHED"
    assert row["home_goals"] == 2


def test_scheduled_match_allows_null_goals(engine, seeded):
    keys = ["match_date", "home_team_id", "away_team_id"]
    with engine.begin() as conn:
        dbm.upsert(conn, dbm.fact_match,
                   [_match_row(seeded, home_goals=None, away_goals=None,
                               home_ht=None, away_ht=None, status="SCHEDULED")], keys)
    with engine.connect() as conn:
        row = conn.execute(sa.select(dbm.fact_match)).mappings().one()
    assert row["home_goals"] is None
    assert row["status"] == "SCHEDULED"


def test_split_batches_handles_go_separator():
    script = "CREATE TABLE a (x INT);\nGO\nCREATE TABLE b (y INT);\nGO\n"
    assert dbm.split_batches(script) == ["CREATE TABLE a (x INT);", "CREATE TABLE b (y INT);"]
