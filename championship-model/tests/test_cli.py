"""End-to-end CLI: schema, fit, backtest, predict, status against SQLite.

This is the test that would catch a change that breaks the daily run without
breaking any single unit -- a renamed column, a mis-ordered argument, a CSV
column that quietly stops being written.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from typer.testing import CliRunner

from champmodel import db as dbm
from champmodel import synth
from champmodel.cli import app
from champmodel.config import DbConfig
from champmodel.ingest.seasons import season_id_for_date, season_row, season_start_year
from champmodel.ingest.teams import ALIASES, TeamRegistry

runner = CliRunner()
FIXTURE_DAY = dt.date(2026, 9, 4)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A working directory with .env, a seeded database and today's fixtures."""
    data = tmp_path / "data"
    data.mkdir()
    db_url = f"sqlite:///{data / 'champ.db'}"
    monkeypatch.setenv("CHAMP_DB_URL", db_url)
    monkeypatch.setenv("CHAMP_DATA_DIR", str(data))
    monkeypatch.setenv("CHAMP_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CHAMP_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "")
    monkeypatch.chdir(tmp_path)

    engine = dbm.make_engine(DbConfig(url=db_url))
    dbm.create_schema(engine)

    real_names = sorted(ALIASES)[:16]
    league = synth.generate(n_seasons=3, n_teams=16,
                            first_season_start=dt.date(2023, 8, 5), seed=29)
    mapping = {f"Team {chr(65 + i)}": real_names[i] for i in range(16)}
    matches = league.matches.copy()
    matches["home_team"] = matches["home_team"].map(mapping)
    matches["away_team"] = matches["away_team"].map(mapping)

    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        seasons, rows = {}, []
        for match in matches.itertuples(index=False):
            sid = season_id_for_date(match.match_date)
            seasons.setdefault(sid, season_row(season_start_year(match.match_date)))
            rows.append({
                "season_id": sid, "match_date": match.match_date,
                "kickoff_utc": dt.datetime.combine(match.match_date, dt.time(15, 0)),
                "home_team_id": registry.team_id(match.home_team),
                "away_team_id": registry.team_id(match.away_team),
                "home_goals": int(match.home_goals), "away_goals": int(match.away_goals),
                "home_ht": None, "away_ht": None,
                "status": "FINISHED", "source": "synthetic",
            })
        seasons.setdefault(season_id_for_date(FIXTURE_DAY),
                           season_row(season_start_year(FIXTURE_DAY)))
        dbm.upsert(conn, dbm.dim_season, list(seasons.values()), ["season_id"])
        dbm.upsert(conn, dbm.fact_match, rows, ["match_date", "home_team_id", "away_team_id"])

        keyed = {(r[1], r[2], r[3]): r[0] for r in conn.execute(sa.select(
            dbm.fact_match.c.match_id, dbm.fact_match.c.match_date,
            dbm.fact_match.c.home_team_id, dbm.fact_match.c.away_team_id)).all()}
        dbm.upsert(conn, dbm.market_odds, [{
            "match_id": keyed[(r["match_date"], r["home_team_id"], r["away_team_id"])],
            "book": "Avg", "odds_over25": float(m.odds_over25),
            "odds_under25": float(m.odds_under25), "odds_btts_yes": float(m.odds_btts_yes),
            "odds_btts_no": float(m.odds_btts_no), "odds_home": None,
            "odds_draw": None, "odds_away": None,
        } for r, m in zip(rows, matches.itertuples(index=False))],
            ["match_id", "book"])

        # File the last matchday again as today's unplayed fixtures.
        last_day = max(r["match_date"] for r in rows)
        fixtures = [dict(r, match_date=FIXTURE_DAY,
                         kickoff_utc=dt.datetime.combine(FIXTURE_DAY, dt.time(19, 45)),
                         season_id=season_id_for_date(FIXTURE_DAY),
                         home_goals=None, away_goals=None, status="SCHEDULED",
                         source="synthetic-fixture")
                    for r in rows if r["match_date"] == last_day]
        dbm.upsert(conn, dbm.fact_match, fixtures,
                   ["match_date", "home_team_id", "away_team_id"])

    engine.dispose()
    return {"dir": tmp_path, "data": data, "fixtures": len(fixtures), "url": db_url}


def _run(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def test_help_lists_the_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "fit", "backtest", "predict", "status", "tune", "init-db"):
        assert command in result.output


def test_version_runs():
    assert runner.invoke(app, ["version"]).exit_code == 0


def test_status_reports_counts_and_a_clean_alias_gate(project):
    output = _run("status").output
    assert "unresolved aliases: 0" in output
    assert "must be zero" not in output
    assert "finished_matches" in output


def test_fit_stores_an_artifact(project):
    output = _run("fit", "--until", "2026-06-30", "--show", "5").output
    assert "home advantage gamma" in output
    artifact = project["data"] / "artifacts" / "dixon_coles_latest.json"
    assert artifact.exists()

    from champmodel.model.dixon_coles import DixonColesFit
    fit = DixonColesFit.load(artifact)
    assert fit.train_rows > 500
    assert 0.05 < fit.home_advantage < 0.50


def test_backtest_runs_and_reports_against_the_market(project):
    """Two seasons so the held-out half clears the 200-row floor below which
    Platt scaling refuses to fit."""
    output = _run("backtest", "--seasons", "2", "--refit-every", "14").output
    assert "no-leakage assertion passed" in output
    assert "market LL" in output
    assert "Calibration -- over25" in output
    assert "Platt-calibrated" in output
    assert (project["data"] / "artifacts" / "calibration_latest.json").exists()


def test_backtest_declines_to_calibrate_on_a_thin_window(project):
    """Better an honest identity calibrator than two coefficients fitted to
    noise, so no artifact is written when there is too little held-out data."""
    output = _run("backtest", "--seasons", "1", "--refit-every", "21").output
    assert "identity (uncalibrated)" in output
    assert not (project["data"] / "artifacts" / "calibration_latest.json").exists()


def test_predict_writes_the_table_the_csv_and_the_database(project):
    _run("fit", "--until", "2026-09-03")
    output = _run("predict", "--date", FIXTURE_DAY.isoformat(), "--no-ingest").output

    assert "P(BTTS)" in output and "P(O2.5)" in output
    assert "H2H BTTS" in output and "Edge" in output
    assert "not betting advice" in output

    csv_path = project["dir"] / "output" / f"{FIXTURE_DAY:%Y-%m-%d}.csv"
    assert csv_path.exists()
    import pandas as pd
    frame = pd.read_csv(csv_path)
    assert len(frame) == project["fixtures"]
    for column in ("p_btts", "p_over25", "lambda_home", "lambda_away",
                   "missing_inputs", "availability_applied", "h2h_meetings"):
        assert column in frame.columns
    assert frame["p_btts"].between(0, 1).all()
    assert frame["p_over25"].between(0, 1).all()

    engine = dbm.make_engine(DbConfig(url=project["url"]))
    with engine.connect() as conn:
        assert dbm.table_count(conn, dbm.prediction) == project["fixtures"]
        assert dbm.table_count(conn, dbm.model_run) == 1
    engine.dispose()


def test_predict_twice_replaces_rather_than_duplicates_per_run(project):
    _run("fit", "--until", "2026-09-03")
    _run("predict", "--date", FIXTURE_DAY.isoformat(), "--no-ingest")
    _run("predict", "--date", FIXTURE_DAY.isoformat(), "--no-ingest")

    engine = dbm.make_engine(DbConfig(url=project["url"]))
    with engine.connect() as conn:
        # Two runs, each holding one row per fixture -- history is kept, but no
        # run ever holds a duplicate.
        assert dbm.table_count(conn, dbm.model_run) == 2
        assert dbm.table_count(conn, dbm.prediction) == project["fixtures"] * 2
        duplicates = conn.execute(sa.text(
            "SELECT COUNT(*) FROM (SELECT run_id, match_id FROM champ.prediction "
            "GROUP BY run_id, match_id HAVING COUNT(*) > 1) d"
        )).scalar_one()
    engine.dispose()
    assert duplicates == 0


def test_predict_flags_absent_team_news(project):
    _run("fit", "--until", "2026-09-03")
    output = _run("predict", "--date", FIXTURE_DAY.isoformat(), "--no-ingest").output
    # The template is written on ingest, not predict, so a first run has none.
    assert "no_team_news" in output or "stale_team_news" in output


def test_predict_on_a_day_with_no_fixtures_is_not_an_error(project):
    _run("fit", "--until", "2026-09-03")
    result = runner.invoke(app, ["predict", "--date", "2026-12-25", "--no-ingest"])
    assert result.exit_code == 0
    assert "no Championship fixtures" in result.output


def test_backtest_synthetic_mode_warns_that_it_is_not_real(project):
    output = _run("backtest", "--synthetic", "--seasons", "1",
                  "--refit-every", "21", "--no-calibrate").output
    assert "SYNTHETIC" in output


def test_missing_driver_reports_cleanly_instead_of_a_traceback(project, monkeypatch):
    """The failure a user hits before installing pyodbc must be a short,
    actionable message and a non-zero exit -- not a rich traceback."""
    from sqlalchemy.connectors.pyodbc import PyODBCConnector

    def boom(cls):
        raise ModuleNotFoundError("No module named 'pyodbc'", name="pyodbc")

    monkeypatch.setattr(PyODBCConnector, "import_dbapi", classmethod(boom))
    monkeypatch.setenv("CHAMP_DB_URL",
                       "mssql+pyodbc://sa:x@localhost:1433/champ"
                       "?driver=ODBC+Driver+18+for+SQL+Server")

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "pyodbc" in result.output
    assert 'pip install -e ".[dev,mssql]"' in result.output
    assert "sqlite:///./data/champ.db" in result.output
    assert "Traceback" not in result.output


def test_unreachable_server_reports_cleanly(project, monkeypatch):
    monkeypatch.setenv("CHAMP_DB_URL", "postgresql+psycopg2://x@127.0.0.1:1/none")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "could not connect to" in result.output
    assert "Traceback" not in result.output


def test_no_env_file_warns_before_using_the_defaults(tmp_path, monkeypatch):
    """Silently falling back to localhost:1433 is how a user ends up debugging
    a connection they never configured."""
    monkeypatch.chdir(tmp_path)
    for key in [k for k in os.environ if k.startswith("CHAMP_DB_")]:
        monkeypatch.delenv(key, raising=False)

    result = runner.invoke(app, ["status"])
    assert "No .env found" in result.output
    assert ".env.example" in result.output


def test_ingest_backfill_uses_the_github_mirror_when_configured(tmp_path, monkeypatch):
    """CHAMP_HISTORICAL_SOURCE=github-mirror end to end: ingest --backfill
    loads from the cached mirror file, and backtest finds its odds under the
    right book without the user having to know the book name exists."""
    import shutil

    from champmodel import db as dbm
    from champmodel.config import DbConfig

    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    shutil.copy(Path(__file__).parent / "fixtures" / "mirror_sample.csv",
               data / "raw" / "mirror_matches.csv")

    db_url = f"sqlite:///{data / 'champ.db'}"
    monkeypatch.setenv("CHAMP_DB_URL", db_url)
    monkeypatch.setenv("CHAMP_DATA_DIR", str(data))
    monkeypatch.setenv("CHAMP_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CHAMP_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("CHAMP_HISTORICAL_SOURCE", "github-mirror")
    monkeypatch.chdir(tmp_path)

    engine = dbm.make_engine(DbConfig(url=db_url))
    dbm.create_schema(engine)
    engine.dispose()

    # The fixture's matches sit in the 2025-26 season; --seasons 2 reaches back
    # far enough to include it regardless of "today"'s real date.
    output = _run("ingest", "--backfill", "--seasons", "2",
                  "--max-age-days", "-1", "--no-availability").output
    assert "github-mirror" in output
    assert "unresolved aliases: 0" in output

    engine = dbm.make_engine(DbConfig(url=db_url))
    with engine.connect() as conn:
        assert dbm.table_count(conn, dbm.fact_match) == 2      # E0 row excluded
        book = conn.execute(sa.select(dbm.market_odds.c.book).limit(1)).scalar_one()
    engine.dispose()
    assert book == "Mirror"
