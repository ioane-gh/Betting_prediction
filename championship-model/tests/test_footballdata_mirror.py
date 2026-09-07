"""The GitHub-mirror fallback source: parsing, filtering, loading, and the
same resilience (retry/backoff/hard-timeout) machinery reused from the
primary source.

The fixture mirrors the real, verified column layout of
xgabora/Club-Football-Match-Data-2000-2025's Matches.csv -- confirmed by
fetching the live file directly while building this module, not guessed at.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import requests

from champmodel.config import Config
from champmodel.db import fact_match, fact_match_stats, market_odds, table_count
from champmodel.ingest.footballdata_mirror import (MirrorFileUnusable,
                                                    MirrorReport,
                                                    _championship_rows,
                                                    _is_stale, backfill,
                                                    check_names,
                                                    download_mirror,
                                                    load_mirror,
                                                    read_mirror_csv)
from champmodel.ingest.footballdata_uk import SourceUnavailable
from champmodel.ingest.teams import TeamRegistry, UnknownTeamError

FIXTURE = Path(__file__).parent / "fixtures" / "mirror_sample.csv"


class _FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class _ScriptedSession:
    def __init__(self, script):
        self.script = script
        self.calls = 0

    def get(self, *_args, **_kwargs):
        response = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return response


# -- parsing and filtering ---------------------------------------------------
def test_reads_the_real_column_layout():
    df = read_mirror_csv(FIXTURE)
    assert {"Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"} <= set(df.columns)
    assert len(df) == 4


def test_rejects_a_file_missing_expected_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Division,MatchDate\nE1,2025-08-09\n", encoding="utf-8")
    with pytest.raises(MirrorFileUnusable, match="missing column"):
        read_mirror_csv(bad)


def test_filters_to_championship_and_the_requested_window():
    df = read_mirror_csv(FIXTURE)
    frame = _championship_rows(df, [2025])
    # E0 (Arsenal/Chelsea) and the 2010 row are both excluded.
    assert len(frame) == 2
    assert set(frame["HomeTeam"]) == {"Sheffield Weds", "QPR"}


def test_check_names_flags_only_the_unresolved_ones():
    df = read_mirror_csv(FIXTURE)
    frame = _championship_rows(df, [2025])
    assert check_names(frame) == []           # every name here is a known alias


# -- loading ------------------------------------------------------------------
def test_load_mirror_populates_matches_stats_and_odds(engine):
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        report = load_mirror(conn, registry, FIXTURE, [2025])

    assert report.ok
    assert report.matches == 2                # E0 and the 2010 row excluded
    assert report.stats_rows == 2
    assert report.odds_rows == 2

    with engine.connect() as conn:
        assert table_count(conn, fact_match) == 2
        assert table_count(conn, fact_match_stats) == 2
        assert table_count(conn, market_odds) == 2


def test_btts_odds_are_null_the_rest_are_not(engine):
    import sqlalchemy as sa

    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        load_mirror(conn, registry, FIXTURE, [2025])
    with engine.connect() as conn:
        row = conn.execute(
            sa.select(market_odds).where(market_odds.c.book == "Mirror").limit(1)
        ).mappings().one()
    assert row["odds_btts_yes"] is None
    assert row["odds_btts_no"] is None
    assert row["odds_over25"] is not None
    assert row["odds_home"] is not None


def test_load_mirror_is_idempotent(engine):
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        load_mirror(conn, registry, FIXTURE, [2025])
        load_mirror(conn, registry, FIXTURE, [2025])
    with engine.connect() as conn:
        assert table_count(conn, fact_match) == 2


def test_load_mirror_hard_fails_on_an_unmapped_team(tmp_path, engine):
    text = FIXTURE.read_text().replace("Sheffield Weds", "Real Madrid")
    bad = tmp_path / "bad_mirror.csv"
    bad.write_text(text, encoding="utf-8")
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        with pytest.raises(UnknownTeamError):
            load_mirror(conn, registry, bad, [2025])


def test_kickoff_converts_uk_local_to_utc(engine):
    import sqlalchemy as sa

    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        load_mirror(conn, registry, FIXTURE, [2025])
    with engine.connect() as conn:
        kickoffs = sorted(conn.execute(sa.select(fact_match.c.kickoff_utc)).scalars().all())
    assert kickoffs[0] == dt.datetime(2025, 8, 9, 11, 30)  # 12:30 BST -> 11:30 UTC


def test_no_data_in_window_is_reported_not_fatal(engine):
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        report = load_mirror(conn, registry, FIXTURE, [1995])
    assert not report.ok
    assert "no Championship rows" in report.error


# -- download: retry, hard-timeout, proxy diagnostics reused from primary ----
def test_download_retries_a_transient_error_then_succeeds(tmp_path):
    good = FIXTURE.read_bytes()
    session = _ScriptedSession([_FakeResponse(503), _FakeResponse(200, content=good)])
    path = download_mirror(tmp_path, session=session, sleeper=lambda _s: None)
    assert path.read_bytes() == good


def test_download_rejects_a_non_csv_body_without_retrying(tmp_path):
    session = _ScriptedSession([_FakeResponse(200, content=b"<html>nope</html>")])
    with pytest.raises(MirrorFileUnusable):
        download_mirror(tmp_path, session=session, sleeper=lambda _s: None)
    assert session.calls == 1


def test_download_survives_a_hanging_connection(tmp_path):
    import time as time_mod

    class AlwaysHangs:
        @staticmethod
        def get(*_a, **_k):
            time_mod.sleep(999)

    start = time_mod.monotonic()
    with pytest.raises(SourceUnavailable):
        download_mirror(tmp_path, session=AlwaysHangs(), sleeper=lambda _s: None,
                        hard_timeout=0.2, max_retries=2)
    assert time_mod.monotonic() - start < 3.0


def test_download_uses_a_fresh_cache_without_touching_the_network(tmp_path):
    cached = tmp_path / "mirror_matches.csv"
    cached.write_bytes(FIXTURE.read_bytes())

    class Explode:
        @staticmethod
        def get(*_a, **_k):
            raise AssertionError("must not hit the network with a fresh cache")

    path = download_mirror(tmp_path, max_age_days=3, session=Explode())
    assert path == cached


def test_is_stale_matches_the_primary_sources_semantics(tmp_path):
    path = tmp_path / "mirror_matches.csv"
    path.write_bytes(b"data")
    assert _is_stale(path, 0) is True
    assert _is_stale(path, -1) is False
    assert _is_stale(tmp_path / "absent.csv", -1) is True


# -- end to end via backfill() ------------------------------------------------
def test_backfill_end_to_end(engine, tmp_path):
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        report = backfill(conn, registry, [2025], tmp_path, session=_ScriptedSession(
            [_FakeResponse(200, content=FIXTURE.read_bytes())]
        ))
    assert report.ok
    assert report.matches == 2


def test_backfill_reports_rather_than_raises_when_unreachable(engine, tmp_path):
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        report = backfill(conn, registry, [2025], tmp_path,
                          session=_ScriptedSession([_FakeResponse(503)] * 5),
                          sleeper=lambda _s: None)
    assert not report.ok
    assert report.error


# -- config wiring -------------------------------------------------------------
def test_benchmark_book_follows_the_configured_source():
    assert Config(historical_source="co-uk").benchmark_book == "Avg"
    assert Config(historical_source="github-mirror").benchmark_book == "Mirror"


def test_historical_source_reads_the_env_var(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("CHAMP_HISTORICAL_SOURCE=github-mirror\n", encoding="utf-8")
    assert Config.load().historical_source == "github-mirror"
    assert Config.load().benchmark_book == "Mirror"
