"""Parsing, caching, rate limiting, and graceful degradation."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import requests

from champmodel.config import Config, DbConfig, ModelParams
from champmodel.ingest import availability as availability_mod
from champmodel.ingest.fbref import derive_suspension_risk
from champmodel.ingest.footballdata_org import (RateLimiter, FootballDataOrgClient,
                                                parse_matches)
from champmodel.ingest.footballdata_uk import (IngestReport, SeasonFileUnusable,
                                               SourceUnavailable, _coalesce,
                                               backfill, download_season,
                                               load_season, parse_dates,
                                               read_season_csv)
from champmodel.ingest.seasons import (recent_start_years, season_code, season_id,
                                       season_id_for_date, season_label)
from champmodel.ingest.teams import TeamRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "E1_sample.csv"


# -- seasons ---------------------------------------------------------------
def test_season_codes():
    assert season_code(2025) == "2526"
    assert season_code(1999) == "9900"
    assert season_id(2025) == 2526
    assert season_label(2025) == "2025-26"


def test_season_boundary_is_july():
    assert season_id_for_date(dt.date(2026, 5, 31)) == 2526
    assert season_id_for_date(dt.date(2026, 7, 1)) == 2627


def test_recent_start_years_is_ordered_and_inclusive():
    years = recent_start_years(10, dt.date(2026, 9, 4))
    assert len(years) == 10
    assert years == sorted(years)
    assert years[-1] == 2026


# -- football-data.co.uk ---------------------------------------------------
def test_parse_dates_handles_both_year_formats():
    parsed = parse_dates(pd.Series(["09/08/2025", "10/08/25"]))
    assert parsed.iloc[0].date() == dt.date(2025, 8, 9)
    assert parsed.iloc[1].date() == dt.date(2025, 8, 10)


def test_read_season_csv_drops_the_trailing_blank_row():
    assert len(read_season_csv(FIXTURE)) == 4


def test_coalesce_prefers_closing_odds_then_falls_back():
    frame = pd.DataFrame({"AvgC>2.5": [1.90, None], "Avg>2.5": [2.00, 1.75]})
    values = _coalesce(frame, ("AvgC>2.5", "Avg>2.5"))
    assert values.iloc[0] == pytest.approx(1.90)   # closing wins
    assert values.iloc[1] == pytest.approx(1.75)   # falls back per row


def test_kickoff_times_convert_from_uk_local_to_utc(engine):
    """12:30 on an August Saturday is BST, so 11:30 UTC. Getting this wrong
    would file evening kickoffs on the wrong day for half the year."""
    import sqlalchemy as sa

    from champmodel.db import fact_match

    report = IngestReport()
    with engine.begin() as conn:
        load_season(conn, TeamRegistry.sync(conn), 2025, FIXTURE, report)
        kickoffs = conn.execute(
            sa.select(fact_match.c.kickoff_utc).order_by(fact_match.c.match_id)
        ).scalars().all()
    assert kickoffs[0] == dt.datetime(2025, 8, 9, 11, 30)


def test_stats_and_odds_land_in_their_tables(engine):
    report = IngestReport()
    with engine.begin() as conn:
        load_season(conn, TeamRegistry.sync(conn), 2025, FIXTURE, report)
    assert report.matches == 4
    assert report.stats_rows == 3       # the unplayed fixture has no stats
    assert report.odds_rows == 7
    assert report.ok


def test_download_falls_back_to_a_stale_cache(tmp_path, monkeypatch):
    """A blocked or flaky source must degrade to the cached copy, not die."""
    cached = tmp_path / "E1_2526.csv"
    cached.write_bytes(FIXTURE.read_bytes())
    # Make it look old enough to be refreshed.
    import os
    old = dt.datetime.now().timestamp() - 60 * 60 * 24 * 30
    os.utime(cached, (old, old))

    class Boom:
        @staticmethod
        def get(*_args, **_kwargs):
            raise requests.ConnectionError("egress policy denied CONNECT")

    path = download_season(2025, tmp_path, max_age_days=7, session=Boom(),
                           sleeper=lambda _s: None)
    assert path == cached


def test_download_raises_when_there_is_no_cache(tmp_path):
    class Boom:
        @staticmethod
        def get(*_args, **_kwargs):
            raise requests.ConnectionError("no route to host")

    with pytest.raises(SourceUnavailable):
        download_season(2025, tmp_path, session=Boom(),
                        sleeper=lambda _s: None)


# -- football-data.org -----------------------------------------------------
def test_rate_limiter_allows_the_quota_then_sleeps():
    slept: list[float] = []
    clock = [0.0]
    limiter = RateLimiter(limit=10, period=60.0)

    def sleeper(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    for _ in range(10):
        limiter.acquire(sleeper=sleeper, clock=lambda: clock[0])
    assert slept == []                    # ten free calls
    limiter.acquire(sleeper=sleeper, clock=lambda: clock[0])
    assert len(slept) == 1 and slept[0] == pytest.approx(60.0, abs=0.1)


def test_parse_matches_flattens_the_api_shape():
    parsed = parse_matches([{
        "utcDate": "2026-09-04T18:45:00Z", "status": "TIMED",
        "homeTeam": {"id": 345, "name": "Sheffield Wednesday FC"},
        "awayTeam": {"id": 351, "name": "Queens Park Rangers FC"},
        "score": {"fullTime": {"home": None, "away": None},
                  "halfTime": {"home": None, "away": None}},
    }])
    assert len(parsed) == 1
    assert parsed[0]["match_date"] == dt.date(2026, 9, 4)
    assert parsed[0]["kickoff_utc"] == dt.datetime(2026, 9, 4, 18, 45)
    assert parsed[0]["status"] == "SCHEDULED"


def test_client_serves_the_cache_without_a_key(tmp_path):
    day = dt.date(2026, 9, 4)
    (tmp_path / f"fdorg_ELC_{day:%Y%m%d}_{day:%Y%m%d}.json").write_text(
        json.dumps({"matches": [{"id": 1}]})
    )
    client = FootballDataOrgClient(api_key="", cache_dir=tmp_path)
    matches, cached = client.matches(day, day)
    assert cached and len(matches) == 1


def test_client_without_a_key_or_cache_says_so(tmp_path):
    client = FootballDataOrgClient(api_key="", cache_dir=tmp_path)
    with pytest.raises(SourceUnavailable, match="FOOTBALL_DATA_ORG_KEY"):
        client.matches(dt.date(2026, 9, 4), dt.date(2026, 9, 4))


# -- availability ----------------------------------------------------------
def test_availability_rejects_a_bad_minutes_share(engine, tmp_path):
    """The 4.5-instead-of-0.45 typo must be rejected, not clipped: clipping
    would hide it and the cap downstream would quietly absorb it."""
    path = tmp_path / "availability.csv"
    path.write_text(
        "team,player_name,reason,minutes_share,is_attacker,is_defender,valid_from,valid_to,note,source_url\n"
        "Leeds,Good Row,INJURY,0.4,1,0,2026-09-01,,,\n"
        "Leeds,Bad Share,INJURY,4.5,1,0,2026-09-01,,,\n"
        "Leeds,Bad Reason,SICK,0.2,1,0,2026-09-01,,,\n"
        "Atlantis,Ghost,INJURY,0.2,1,0,2026-09-01,,,\n"
        "Leeds,No Date,INJURY,0.2,1,0,,,,\n",
        encoding="utf-8",
    )
    with engine.begin() as conn:
        report = availability_mod.load_csv(conn, TeamRegistry.sync(conn), path)
    assert report.rows_loaded == 1
    assert len(report.errors) == 4
    assert any("minutes_share" in e for e in report.errors)
    assert any("Atlantis" in e for e in report.errors)


def test_active_overrides_respects_the_validity_window(engine, tmp_path):
    path = tmp_path / "availability.csv"
    path.write_text(
        "team,player_name,reason,minutes_share,is_attacker,is_defender,valid_from,valid_to,note,source_url\n"
        "Leeds,Long Term,INJURY,0.4,1,0,2026-09-01,,,\n"
        "Leeds,Short Ban,SUSPENSION,0.3,0,1,2026-09-01,2026-09-05,,\n",
        encoding="utf-8",
    )
    with engine.begin() as conn:
        availability_mod.load_csv(conn, TeamRegistry.sync(conn), path)
        during = availability_mod.active_overrides(conn, dt.date(2026, 9, 4))
        after = availability_mod.active_overrides(conn, dt.date(2026, 9, 20))
        before = availability_mod.active_overrides(conn, dt.date(2026, 8, 20))
    assert len(during) == 2
    assert [row["player_name"] for row in after] == ["Long Term"]
    assert before == []


def test_template_is_written_only_once(tmp_path):
    path = tmp_path / "availability.csv"
    availability_mod.write_template(path)
    original = path.read_text()
    path.write_text(original + "Leeds,X,INJURY,0.1,1,0,2026-09-01,,,\n")
    availability_mod.write_template(path)
    assert "Leeds,X,INJURY" in path.read_text()


def test_missing_availability_file_is_reported_not_fatal(engine, tmp_path):
    with engine.begin() as conn:
        report = availability_mod.load_csv(conn, TeamRegistry.sync(conn),
                                           tmp_path / "nope.csv")
    assert report.missing_file
    assert not report.ok
    assert report.rows_loaded == 0


# -- fbref -----------------------------------------------------------------
def test_derive_suspension_risk_flags_the_thresholds():
    players = pd.DataFrame({
        "team": ["Leeds"] * 4,
        "player": ["At Five", "One Away", "Sent Off", "Clean"],
        "min": [2000, 1800, 1500, 1900],
        "CrdY": [5, 4, 1, 0],
        "CrdR": [0, 0, 1, 0],
    })
    risk = derive_suspension_risk(players)
    reasons = dict(zip(risk["player"], risk["reason"]))
    assert reasons["At Five"] == "SUSPENSION"
    assert reasons["One Away"] == "DOUBT"
    assert reasons["Sent Off"] == "SUSPENSION"
    assert "Clean" not in reasons
    assert (risk["minutes_share"] <= 1.0).all()


def test_derive_suspension_risk_survives_an_empty_frame():
    assert derive_suspension_risk(pd.DataFrame()).empty


# -- config ----------------------------------------------------------------
def test_model_params_replace_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown model parameter"):
        ModelParams().replace(not_a_real_knob=1)


def test_decay_xi_matches_the_half_life():
    import math
    assert ModelParams(decay_half_life_days=180).decay_xi == pytest.approx(math.log(2) / 180)
    assert ModelParams(decay_half_life_days=0).decay_xi == 0.0


def test_redacted_url_hides_the_password():
    db = DbConfig(user="sa", password="hunter2", host="db", name="champ")
    assert "hunter2" not in db.redacted_url()
    assert "***" in db.redacted_url()


def test_sqlalchemy_url_carries_the_credentials():
    db = DbConfig(user="sa", password="hunter2")
    import urllib.parse
    assert "hunter2" in urllib.parse.unquote_plus(db.sqlalchemy_url())


def test_max_age_days_semantics(tmp_path):
    """0 means always refresh, -1 means never -- the flag reads the way it
    behaves, which it did not at first."""
    from champmodel.ingest.footballdata_uk import _is_stale

    cached = tmp_path / "E1_2526.csv"
    cached.write_bytes(FIXTURE.read_bytes())
    assert _is_stale(cached, 0) is True         # just written, but 0 = always
    assert _is_stale(cached, -1) is False       # never refresh
    assert _is_stale(cached, 7) is False        # fresh enough
    assert _is_stale(tmp_path / "absent.csv", -1) is True


def test_never_refresh_uses_the_cache_without_touching_the_network(tmp_path):
    cached = tmp_path / "E1_2526.csv"
    cached.write_bytes(FIXTURE.read_bytes())

    class Explode:
        @staticmethod
        def get(*_args, **_kwargs):
            raise AssertionError("must not hit the network when pinned to cache")

    assert download_season(2025, tmp_path, max_age_days=-1, session=Explode()) == cached


def test_written_template_loads_as_empty(engine, tmp_path):
    """An auto-written template must not inject a phantom injury into the next
    prediction run, so its example row is commented out."""
    path = tmp_path / "availability.csv"
    availability_mod.write_template(path)
    with engine.begin() as conn:
        report = availability_mod.load_csv(conn, TeamRegistry.sync(conn), path)
    assert report.rows_loaded == 0
    assert report.errors == []
    assert not report.missing_file


# -- Windows connection strings --------------------------------------------
def _odbc(db: DbConfig) -> str:
    import urllib.parse
    return urllib.parse.unquote_plus(db.sqlalchemy_url().split("odbc_connect=")[1])


def test_plain_host_gets_a_port():
    assert DbConfig(host="localhost", port=1433).server == "localhost,1433"
    assert "SERVER=localhost,1433" in _odbc(DbConfig(host="localhost", port=1433))


@pytest.mark.parametrize("host", [r".\SQLEXPRESS", r"localhost\SQLEXPRESS",
                                  r"(localdb)\MSSQLLocalDB"])
def test_named_instances_never_get_a_port(host):
    """A named instance and LocalDB resolve through the SQL Browser service by
    instance name. Appending ,1433 stops the connection working at all, which
    is how most Windows installs are reached."""
    db = DbConfig(host=host, port=1433, trusted_connection=True)
    assert db.server == host
    assert f"SERVER={host};" in _odbc(db)
    assert ",1433" not in _odbc(db)


def test_zero_port_is_omitted():
    assert DbConfig(host="dbhost", port=0).server == "dbhost"


def test_trusted_connection_sends_no_credentials():
    odbc = _odbc(DbConfig(host=r".\SQLEXPRESS", trusted_connection=True,
                          user="ignored", password="ignored"))
    assert "Trusted_Connection=yes" in odbc
    assert "UID=" not in odbc and "PWD=" not in odbc


def test_redacted_url_covers_trusted_connections():
    shown = DbConfig(host=r"(localdb)\MSSQLLocalDB", trusted_connection=True).redacted_url()
    assert "(trusted)" in shown
    assert "PWD" not in shown


# -- configuration errors report, they do not crash -------------------------
def _break_pyodbc_import(monkeypatch, module_name: str) -> None:
    """Make SQLAlchemy's pyodbc connector fail exactly as it does when the
    driver is absent -- the code path from the real traceback."""
    from sqlalchemy.connectors.pyodbc import PyODBCConnector

    def boom(cls):
        raise ModuleNotFoundError(f"No module named {module_name!r}", name=module_name)

    monkeypatch.setattr(PyODBCConnector, "import_dbapi", classmethod(boom))


def test_missing_pyodbc_raises_an_actionable_error(monkeypatch):
    """A missing optional driver is an expected setup state, not a bug. It has
    to say what to install and what the alternative is."""
    from champmodel.db import DriverNotInstalled, make_engine

    _break_pyodbc_import(monkeypatch, "pyodbc")
    with pytest.raises(DriverNotInstalled) as excinfo:
        make_engine(DbConfig(host="localhost", user="sa", password="x"))

    message = str(excinfo.value)
    assert "pyodbc" in message
    assert 'pip install -e ".[dev,mssql]"' in message
    assert "sqlite:///./data/champ.db" in message      # the way out


def test_a_different_missing_module_is_not_swallowed(monkeypatch):
    from champmodel.db import DriverNotInstalled, make_engine

    _break_pyodbc_import(monkeypatch, "somethingelse")
    with pytest.raises(ModuleNotFoundError) as excinfo:
        make_engine(DbConfig(host="localhost", user="sa", password="x"))
    assert not isinstance(excinfo.value, DriverNotInstalled)


def test_sqlite_needs_no_driver():
    """The documented fallback must work in an environment with no pyodbc."""
    from champmodel.db import create_schema, make_engine

    engine = make_engine(DbConfig(url="sqlite://"))
    create_schema(engine)
    engine.dispose()


def test_config_records_which_env_file_it_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Config.load().env_file is None

    (tmp_path / ".env").write_text("CHAMP_LOG_LEVEL=ERROR\n", encoding="utf-8")
    assert Config.load().env_file is not None


def test_unconfigured_database_is_detectable(tmp_path, monkeypatch):
    """No .env and no CHAMP_DB_* means the defaults are silently pointing at a
    local SQL Server. The CLI warns rather than failing obscurely later."""
    monkeypatch.chdir(tmp_path)
    for key in [k for k in os.environ if k.startswith("CHAMP_DB_")]:
        monkeypatch.delenv(key, raising=False)
    assert Config.load().database_is_unconfigured

    monkeypatch.setenv("CHAMP_DB_URL", "sqlite://")
    assert not Config.load().database_is_unconfigured


# -- a season that is not published yet must degrade, not crash -------------
def test_an_html_error_page_is_not_cached(tmp_path):
    """football-data.co.uk can answer 200 with an error page. Caching that
    would poison every later run, and parsing it would KeyError several frames
    deep."""
    from champmodel.ingest.footballdata_uk import SeasonFileUnusable

    class ErrorPage:
        @staticmethod
        def get(*_args, **_kwargs):
            class R:
                content = b"<html><body>404 Not Found</body></html>"
                status_code = 200

                @staticmethod
                def raise_for_status():
                    return None
            return R()

    with pytest.raises(SeasonFileUnusable):
        download_season(2026, tmp_path, session=ErrorPage())
    assert not (tmp_path / "E1_2627.csv").exists()


def test_a_csv_without_result_columns_is_rejected(tmp_path):
    from champmodel.ingest.footballdata_uk import SeasonFileUnusable

    path = tmp_path / "E1_2627.csv"
    path.write_text("Div,Date,Something\nE1,09/08/2026,x\n", encoding="utf-8")
    with pytest.raises(SeasonFileUnusable, match="missing the column"):
        read_season_csv(path)


def test_backfill_records_a_missing_season_and_keeps_going(engine, tmp_path):
    """The current season often has no file yet. That is a degraded input to
    record, not a reason to abandon nine good seasons."""
    from champmodel.ingest.footballdata_uk import backfill

    (tmp_path / "E1_2526.csv").write_bytes(FIXTURE.read_bytes())
    (tmp_path / "E1_2627.csv").write_text("<html>nope</html>", encoding="utf-8")

    with engine.begin() as conn:
        report = backfill(conn, TeamRegistry.sync(conn), [2025, 2026], tmp_path,
                          max_age_days=-1, request_delay=0)

    assert report.seasons_loaded == [2025]
    assert 2026 in report.seasons_failed
    assert report.matches == 4
    assert not report.ok            # surfaced to the user, not swallowed


def test_backfill_still_hard_fails_on_an_unmapped_team(engine, tmp_path):
    """Degrading on a missing file must not soften the alias rule."""
    from champmodel.ingest.footballdata_uk import backfill
    from champmodel.ingest.teams import UnknownTeamError

    text = FIXTURE.read_text(encoding="latin-1").replace("Sheffield Weds", "Real Madrid")
    (tmp_path / "E1_2526.csv").write_text(text, encoding="latin-1")

    with engine.begin() as conn:
        with pytest.raises(UnknownTeamError):
            backfill(conn, TeamRegistry.sync(conn), [2025], tmp_path, max_age_days=-1,
                    request_delay=0)


# -- transient 503s are retried, not fatal -----------------------------------
class _ScriptedResponses:
    """A fake requests session that returns a scripted sequence of responses,
    one per call, then repeats the last one."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def get(self, *_args, **_kwargs):
        response = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return response


class _FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_a_transient_503_is_retried_and_then_succeeds(tmp_path):
    """Exactly what the real ingest hit: a burst of season requests drew a
    handful of 503s. The retry has to recover without the caller re-running
    anything."""
    good_body = FIXTURE.read_bytes()
    script = [
        _FakeResponse(503, headers={"Retry-After": "0"}),
        _FakeResponse(503, headers={"Retry-After": "0"}),
        _FakeResponse(200, content=good_body),
    ]
    session = _ScriptedResponses(script)

    path = download_season(2025, tmp_path, session=session,
                           sleeper=lambda _seconds: None)
    assert path.read_bytes() == good_body
    assert session.calls == 3


def test_persistent_503_falls_back_to_a_stale_cache(tmp_path):
    cached = tmp_path / "E1_2526.csv"
    cached.write_bytes(FIXTURE.read_bytes())
    import os
    old_time = dt.datetime.now().timestamp() - 60 * 60 * 24 * 30
    os.utime(cached, (old_time, old_time))

    session = _ScriptedResponses([_FakeResponse(503)] * 5)
    path = download_season(2025, tmp_path, session=session,
                           sleeper=lambda _seconds: None)
    assert path == cached


def test_persistent_503_raises_when_there_is_no_cache_to_fall_back_to(tmp_path):
    session = _ScriptedResponses([_FakeResponse(503)] * 5)
    with pytest.raises(SourceUnavailable):
        download_season(2025, tmp_path, session=session,
                        sleeper=lambda _seconds: None)


def test_an_unpublished_season_is_not_retried(tmp_path):
    """SeasonFileUnusable means the response was well-formed but not a
    results CSV -- retrying it burns time for nothing, unlike a 503."""
    session = _ScriptedResponses([_FakeResponse(200, content=b"<html>no season yet</html>")])
    with pytest.raises(SeasonFileUnusable):
        download_season(2025, tmp_path, session=session,
                        sleeper=lambda _seconds: (_ for _ in ()).throw(
                            AssertionError("must not sleep/retry on SeasonFileUnusable")))
    assert session.calls == 1


def test_backfill_pauses_between_seasons_but_not_before_the_first(engine, tmp_path):
    from champmodel.ingest.seasons import season_code

    delays = []
    for start_year in (2024, 2025, 2026):
        (tmp_path / f"E1_{season_code(start_year)}.csv").write_bytes(FIXTURE.read_bytes())

    with engine.begin() as conn:
        backfill(conn, TeamRegistry.sync(conn), [2024, 2025, 2026], tmp_path,
                max_age_days=-1, request_delay=2.5, sleeper=delays.append)

    assert delays == [2.5, 2.5]      # one gap before season 2 and season 3, none before season 1


def test_backfill_recovers_from_a_run_of_503s_partway_through(engine, tmp_path):
    """Reproduces the real failure: several seasons in a row answer 503 near
    the end of a ten-season run. backfill must retry each and finish clean."""
    good = FIXTURE.read_bytes()

    def make_session():
        return _ScriptedResponses([_FakeResponse(503), _FakeResponse(503),
                                   _FakeResponse(200, content=good)])

    class PerCallSession:
        """download_season is given one shared session; give each call its
        own scripted 503-503-200 sequence to emulate the outage clearing."""
        def __init__(self):
            self._sessions = {}

        def get(self, url, *args, **kwargs):
            self._sessions.setdefault(url, make_session())
            return self._sessions[url].get(url, *args, **kwargs)

    with engine.begin() as conn:
        report = backfill(conn, TeamRegistry.sync(conn), [2024, 2025, 2026], tmp_path,
                          max_age_days=-1, session=PerCallSession(),
                          request_delay=0, sleeper=lambda _s: None)

    assert report.seasons_loaded == [2024, 2025, 2026]
    assert report.seasons_failed == {}
    assert report.ok


def test_default_timeout_bounds_a_single_stuck_attempt(tmp_path):
    """A user reported a run 'stuck' on this exact warning for minutes. The
    cause was a plain float timeout, which requests applies separately to
    connect and read -- 30.0 meant up to 60s per attempt. Assert the fix
    directly: the request actually receives the (connect, read) tuple, not a
    lone float that doubles under the hood."""
    from champmodel.ingest.footballdata_uk import DEFAULT_TIMEOUT

    assert isinstance(DEFAULT_TIMEOUT, tuple) and len(DEFAULT_TIMEOUT) == 2
    connect, read = DEFAULT_TIMEOUT
    # Four attempts against a host that never answers must stay well under
    # the ~4.35 minutes the old single-float default could reach.
    worst_case_requests = 4 * (connect + read)
    worst_case_backoff = 3 + 6 + 12          # the 2**attempt*3 schedule, 3 retries
    assert worst_case_requests + worst_case_backoff < 120


def test_download_season_passes_the_timeout_tuple_through(tmp_path):
    seen = {}

    class Recorder:
        @staticmethod
        def get(url, timeout=None, headers=None):
            seen["timeout"] = timeout
            return _FakeResponse(200, content=FIXTURE.read_bytes())

    download_season(2025, tmp_path, session=Recorder(), sleeper=lambda _s: None)
    assert seen["timeout"] == (5.0, 15.0)


# -- a hard ceiling that no network condition can bypass ---------------------
def test_bounded_get_gives_up_on_a_call_that_never_returns():
    """Reproduces the real report: a request that hangs for minutes despite a
    short configured timeout, because something on the network (VPN, proxy,
    broken route) silently drops the connection instead of refusing it.
    _bounded_get must return control within hard_timeout regardless."""
    import time as time_mod

    from champmodel.ingest.footballdata_uk import _bounded_get

    class NeverReturns:
        @staticmethod
        def get(*_args, **_kwargs):
            time_mod.sleep(999)     # simulates a connection requests' own
                                     # timeout failed to bound

    start = time_mod.monotonic()
    with pytest.raises(SourceUnavailable, match="did not respond within"):
        _bounded_get(NeverReturns(), "https://example.invalid/x",
                    timeout=(5.0, 15.0), headers={}, hard_timeout=0.2)
    elapsed = time_mod.monotonic() - start
    assert elapsed < 2.0, f"the hard cap did not actually bound the wait ({elapsed:.2f}s)"


def test_bounded_get_returns_normally_when_the_call_is_fast():
    from champmodel.ingest.footballdata_uk import _bounded_get

    session = _ScriptedResponses([_FakeResponse(200, content=b"ok")])
    result = _bounded_get(session, "https://example.invalid/x",
                          timeout=(5.0, 15.0), headers={}, hard_timeout=5.0)
    assert result.status_code == 200


def test_download_season_survives_a_hanging_connection(tmp_path):
    """The end-to-end version of the bug report: every attempt hangs forever
    at the transport layer. download_season must still finish -- fast,
    bounded by hard_timeout, not by whatever the network is doing -- and
    report SourceUnavailable rather than blocking indefinitely."""
    import time as time_mod

    class AlwaysHangs:
        @staticmethod
        def get(*_args, **_kwargs):
            time_mod.sleep(999)

    start = time_mod.monotonic()
    with pytest.raises(SourceUnavailable):
        download_season(2025, tmp_path, session=AlwaysHangs(),
                        sleeper=lambda _s: None, hard_timeout=0.2, max_retries=3)
    elapsed = time_mod.monotonic() - start
    # 3 attempts * 0.2s hard cap, plus near-zero backoff (stubbed sleeper) --
    # must stay a tiny fraction of what the old bug let this take (minutes).
    assert elapsed < 3.0, f"took {elapsed:.2f}s -- the hard cap was not applied per attempt"


def test_default_hard_timeout_is_derived_from_the_timeout_budget():
    import inspect

    from champmodel.ingest.footballdata_uk import DEFAULT_TIMEOUT, download_season

    sig = inspect.signature(download_season)
    assert sig.parameters["hard_timeout"].default is None   # computed, not hardcoded
    # Documented relationship: connect + read + 10s cushion.
    assert sum(DEFAULT_TIMEOUT) + 10.0 == 30.0
