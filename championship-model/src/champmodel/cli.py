"""Command line: ingest, fit, backtest, tune, predict.

    champmodel init-db                 create the schema and seed the teams
    champmodel ingest --backfill       10 seasons of results from football-data.co.uk
    champmodel ingest --today          today's fixtures from football-data.org
    champmodel fit                     fit and store the Dixon-Coles model
    champmodel backtest                walk-forward, scored against the closing line
    champmodel tune                    grid-search hyperparameters out-of-sample
    champmodel predict --date today    the daily run
    champmodel status                  row counts and the current fit
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from . import MODEL_VERSION, __version__
from .config import Config
from .logging_setup import get_logger, set_run_id, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="BTTS and Over 2.5 probability engine for the English Championship.",
)
log = get_logger(__name__)

ARTIFACT_NAME = "dixon_coles_latest.json"
CALIBRATION_NAME = "calibration_latest.json"


def _load_config() -> Config:
    cfg = Config.load()
    cfg.ensure_dirs()
    setup_logging(cfg.log_level, cfg.log_dir)
    return cfg


def _parse_date(value: str | None) -> dt.date:
    if value is None or value.lower() in {"today", "now"}:
        return dt.date.today()
    if value.lower() == "yesterday":
        return dt.date.today() - dt.timedelta(days=1)
    if value.lower() == "tomorrow":
        return dt.date.today() + dt.timedelta(days=1)
    return dt.date.fromisoformat(value)


def _echo_err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


def _engine(cfg: Config):
    """Build the engine, reporting configuration problems as messages.

    A missing driver or an unreachable server is a setup mistake, not a bug,
    and a forty-line rich traceback buries the one line that says what to do
    about it.
    """
    from .db import DriverNotInstalled, make_engine

    if cfg.database_is_unconfigured:
        typer.secho(
            "No .env found and no CHAMP_DB_* set, so the built-in defaults are "
            "in use: SQL Server on localhost:1433 with no credentials.\n"
            "Create one with `Copy-Item .env.example .env` (PowerShell) or "
            "`cp .env.example .env`, then edit it.",
            fg=typer.colors.YELLOW, err=True,
        )
    try:
        return make_engine(cfg.db)
    except DriverNotInstalled as exc:
        _echo_err(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # bad URL, unreachable host, wrong driver name
        _echo_err(f"could not connect to {cfg.db.redacted_url()}\n  {exc}")
        raise typer.Exit(code=1) from exc


# --------------------------------------------------------------------------
@app.command("version")
def version_cmd() -> None:
    """Print the package and model versions."""
    typer.echo(f"champmodel {__version__} (model {MODEL_VERSION})")


@app.command("init-db")
def init_db_cmd(
    seed_teams: bool = typer.Option(True, help="Also populate dim_team and team_alias."),
) -> None:
    """Create the champ schema, then seed the canonical team registry."""
    from .db import create_schema
    from .ingest.teams import TeamRegistry

    cfg = _load_config()
    typer.echo(f"connecting to {cfg.db.redacted_url()}")
    engine = _engine(cfg)
    create_schema(engine)
    typer.secho("schema applied", fg=typer.colors.GREEN)
    if seed_teams:
        with engine.begin() as conn:
            registry = TeamRegistry.sync(conn)
        typer.secho(f"seeded {len(registry.canonical_to_id)} teams", fg=typer.colors.GREEN)


@app.command("ingest")
def ingest_cmd(
    backfill: bool = typer.Option(False, "--backfill", help="Load historical seasons."),
    today: bool = typer.Option(False, "--today", help="Load today's fixtures."),
    date: Optional[str] = typer.Option(None, help="Fixture date (default: today)."),
    seasons: Optional[int] = typer.Option(None, help="How many seasons to backfill."),
    availability: bool = typer.Option(True, help="Also load data/availability.csv."),
    xg: Optional[bool] = typer.Option(None, help="Override CHAMP_ENABLE_FBREF."),
    max_age_days: int = typer.Option(
        7, help="Re-download season files older than N days. 0 always, -1 never."),
    window_days: int = typer.Option(0, help="Fixture window either side of the date."),
) -> None:
    """Load results, fixtures, availability and (optionally) xG."""
    from .ingest import availability as availability_mod
    from .ingest import fbref, footballdata_org, footballdata_uk
    from .ingest.seasons import recent_start_years
    from .ingest.teams import TeamRegistry, UnknownTeamError

    cfg = _load_config()
    if not backfill and not today:
        backfill = today = True

    engine = _engine(cfg)
    failures: list[str] = []

    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)

        if backfill:
            years = recent_start_years(seasons or cfg.backfill_seasons)
            typer.echo(f"backfilling seasons {years[0]}-{years[-1]} ...")
            if cfg.ignore_system_proxy:
                typer.echo("  CHAMP_IGNORE_SYSTEM_PROXY=1: routing around the system/VPN proxy")
            session = footballdata_uk.make_session(cfg.ignore_system_proxy)
            try:
                report = footballdata_uk.backfill(conn, registry, years, cfg.raw_dir,
                                                  max_age_days=max_age_days, session=session)
                typer.echo(f"  {report.summary()}")
                for year, reason in report.seasons_failed.items():
                    failures.append(f"season {year}: {reason}")
            except UnknownTeamError as exc:
                _echo_err(f"ingest aborted: {exc}")
                raise typer.Exit(code=2) from exc

        if today:
            day = _parse_date(date)
            client = footballdata_org.FootballDataOrgClient(cfg.football_data_org_key,
                                                            cfg.cache_dir)
            typer.echo(f"fetching fixtures for {day} ...")
            try:
                fixtures = footballdata_org.ingest_day(conn, registry, client, day,
                                                       window_days=window_days)
            except UnknownTeamError as exc:
                _echo_err(f"fixture ingest aborted: {exc}")
                raise typer.Exit(code=2) from exc
            if fixtures.error:
                failures.append(f"fixtures: {fixtures.error}")
                _echo_err(f"  fixtures unavailable: {fixtures.error}")
            else:
                typer.echo(f"  fetched {fixtures.fetched}, wrote {fixtures.written}"
                           f"{' (cached)' if fixtures.from_cache else ''}")

        if availability:
            path = cfg.availability_csv
            availability_mod.write_template(path)
            report = availability_mod.load_csv(conn, registry, path)
            if report.missing_file:
                failures.append("availability.csv missing")
            typer.echo(f"availability: {report.rows_loaded} row(s) loaded"
                       f" from {path}")
            for error in report.errors:
                _echo_err(f"  {error}")
                failures.append(f"availability: {error}")

        use_xg = cfg.enable_fbref if xg is None else xg
        if use_xg:
            years = recent_start_years(seasons or min(cfg.backfill_seasons, 3))
            typer.echo("fetching FBref xG ...")
            fb = fbref.ingest(conn, registry, years, cfg.data_dir / "fbref",
                              crawl_delay=cfg.fbref_crawl_delay)
            if fb.error:
                failures.append(f"fbref: {fb.error}")
                _echo_err(f"  xG unavailable: {fb.error}")
            else:
                typer.echo(f"  {fb.xg_rows} match(es) enriched")

        from .repository import counts, unresolved_alias_count
        unresolved = unresolved_alias_count(conn)
        row_counts = counts(conn)

    typer.echo("")
    typer.echo(f"matches={row_counts['matches']} "
               f"finished={row_counts['finished_matches']} "
               f"odds={row_counts['odds']} teams={row_counts['teams']}")
    if unresolved:
        _echo_err(f"{unresolved} unresolved alias row(s) -- this must be zero")
        raise typer.Exit(code=2)
    typer.secho("unresolved aliases: 0", fg=typer.colors.GREEN)

    if failures:
        typer.secho(f"{len(failures)} input(s) missing or degraded:", fg=typer.colors.YELLOW)
        for item in failures:
            typer.echo(f"  - {item}")


@app.command("fit")
def fit_cmd(
    until: Optional[str] = typer.Option(None, help="Train on matches up to this date."),
    half_life: Optional[float] = typer.Option(None, help="Override the decay half-life."),
    show: int = typer.Option(10, help="How many teams to print."),
) -> None:
    """Fit the Dixon-Coles model and store it under data/artifacts."""
    from tabulate import tabulate

    from .model.dixon_coles import fit_dixon_coles
    from .repository import load_matches

    cfg = _load_config()
    params = cfg.model if half_life is None else cfg.model.replace(decay_half_life_days=half_life)
    cutoff = _parse_date(until) if until else dt.date.today()

    engine = _engine(cfg)
    with engine.connect() as conn:
        matches = load_matches(conn, finished_only=True, until=cutoff, with_odds=False)
    if matches.empty:
        _echo_err("no finished matches in the database -- run `champmodel ingest --backfill`")
        raise typer.Exit(code=1)

    fit = fit_dixon_coles(matches, ref_date=cutoff, params=params)
    path = fit.save(cfg.artifact_dir / ARTIFACT_NAME)

    typer.echo(f"trained on {fit.train_rows} matches through {fit.train_end_date}")
    typer.echo(f"home advantage gamma = {fit.home_advantage:.4f}   rho = {fit.rho:.4f}")
    if not 0.15 <= fit.home_advantage <= 0.40:
        typer.secho(f"  gamma outside the expected 0.20-0.30 band -- check the inputs",
                    fg=typer.colors.YELLOW)
    shrunk = fit.shrunk_teams()
    if shrunk:
        typer.echo(f"shrunk toward the league mean: {', '.join(shrunk)}")
    table = fit.team_table().head(show)
    typer.echo(tabulate(table, headers="keys", tablefmt="simple",
                        showindex=False, floatfmt=".4f"))
    typer.secho(f"saved {path}", fg=typer.colors.GREEN)


@app.command("backtest")
def backtest_cmd(
    start: Optional[str] = typer.Option(None, help="First date to predict."),
    end: Optional[str] = typer.Option(None, help="Last date to predict."),
    seasons: int = typer.Option(3, help="Backtest the last N seasons if --start is absent."),
    refit_every: int = typer.Option(7, help="Days between refits."),
    buckets: int = typer.Option(10, help="Calibration buckets."),
    calibrate: bool = typer.Option(True, help="Fit Platt scaling on a held-out half."),
    h2h: bool = typer.Option(False, help="Include the head-to-head nudge."),
    synthetic: bool = typer.Option(False, "--synthetic",
                                   help="Run against generated data (offline self-test)."),
    save_calibration: bool = typer.Option(True, help="Store the fitted calibrators."),
) -> None:
    """Walk-forward backtest, scored against the de-vigged closing line."""
    from .backtest.metrics import format_report, summarise
    from .backtest.walk_forward import (apply_calibration, assert_no_leakage,
                                        split_calibration, walk_forward)
    from .features.h2h import H2HIndex
    from .model.pipeline import make_h2h_adjuster
    from .repository import load_matches

    cfg = _load_config()
    params = cfg.model.replace(enable_h2h_adjustment=h2h)

    if synthetic:
        from . import synth
        typer.secho("running on SYNTHETIC data -- these numbers describe the "
                    "machinery, not the Championship", fg=typer.colors.YELLOW)
        matches = synth.generate(n_seasons=max(seasons + 2, 4)).matches
    else:
        engine = _engine(cfg)
        with engine.connect() as conn:
            matches = load_matches(conn, finished_only=True)
        if matches.empty:
            _echo_err("no finished matches -- run `champmodel ingest --backfill` first")
            raise typer.Exit(code=1)

    days = sorted(matches["match_date"].unique())
    test_start = _parse_date(start) if start else (
        days[-1] - dt.timedelta(days=365 * seasons)
    )
    test_end = _parse_date(end) if end else days[-1]

    adjust = make_h2h_adjuster(H2HIndex(matches), params) if h2h else None
    with typer.progressbar(length=100, label="walk-forward") as bar:
        state = {"last": 0}

        def progress(done: int, total: int) -> None:
            pct = int(100 * done / max(total, 1))
            bar.update(pct - state["last"])
            state["last"] = pct

        result = walk_forward(matches, params, test_start=test_start, test_end=test_end,
                              refit_every_days=refit_every, adjust=adjust,
                              progress=progress)

    if result.predictions.empty:
        _echo_err("the backtest produced no predictions -- widen the window")
        raise typer.Exit(code=1)

    assert_no_leakage(result.predictions)
    typer.secho("no-leakage assertion passed", fg=typer.colors.GREEN)
    typer.echo(f"{result.n_predictions} predictions from {result.n_fits} fits "
               f"({test_start} .. {test_end})")
    typer.echo("")
    typer.echo("=== raw model ===")
    typer.echo(format_report(summarise(result.predictions, buckets)))

    if calibrate:
        calibration, calib_rows, eval_rows = split_calibration(result)
        typer.echo("")
        typer.echo(f"=== Platt-calibrated (fitted on {len(calib_rows)} earlier rows, "
                   f"scored on {len(eval_rows)} later ones) ===")
        typer.echo(f"BTTS   {calibration.btts.describe()}")
        typer.echo(f"O2.5   {calibration.over25.describe()}")
        typer.echo("")
        typer.echo(format_report(summarise(apply_calibration(eval_rows, calibration), buckets)))
        if save_calibration and calibration.any_fitted and not synthetic:
            path = cfg.artifact_dir / CALIBRATION_NAME
            path.write_text(calibration.to_json(), encoding="utf-8")
            typer.secho(f"saved {path}", fg=typer.colors.GREEN)

    out = cfg.output_dir / f"backtest-{dt.date.today():%Y-%m-%d}.csv"
    result.predictions.to_csv(out, index=False)
    typer.echo(f"predictions written to {out}")


@app.command("tune")
def tune_cmd(
    seasons: int = typer.Option(2, help="Backtest window, in seasons."),
    refit_every: int = typer.Option(14, help="Days between refits (higher = faster)."),
    half_lives: str = typer.Option("90,140,180,240,320", help="Comma-separated xi grid."),
    shrinkage: str = typer.Option("0.0,0.3,1.0", help="Comma-separated shrinkage grid."),
    synthetic: bool = typer.Option(False, "--synthetic", help="Tune against generated data."),
) -> None:
    """Grid-search hyperparameters on out-of-sample log loss."""
    from tabulate import tabulate

    from .backtest.walk_forward import grid_search
    from .repository import load_matches

    cfg = _load_config()
    if synthetic:
        from . import synth
        typer.secho("tuning on SYNTHETIC data -- the winning values describe the "
                    "generator, not the Championship", fg=typer.colors.YELLOW)
        matches = synth.generate(n_seasons=seasons + 2).matches
    else:
        engine = _engine(cfg)
        with engine.connect() as conn:
            matches = load_matches(conn, finished_only=True)
        if matches.empty:
            _echo_err("no finished matches -- run `champmodel ingest --backfill` first")
            raise typer.Exit(code=1)

    days = sorted(matches["match_date"].unique())
    test_start = days[-1] - dt.timedelta(days=365 * seasons)
    grid = {
        "decay_half_life_days": [float(v) for v in half_lives.split(",") if v.strip()],
        "shrinkage_prior": [float(v) for v in shrinkage.split(",") if v.strip()],
    }
    total = len(grid["decay_half_life_days"]) * len(grid["shrinkage_prior"])
    typer.echo(f"searching {total} combination(s) over {test_start} .. {days[-1]}")

    def progress(done: int, _total: int, params: Any) -> None:
        typer.echo(f"  [{done}/{total}] half_life={params.decay_half_life_days} "
                   f"shrinkage={params.shrinkage_prior}")

    frame = grid_search(matches, cfg.model, grid, test_start=test_start,
                        refit_every_days=refit_every, progress=progress)
    if frame.empty:
        _echo_err("grid search produced no results")
        raise typer.Exit(code=1)
    typer.echo("")
    typer.echo(tabulate(frame, headers="keys", tablefmt="simple",
                        showindex=False, floatfmt=".5f"))
    best = frame.iloc[0]
    typer.secho(f"\nbest: half_life={best['decay_half_life_days']:.0f} "
                f"shrinkage={best['shrinkage_prior']:.2f} "
                f"(combined log loss {best['combined']:.5f})", fg=typer.colors.GREEN)
    typer.echo("Set CHAMP_DECAY_HALF_LIFE_DAYS / CHAMP_SHRINKAGE_PRIOR in .env to adopt.")


@app.command("predict")
def predict_cmd(
    date: Optional[str] = typer.Option(None, "--date", help="Fixture date (default: today)."),
    refit_after: Optional[int] = typer.Option(None, help="Refit if the fit is older than N days."),
    ingest: bool = typer.Option(True, help="Refresh today's fixtures first."),
    write_db: bool = typer.Option(True, help="Write to champ.prediction."),
    csv: bool = typer.Option(True, help="Write output/YYYY-MM-DD.csv."),
) -> None:
    """The daily run: fixtures in, probabilities out."""
    from tabulate import tabulate

    from .features.availability_adj import build_index
    from .features.h2h import H2HIndex
    from .ingest import availability as availability_mod
    from .ingest import footballdata_org
    from .ingest.teams import TeamRegistry, UnknownTeamError
    from .model.calibrate import CalibrationSet
    from .model.dixon_coles import DixonColesFit, fit_dixon_coles
    from .model.pipeline import FLAG_NO_MARKET, predict_fixture
    from .backtest.metrics import devig_proportional
    from .repository import (create_model_run, load_fixtures, load_matches,
                             write_predictions)

    cfg = _load_config()
    day = _parse_date(date)
    params = cfg.model
    engine = _engine(cfg)
    degraded: list[str] = []

    with engine.begin() as conn:
        registry = TeamRegistry.load(conn)

        # 1. today's fixtures
        if ingest:
            client = footballdata_org.FootballDataOrgClient(cfg.football_data_org_key,
                                                            cfg.cache_dir)
            try:
                report = footballdata_org.ingest_day(conn, registry, client, day)
                if report.error:
                    degraded.append(f"fixtures: {report.error}")
            except UnknownTeamError as exc:
                _echo_err(f"fixture ingest aborted: {exc}")
                raise typer.Exit(code=2) from exc

        # 2. availability overrides valid today
        avail_report = availability_mod.load_csv(conn, registry, cfg.availability_csv)
        if avail_report.missing_file:
            degraded.append("availability.csv not found")
        for error in avail_report.errors:
            degraded.append(f"availability: {error}")
        overrides = availability_mod.active_overrides(conn, day)
        news_flags = _team_news_flags(cfg.availability_csv, day, degraded)
        availability_index = build_index(overrides, params, day) if overrides or \
            not avail_report.missing_file else None

        fixtures = load_fixtures(conn, day)
        history = load_matches(conn, finished_only=True, until=day - dt.timedelta(days=1),
                               with_odds=False)

    if fixtures.empty:
        typer.secho(f"no Championship fixtures found for {day}", fg=typer.colors.YELLOW)
        if degraded:
            for item in degraded:
                typer.echo(f"  - {item}")
        raise typer.Exit(code=0)

    # 3. load or refit the model
    artifact = cfg.artifact_dir / ARTIFACT_NAME
    max_age = cfg.refit_after_days if refit_after is None else refit_after
    fit: DixonColesFit | None = None
    if artifact.exists():
        try:
            candidate = DixonColesFit.load(artifact)
            age = (day - candidate.train_end_date).days
            if age <= max_age:
                fit = candidate
                typer.echo(f"using stored fit through {candidate.train_end_date} "
                           f"({age} day(s) old)")
            else:
                typer.echo(f"stored fit is {age} day(s) old -- refitting")
        except Exception as exc:
            log.warning("stored fit unreadable", extra={"error": str(exc)})
    if fit is None:
        if history.empty:
            _echo_err("no history to fit on -- run `champmodel ingest --backfill`")
            raise typer.Exit(code=1)
        fit = fit_dixon_coles(history, ref_date=day, params=params)
        fit.save(artifact)
        typer.echo(f"refit on {fit.train_rows} matches through {fit.train_end_date}")

    # 4. calibration, if a backtest produced one
    calibration_path = cfg.artifact_dir / CALIBRATION_NAME
    calibration = CalibrationSet()
    if calibration_path.exists():
        calibration = CalibrationSet.from_dict(json.loads(
            calibration_path.read_text(encoding="utf-8")))
    else:
        degraded.append("no calibration artifact -- run `champmodel backtest` first")

    h2h_index = H2HIndex(history)
    market_over = devig_proportional(fixtures["odds_over25"], fixtures["odds_under25"])
    market_btts = devig_proportional(fixtures["odds_btts_yes"], fixtures["odds_btts_no"])

    predictions = []
    for position, (_, row) in enumerate(fixtures.iterrows()):
        mo = float(market_over[position]) if market_over.size > position else float("nan")
        mb = float(market_btts[position]) if market_btts.size > position else float("nan")
        predictions.append(predict_fixture(
            fit, row["home_team"], row["away_team"], day, params,
            kickoff_utc=row["kickoff_utc"],
            availability=availability_index,
            home_team_id=int(row["home_team_id"]),
            away_team_id=int(row["away_team_id"]),
            h2h_index=h2h_index,
            calibration=calibration,
            market_p_over25=None if mo != mo else mo,
            market_p_btts=None if mb != mb else mb,
            match_id=int(row["match_id"]),
            extra_flags=news_flags,
        ))

    # 5. store
    run_id = None
    if write_db:
        with engine.begin() as conn:
            run_id = create_model_run(
                conn, params,
                train_rows=fit.train_rows, train_end_date=fit.train_end_date,
                extra={"calibration": calibration.to_dict(),
                       "degraded_inputs": degraded,
                       "fixture_date": day.isoformat()},
            )
            set_run_id(run_id)
            write_predictions(conn, run_id, [p.to_row(run_id) for p in predictions])
        typer.echo(f"stored run_id={run_id}")

    # 6. print and write the CSV
    typer.echo("")
    typer.echo(_render_table(predictions, day))
    if csv:
        out = _write_csv(predictions, cfg.output_dir, day, run_id)
        typer.secho(f"\nwritten to {out}", fg=typer.colors.GREEN)

    if degraded:
        typer.secho("\ndegraded inputs for this run:", fg=typer.colors.YELLOW)
        for item in degraded:
            typer.echo(f"  - {item}")
    typer.echo("\nProbability estimates, not betting advice. Bookmaker margins on "
               "these markets run 4-7%; an edge has to clear that before it exists.")


def _team_news_flags(path: Path, day: dt.date, degraded: list[str]) -> list[str]:
    """Flag team news that was never entered, or entered before today.

    An availability file last touched three days ago says nothing about
    tonight's line-ups, and a prediction made without today's team news is a
    different product from one made with it. The output has to say which.
    """
    from .model.pipeline import FLAG_STALE_AVAILABILITY

    if not path.exists():
        return []          # predict_fixture already flags the absent index
    edited = dt.date.fromtimestamp(path.stat().st_mtime)
    if edited < day:
        degraded.append(f"availability.csv last edited {edited}, before {day}")
        return [FLAG_STALE_AVAILABILITY]
    return []


def _render_table(predictions: list[Any], day: dt.date) -> str:
    from tabulate import tabulate

    def pct(value: float | None) -> str:
        return "-" if value is None or value != value else f"{value:.1%}"

    rows = []
    # Sort by absolute edge where the market priced it; unpriced fixtures last.
    def sort_key(p: Any) -> tuple[int, float]:
        edge = p.edge_over25
        return (0, -abs(edge)) if edge is not None else (1, 0.0)

    for pred in sorted(predictions, key=sort_key):
        kickoff = pred.kickoff_utc.strftime("%H:%M") if pred.kickoff_utc else "-"
        h2h = pred.h2h
        rows.append([
            kickoff, pred.home_team, pred.away_team,
            pct(pred.p_btts), pct(pred.p_over25),
            f"{pred.lambda_home:.2f}", f"{pred.lambda_away:.2f}",
            h2h.display_btts() if h2h else "-",
            h2h.display_over25() if h2h else "-",
            pct(pred.market_p_over25),
            f"{pred.edge_over25:+.1%}" if pred.edge_over25 is not None else "-",
            "yes" if pred.availability_applied else "no",
            ",".join(pred.missing_inputs) or "-",
        ])
    header = [f"Kickoff (UTC {day})", "Home", "Away", "P(BTTS)", "P(O2.5)",
              "lH", "lA", "H2H BTTS", "H2H O2.5", "Mkt O2.5", "Edge", "Avail?", "Flags"]
    return tabulate(rows, headers=header, tablefmt="simple")


def _write_csv(predictions: list[Any], output_dir: Path, day: dt.date,
               run_id: int | None) -> Path:
    import csv as csv_mod

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{day:%Y-%m-%d}.csv"
    fields = [
        "run_id", "match_id", "match_date", "kickoff_utc", "home_team", "away_team",
        "p_btts", "p_over25", "p_btts_raw", "p_over25_raw",
        "lambda_home", "lambda_away", "p_home", "p_draw", "p_away",
        "h2h_meetings", "h2h_btts", "h2h_over25",
        "market_p_over25", "edge_over25", "market_p_btts", "edge_btts",
        "availability_applied", "h2h_applied", "missing_inputs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_mod.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pred in predictions:
            h2h = pred.h2h
            writer.writerow({
                "run_id": run_id,
                "match_id": pred.match_id,
                "match_date": pred.match_date.isoformat(),
                "kickoff_utc": pred.kickoff_utc.isoformat() if pred.kickoff_utc else "",
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "p_btts": round(pred.p_btts, 5),
                "p_over25": round(pred.p_over25, 5),
                "p_btts_raw": round(pred.p_btts_raw, 5),
                "p_over25_raw": round(pred.p_over25_raw, 5),
                "lambda_home": round(pred.lambda_home, 4),
                "lambda_away": round(pred.lambda_away, 4),
                "p_home": round(pred.p_home, 5),
                "p_draw": round(pred.p_draw, 5),
                "p_away": round(pred.p_away, 5),
                "h2h_meetings": h2h.meetings if h2h else 0,
                "h2h_btts": "" if not h2h or not h2h.has_data else round(h2h.btts_rate, 4),
                "h2h_over25": "" if not h2h or not h2h.has_data else round(h2h.over25_rate, 4),
                "market_p_over25": "" if pred.market_p_over25 is None
                                   else round(pred.market_p_over25, 5),
                "edge_over25": "" if pred.edge_over25 is None else round(pred.edge_over25, 5),
                "market_p_btts": "" if pred.market_p_btts is None
                                 else round(pred.market_p_btts, 5),
                "edge_btts": "" if pred.edge_btts is None else round(pred.edge_btts, 5),
                "availability_applied": int(pred.availability_applied),
                "h2h_applied": int(pred.h2h_applied),
                "missing_inputs": pred.missing_inputs_text() or "",
            })
    return path


@app.command("status")
def status_cmd() -> None:
    """Row counts, the stored fit, and the alias gate."""
    from tabulate import tabulate

    from .model.dixon_coles import DixonColesFit
    from .repository import counts, latest_run, unresolved_alias_count

    cfg = _load_config()
    engine = _engine(cfg)
    with engine.connect() as conn:
        row_counts = counts(conn)
        run = latest_run(conn)
        unresolved = unresolved_alias_count(conn)

    typer.echo(tabulate(sorted(row_counts.items()), headers=["table", "rows"],
                        tablefmt="simple"))
    typer.echo("")
    typer.echo(f"unresolved aliases: {unresolved}"
               + ("  <-- must be zero" if unresolved else ""))

    artifact = cfg.artifact_dir / ARTIFACT_NAME
    if artifact.exists():
        fit = DixonColesFit.load(artifact)
        age = (dt.date.today() - fit.train_end_date).days
        typer.echo(f"stored fit: {fit.train_rows} matches through {fit.train_end_date} "
                   f"({age} day(s) old), gamma={fit.home_advantage:.4f}, rho={fit.rho:.4f}")
    else:
        typer.echo("stored fit: none -- run `champmodel fit`")

    calibration_path = cfg.artifact_dir / CALIBRATION_NAME
    typer.echo(f"calibration: {'present' if calibration_path.exists() else 'none -- run `champmodel backtest`'}")
    if run:
        typer.echo(f"latest run: run_id={run['run_id']} at {run['run_utc']} "
                   f"({run['model_version']})")


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
