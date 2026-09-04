"""Walk-forward backtest.

For every matchday in the test window: fit on matches strictly *before* that
date, predict that day's fixtures, record. The strict inequality is asserted
rather than assumed -- a leaked result is the single easiest way to produce a
model that looks excellent and is worthless.

Refitting on every distinct date is wasteful (a Championship season has ~130 of
them and consecutive fits barely differ), so the fit is reused for up to
``refit_every_days``. That never relaxes the leakage rule: the reused fit was
trained on data older still.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import ModelParams
from ..logging_setup import get_logger
from ..model.calibrate import CalibrationSet, fit_calibration
from ..model.dixon_coles import DixonColesFit, NotEnoughData, fit_dixon_coles
from ..model.score_matrix import match_probabilities
from .metrics import devig_proportional, summarise

log = get_logger(__name__)

PREDICTION_COLUMNS = [
    "match_date", "home_team", "away_team",
    "lambda_home", "lambda_away",
    "p_btts", "p_over25", "p_home", "p_draw", "p_away",
    "actual_btts", "actual_over25", "home_goals", "away_goals",
    "market_p_btts", "market_p_over25",
    "train_rows", "train_end_date", "fit_date",
]


class LeakageError(AssertionError):
    """The training window reached the match being predicted."""


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    params: ModelParams
    n_fits: int = 0
    skipped_days: list[dt.date] = field(default_factory=list)
    calibration: CalibrationSet = field(default_factory=CalibrationSet)

    def summary(self, n_buckets: int = 10) -> dict[str, Any]:
        return summarise(self.predictions, n_buckets)

    @property
    def n_predictions(self) -> int:
        return int(len(self.predictions))


def _prepare(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date
    frame = frame.dropna(subset=["home_goals", "away_goals"])
    return frame.sort_values("match_date").reset_index(drop=True)


def _market_probabilities(day_matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """De-vigged closing probabilities, where the source carried the prices."""
    n = len(day_matches)
    nan = np.full(n, np.nan)
    btts = nan
    over = nan
    if {"odds_btts_yes", "odds_btts_no"} <= set(day_matches.columns):
        btts = devig_proportional(day_matches["odds_btts_yes"], day_matches["odds_btts_no"])
    if {"odds_over25", "odds_under25"} <= set(day_matches.columns):
        over = devig_proportional(day_matches["odds_over25"], day_matches["odds_under25"])
    return btts, over


def walk_forward(
    matches: pd.DataFrame,
    params: ModelParams | None = None,
    *,
    test_start: dt.date | None = None,
    test_end: dt.date | None = None,
    min_train_matches: int = 200,
    refit_every_days: int = 7,
    adjust: Callable[[DixonColesFit, pd.Series], tuple[float, float]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """Run the backtest and return every prediction it made.

    ``adjust`` is the hook Phases 5 and 6 use: given the fit and the fixture
    row it returns adjusted (lambda_home, lambda_away). Leaving it ``None``
    backtests the bare model, which is the baseline every later addition is
    measured against.
    """
    params = params or ModelParams()
    frame = _prepare(matches)
    if frame.empty:
        raise NotEnoughData("no finished matches to backtest")

    all_days = sorted(frame["match_date"].unique())
    test_start = test_start or all_days[0]
    test_end = test_end or all_days[-1]
    test_days = [d for d in all_days if test_start <= d <= test_end]

    fit: DixonColesFit | None = None
    fit_trained_to: dt.date | None = None
    rows: list[dict[str, Any]] = []
    result = BacktestResult(predictions=pd.DataFrame(columns=PREDICTION_COLUMNS), params=params)

    for position, day in enumerate(test_days):
        train = frame[frame["match_date"] < day]
        if len(train) < min_train_matches:
            result.skipped_days.append(day)
            continue

        needs_refit = (
            fit is None
            or fit_trained_to is None
            or (day - fit_trained_to).days >= refit_every_days
        )
        if needs_refit:
            train_end = train["match_date"].max()
            if train_end >= day:
                raise LeakageError(
                    f"training window ends {train_end} but predicting {day}"
                )
            try:
                fit = fit_dixon_coles(train, ref_date=day, params=params, initial=fit)
            except NotEnoughData:
                result.skipped_days.append(day)
                continue
            fit_trained_to = day
            result.n_fits += 1

        assert fit is not None
        if fit.train_end_date >= day:  # belt and braces: the fit knows its own window
            raise LeakageError(
                f"fit trained through {fit.train_end_date} used to predict {day}"
            )

        day_matches = frame[frame["match_date"] == day]
        market_btts, market_over = _market_probabilities(day_matches)

        for offset, (_, match) in enumerate(day_matches.iterrows()):
            lam, mu = fit.lambdas(match["home_team"], match["away_team"])
            if adjust is not None:
                lam, mu = adjust(fit, match)
            probs = match_probabilities(lam, mu, fit.rho, params.max_goals)
            home_goals = int(match["home_goals"])
            away_goals = int(match["away_goals"])
            rows.append({
                "match_date": day,
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "lambda_home": probs.lambda_home,
                "lambda_away": probs.lambda_away,
                "p_btts": probs.p_btts,
                "p_over25": probs.p_over25,
                "p_home": probs.p_home,
                "p_draw": probs.p_draw,
                "p_away": probs.p_away,
                "actual_btts": int(home_goals >= 1 and away_goals >= 1),
                "actual_over25": int(home_goals + away_goals >= 3),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "market_p_btts": float(market_btts[offset]) if offset < len(market_btts) else np.nan,
                "market_p_over25": float(market_over[offset]) if offset < len(market_over) else np.nan,
                "train_rows": fit.train_rows,
                "train_end_date": fit.train_end_date,
                "fit_date": fit_trained_to,
            })

        if progress is not None:
            progress(position + 1, len(test_days))

    result.predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    log.info("walk-forward complete",
             extra={"predictions": len(result.predictions), "fits": result.n_fits,
                    "skipped_days": len(result.skipped_days)})
    return result


def assert_no_leakage(predictions: pd.DataFrame) -> None:
    """Post-hoc check: every prediction's training window closed before kickoff."""
    if predictions.empty:
        return
    bad = predictions[pd.to_datetime(predictions["train_end_date"])
                      >= pd.to_datetime(predictions["match_date"])]
    if not bad.empty:
        raise LeakageError(
            f"{len(bad)} prediction(s) trained on data at or after the match date, "
            f"first on {bad.iloc[0]['match_date']}"
        )


def split_calibration(
    result: BacktestResult,
    holdout_fraction: float = 0.5,
) -> tuple[CalibrationSet, pd.DataFrame, pd.DataFrame]:
    """Fit Platt scaling on the earlier half, evaluate on the later half.

    Calibrating and scoring on the same rows would flatter the model; the split
    is by date so the evaluation period is strictly later, as it would be live.
    """
    predictions = result.predictions.sort_values("match_date").reset_index(drop=True)
    if predictions.empty:
        return CalibrationSet(), predictions, predictions
    cut = int(len(predictions) * holdout_fraction)
    calib_rows = predictions.iloc[:cut]
    eval_rows = predictions.iloc[cut:].reset_index(drop=True)
    calibration = fit_calibration(calib_rows) if len(calib_rows) else CalibrationSet()
    return calibration, calib_rows, eval_rows


def apply_calibration(predictions: pd.DataFrame, calibration: CalibrationSet) -> pd.DataFrame:
    """Return a copy with the calibrated probabilities in place."""
    out = predictions.copy()
    out["p_btts_raw"] = out["p_btts"]
    out["p_over25_raw"] = out["p_over25"]
    out["p_btts"] = calibration.btts.transform(out["p_btts"])
    out["p_over25"] = calibration.over25.transform(out["p_over25"])
    return out


# --------------------------------------------------------------------------
# Hyperparameter search
# --------------------------------------------------------------------------
@dataclass
class TuningRow:
    params: ModelParams
    over25_log_loss: float
    btts_log_loss: float
    combined: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "decay_half_life_days": self.params.decay_half_life_days,
            "shrinkage_prior": self.params.shrinkage_prior,
            "min_weighted_matches": self.params.min_weighted_matches,
            "over25_log_loss": self.over25_log_loss,
            "btts_log_loss": self.btts_log_loss,
            "combined": self.combined,
            "n": self.n,
        }


def grid_search(
    matches: pd.DataFrame,
    base: ModelParams,
    grid: dict[str, Sequence[Any]],
    *,
    test_start: dt.date | None = None,
    test_end: dt.date | None = None,
    refit_every_days: int = 14,
    min_train_matches: int = 200,
    progress: Callable[[int, int, ModelParams], None] | None = None,
) -> pd.DataFrame:
    """Grid-search hyperparameters on **out-of-sample** log loss.

    Tuning on training fit would pick the shortest half-life every time; the
    score here comes from the same walk-forward the model is judged by.
    """
    keys = sorted(grid)
    combinations = list(itertools.product(*(grid[k] for k in keys)))
    results: list[TuningRow] = []

    for i, values in enumerate(combinations):
        params = base.replace(**dict(zip(keys, values)))
        if progress is not None:
            progress(i + 1, len(combinations), params)
        result = walk_forward(
            matches, params,
            test_start=test_start, test_end=test_end,
            refit_every_days=refit_every_days,
            min_train_matches=min_train_matches,
        )
        if result.predictions.empty:
            continue
        summary = result.summary()
        over = summary["scores"].get("over25")
        btts = summary["scores"].get("btts")
        over_ll = over.model_log_loss if over else float("nan")
        btts_ll = btts.model_log_loss if btts else float("nan")
        results.append(TuningRow(
            params=params,
            over25_log_loss=over_ll,
            btts_log_loss=btts_ll,
            combined=float(np.nanmean([over_ll, btts_ll])),
            n=result.n_predictions,
        ))

    frame = pd.DataFrame([r.as_dict() for r in results])
    return frame.sort_values("combined").reset_index(drop=True) if not frame.empty else frame
