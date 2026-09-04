"""Phase 9: the backtest's training window must never reach the target match.

A leaked result is the single easiest way to produce a model that looks
excellent and is worthless, so this is checked three ways: the walk-forward
raises if a fit's window closes on or after the match date, the post-hoc
assertion catches it in a frame, and a deliberately leaked frame is rejected.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from champmodel import synth
from champmodel.config import ModelParams
from champmodel.backtest.walk_forward import (LeakageError, assert_no_leakage,
                                              split_calibration, walk_forward)


@pytest.fixture(scope="module")
def league():
    return synth.generate(n_seasons=3, n_teams=12, seed=7)


@pytest.fixture(scope="module")
def result(league):
    days = sorted(league.matches["match_date"].unique())
    return walk_forward(league.matches, ModelParams(),
                        test_start=days[len(days) // 2],
                        refit_every_days=14, min_train_matches=120)


def test_backtest_produces_predictions(result):
    assert result.n_predictions > 100
    assert result.n_fits > 0


def test_every_training_window_closes_before_the_match(result):
    frame = result.predictions
    assert (pd.to_datetime(frame["train_end_date"])
            < pd.to_datetime(frame["match_date"])).all()
    assert_no_leakage(frame)


def test_reused_fits_are_also_clean(result):
    """Refitting weekly means a fit predicts several later days; each of those
    must still sit strictly after the window."""
    frame = result.predictions
    reused = frame[frame["fit_date"] != frame["match_date"]]
    assert not reused.empty, "the test window should include reused fits"
    assert (pd.to_datetime(reused["train_end_date"])
            < pd.to_datetime(reused["match_date"])).all()


def test_assert_no_leakage_rejects_a_leaked_frame(result):
    leaked = result.predictions.copy()
    leaked.loc[leaked.index[0], "train_end_date"] = leaked.loc[leaked.index[0], "match_date"]
    with pytest.raises(LeakageError):
        assert_no_leakage(leaked)


def test_assert_no_leakage_accepts_an_empty_frame():
    assert_no_leakage(pd.DataFrame())


def test_calibration_split_is_chronological(result):
    _, calib_rows, eval_rows = split_calibration(result, 0.5)
    assert not calib_rows.empty and not eval_rows.empty
    # The calibrator may never see a match later than the rows it is scored on.
    assert calib_rows["match_date"].max() <= eval_rows["match_date"].min()


def test_predictions_only_cover_the_requested_window(league):
    days = sorted(league.matches["match_date"].unique())
    start, end = days[len(days) // 2], days[len(days) // 2 + 20]
    result = walk_forward(league.matches, ModelParams(), test_start=start, test_end=end,
                          refit_every_days=14, min_train_matches=120)
    assert result.predictions["match_date"].min() >= start
    assert result.predictions["match_date"].max() <= end
