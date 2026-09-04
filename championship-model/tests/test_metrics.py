"""Scoring rules, de-vigging, and the calibration table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from champmodel.backtest.metrics import (brier_score, calibration_table,
                                         devig_proportional,
                                         expected_calibration_error, log_loss,
                                         max_bucket_error, overround,
                                         score_market, summarise)


def test_log_loss_of_a_perfect_forecast_is_zero():
    assert log_loss([1, 0, 1], [1 - 1e-12, 1e-12, 1 - 1e-12]) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_of_a_coin_flip():
    assert log_loss([1, 0, 1, 0], [0.5] * 4) == pytest.approx(np.log(2))


def test_log_loss_punishes_confident_mistakes():
    assert log_loss([0], [0.99]) > log_loss([0], [0.6]) > log_loss([0], [0.4])


def test_brier_score_bounds():
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_devig_removes_the_overround():
    """1.90 / 1.90 is a 5.3% book on an even-money market -- de-vigged, it is
    exactly 0.5 a side."""
    assert float(devig_proportional(1.90, 1.90)) == pytest.approx(0.5)
    assert float(overround(1.90, 1.90)) == pytest.approx(1.0 / 1.9 * 2)


def test_devigged_probabilities_sum_to_one():
    yes = devig_proportional([2.10, 1.55, 3.40], [1.80, 2.50, 1.35])
    no = devig_proportional([1.80, 2.50, 1.35], [2.10, 1.55, 3.40])
    assert np.allclose(yes + no, 1.0)


def test_devig_is_nan_for_missing_prices():
    result = devig_proportional([np.nan, 1.90, 0.0], [1.90, np.nan, 2.0])
    assert np.isnan(result[0]) and np.isnan(result[1]) and np.isnan(result[2])


def test_score_market_compares_like_with_like():
    """The market can only be scored where it has a price, so the model must be
    scored on the same subset -- otherwise the comparison is meaningless."""
    y = [1, 0, 1, 0]
    model = [0.6, 0.4, 0.7, 0.3]
    market = [0.55, 0.45, np.nan, np.nan]
    score = score_market(y, model, market, market="over25")
    assert score.n == 2
    assert score.n_market == 2
    assert score.model_log_loss == pytest.approx(log_loss(y[:2], model[:2]))
    assert score.market_log_loss == pytest.approx(log_loss(y[:2], market[:2]))


def test_score_market_without_a_market_leaves_the_gap_undefined():
    score = score_market([1, 0], [0.6, 0.4], None)
    assert np.isnan(score.market_log_loss)
    assert not score.matches_market


def test_matches_market_uses_the_one_hundredth_bar():
    close = score_market([1, 0] * 50, [0.6, 0.4] * 50, [0.6, 0.4] * 50)
    assert close.matches_market
    far = score_market([1, 0] * 50, [0.9, 0.1] * 50, [0.6, 0.4] * 50)
    assert not far.matches_market


def test_calibration_table_recovers_a_known_frequency():
    rng = np.random.default_rng(5)
    p = np.repeat([0.25, 0.75], 2000)
    y = rng.binomial(1, p)
    table = calibration_table(y, p, 10)
    populated = table[table["n"] > 0]
    assert len(populated) == 2
    for _, row in populated.iterrows():
        assert row["observed"] == pytest.approx(row["mean_predicted"], abs=0.03)
    assert max_bucket_error(table) < 0.03


def test_calibration_table_has_one_row_per_bucket_and_counts_everything():
    p = np.linspace(0.01, 0.99, 500)
    y = (p > 0.5).astype(int)
    table = calibration_table(y, p, 10)
    assert len(table) == 10
    assert table["n"].sum() == 500


def test_calibration_table_flags_a_biased_forecast():
    p = np.full(1000, 0.8)
    y = np.zeros(1000, dtype=int)
    y[:300] = 1                       # says 80%, happens 30%
    table = calibration_table(y, p, 10)
    assert max_bucket_error(table) == pytest.approx(0.5, abs=0.01)
    assert expected_calibration_error(table) == pytest.approx(0.5, abs=0.01)


def test_summarise_covers_both_markets():
    frame = pd.DataFrame({
        "p_btts": [0.5, 0.6, 0.4],
        "p_over25": [0.45, 0.55, 0.5],
        "actual_btts": [1, 1, 0],
        "actual_over25": [0, 1, 1],
        "market_p_btts": [0.52, 0.58, 0.42],
        "market_p_over25": [0.47, 0.53, 0.5],
    })
    summary = summarise(frame)
    assert set(summary["scores"]) == {"btts", "over25"}
    assert set(summary["tables"]) == {"btts", "over25"}
    assert summary["n_rows"] == 3
    assert np.isfinite(summary["scores"]["btts"].market_log_loss)
