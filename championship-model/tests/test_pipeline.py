"""The prediction path: flags, ordering of adjustments, and the stored row.

The point of this file is the missing_inputs contract. A prediction made
without team news, without a market price or without a calibrator has to say
so on the row, because it is not the same product as one made with them.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from champmodel import synth
from champmodel.config import ModelParams
from champmodel.features.availability_adj import build_index
from champmodel.features.h2h import H2HIndex
from champmodel.model.calibrate import CalibrationSet, PlattCalibrator
from champmodel.model.dixon_coles import fit_dixon_coles
from champmodel.model.pipeline import (FLAG_NO_AVAILABILITY, FLAG_NO_H2H,
                                       FLAG_NO_MARKET, FLAG_UNCALIBRATED,
                                       FLAG_UNKNOWN_HOME, predict_fixture)

DAY = dt.date(2026, 9, 4)


@pytest.fixture(scope="module")
def league():
    return synth.generate(n_seasons=3, n_teams=12, seed=17)


@pytest.fixture(scope="module")
def fit(league):
    return fit_dixon_coles(league.matches, league.matches["match_date"].max(),
                           ModelParams())


def _predict(fit, league, **kwargs):
    return predict_fixture(fit, league.teams[0], league.teams[1], DAY,
                           ModelParams(), **kwargs)


def test_bare_prediction_flags_every_absent_input(fit, league):
    prediction = _predict(fit, league)
    assert FLAG_NO_AVAILABILITY in prediction.missing_inputs
    assert FLAG_NO_H2H in prediction.missing_inputs
    assert FLAG_NO_MARKET in prediction.missing_inputs
    assert FLAG_UNCALIBRATED in prediction.missing_inputs
    assert prediction.missing_inputs_text()


def test_a_complete_prediction_has_nothing_to_flag(fit, league):
    calibration = CalibrationSet(
        btts=PlattCalibrator(0.9, 0.0, 500, True, "btts"),
        over25=PlattCalibrator(0.9, 0.0, 500, True, "over25"),
    )
    prediction = predict_fixture(
        fit, league.teams[0], league.teams[1], DAY, ModelParams(),
        availability=build_index([], ModelParams(), DAY),
        home_team_id=1, away_team_id=2,
        h2h_index=H2HIndex(league.matches),
        calibration=calibration,
        market_p_over25=0.51, market_p_btts=0.49,
    )
    assert prediction.missing_inputs == []
    assert prediction.missing_inputs_text() is None


def test_unknown_team_is_flagged(fit, league):
    prediction = predict_fixture(fit, "Newly Promoted FC", league.teams[1], DAY,
                                 ModelParams())
    assert FLAG_UNKNOWN_HOME in prediction.missing_inputs


def test_missing_inputs_text_fits_the_column(fit, league):
    prediction = _predict(fit, league, extra_flags=["x" * 300])
    assert len(prediction.missing_inputs_text()) <= 200


def test_calibration_is_applied_and_the_raw_value_is_kept(fit, league):
    calibration = CalibrationSet(
        btts=PlattCalibrator(0.5, 0.0, 500, True, "btts"),
        over25=PlattCalibrator(0.5, 0.0, 500, True, "over25"),
    )
    prediction = _predict(fit, league, calibration=calibration)
    assert prediction.p_over25 != prediction.p_over25_raw
    assert prediction.p_btts != prediction.p_btts_raw
    assert FLAG_UNCALIBRATED not in prediction.missing_inputs
    # a = 0.5 pulls everything toward 0.5.
    assert abs(prediction.p_over25 - 0.5) < abs(prediction.p_over25_raw - 0.5)


def test_availability_moves_the_lambdas_and_sets_the_flag(fit, league):
    params = ModelParams(availability_k=0.5)
    index = build_index([{"team_id": 1, "player_name": "Striker",
                          "minutes_share": 0.5, "is_attacker": True,
                          "is_defender": False}], params, DAY)
    baseline = predict_fixture(fit, league.teams[0], league.teams[1], DAY, params,
                               availability=build_index([], params, DAY),
                               home_team_id=1, away_team_id=2)
    adjusted = predict_fixture(fit, league.teams[0], league.teams[1], DAY, params,
                               availability=index, home_team_id=1, away_team_id=2)
    assert adjusted.lambda_home < baseline.lambda_home
    assert adjusted.availability_applied
    assert not baseline.availability_applied


def test_edges_are_signed_differences(fit, league):
    prediction = _predict(fit, league, market_p_over25=0.40, market_p_btts=0.60)
    assert prediction.edge_over25 == pytest.approx(prediction.p_over25 - 0.40)
    assert prediction.edge_btts == pytest.approx(prediction.p_btts - 0.60)


def test_edge_is_none_without_a_price(fit, league):
    assert _predict(fit, league).edge_over25 is None


def test_h2h_is_off_by_default_even_with_an_index(fit, league):
    with_index = _predict(fit, league, h2h_index=H2HIndex(league.matches))
    assert not with_index.h2h_applied
    assert with_index.h2h is not None


def test_to_row_is_shaped_for_the_prediction_table(fit, league):
    prediction = _predict(fit, league, match_id=42)
    row = prediction.to_row(run_id=7)
    assert row["run_id"] == 7 and row["match_id"] == 42
    assert 0.0 <= row["p_btts"] <= 1.0
    assert 0.0 <= row["p_over25"] <= 1.0
    assert row["p_home"] + row["p_draw"] + row["p_away"] == pytest.approx(1.0, abs=1e-4)
    # DECIMAL(6,5) and DECIMAL(6,4) in the schema.
    assert len(str(row["p_btts"]).split(".")[-1]) <= 5
    assert len(str(row["lambda_home"]).split(".")[-1]) <= 4


def test_probabilities_sit_in_the_range_the_readme_promises(fit, league):
    """Championship scoring clusters near 2.5 goals, so P(over 2.5) should sit
    around 0.45-0.55 for most fixtures. Extremes mean an input is wrong."""
    values = [
        predict_fixture(fit, home, away, DAY, ModelParams()).p_over25
        for home in league.teams[:6] for away in league.teams[6:]
        if home != away
    ]
    assert 0.35 < pd.Series(values).median() < 0.65
    assert all(0.05 < v < 0.95 for v in values)
