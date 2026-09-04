"""Availability adjustment, head-to-head, form, and Platt calibration."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from champmodel.config import ModelParams
from champmodel.features.availability_adj import (adjust_lambdas, build_index,
                                                  missing_input_flags)
from champmodel.features.form import FormIndex, league_table
from champmodel.features.h2h import (H2HIndex, MAX_H2H_WEIGHT, h2h_adjustment)
from champmodel.model.calibrate import (CalibrationSet, PlattCalibrator,
                                        fit_platt, logit, sigmoid)

DAY = dt.date(2026, 9, 4)


def _row(team_id, share, attacker=False, defender=False, name="Player"):
    return {"team_id": team_id, "player_name": name, "minutes_share": share,
            "is_attacker": attacker, "is_defender": defender}


# -- availability ----------------------------------------------------------
def test_missing_attacker_lowers_attack_more_than_a_squad_player():
    params = ModelParams(availability_k=0.5, availability_other_weight=0.3)
    attacker = build_index([_row(1, 0.4, attacker=True)], params, DAY).get(1)
    other = build_index([_row(1, 0.4)], params, DAY).get(1)
    assert attacker.attack_delta < other.attack_delta < 0
    assert attacker.attack_delta == pytest.approx(-0.5 * 0.4)
    assert other.attack_delta == pytest.approx(-0.5 * 0.4 * 0.3)


def test_missing_defender_raises_the_concession_rate():
    params = ModelParams(availability_k=0.5)
    entry = build_index([_row(2, 0.5, defender=True)], params, DAY).get(2)
    assert entry.defence_delta == pytest.approx(0.25)
    assert entry.defence_delta > 0


def test_the_cap_holds_against_a_fat_fingered_minutes_share():
    """The typo this cap exists for: four key players entered at 0.9 each."""
    params = ModelParams(availability_k=0.5, availability_cap=0.25)
    rows = [_row(1, 0.9, attacker=True, name=f"P{i}") for i in range(4)]
    entry = build_index(rows, params, DAY).get(1)
    assert entry.missing_attack_share == pytest.approx(3.6)
    assert entry.attack_delta == pytest.approx(-0.25)   # not -1.80
    assert entry.capped
    assert missing_input_flags(entry, None) == ["home_avail_capped"]


def test_an_out_of_range_share_in_the_database_is_still_clamped():
    params = ModelParams(availability_k=0.5, availability_cap=0.25)
    entry = build_index([_row(1, 17.0, attacker=True)], params, DAY).get(1)
    assert entry.missing_attack_share <= 1.0


def test_adjust_lambdas_moves_the_right_side():
    """A missing player has a primary effect and a smaller secondary one: an
    absent striker mostly cuts his own side's attack, but he was also part of
    the defensive shape, so the opponent's lambda drifts up at weight 0.3."""
    params = ModelParams(availability_k=0.5, availability_other_weight=0.3)
    home_striker_out = build_index([_row(1, 0.4, attacker=True)], params, DAY).get(1)
    away_defender_out = build_index([_row(2, 0.4, defender=True)], params, DAY).get(2)

    lam, mu = adjust_lambdas(1.60, 1.20, home_striker_out, None)
    assert lam == pytest.approx(1.60 * np.exp(-0.5 * 0.4))          # attack cut
    assert mu == pytest.approx(1.20 * np.exp(0.5 * 0.4 * 0.3))      # secondary
    assert lam < 1.60 < 1.60 * 1.2 and mu > 1.20

    lam, mu = adjust_lambdas(1.60, 1.20, None, away_defender_out)
    assert lam == pytest.approx(1.60 * np.exp(0.5 * 0.4))           # concedes more
    assert mu == pytest.approx(1.20 * np.exp(-0.5 * 0.4 * 0.3))
    assert lam > 1.60 and mu < 1.20


def test_no_overrides_means_no_change():
    index = build_index([], ModelParams(), DAY)
    assert index.deltas(1) == (0.0, 0.0)
    assert adjust_lambdas(1.5, 1.1, None, None) == (1.5, 1.1)


# -- head to head ----------------------------------------------------------
@pytest.fixture()
def h2h_matches():
    rows = []
    for year, (hg, ag) in zip(range(2018, 2026), [(2, 1), (0, 0), (1, 1), (3, 2),
                                                   (2, 2), (1, 0), (4, 1), (2, 3)]):
        rows.append({"match_date": dt.date(year, 3, 1), "home_team": "Leeds United",
                     "away_team": "Millwall", "home_goals": hg, "away_goals": ag})
    rows.append({"match_date": dt.date(2026, 4, 1), "home_team": "Leeds United",
                 "away_team": "Norwich City", "home_goals": 1, "away_goals": 1})
    return pd.DataFrame(rows)


def test_h2h_record_is_symmetric_in_the_pair(h2h_matches):
    index = H2HIndex(h2h_matches)
    forward = index.record("Leeds United", "Millwall")
    reverse = index.record("Millwall", "Leeds United")
    assert forward.meetings == reverse.meetings == 8
    assert forward.btts_rate == pytest.approx(reverse.btts_rate)


def test_h2h_rates_are_right(h2h_matches):
    record = H2HIndex(h2h_matches).record("Leeds United", "Millwall")
    # Scores 2-1 0-0 1-1 3-2 2-2 1-0 4-1 2-3.
    # BTTS in 2-1, 1-1, 3-2, 2-2, 4-1, 2-3 = 6 of 8.
    # Totals 3 0 2 5 4 1 5 5, so five reach three or more.
    assert record.btts_rate == pytest.approx(6 / 8)
    assert record.over25_rate == pytest.approx(5 / 8)
    assert record.avg_total_goals == pytest.approx(25 / 8)


def test_h2h_window_only_looks_backwards(h2h_matches):
    index = H2HIndex(h2h_matches)
    record = index.record("Leeds United", "Millwall", before=dt.date(2021, 1, 1))
    assert record.meetings == 3
    assert record.last_meeting == dt.date(2020, 3, 1)


def test_unknown_pair_has_no_record(h2h_matches):
    record = H2HIndex(h2h_matches).record("Hull City", "Watford")
    assert not record.has_data
    assert record.display_btts() == "-"


def test_h2h_adjustment_is_off_by_default(h2h_matches):
    record = H2HIndex(h2h_matches).record("Leeds United", "Millwall")
    lam, mu, applied = h2h_adjustment(1.6, 1.2, record, 0.48, ModelParams())
    assert (lam, mu, applied) == (1.6, 1.2, False)


def test_h2h_adjustment_weight_is_clamped(h2h_matches):
    """However the config is set, H2H may not overrule the fitted strengths."""
    record = H2HIndex(h2h_matches).record("Leeds United", "Millwall")
    sane = ModelParams(enable_h2h_adjustment=True, h2h_weight=MAX_H2H_WEIGHT)
    absurd = ModelParams(enable_h2h_adjustment=True, h2h_weight=50.0)
    assert h2h_adjustment(1.6, 1.2, record, 0.48, sane)[:2] == \
           pytest.approx(h2h_adjustment(1.6, 1.2, record, 0.48, absurd)[:2])


def test_h2h_ignores_a_thin_record():
    matches = pd.DataFrame([{"match_date": dt.date(2025, 3, 1), "home_team": "A",
                             "away_team": "B", "home_goals": 1, "away_goals": 1}])
    record = H2HIndex(matches).record("A", "B")
    params = ModelParams(enable_h2h_adjustment=True, h2h_weight=0.1)
    assert h2h_adjustment(1.5, 1.1, record, 0.5, params) == (1.5, 1.1, False)


# -- form (display only) ---------------------------------------------------
def test_form_reads_most_recent_first(h2h_matches):
    form = FormIndex(h2h_matches).form("Leeds United", window=3)
    assert form.matches == 3
    assert len(form.results) == 3
    assert form.results[0] in "WDL"


def test_league_table_orders_by_points():
    matches = pd.DataFrame([
        {"match_date": dt.date(2026, 1, 1), "home_team": "A", "away_team": "B",
         "home_goals": 3, "away_goals": 0},
        {"match_date": dt.date(2026, 1, 8), "home_team": "B", "away_team": "C",
         "home_goals": 1, "away_goals": 1},
    ])
    table = league_table(matches)
    assert list(table["team"]) == ["A", "B", "C"] or list(table["team"])[0] == "A"
    assert table.iloc[0]["points"] == 3


# -- calibration -----------------------------------------------------------
def test_logit_and_sigmoid_round_trip():
    values = np.array([0.01, 0.3, 0.5, 0.87, 0.999])
    assert np.allclose(sigmoid(logit(values)), values, atol=1e-9)


def test_identity_calibrator_changes_nothing():
    calibrator = PlattCalibrator()
    assert calibrator.is_identity
    assert np.allclose(calibrator.transform([0.2, 0.5, 0.8]), [0.2, 0.5, 0.8])


def test_platt_corrects_a_systematically_overconfident_forecast():
    rng = np.random.default_rng(4)
    truth = rng.uniform(0.25, 0.75, 4000)
    # Push predictions away from the base rate: a classic overconfident model.
    overconfident = sigmoid(1.8 * logit(truth))
    outcomes = rng.binomial(1, truth)

    calibrator = fit_platt(overconfident, outcomes, market="test")
    assert calibrator.fitted
    assert calibrator.a < 1.0            # shrink back toward the base rate

    from champmodel.backtest.metrics import log_loss
    assert log_loss(outcomes, calibrator.transform(overconfident)) < \
           log_loss(outcomes, overconfident)


def test_platt_refuses_to_fit_on_too_little_data():
    calibrator = fit_platt([0.4, 0.6, 0.5], [1, 0, 1])
    assert not calibrator.fitted
    assert calibrator.is_identity


def test_platt_refuses_when_every_outcome_is_the_same():
    calibrator = fit_platt(np.linspace(0.2, 0.8, 500), np.ones(500, dtype=int))
    assert not calibrator.fitted


def test_calibration_set_round_trips():
    original = CalibrationSet(btts=PlattCalibrator(0.9, 0.05, 500, True, "btts"),
                              over25=PlattCalibrator(0.8, -0.02, 500, True, "over25"))
    restored = CalibrationSet.from_dict(original.to_dict())
    assert restored.btts.a == pytest.approx(0.9)
    assert restored.over25.b == pytest.approx(-0.02)
    assert restored.any_fitted
