"""The fit itself: gradient, identifiability, parameter recovery, shrinkage.

Ground truth comes from champmodel.synth, which simulates matches from known
Dixon-Coles parameters. That is the only way to assert that the fit recovers
what it should rather than merely converging to something.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import spearmanr

from champmodel import synth
from champmodel.config import ModelParams
from champmodel.features.form import league_table
from champmodel.model.dixon_coles import (DixonColesFit, NotEnoughData,
                                          build_training_data, decay_weights,
                                          fit_dixon_coles,
                                          negative_log_likelihood, shrinkage_ridge)


@pytest.fixture(scope="module")
def league():
    return synth.generate(n_seasons=4, n_teams=20, seed=11)


@pytest.fixture(scope="module")
def fit(league):
    ref = league.matches["match_date"].max()
    return fit_dixon_coles(league.matches, ref_date=ref, params=ModelParams())


def test_analytic_gradient_matches_finite_differences(league):
    """The whole backtest rests on this gradient; a wrong one converges to the
    wrong place while still reporting success."""
    ref = league.matches["match_date"].max()
    data = build_training_data(league.matches.head(400), ref, 180.0)
    ridge = shrinkage_ridge(data, ModelParams())
    rng = np.random.default_rng(3)
    theta = np.concatenate([
        rng.normal(0, 0.2, data.n_teams), rng.normal(0, 0.2, data.n_teams),
        [0.24], [-0.07],
    ])
    _, analytic = negative_log_likelihood(theta, data, ridge)
    numeric = approx_fprime(
        theta, lambda t: negative_log_likelihood(t, data, ridge, with_gradient=False), 1e-6
    )
    assert np.abs(analytic - numeric).max() < 1e-4


def test_fit_converges(fit):
    assert fit.converged
    # A decayed fit's effective sample is roughly one season, so gamma carries
    # a standard error near 0.05. Assert the plausible football range here and
    # leave the tight recovery check to the undecayed fit below.
    assert 0.10 <= fit.home_advantage <= 0.45


def test_fit_recovers_the_true_home_advantage(league):
    """With decay switched off the full history is in play (~1900 matches),
    which pins gamma to within about 0.02. Checked across five seeds while
    writing the test: the worst deviation from the true 0.25 was 0.033.
    """
    fit = fit_dixon_coles(league.matches, league.matches["match_date"].max(),
                          ModelParams(decay_half_life_days=0))
    assert fit.home_advantage == pytest.approx(league.home_advantage, abs=0.06)


def test_fit_recovers_a_negative_rho(fit):
    assert -0.25 < fit.rho < 0.0


def test_attack_ratings_track_the_true_ratings(league):
    """``true_attack`` holds the final season's ratings, so the comparable fit
    is the final season with decay off. One season of 20 teams is a small
    sample; across five seeds the correlation ran 0.74 to 0.94, so 0.65 is a
    threshold a working fit clears and a broken one (correlation ~0) cannot.
    """
    final = league.matches[league.matches["match_date"] >= dt.date(2024, 8, 1)]
    fit = fit_dixon_coles(final, final["match_date"].max(),
                          ModelParams(decay_half_life_days=0))
    truth = dict(zip(league.teams, league.true_attack))
    estimated = dict(zip(fit.teams, fit.attack))
    common = sorted(set(truth) & set(estimated))
    correlation = np.corrcoef([truth[t] for t in common],
                              [estimated[t] for t in common])[0, 1]
    assert correlation > 0.65


def test_top_of_the_table_has_the_strongest_ratings(league):
    """Phase 4's sanity check, on a single season with decay switched off."""
    last_season = league.matches[
        league.matches["match_date"] >= dt.date(2024, 8, 1)
    ]
    fit = fit_dixon_coles(last_season, last_season["match_date"].max(),
                          ModelParams(decay_half_life_days=0))
    merged = fit.team_table().merge(league_table(last_season), on="team")
    assert spearmanr(merged["net_strength"], merged["points"]).correlation > 0.8


def test_attack_ratings_are_centred(fit):
    assert float(np.mean(fit.attack)) == pytest.approx(0.0, abs=1e-6)


def test_lambdas_are_plausible_for_the_championship(fit, league):
    lambdas = [fit.lambdas(h, a) for h in league.teams[:6] for a in league.teams[6:12]]
    totals = [lam + mu for lam, mu in lambdas]
    assert 1.5 < np.mean(totals) < 3.6
    assert all(lam > mu for lam, mu in
               [fit.lambdas(t, t2) for t, t2 in [(league.teams[0], league.teams[0])]])


def test_home_advantage_makes_the_same_pair_favour_the_home_side(fit, league):
    home, away = league.teams[0], league.teams[1]
    lam_a, mu_a = fit.lambdas(home, away)
    lam_b, mu_b = fit.lambdas(away, home)
    # Swapping venue must swap which side gets the boost.
    assert lam_a / mu_b > 1.0
    assert lam_b / mu_a > 1.0


def test_unknown_team_falls_back_to_the_league_mean(fit, league):
    known = fit.lambdas(league.teams[0], league.teams[1])
    unknown = fit.lambdas(league.teams[0], "Newly Promoted FC")
    assert all(np.isfinite(unknown))
    assert unknown != known
    assert not fit.has_team("Newly Promoted FC")


def test_shrinkage_pulls_a_thin_record_toward_the_mean(league):
    """A side with three matches must not top the attack table on that evidence."""
    matches = league.matches.copy()
    newcomer = matches["home_team"] == league.teams[0]
    # Give one club a tiny record of lopsided wins.
    trimmed = matches[~(newcomer | (matches["away_team"] == league.teams[0]))]
    hot_start = matches[newcomer].head(3).copy()
    hot_start["home_goals"] = 5
    hot_start["away_goals"] = 0
    combined = trimmed._append(hot_start) if hasattr(trimmed, "_append") \
        else __import__("pandas").concat([trimmed, hot_start])
    ref = combined["match_date"].max()

    strong = fit_dixon_coles(combined, ref, ModelParams(shrinkage_prior=3.0,
                                                        min_weighted_matches=20))
    weak = fit_dixon_coles(combined, ref, ModelParams(shrinkage_prior=0.0,
                                                      min_weighted_matches=20))
    i_strong = strong.index[league.teams[0]]
    i_weak = weak.index[league.teams[0]]
    assert abs(strong.attack[i_strong]) < abs(weak.attack[i_weak])
    assert league.teams[0] in strong.shrunk_teams()


def test_decay_weights_halve_at_the_half_life():
    ref = dt.date(2026, 9, 4)
    dates = [ref, ref - dt.timedelta(days=180), ref - dt.timedelta(days=360)]
    weights = decay_weights(dates, ref, 180.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, abs=1e-9)
    assert weights[2] == pytest.approx(0.25, abs=1e-9)


def test_zero_half_life_disables_decay():
    ref = dt.date(2026, 9, 4)
    dates = [ref - dt.timedelta(days=d) for d in (0, 100, 1000)]
    assert np.allclose(decay_weights(dates, ref, 0.0), 1.0)


def test_decay_makes_recent_matches_dominate(league):
    """Form enters the model through decay, so a short half-life has to move
    the ratings much more than a long one."""
    ref = league.matches["match_date"].max()
    short = fit_dixon_coles(league.matches, ref, ModelParams(decay_half_life_days=60))
    long = fit_dixon_coles(league.matches, ref, ModelParams(decay_half_life_days=1000))
    assert np.abs(short.attack - long.attack).max() > 0.05


def test_fit_round_trips_through_json(fit, tmp_path):
    path = fit.save(tmp_path / "fit.json")
    reloaded = DixonColesFit.load(path)
    assert reloaded.teams == fit.teams
    assert np.allclose(reloaded.attack, fit.attack)
    assert np.allclose(reloaded.defence, fit.defence)
    assert reloaded.home_advantage == pytest.approx(fit.home_advantage)
    assert reloaded.rho == pytest.approx(fit.rho)
    assert reloaded.train_end_date == fit.train_end_date
    assert reloaded.params == fit.params


def test_too_little_data_raises(league):
    with pytest.raises(NotEnoughData):
        fit_dixon_coles(league.matches.head(5), league.matches["match_date"].max(),
                        ModelParams())


def test_predictions_are_probabilities(fit, league):
    probs = fit.predict(league.teams[0], league.teams[1])
    for value in (probs.p_btts, probs.p_over25, probs.p_home, probs.p_draw, probs.p_away):
        assert 0.0 <= value <= 1.0
    assert probs.p_home + probs.p_draw + probs.p_away == pytest.approx(1.0, abs=1e-9)
