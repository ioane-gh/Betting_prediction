"""Phase 9: the score matrix must be a probability distribution, and with tau
disabled it must agree with an independent Poisson calculation."""

from __future__ import annotations

import numpy as np
import pytest

from champmodel.model.score_matrix import (
    clip_rho, independent_poisson_btts, independent_poisson_over25,
    match_probabilities, p_over_line, probabilities_from_matrix, score_matrix,
    tau_matrix,
)

LAMBDAS = [(1.45, 1.15), (0.80, 0.60), (2.30, 1.90), (1.00, 1.00), (3.10, 0.45)]


@pytest.mark.parametrize("lam,mu", LAMBDAS)
@pytest.mark.parametrize("rho", [0.0, -0.08, 0.06])
def test_matrix_sums_to_one(lam, mu, rho):
    matrix = score_matrix(lam, mu, rho, max_goals=10)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-9)
    assert (matrix >= 0).all()


@pytest.mark.parametrize("lam,mu", LAMBDAS)
def test_over25_matches_independent_poisson_without_tau(lam, mu):
    """With rho = 0 the two goal counts are independent, so the closed-form
    Poisson answer is exact -- up to the mass truncated beyond max_goals."""
    probs = match_probabilities(lam, mu, rho=0.0, max_goals=15)
    assert probs.p_over25 == pytest.approx(independent_poisson_over25(lam, mu), abs=1e-6)


@pytest.mark.parametrize("lam,mu", LAMBDAS)
def test_btts_matches_independent_poisson_without_tau(lam, mu):
    probs = match_probabilities(lam, mu, rho=0.0, max_goals=15)
    assert probs.p_btts == pytest.approx(independent_poisson_btts(lam, mu), abs=1e-6)


@pytest.mark.parametrize("lam,mu", LAMBDAS)
def test_one_x_two_sums_to_one(lam, mu):
    probs = match_probabilities(lam, mu, rho=-0.08)
    assert probs.p_home + probs.p_draw + probs.p_away == pytest.approx(1.0, abs=1e-9)


def test_tau_only_touches_the_low_score_corner():
    tau = tau_matrix(1.5, 1.2, -0.1, 8)
    corner = tau[:2, :2]
    assert not np.allclose(corner, 1.0)
    rest = tau.copy()
    rest[:2, :2] = 1.0
    assert np.allclose(rest, 1.0)


def test_tau_is_identity_at_rho_zero():
    assert np.allclose(tau_matrix(1.5, 1.2, 0.0, 6), 1.0)


def test_negative_rho_shifts_mass_into_the_draw_cells():
    """The correction exists because real football has more 0-0 and 1-1 than
    independent Poissons predict; a negative rho must do exactly that."""
    plain = score_matrix(1.4, 1.1, 0.0, 10)
    corrected = score_matrix(1.4, 1.1, -0.10, 10)
    assert corrected[0, 0] > plain[0, 0]
    assert corrected[1, 1] > plain[1, 1]
    assert corrected[1, 0] < plain[1, 0]
    assert corrected[0, 1] < plain[0, 1]


def test_expected_goals_matches_lambdas_without_tau():
    probs = match_probabilities(1.6, 1.1, rho=0.0, max_goals=15)
    assert probs.expected_goals == pytest.approx(1.6 + 1.1, abs=1e-4)


def test_p_over_line_is_monotone():
    matrix = score_matrix(1.5, 1.2, -0.05, 12)
    values = [p_over_line(matrix, line) for line in (0.5, 1.5, 2.5, 3.5, 4.5)]
    assert values == sorted(values, reverse=True)
    assert p_over_line(matrix, 2.5) == pytest.approx(
        probabilities_from_matrix(matrix, 1.5, 1.2).p_over25
    )


def test_larger_matrix_barely_moves_the_answer():
    """max_goals = 10 must not be truncating anything that matters.

    At the top of the plausible range for a Championship fixture the answer
    moves by ~3e-6 between a 10-goal and a 20-goal grid: three ten-thousandths
    of a percentage point, far below the precision of the stored column.
    """
    small = match_probabilities(2.0, 1.8, -0.05, max_goals=10)
    large = match_probabilities(2.0, 1.8, -0.05, max_goals=20)
    assert small.p_over25 == pytest.approx(large.p_over25, abs=1e-5)
    assert small.p_btts == pytest.approx(large.p_btts, abs=1e-5)


def test_rejects_non_positive_lambda():
    with pytest.raises(ValueError):
        score_matrix(0.0, 1.0)


def test_clip_rho_keeps_cells_positive():
    lam, mu = 3.5, 3.0
    rho = clip_rho(0.9, lam, mu)
    assert (tau_matrix(lam, mu, rho, 4) > 0).all()
