"""Score matrix and the market probabilities read off it.

Two independent Poissons would be enough for a first pass, but they get the
low-scoring cells wrong -- and 0-0, 1-0, 0-1 and 1-1 are exactly the region
BTTS and Over 2.5 live in. The Dixon-Coles ``tau`` correction adjusts those
four cells and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy.stats import poisson

# tau can only push a cell so far before the joint pmf stops being a pmf.
RHO_BOUNDS = (-0.25, 0.25)
_MIN_TAU = 1e-9


def tau_matrix(lam: float, mu: float, rho: float, size: int) -> np.ndarray:
    """The multiplicative correction, 1.0 everywhere except the 2x2 corner."""
    tau = np.ones((size, size), dtype=float)
    if rho == 0.0:
        return tau
    tau[0, 0] = 1.0 - lam * mu * rho
    tau[0, 1] = 1.0 + lam * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    # A rho outside its admissible range for these lambdas can drive a cell
    # negative. Clamp rather than emit a negative probability.
    np.maximum(tau, _MIN_TAU, out=tau)
    return tau


def score_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    """P(home = h, away = a) for h, a in 0..max_goals, renormalised to sum to 1.

    Renormalisation matters twice over: it absorbs the probability mass beyond
    ``max_goals`` and the small distortion tau introduces.
    """
    if lam <= 0 or mu <= 0:
        raise ValueError(f"lambdas must be positive, got lam={lam}, mu={mu}")
    size = int(max_goals) + 1
    goals = np.arange(size)
    home_pmf = poisson.pmf(goals, lam)
    away_pmf = poisson.pmf(goals, mu)
    matrix = np.outer(home_pmf, away_pmf) * tau_matrix(lam, mu, rho, size)
    total = matrix.sum()
    if total <= 0:
        raise ValueError("degenerate score matrix")
    return matrix / total


@dataclass(frozen=True)
class MatchProbabilities:
    """Everything the daily table needs, read off one score matrix."""

    p_btts: float
    p_over25: float
    p_home: float
    p_draw: float
    p_away: float
    lambda_home: float
    lambda_away: float
    expected_goals: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def probabilities_from_matrix(matrix: np.ndarray, lam: float, mu: float) -> MatchProbabilities:
    size = matrix.shape[0]
    home_goals = np.arange(size)[:, None]
    away_goals = np.arange(size)[None, :]

    total_goals = home_goals + away_goals
    p_over25 = float(matrix[total_goals >= 3].sum())
    p_btts = float(matrix[1:, 1:].sum())
    p_home = float(np.tril(matrix, -1).sum())   # home goals > away goals
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    expected = float((matrix * total_goals).sum())

    return MatchProbabilities(
        p_btts=p_btts,
        p_over25=p_over25,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        lambda_home=float(lam),
        lambda_away=float(mu),
        expected_goals=expected,
    )


def match_probabilities(lam: float, mu: float, rho: float = 0.0,
                        max_goals: int = 10) -> MatchProbabilities:
    """Score matrix plus the derived market probabilities, in one call."""
    matrix = score_matrix(lam, mu, rho, max_goals)
    return probabilities_from_matrix(matrix, lam, mu)


def p_over_line(matrix: np.ndarray, line: float) -> float:
    """P(total goals > line) for any half-goal line."""
    size = matrix.shape[0]
    totals = np.arange(size)[:, None] + np.arange(size)[None, :]
    return float(matrix[totals > line].sum())


def independent_poisson_over25(lam: float, mu: float) -> float:
    """P(total > 2.5) with no tau correction, in closed form.

    The sum of two independent Poissons is Poisson(lam + mu), so this is exact
    and independent of the matrix code -- which is what makes it useful as the
    reference in tests/test_score_matrix.py.
    """
    total = lam + mu
    return float(1.0 - poisson.cdf(2, total))


def independent_poisson_btts(lam: float, mu: float) -> float:
    """P(both score) under independence, in closed form."""
    return float((1.0 - np.exp(-lam)) * (1.0 - np.exp(-mu)))


def clip_rho(rho: float, lam: float, mu: float) -> float:
    """Keep rho inside the range where every corrected cell stays positive."""
    lo, hi = RHO_BOUNDS
    if lam > 0 and mu > 0:
        lo = max(lo, -1.0 / max(lam, 1e-9), -1.0 / max(mu, 1e-9))
        hi = min(hi, 1.0 / max(lam * mu, 1e-9), 1.0)
    return float(np.clip(rho, lo, hi))
