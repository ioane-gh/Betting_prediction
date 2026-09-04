"""Synthetic Championship seasons with known ground truth.

This is a **test harness, not a data source**. Every number it produces is
invented. It exists so the machinery around the model -- the fit, the
walk-forward backtest, the metrics, the calibration -- can be exercised and
proved correct without a network connection, and so the tests can assert
against parameters that are actually known rather than estimated.

The generator draws true attack and defence ratings per team, simulates every
match from the same Dixon-Coles distribution the model assumes, and then
prices a synthetic closing line: the true probability, blurred slightly (a real
market is not omniscient) and loaded with a 5% overround. That gives the
backtest a benchmark with the same shape as the real one -- a baseline the
model should approach and should not comfortably beat.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .model.score_matrix import score_matrix

DEFAULT_TEAMS = 24
DEFAULT_GAMMA = 0.25
DEFAULT_RHO = -0.08
DEFAULT_OVERROUND = 1.05


@dataclass
class SyntheticLeague:
    """The generated matches plus the parameters that produced them."""

    matches: pd.DataFrame
    teams: list[str]
    true_attack: np.ndarray
    true_defence: np.ndarray
    home_advantage: float
    rho: float

    def truth_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": self.teams,
            "true_attack": self.true_attack,
            "true_defence": self.true_defence,
        }).sort_values("true_attack", ascending=False).reset_index(drop=True)


def _season_fixture_dates(start: dt.date, rounds: int) -> list[dt.date]:
    """One matchday a week from early August."""
    return [start + dt.timedelta(days=7 * r) for r in range(rounds)]


def _round_robin(teams: Sequence[int]) -> list[list[tuple[int, int]]]:
    """Circle-method schedule: each team meets every other once."""
    squad = list(teams)
    if len(squad) % 2:
        squad.append(-1)  # bye
    half = len(squad) // 2
    rounds = []
    for _ in range(len(squad) - 1):
        pairs = [(squad[i], squad[len(squad) - 1 - i]) for i in range(half)]
        rounds.append([(h, a) for h, a in pairs if h != -1 and a != -1])
        squad = [squad[0]] + [squad[-1]] + squad[1:-1]
    return rounds


def _sample_score(lam: float, mu: float, rho: float, rng: np.random.Generator,
                  max_goals: int = 10) -> tuple[int, int]:
    matrix = score_matrix(lam, mu, rho, max_goals)
    flat = matrix.ravel()
    pick = rng.choice(flat.size, p=flat / flat.sum())
    return int(pick // matrix.shape[1]), int(pick % matrix.shape[1])


def generate(
    n_seasons: int = 4,
    n_teams: int = DEFAULT_TEAMS,
    first_season_start: dt.date = dt.date(2021, 8, 7),
    *,
    home_advantage: float = DEFAULT_GAMMA,
    rho: float = DEFAULT_RHO,
    attack_sd: float = 0.22,
    defence_sd: float = 0.18,
    drift_sd: float = 0.06,
    base_rate: float = 1.07,
    overround: float = DEFAULT_OVERROUND,
    market_noise: float = 0.02,
    seed: int = 20260904,
) -> SyntheticLeague:
    """Simulate ``n_seasons`` of a ``n_teams``-club league.

    ``base_rate`` sets the scoring level: it is tuned so total goals land near
    the Championship.s real 2.5-2.6 per match and Over 2.5 near 0.50,
    which is what makes the synthetic calibration curves comparable in shape to
    the real ones.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]

    attack = rng.normal(0.0, attack_sd, n_teams)
    attack -= attack.mean()
    defence = rng.normal(0.0, defence_sd, n_teams)
    defence -= defence.mean()
    # base_rate is folded into the intercept so lambdas land at a realistic level.
    intercept = float(np.log(base_rate))

    rows: list[dict] = []
    for season in range(n_seasons):
        # Squads change between seasons; ratings drift rather than reset.
        if season > 0:
            attack = attack + rng.normal(0.0, drift_sd, n_teams)
            attack -= attack.mean()
            defence = defence + rng.normal(0.0, drift_sd, n_teams)
            defence -= defence.mean()

        season_start = dt.date(first_season_start.year + season,
                               first_season_start.month, first_season_start.day)
        legs = _round_robin(range(n_teams))
        # Double round robin: reverse fixtures in the second half of the season.
        schedule = legs + [[(a, h) for h, a in rnd] for rnd in legs]
        dates = _season_fixture_dates(season_start, len(schedule))

        for day, fixtures in zip(dates, schedule):
            for home, away in fixtures:
                lam = float(np.exp(intercept + attack[home] + defence[away] + home_advantage))
                mu = float(np.exp(intercept + attack[away] + defence[home]))
                hg, ag = _sample_score(lam, mu, rho, rng)
                rows.append({
                    "match_date": day,
                    "home_team": teams[home],
                    "away_team": teams[away],
                    "home_goals": hg,
                    "away_goals": ag,
                    "true_lambda_home": lam,
                    "true_lambda_away": mu,
                })

    matches = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    matches = _price_market(matches, rho, rng, overround, market_noise)

    # Report the ratings on the same scale the model reports: alpha centred at
    # zero, with the scoring level absorbed into the intercept.
    return SyntheticLeague(
        matches=matches,
        teams=teams,
        true_attack=attack,
        true_defence=defence,
        home_advantage=home_advantage,
        rho=rho,
    )


def _price_market(matches: pd.DataFrame, rho: float, rng: np.random.Generator,
                  overround: float, noise: float) -> pd.DataFrame:
    """Attach a synthetic closing line to every match.

    The market sees the true probability with a small logit-space error and
    charges an overround on top -- so beating it should be hard, exactly as in
    the real data.
    """
    from .model.score_matrix import probabilities_from_matrix

    p_over, p_btts = [], []
    for lam, mu in zip(matches["true_lambda_home"], matches["true_lambda_away"]):
        matrix = score_matrix(float(lam), float(mu), rho, 10)
        probs = probabilities_from_matrix(matrix, float(lam), float(mu))
        p_over.append(probs.p_over25)
        p_btts.append(probs.p_btts)

    def _blur(values: list[float]) -> np.ndarray:
        arr = np.clip(np.asarray(values), 1e-6, 1 - 1e-6)
        logit = np.log(arr / (1 - arr)) + rng.normal(0.0, noise, arr.size)
        return 1.0 / (1.0 + np.exp(-logit))

    market_over = _blur(p_over)
    market_btts = _blur(p_btts)

    matches = matches.copy()
    matches["true_p_over25"] = p_over
    matches["true_p_btts"] = p_btts
    # Split the overround evenly across the two sides of the book.
    matches["odds_over25"] = np.round(1.0 / (market_over * overround), 2)
    matches["odds_under25"] = np.round(1.0 / ((1 - market_over) * overround), 2)
    matches["odds_btts_yes"] = np.round(1.0 / (market_btts * overround), 2)
    matches["odds_btts_no"] = np.round(1.0 / ((1 - market_btts) * overround), 2)
    return matches
