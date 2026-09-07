"""Dixon-Coles bivariate Poisson fit.

    lambda_home = exp(alpha_i + beta_j + gamma)
    lambda_away = exp(alpha_j + beta_i)

``alpha`` is attack strength and ``beta`` is defensive concession -- a *lower*
(more negative) beta means a meaner defence -- with ``gamma`` the home
advantage. The parametrisation has one redundant direction (adding c to every
alpha and subtracting it from every beta leaves both lambdas untouched), so
the fit is anchored with mean(alpha) = 0 and the result re-centred exactly.

Three details do the real work:

* **Time decay.** Each historical match is weighted ``exp(-xi * days_ago)``.
  This *is* the form model: recent matches dominate the fit. A separate
  "last 5 games" feature on top would double-count the same information and
  degrade calibration, so there isn't one.
* **The tau correction** on the 0-0, 1-0, 0-1 and 1-1 cells, which is where
  BTTS and Over 2.5 are decided.
* **Shrinkage.** A side with only a handful of weighted matches -- a promoted
  team in August -- is pulled toward the league mean by a ridge penalty whose
  strength falls to zero once the team has enough history.

The negative log-likelihood supplies an analytic gradient; ``L-BFGS-B`` with
a numeric one is roughly fifty times slower, which matters because Phase 7
refits hundreds of times.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import ModelParams
from ..logging_setup import get_logger
from .score_matrix import RHO_BOUNDS, match_probabilities, score_matrix

log = get_logger(__name__)

# Pins the redundant alpha/beta shift. Free -- the direction does not change
# any lambda -- but it keeps the optimiser off a flat ridge.
_ANCHOR = 1.0
# Keeps the Hessian non-singular for a team with no matches at all.
_BASE_RIDGE = 1e-4
_MIN_TAU = 1e-9


class NotEnoughData(ValueError):
    """Too few matches to fit anything meaningful."""


@dataclass
class TrainingData:
    """Match arrays indexed against a fixed team list."""

    teams: list[str]
    home_idx: np.ndarray
    away_idx: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    weights: np.ndarray
    dates: np.ndarray

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_matches(self) -> int:
        return int(self.home_idx.size)

    def weighted_match_counts(self) -> np.ndarray:
        counts = np.bincount(self.home_idx, weights=self.weights, minlength=self.n_teams)
        counts += np.bincount(self.away_idx, weights=self.weights, minlength=self.n_teams)
        return counts


def decay_weights(dates: Sequence[dt.date] | np.ndarray, ref_date: dt.date,
                  half_life_days: float) -> np.ndarray:
    """exp(-ln2 * days_ago / half_life). A non-positive half-life disables decay."""
    days = np.array([(ref_date - d).days for d in dates], dtype=float)
    if half_life_days is None or half_life_days <= 0:
        return np.ones_like(days)
    xi = math.log(2.0) / float(half_life_days)
    return np.exp(-xi * np.maximum(days, 0.0))


def build_training_data(
    matches: pd.DataFrame,
    ref_date: dt.date,
    half_life_days: float,
    *,
    teams: Sequence[str] | None = None,
) -> TrainingData:
    """Turn a results frame into index arrays plus decay weights.

    ``matches`` needs ``match_date``, ``home_team``, ``away_team``,
    ``home_goals``, ``away_goals``.
    """
    required = {"match_date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"training frame missing columns: {sorted(missing)}")

    frame = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date
    if frame.empty:
        raise NotEnoughData("no finished matches in the training window")

    team_list = list(teams) if teams is not None else sorted(
        set(frame["home_team"]) | set(frame["away_team"])
    )
    index = {name: i for i, name in enumerate(team_list)}
    known = frame["home_team"].isin(index) & frame["away_team"].isin(index)
    frame = frame[known]
    if frame.empty:
        raise NotEnoughData("no training matches involve the requested teams")

    dates = frame["match_date"].to_numpy()
    return TrainingData(
        teams=team_list,
        home_idx=frame["home_team"].map(index).to_numpy(dtype=np.int64),
        away_idx=frame["away_team"].map(index).to_numpy(dtype=np.int64),
        home_goals=frame["home_goals"].to_numpy(dtype=np.int64),
        away_goals=frame["away_goals"].to_numpy(dtype=np.int64),
        weights=decay_weights(dates, ref_date, half_life_days),
        dates=dates,
    )


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def _unpack(theta: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    return theta[:n], theta[n:2 * n], float(theta[2 * n]), float(theta[2 * n + 1])


def _tau_terms(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray,
               rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """tau and its partials for every match, evaluated only on the 2x2 corner."""
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_mu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho

    d_lam[m00] = -mu[m00] * rho
    d_lam[m01] = rho
    d_mu[m00] = -lam[m00] * rho
    d_mu[m10] = rho

    d_rho[m00] = -lam[m00] * mu[m00]
    d_rho[m01] = lam[m01]
    d_rho[m10] = mu[m10]
    d_rho[m11] = -1.0

    np.maximum(tau, _MIN_TAU, out=tau)
    return tau, d_lam, d_mu, d_rho


def negative_log_likelihood(
    theta: np.ndarray,
    data: TrainingData,
    ridge: np.ndarray,
    *,
    with_gradient: bool = True,
) -> tuple[float, np.ndarray] | float:
    """Weighted NLL plus the shrinkage penalty, with its analytic gradient."""
    n = data.n_teams
    alpha, beta, gamma, rho = _unpack(theta, n)

    log_lam = alpha[data.home_idx] + beta[data.away_idx] + gamma
    log_mu = alpha[data.away_idx] + beta[data.home_idx]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    x, y, w = data.home_goals, data.away_goals, data.weights
    tau, d_tau_lam, d_tau_mu, d_tau_rho = _tau_terms(x, y, lam, mu, rho)

    loglik = w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu)
    nll = -float(loglik.sum())

    penalty = 0.5 * float(np.sum(ridge * (alpha ** 2 + beta ** 2)))
    anchor = 0.5 * _ANCHOR * float(alpha.sum()) ** 2
    total = nll + penalty + anchor

    if not with_gradient:
        return total

    # d(nll)/d(log lambda) and d(nll)/d(log mu), per match.
    g_log_lam = -w * (d_tau_lam * lam / tau + x - lam)
    g_log_mu = -w * (d_tau_mu * mu / tau + y - mu)

    g_alpha = (np.bincount(data.home_idx, weights=g_log_lam, minlength=n)
               + np.bincount(data.away_idx, weights=g_log_mu, minlength=n))
    g_beta = (np.bincount(data.away_idx, weights=g_log_lam, minlength=n)
              + np.bincount(data.home_idx, weights=g_log_mu, minlength=n))
    g_gamma = float(g_log_lam.sum())
    g_rho = -float((w * d_tau_rho / tau).sum())

    g_alpha += ridge * alpha + _ANCHOR * float(alpha.sum())
    g_beta += ridge * beta

    grad = np.concatenate([g_alpha, g_beta, [g_gamma], [g_rho]])
    return total, grad


def shrinkage_ridge(data: TrainingData, params: ModelParams) -> np.ndarray:
    """Per-team ridge weight: full strength at zero matches, zero once settled.

    A promoted side with five matches played should not be handed a league-best
    attack rating on the strength of one 4-0 win.
    """
    counts = data.weighted_match_counts()
    deficit = np.maximum(0.0, float(params.min_weighted_matches) - counts)
    return _BASE_RIDGE + float(params.shrinkage_prior) * deficit


@dataclass
class DixonColesFit:
    """A fitted model, plus enough metadata to reproduce and audit it."""

    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_advantage: float
    rho: float
    params: ModelParams
    train_rows: int
    train_end_date: dt.date
    weighted_counts: np.ndarray
    nll: float
    converged: bool
    message: str = ""
    fitted_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    # -- lookups -----------------------------------------------------------
    @property
    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.teams)}

    def has_team(self, name: str) -> bool:
        return name in self.index

    def shrunk_teams(self) -> list[str]:
        """Teams still short of ``min_weighted_matches`` at fit time."""
        floor = float(self.params.min_weighted_matches)
        return [t for t, c in zip(self.teams, self.weighted_counts) if c < floor]

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for a fixture. Unknown teams fall back to league mean."""
        idx = self.index
        i, j = idx.get(home), idx.get(away)
        a_h = self.attack[i] if i is not None else 0.0
        d_h = self.defence[i] if i is not None else 0.0
        a_a = self.attack[j] if j is not None else 0.0
        d_a = self.defence[j] if j is not None else 0.0
        return (
            float(np.exp(a_h + d_a + self.home_advantage)),
            float(np.exp(a_a + d_h)),
        )

    def predict(self, home: str, away: str, *, max_goals: int | None = None):
        lam, mu = self.lambdas(home, away)
        return match_probabilities(lam, mu, self.rho,
                                   max_goals or self.params.max_goals)

    def score_matrix(self, home: str, away: str, *, max_goals: int | None = None) -> np.ndarray:
        lam, mu = self.lambdas(home, away)
        return score_matrix(lam, mu, self.rho, max_goals or self.params.max_goals)

    def team_table(self) -> pd.DataFrame:
        """Attack / defence ratings, strongest attack first. The Phase 4 sanity check."""
        return pd.DataFrame({
            "team": self.teams,
            "attack": self.attack,
            "defence": self.defence,
            # Positive = better than league average overall.
            "net_strength": self.attack - self.defence,
            "weighted_matches": self.weighted_counts,
        }).sort_values("attack", ascending=False).reset_index(drop=True)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "teams": list(self.teams),
            "attack": [float(v) for v in self.attack],
            "defence": [float(v) for v in self.defence],
            "home_advantage": float(self.home_advantage),
            "rho": float(self.rho),
            "params": self.params.to_dict(),
            "train_rows": int(self.train_rows),
            "train_end_date": self.train_end_date.isoformat(),
            "weighted_counts": [float(v) for v in self.weighted_counts],
            "nll": float(self.nll),
            "converged": bool(self.converged),
            "message": self.message,
            "fitted_at": self.fitted_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DixonColesFit":
        return cls(
            teams=list(payload["teams"]),
            attack=np.asarray(payload["attack"], dtype=float),
            defence=np.asarray(payload["defence"], dtype=float),
            home_advantage=float(payload["home_advantage"]),
            rho=float(payload["rho"]),
            params=ModelParams(**payload["params"]),
            train_rows=int(payload["train_rows"]),
            train_end_date=dt.date.fromisoformat(payload["train_end_date"]),
            weighted_counts=np.asarray(payload["weighted_counts"], dtype=float),
            nll=float(payload["nll"]),
            converged=bool(payload["converged"]),
            message=payload.get("message", ""),
            fitted_at=dt.datetime.fromisoformat(payload["fitted_at"]),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "DixonColesFit":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_dixon_coles(
    matches: pd.DataFrame,
    ref_date: dt.date,
    params: ModelParams | None = None,
    *,
    teams: Sequence[str] | None = None,
    initial: DixonColesFit | None = None,
    max_iter: int = 500,
) -> DixonColesFit:
    """Fit on every match in ``matches``, decayed toward ``ref_date``.

    The caller is responsible for the training window; nothing here filters by
    date, which keeps the leakage rule in one place (``backtest.walk_forward``).
    """
    params = params or ModelParams()
    data = build_training_data(matches, ref_date, params.decay_half_life_days, teams=teams)
    if data.n_matches < 10 or data.n_teams < 2:
        raise NotEnoughData(
            f"need at least 10 matches and 2 teams, got {data.n_matches} and {data.n_teams}"
        )

    n = data.n_teams
    ridge = shrinkage_ridge(data, params)

    theta0 = np.zeros(2 * n + 2)
    if initial is not None:
        # Warm start from the previous fit; walk-forward refits differ little.
        prev = initial.index
        for i, team in enumerate(data.teams):
            j = prev.get(team)
            if j is not None:
                theta0[i] = initial.attack[j]
                theta0[n + i] = initial.defence[j]
        theta0[2 * n] = initial.home_advantage
        theta0[2 * n + 1] = initial.rho
    else:
        theta0[2 * n] = 0.25          # a Championship home edge is ~0.25
        theta0[2 * n + 1] = -0.05     # rho is reliably slightly negative

    bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), RHO_BOUNDS]

    result = minimize(
        negative_log_likelihood,
        theta0,
        args=(data, ridge),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7},
    )

    alpha, beta, gamma, rho = _unpack(result.x, n)
    # Re-centre exactly: shifting alpha down by its mean and beta up by the same
    # amount leaves every lambda unchanged, and makes alpha readable as
    # "attack relative to the league".
    shift = float(alpha.mean())
    alpha = alpha - shift
    beta = beta + shift

    fit = DixonColesFit(
        teams=data.teams,
        attack=alpha,
        defence=beta,
        home_advantage=gamma,
        rho=rho,
        params=params,
        train_rows=data.n_matches,
        train_end_date=max(data.dates),
        weighted_counts=data.weighted_match_counts(),
        nll=float(result.fun),
        converged=bool(result.success),
        message=str(result.message),
    )
    log.info(
        "dixon-coles fitted",
        extra={"matches": data.n_matches, "teams": n, "gamma": round(gamma, 4),
               "rho": round(rho, 4), "nll": round(float(result.fun), 2),
               "converged": fit.converged},
    )
    if not fit.converged:
        # "message" collides with a LogRecord attribute the stdlib logger sets
        # itself during formatting -- passing it through `extra` raises
        # KeyError from inside logging.Logger.makeRecord, turning a warning
        # about a bad fit into a crash instead. Every other extra={} call in
        # this codebase was audited for the same trap; this was the only hit.
        log.warning("optimiser did not converge", extra={"optimizer_message": fit.message})
    return fit
