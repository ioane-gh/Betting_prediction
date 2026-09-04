"""Platt scaling.

A Dixon-Coles fit can be systematically over- or under-confident even when it
ranks fixtures correctly. Platt scaling fixes that with two parameters:

    p_calibrated = sigmoid(a * logit(p_raw) + b)

``a < 1`` pulls predictions toward the base rate, ``b`` shifts the whole curve.
Fit on a **held-out** period, never on the training fit, and store the
coefficients alongside the model run so a stored prediction can be reproduced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-9
MIN_CALIBRATION_ROWS = 200


def logit(p: np.ndarray | Sequence[float] | float) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


@dataclass(frozen=True)
class PlattCalibrator:
    """Two coefficients and the evidence behind them. ``(1, 0)`` is identity."""

    a: float = 1.0
    b: float = 0.0
    n_rows: int = 0
    fitted: bool = False
    market: str = ""

    def transform(self, p: np.ndarray | Sequence[float] | float) -> np.ndarray:
        if not self.fitted:
            return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        return sigmoid(self.a * logit(p) + self.b)

    def __call__(self, p):
        return self.transform(p)

    @property
    def is_identity(self) -> bool:
        return not self.fitted or (abs(self.a - 1.0) < 1e-6 and abs(self.b) < 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {"a": float(self.a), "b": float(self.b), "n_rows": int(self.n_rows),
                "fitted": bool(self.fitted), "market": self.market}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PlattCalibrator":
        if not payload:
            return cls()
        return cls(
            a=float(payload.get("a", 1.0)),
            b=float(payload.get("b", 0.0)),
            n_rows=int(payload.get("n_rows", 0)),
            fitted=bool(payload.get("fitted", False)),
            market=str(payload.get("market", "")),
        )

    def describe(self) -> str:
        if not self.fitted:
            return "identity (uncalibrated)"
        return f"a={self.a:.4f} b={self.b:+.4f} on {self.n_rows} rows"


def fit_platt(p_raw: Sequence[float], y: Sequence[int], *, market: str = "",
              min_rows: int = MIN_CALIBRATION_ROWS) -> PlattCalibrator:
    """Fit the two coefficients by maximum likelihood.

    Returns the identity calibrator when there is too little held-out data --
    a calibration fitted on 40 matches is noise dressed as a correction.
    """
    x = logit(p_raw)
    target = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(target)
    x, target = x[mask], target[mask]

    if x.size < min_rows or target.min() == target.max():
        return PlattCalibrator(n_rows=int(x.size), fitted=False, market=market)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = theta
        z = a * x + b
        # log-sum-exp form: stable for large |z|.
        nll = float(np.sum(np.logaddexp(0.0, z) - target * z))
        residual = sigmoid(z) - target
        return nll, np.array([float(np.dot(residual, x)), float(residual.sum())])

    result = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    a, b = result.x
    return PlattCalibrator(a=float(a), b=float(b), n_rows=int(x.size),
                           fitted=bool(result.success), market=market)


@dataclass(frozen=True)
class CalibrationSet:
    """One calibrator per market, stored together on the model run."""

    btts: PlattCalibrator = PlattCalibrator()
    over25: PlattCalibrator = PlattCalibrator()

    def to_dict(self) -> dict[str, Any]:
        return {"btts": self.btts.to_dict(), "over25": self.over25.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CalibrationSet":
        payload = payload or {}
        return cls(
            btts=PlattCalibrator.from_dict(payload.get("btts")),
            over25=PlattCalibrator.from_dict(payload.get("over25")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @property
    def any_fitted(self) -> bool:
        return self.btts.fitted or self.over25.fitted


def fit_calibration(predictions, *, min_rows: int = MIN_CALIBRATION_ROWS) -> CalibrationSet:
    """Fit both markets from a backtest frame (held-out by construction)."""
    return CalibrationSet(
        btts=fit_platt(predictions["p_btts"], predictions["actual_btts"],
                       market="btts", min_rows=min_rows),
        over25=fit_platt(predictions["p_over25"], predictions["actual_over25"],
                         market="over25", min_rows=min_rows),
    )
