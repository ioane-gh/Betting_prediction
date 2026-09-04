"""Scoring rules, the de-vigged market benchmark, and calibration tables.

The benchmark is the point of this module. A model that predicts Over 2.5 at
0.50 for every fixture will look plausible and score a log loss near 0.69; the
closing line is the honest comparison, because it already contains team news,
weather and money. Matching it means the model works. Beating it by ten points
means there is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-15


def log_loss(y: Sequence[int], p: Sequence[float]) -> float:
    """Mean binary cross-entropy. Lower is better; 0.693 is a coin flip."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    if not mask.any():
        return float("nan")
    y_arr, p_arr = y_arr[mask], p_arr[mask]
    return float(-np.mean(y_arr * np.log(p_arr) + (1 - y_arr) * np.log(1 - p_arr)))


def brier_score(y: Sequence[int], p: Sequence[float]) -> float:
    """Mean squared error of the probability. Lower is better."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    if not mask.any():
        return float("nan")
    return float(np.mean((p_arr[mask] - y_arr[mask]) ** 2))


def devig_proportional(odds_yes: Sequence[float] | float,
                       odds_no: Sequence[float] | float) -> np.ndarray:
    """Two-way decimal odds -> implied probability with the overround removed.

    Proportional (normalised) de-vigging: divide each implied probability by
    their sum. It is the standard choice and, unlike shin or power methods,
    needs nothing but the two prices.
    """
    yes = np.asarray(odds_yes, dtype=float)
    no = np.asarray(odds_no, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        imp_yes = np.where(yes > 1.0, 1.0 / yes, np.nan)
        imp_no = np.where(no > 1.0, 1.0 / no, np.nan)
        total = imp_yes + imp_no
        return np.where(total > 0, imp_yes / total, np.nan)


def overround(odds_yes: Sequence[float] | float,
              odds_no: Sequence[float] | float) -> np.ndarray:
    """The bookmaker's margin: 1.05 means a 5% book."""
    yes = np.asarray(odds_yes, dtype=float)
    no = np.asarray(odds_no, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((yes > 1.0) & (no > 1.0), 1.0 / yes + 1.0 / no, np.nan)


@dataclass
class MarketScore:
    """How the model and the closing line scored on the same rows."""

    market: str
    n: int
    base_rate: float
    model_log_loss: float
    model_brier: float
    market_log_loss: float = float("nan")
    market_brier: float = float("nan")
    n_market: int = 0

    @property
    def log_loss_gap(self) -> float:
        """Model minus market. Negative means the model scored better."""
        return self.model_log_loss - self.market_log_loss

    @property
    def matches_market(self) -> bool:
        """Within 0.01 of the closing line, the Phase 7 acceptance bar."""
        if not np.isfinite(self.market_log_loss):
            return False
        return abs(self.log_loss_gap) <= 0.01

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["log_loss_gap"] = self.log_loss_gap
        payload["matches_market"] = self.matches_market
        return payload


def score_market(
    y: Sequence[int],
    p_model: Sequence[float],
    p_market: Sequence[float] | None = None,
    *,
    market: str = "",
) -> MarketScore:
    """Score the model, and the market on the subset where it has a price."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p_model, dtype=float)
    valid = np.isfinite(y_arr) & np.isfinite(p_arr)

    score = MarketScore(
        market=market,
        n=int(valid.sum()),
        base_rate=float(y_arr[valid].mean()) if valid.any() else float("nan"),
        model_log_loss=log_loss(y_arr[valid], p_arr[valid]),
        model_brier=brier_score(y_arr[valid], p_arr[valid]),
    )

    if p_market is not None:
        m_arr = np.asarray(p_market, dtype=float)
        # Compare like with like: only rows the market actually priced.
        both = valid & np.isfinite(m_arr)
        if both.any():
            score.n_market = int(both.sum())
            score.market_log_loss = log_loss(y_arr[both], m_arr[both])
            score.market_brier = brier_score(y_arr[both], m_arr[both])
            score.model_log_loss = log_loss(y_arr[both], p_arr[both])
            score.model_brier = brier_score(y_arr[both], p_arr[both])
            score.n = int(both.sum())
            score.base_rate = float(y_arr[both].mean())
    return score


def calibration_table(y: Sequence[int], p: Sequence[float], n_buckets: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency, in equal-width buckets.

    Equal-width (not equal-count) buckets on purpose: the question is whether
    "we said 60%" happens 60% of the time, and the count column shows how much
    evidence sits behind each answer.
    """
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    y_arr, p_arr = y_arr[mask], p_arr[mask]

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    idx = np.clip(np.digitize(p_arr, edges[1:-1], right=False), 0, n_buckets - 1)

    rows = []
    for bucket in range(n_buckets):
        sel = idx == bucket
        count = int(sel.sum())
        rows.append({
            "bucket": f"{edges[bucket]:.1f}-{edges[bucket + 1]:.1f}",
            "n": count,
            "mean_predicted": float(p_arr[sel].mean()) if count else float("nan"),
            "observed": float(y_arr[sel].mean()) if count else float("nan"),
            "diff": (float(p_arr[sel].mean() - y_arr[sel].mean()) if count else float("nan")),
        })
    return pd.DataFrame(rows)


def max_bucket_error(table: pd.DataFrame, min_count: int = 30) -> float:
    """Worst |predicted - observed| among buckets with enough rows to mean anything."""
    populated = table[table["n"] >= min_count]
    if populated.empty:
        return float("nan")
    return float(populated["diff"].abs().max())


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Count-weighted mean |predicted - observed| across buckets."""
    populated = table.dropna(subset=["diff"])
    if populated.empty or populated["n"].sum() == 0:
        return float("nan")
    return float((populated["diff"].abs() * populated["n"]).sum() / populated["n"].sum())


def summarise(predictions: pd.DataFrame, n_buckets: int = 10) -> dict[str, Any]:
    """Score both markets from a backtest frame and build their calibration tables."""
    out: dict[str, Any] = {"n_rows": int(len(predictions)), "scores": {}, "tables": {}}
    for market, p_col, y_col, m_col in (
        ("btts", "p_btts", "actual_btts", "market_p_btts"),
        ("over25", "p_over25", "actual_over25", "market_p_over25"),
    ):
        if p_col not in predictions or y_col not in predictions:
            continue
        market_probs = predictions[m_col] if m_col in predictions else None
        score = score_market(predictions[y_col], predictions[p_col], market_probs, market=market)
        table = calibration_table(predictions[y_col], predictions[p_col], n_buckets)
        out["scores"][market] = score
        out["tables"][market] = table
        out.setdefault("calibration_error", {})[market] = {
            "max_bucket_error": max_bucket_error(table),
            "expected_calibration_error": expected_calibration_error(table),
        }
    return out


def format_report(summary: dict[str, Any], *, float_fmt: str = ".4f") -> str:
    """A printable backtest report: scores, gaps, and calibration tables."""
    from tabulate import tabulate

    lines: list[str] = []
    rows = []
    for market, score in summary["scores"].items():
        rows.append([
            market, score.n, f"{score.base_rate:{float_fmt}}",
            f"{score.model_log_loss:{float_fmt}}",
            f"{score.market_log_loss:{float_fmt}}" if np.isfinite(score.market_log_loss) else "-",
            f"{score.log_loss_gap:+{float_fmt}}" if np.isfinite(score.market_log_loss) else "-",
            f"{score.model_brier:{float_fmt}}",
            f"{score.market_brier:{float_fmt}}" if np.isfinite(score.market_brier) else "-",
            "yes" if score.matches_market else "no",
        ])
    lines.append(tabulate(
        rows,
        headers=["market", "n", "base rate", "model LL", "market LL", "gap",
                 "model Brier", "market Brier", "within 0.01"],
        tablefmt="simple",
    ))

    for market, table in summary["tables"].items():
        lines.append("")
        lines.append(f"Calibration -- {market}")
        display = table.copy()
        for col in ("mean_predicted", "observed", "diff"):
            display[col] = display[col].map(lambda v: "-" if pd.isna(v) else f"{v:{float_fmt}}")
        lines.append(tabulate(display, headers="keys", tablefmt="simple", showindex=False))
        errors = summary.get("calibration_error", {}).get(market, {})
        if errors:
            lines.append(
                f"  worst populated bucket: {errors['max_bucket_error']:.4f}   "
                f"ECE: {errors['expected_calibration_error']:.4f}"
            )
    return "\n".join(lines)
