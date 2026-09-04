"""Head-to-head.

Worth being clear about what this is for. Two Championship sides meet twice a
season, so even a ten-year window yields about twenty matches, most of them
played by squads that have since turned over completely. Beyond the team
strengths the model already fits, the extra predictive signal is close to zero
and the published literature is consistent on that point.

So the default posture is:

* **Display** the record -- last ten meetings, goals, BTTS and Over 2.5 hit
  rates. It is useful context for a human reading the table.
* **Optionally** apply a heavily shrunk nudge, off by default, capped tightly,
  and only worth switching on if a backtest shows it improves out-of-sample
  log loss. It probably will not.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import ModelParams
from ..logging_setup import get_logger

log = get_logger(__name__)

# However tempting the config file looks, this is as far as H2H gets to move a
# lambda. Above roughly 0.1 the nudge starts overriding the fitted strengths.
MAX_H2H_WEIGHT = 0.10


@dataclass(frozen=True)
class H2HRecord:
    """The last N meetings between two clubs, in either direction."""

    home_team: str
    away_team: str
    meetings: int
    btts_rate: float
    over25_rate: float
    avg_total_goals: float
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    last_meeting: dt.date | None = None

    @property
    def has_data(self) -> bool:
        return self.meetings > 0

    def display_btts(self) -> str:
        return f"{self.btts_rate:.0%} ({self.meetings})" if self.has_data else "-"

    def display_over25(self) -> str:
        return f"{self.over25_rate:.0%} ({self.meetings})" if self.has_data else "-"


def empty_record(home: str, away: str) -> H2HRecord:
    return H2HRecord(home, away, 0, float("nan"), float("nan"), float("nan"))


class H2HIndex:
    """Pre-grouped meeting history, so a day's fixtures need one pass, not N."""

    def __init__(self, matches: pd.DataFrame) -> None:
        frame = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        if frame.empty:
            self._by_pair: dict[frozenset, pd.DataFrame] = {}
            return
        frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date
        frame["_total"] = frame["home_goals"] + frame["away_goals"]
        frame["_btts"] = ((frame["home_goals"] >= 1) & (frame["away_goals"] >= 1)).astype(int)
        frame["_over25"] = (frame["_total"] >= 3).astype(int)
        frame["_pair"] = [frozenset((h, a)) for h, a
                          in zip(frame["home_team"], frame["away_team"])]
        self._by_pair = {
            pair: group.sort_values("match_date")
            for pair, group in frame.groupby("_pair", sort=False)
        }

    def record(self, home: str, away: str, *, before: dt.date | None = None,
               window: int = 10) -> H2HRecord:
        """The last ``window`` meetings strictly before ``before``."""
        group = self._by_pair.get(frozenset((home, away)))
        if group is None or group.empty:
            return empty_record(home, away)
        if before is not None:
            group = group[group["match_date"] < before]
        if group.empty:
            return empty_record(home, away)
        recent = group.tail(window)

        home_wins = int(((recent["home_team"] == home) &
                         (recent["home_goals"] > recent["away_goals"])).sum()
                        + ((recent["away_team"] == home) &
                           (recent["away_goals"] > recent["home_goals"])).sum())
        away_wins = int(((recent["home_team"] == away) &
                         (recent["home_goals"] > recent["away_goals"])).sum()
                        + ((recent["away_team"] == away) &
                           (recent["away_goals"] > recent["home_goals"])).sum())

        return H2HRecord(
            home_team=home,
            away_team=away,
            meetings=int(len(recent)),
            btts_rate=float(recent["_btts"].mean()),
            over25_rate=float(recent["_over25"].mean()),
            avg_total_goals=float(recent["_total"].mean()),
            home_wins=home_wins,
            draws=int((recent["home_goals"] == recent["away_goals"]).sum()),
            away_wins=away_wins,
            last_meeting=recent["match_date"].iloc[-1],
        )


def h2h_adjustment(
    lambda_home: float,
    lambda_away: float,
    record: H2HRecord,
    model_over25: float,
    params: ModelParams,
) -> tuple[float, float, bool]:
    """Optionally nudge the lambdas toward the pair's Over 2.5 history.

        lambda_adj = lambda * (1 + w * (h2h_rate - model_rate))

    Returns ``(lambda_home, lambda_away, applied)``. Disabled by default; the
    weight is clamped to ``MAX_H2H_WEIGHT`` however the config is set, and a
    record with fewer than four meetings is ignored as pure noise.
    """
    if not params.enable_h2h_adjustment:
        return lambda_home, lambda_away, False
    if not record.has_data or record.meetings < 4 or not np.isfinite(record.over25_rate):
        return lambda_home, lambda_away, False

    weight = min(abs(float(params.h2h_weight)), MAX_H2H_WEIGHT)
    factor = 1.0 + weight * (record.over25_rate - model_over25)
    # Never let the nudge invert or explode a lambda.
    factor = float(np.clip(factor, 0.9, 1.1))
    return lambda_home * factor, lambda_away * factor, True


def summarise_pairs(index: H2HIndex, fixtures: list[tuple[str, str]],
                    before: dt.date | None = None, window: int = 10) -> pd.DataFrame:
    """One display row per fixture."""
    rows: list[dict[str, Any]] = []
    for home, away in fixtures:
        record = index.record(home, away, before=before, window=window)
        rows.append({
            "home_team": home,
            "away_team": away,
            "h2h_meetings": record.meetings,
            "h2h_btts": record.btts_rate,
            "h2h_over25": record.over25_rate,
            "h2h_avg_goals": record.avg_total_goals,
            "h2h_last": record.last_meeting,
        })
    return pd.DataFrame(rows)
