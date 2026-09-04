"""Recent form -- **for display only**.

Read this before adding form to the model: current form already enters through
the time decay in the Dixon-Coles fit. Weighting a match by
``exp(-xi * days_ago)`` with a ~180-day half-life means the last five games
carry several times the weight of games from last autumn, and the strengths
that come out are *already* form-adjusted.

Bolting a separate "points from the last five" feature on top double-counts
exactly that information, and in practice degrades calibration: the model
becomes over-confident about teams on a streak, which is the one situation
where regression to the mean is strongest.

So this module computes form for the human reading the output table, and
nothing here feeds the fit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_WINDOW = 5


@dataclass(frozen=True)
class FormRecord:
    """A team's last N results, most recent first."""

    team: str
    matches: int
    points: int
    goals_for: int
    goals_against: int
    btts_rate: float
    over25_rate: float
    results: str = ""       # e.g. "WWDLW"

    @property
    def has_data(self) -> bool:
        return self.matches > 0

    def display(self) -> str:
        return self.results if self.has_data else "-"


def empty_form(team: str) -> FormRecord:
    return FormRecord(team, 0, 0, 0, 0, float("nan"), float("nan"))


class FormIndex:
    """Per-team result history, ordered by date."""

    def __init__(self, matches: pd.DataFrame) -> None:
        frame = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        self._by_team: dict[str, pd.DataFrame] = {}
        if frame.empty:
            return
        frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date

        rows: list[dict[str, Any]] = []
        for match in frame.itertuples(index=False):
            for team, opponent, scored, conceded in (
                (match.home_team, match.away_team, match.home_goals, match.away_goals),
                (match.away_team, match.home_team, match.away_goals, match.home_goals),
            ):
                rows.append({
                    "team": team,
                    "opponent": opponent,
                    "match_date": match.match_date,
                    "scored": int(scored),
                    "conceded": int(conceded),
                })
        long = pd.DataFrame(rows)
        self._by_team = {
            team: group.sort_values("match_date")
            for team, group in long.groupby("team", sort=False)
        }

    def form(self, team: str, *, before: dt.date | None = None,
             window: int = DEFAULT_WINDOW) -> FormRecord:
        group = self._by_team.get(team)
        if group is None or group.empty:
            return empty_form(team)
        if before is not None:
            group = group[group["match_date"] < before]
        if group.empty:
            return empty_form(team)
        recent = group.tail(window)

        wins = int((recent["scored"] > recent["conceded"]).sum())
        draws = int((recent["scored"] == recent["conceded"]).sum())
        letters = "".join(
            "W" if s > c else ("D" if s == c else "L")
            for s, c in zip(recent["scored"][::-1], recent["conceded"][::-1])
        )
        totals = recent["scored"] + recent["conceded"]
        return FormRecord(
            team=team,
            matches=int(len(recent)),
            points=wins * 3 + draws,
            goals_for=int(recent["scored"].sum()),
            goals_against=int(recent["conceded"].sum()),
            btts_rate=float(((recent["scored"] >= 1) & (recent["conceded"] >= 1)).mean()),
            over25_rate=float((totals >= 3).mean()),
            results=letters,
        )


def league_table(matches: pd.DataFrame, season_matches_only: bool = True) -> pd.DataFrame:
    """Points table from a results frame. Used to sanity-check attack ratings."""
    frame = matches.dropna(subset=["home_goals", "away_goals"])
    if frame.empty:
        return pd.DataFrame(columns=["team", "played", "points", "gf", "ga", "gd"])

    records: dict[str, dict[str, int]] = {}
    for match in frame.itertuples(index=False):
        for team, scored, conceded in (
            (match.home_team, match.home_goals, match.away_goals),
            (match.away_team, match.away_goals, match.home_goals),
        ):
            rec = records.setdefault(team, {"played": 0, "points": 0, "gf": 0, "ga": 0})
            rec["played"] += 1
            rec["gf"] += int(scored)
            rec["ga"] += int(conceded)
            rec["points"] += 3 if scored > conceded else (1 if scored == conceded else 0)

    table = pd.DataFrame([{"team": t, **v} for t, v in records.items()])
    table["gd"] = table["gf"] - table["ga"]
    return table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)
