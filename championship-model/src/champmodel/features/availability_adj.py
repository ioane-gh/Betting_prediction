"""Turn availability overrides into bounded adjustments to attack and defence.

The rule (Phase 5):

    missing_attack_share = SUM minutes_share of unavailable attackers   (weight 1.0)
                         + SUM minutes_share of unavailable others      (weight 0.3)
    alpha_adj = alpha - k * missing_attack_share          k ~ 0.5

and the mirror image on ``beta`` for missing defenders and goalkeepers -- a
thinner defence concedes more, so beta moves *up*.

The cap is not optional. ``minutes_share`` is hand-entered, and the first time
someone types 4.5 instead of 0.45 an uncapped adjustment produces a lambda no
sane market would recognise. The total move is clipped at +/-0.25 in log space,
about a 22% swing in expected goals -- enough to matter, small enough to stay
sane.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..config import ModelParams
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class TeamAvailability:
    """The adjustment for one team on one date, and why."""

    team_id: int
    missing_attack_share: float
    missing_defence_share: float
    attack_delta: float
    defence_delta: float
    players_out: tuple[str, ...] = ()
    capped: bool = False

    @property
    def applied(self) -> bool:
        return bool(self.players_out)

    def describe(self) -> str:
        if not self.applied:
            return ""
        flag = " (capped)" if self.capped else ""
        return (f"{len(self.players_out)} out, "
                f"att {self.attack_delta:+.3f} def {self.defence_delta:+.3f}{flag}")


@dataclass
class AvailabilityIndex:
    """Every team's adjustment for a given date."""

    day: dt.date
    by_team: dict[int, TeamAvailability] = field(default_factory=dict)
    source_rows: int = 0

    def get(self, team_id: int) -> TeamAvailability | None:
        return self.by_team.get(team_id)

    def deltas(self, team_id: int) -> tuple[float, float]:
        """(attack_delta, defence_delta); zeros when the team has no entries."""
        entry = self.by_team.get(team_id)
        return (entry.attack_delta, entry.defence_delta) if entry else (0.0, 0.0)

    @property
    def teams_affected(self) -> int:
        return sum(1 for entry in self.by_team.values() if entry.applied)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_index(
    overrides: Iterable[Mapping[str, Any]],
    params: ModelParams,
    day: dt.date,
) -> AvailabilityIndex:
    """Aggregate override rows into per-team attack and defence deltas."""
    index = AvailabilityIndex(day=day)
    other_weight = float(params.availability_other_weight)
    k = float(params.availability_k)
    cap = abs(float(params.availability_cap))

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in overrides:
        index.source_rows += 1
        grouped.setdefault(int(row["team_id"]), []).append(row)

    for team_id, rows in grouped.items():
        attack_share = 0.0
        defence_share = 0.0
        names: list[str] = []
        for row in rows:
            share = float(row.get("minutes_share") or 0.0)
            # A row outside [0, 1] should have been rejected at load time;
            # clamp here too so a hand-inserted DB row cannot escape the bound.
            share = float(np.clip(share, 0.0, 1.0))
            is_attacker = _as_bool(row.get("is_attacker"))
            is_defender = _as_bool(row.get("is_defender"))
            attack_share += share * (1.0 if is_attacker else other_weight)
            defence_share += share * (1.0 if is_defender else other_weight)
            names.append(str(row.get("player_name", "?")))

        raw_attack = k * attack_share
        raw_defence = k * defence_share
        capped = raw_attack > cap or raw_defence > cap
        index.by_team[team_id] = TeamAvailability(
            team_id=team_id,
            missing_attack_share=attack_share,
            missing_defence_share=defence_share,
            # Losing attackers lowers attack; losing defenders raises the
            # rate at which the team concedes.
            attack_delta=-float(min(raw_attack, cap)),
            defence_delta=float(min(raw_defence, cap)),
            players_out=tuple(names),
            capped=capped,
        )
        if capped:
            log.warning("availability adjustment hit the cap",
                        extra={"team_id": team_id, "attack_share": round(attack_share, 3),
                               "defence_share": round(defence_share, 3), "cap": cap})

    log.info("availability index built",
             extra={"day": str(day), "rows": index.source_rows,
                    "teams": index.teams_affected})
    return index


def adjust_lambdas(
    lambda_home: float,
    lambda_away: float,
    home: TeamAvailability | None,
    away: TeamAvailability | None,
) -> tuple[float, float]:
    """Apply both teams' deltas in log space.

    ``lambda_home = exp(alpha_home + beta_away + gamma)``, so the home lambda
    moves with the home side's attack delta and the away side's defence delta.
    """
    home_attack = home.attack_delta if home else 0.0
    home_defence = home.defence_delta if home else 0.0
    away_attack = away.attack_delta if away else 0.0
    away_defence = away.defence_delta if away else 0.0

    return (
        float(lambda_home * np.exp(home_attack + away_defence)),
        float(lambda_away * np.exp(away_attack + home_defence)),
    )


def missing_input_flags(home: TeamAvailability | None,
                        away: TeamAvailability | None) -> list[str]:
    """Flags for the prediction row when availability was capped."""
    flags: list[str] = []
    if home and home.capped:
        flags.append("home_avail_capped")
    if away and away.capped:
        flags.append("away_avail_capped")
    return flags
