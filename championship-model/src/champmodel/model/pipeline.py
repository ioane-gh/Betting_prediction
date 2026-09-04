"""One prediction path, shared by the daily run and the backtest.

Keeping this in one place is what makes the backtest meaningful: if the CLI
built its lambdas differently from ``walk_forward``, the measured log loss
would describe a model nobody is actually running.

Order of operations:

1. lambdas from the fitted Dixon-Coles strengths;
2. availability adjustment (bounded, Phase 5);
3. optional head-to-head nudge (off by default, Phase 6);
4. score matrix with the tau correction;
5. Platt calibration, if a calibrator was fitted.

Every input that was wanted and missing is recorded on the row. A prediction
made without today's team news is not the same product as one made with it,
and the output has to say so.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import ModelParams
from ..features.availability_adj import AvailabilityIndex, TeamAvailability, adjust_lambdas
from ..features.h2h import H2HIndex, H2HRecord, empty_record, h2h_adjustment
from ..logging_setup import get_logger
from .calibrate import CalibrationSet
from .dixon_coles import DixonColesFit
from .score_matrix import match_probabilities

log = get_logger(__name__)

# Flags that can appear in prediction.missing_inputs.
FLAG_NO_AVAILABILITY = "no_team_news"
FLAG_STALE_AVAILABILITY = "stale_team_news"
FLAG_NO_MARKET = "no_market_odds"
FLAG_NO_H2H = "no_h2h"
FLAG_NO_XG = "no_xg"
FLAG_UNKNOWN_HOME = "home_team_unfitted"
FLAG_UNKNOWN_AWAY = "away_team_unfitted"
FLAG_SHRUNK_HOME = "home_team_shrunk"
FLAG_SHRUNK_AWAY = "away_team_shrunk"
FLAG_UNCALIBRATED = "uncalibrated"


@dataclass
class FixturePrediction:
    """Everything the output table and champ.prediction need for one fixture."""

    home_team: str
    away_team: str
    match_date: dt.date
    kickoff_utc: dt.datetime | None
    lambda_home: float
    lambda_away: float
    p_btts: float
    p_over25: float
    p_home: float
    p_draw: float
    p_away: float
    p_btts_raw: float
    p_over25_raw: float
    availability_applied: bool
    h2h_applied: bool
    missing_inputs: list[str] = field(default_factory=list)
    h2h: H2HRecord | None = None
    match_id: int | None = None
    market_p_over25: float | None = None
    market_p_btts: float | None = None

    @property
    def edge_over25(self) -> float | None:
        if self.market_p_over25 is None:
            return None
        return self.p_over25 - self.market_p_over25

    @property
    def edge_btts(self) -> float | None:
        if self.market_p_btts is None:
            return None
        return self.p_btts - self.market_p_btts

    def missing_inputs_text(self) -> str | None:
        """Comma-joined, truncated to the column width in champ.prediction."""
        if not self.missing_inputs:
            return None
        return ",".join(self.missing_inputs)[:200]

    def to_row(self, run_id: int) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "match_id": self.match_id,
            "lambda_home": round(float(self.lambda_home), 4),
            "lambda_away": round(float(self.lambda_away), 4),
            "p_btts": round(float(self.p_btts), 5),
            "p_over25": round(float(self.p_over25), 5),
            "p_home": round(float(self.p_home), 5),
            "p_draw": round(float(self.p_draw), 5),
            "p_away": round(float(self.p_away), 5),
            "availability_applied": bool(self.availability_applied),
            "missing_inputs": self.missing_inputs_text(),
        }


def predict_fixture(
    fit: DixonColesFit,
    home: str,
    away: str,
    match_date: dt.date,
    params: ModelParams,
    *,
    kickoff_utc: dt.datetime | None = None,
    availability: AvailabilityIndex | None = None,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
    h2h_index: H2HIndex | None = None,
    calibration: CalibrationSet | None = None,
    market_p_over25: float | None = None,
    market_p_btts: float | None = None,
    has_xg: bool = True,
    match_id: int | None = None,
    extra_flags: Sequence[str] = (),
) -> FixturePrediction:
    """Run one fixture through the whole path, recording what was missing."""
    missing: list[str] = list(extra_flags)

    if not fit.has_team(home):
        missing.append(FLAG_UNKNOWN_HOME)
    if not fit.has_team(away):
        missing.append(FLAG_UNKNOWN_AWAY)
    shrunk = set(fit.shrunk_teams())
    if home in shrunk:
        missing.append(FLAG_SHRUNK_HOME)
    if away in shrunk:
        missing.append(FLAG_SHRUNK_AWAY)

    lam, mu = fit.lambdas(home, away)

    # -- availability ------------------------------------------------------
    home_avail: TeamAvailability | None = None
    away_avail: TeamAvailability | None = None
    if availability is None:
        missing.append(FLAG_NO_AVAILABILITY)
    else:
        if home_team_id is not None:
            home_avail = availability.get(home_team_id)
        if away_team_id is not None:
            away_avail = availability.get(away_team_id)
        lam, mu = adjust_lambdas(lam, mu, home_avail, away_avail)
        if home_avail and home_avail.capped:
            missing.append("home_avail_capped")
        if away_avail and away_avail.capped:
            missing.append("away_avail_capped")
    availability_applied = bool((home_avail and home_avail.applied)
                                or (away_avail and away_avail.applied))

    # -- head to head ------------------------------------------------------
    record = empty_record(home, away)
    h2h_applied = False
    if h2h_index is not None:
        record = h2h_index.record(home, away, before=match_date, window=params.h2h_window)
        if not record.has_data:
            missing.append(FLAG_NO_H2H)
        else:
            # The nudge is relative to what the model already says, so the
            # uncorrected probability has to be computed first.
            baseline = match_probabilities(lam, mu, fit.rho, params.max_goals)
            lam, mu, h2h_applied = h2h_adjustment(lam, mu, record,
                                                  baseline.p_over25, params)
    else:
        missing.append(FLAG_NO_H2H)

    # -- score matrix ------------------------------------------------------
    probs = match_probabilities(lam, mu, fit.rho, params.max_goals)
    p_btts_raw, p_over25_raw = probs.p_btts, probs.p_over25

    # -- calibration -------------------------------------------------------
    p_btts, p_over25 = p_btts_raw, p_over25_raw
    if calibration is None or not calibration.any_fitted:
        missing.append(FLAG_UNCALIBRATED)
    else:
        p_btts = float(calibration.btts.transform(p_btts_raw))
        p_over25 = float(calibration.over25.transform(p_over25_raw))

    if market_p_over25 is None and market_p_btts is None:
        missing.append(FLAG_NO_MARKET)
    if not has_xg:
        missing.append(FLAG_NO_XG)

    return FixturePrediction(
        home_team=home,
        away_team=away,
        match_date=match_date,
        kickoff_utc=kickoff_utc,
        lambda_home=probs.lambda_home,
        lambda_away=probs.lambda_away,
        p_btts=p_btts,
        p_over25=p_over25,
        p_home=probs.p_home,
        p_draw=probs.p_draw,
        p_away=probs.p_away,
        p_btts_raw=p_btts_raw,
        p_over25_raw=p_over25_raw,
        availability_applied=availability_applied,
        h2h_applied=h2h_applied,
        missing_inputs=missing,
        h2h=record,
        match_id=match_id,
        market_p_over25=market_p_over25,
        market_p_btts=market_p_btts,
    )


def make_h2h_adjuster(h2h_index: H2HIndex, params: ModelParams):
    """An ``adjust`` hook for ``walk_forward``, so H2H can be measured.

    Phase 6 says prove it before enabling it -- this is what does the proving.
    """
    def adjust(fit: DixonColesFit, match) -> tuple[float, float]:
        lam, mu = fit.lambdas(match["home_team"], match["away_team"])
        record = h2h_index.record(match["home_team"], match["away_team"],
                                  before=match["match_date"], window=params.h2h_window)
        baseline = match_probabilities(lam, mu, fit.rho, params.max_goals)
        lam, mu, _ = h2h_adjustment(lam, mu, record, baseline.p_over25, params)
        return lam, mu
    return adjust
