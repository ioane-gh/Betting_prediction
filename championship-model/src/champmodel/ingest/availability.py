"""Availability overrides: injuries and suspensions from a hand-edited CSV.

There is no free, reliable, machine-readable injury feed, so the authoritative
layer is a CSV the user edits before a run (Phase 5, layer 1). It is loaded
here into ``champ.availability_override``.

Validation is strict on purpose. ``minutes_share`` is the knob most likely to
be fat-fingered, and a bad value flows straight into the lambdas, so a row
outside [0, 1] is rejected rather than clipped.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import sqlalchemy as sa

from ..db import availability_override, upsert
from ..logging_setup import get_logger
from .teams import TeamRegistry, UnknownTeamError

log = get_logger(__name__)

SOURCE = "manual"
VALID_REASONS = {"INJURY", "SUSPENSION", "DOUBT"}
COLUMNS = (
    "team", "player_name", "reason", "minutes_share",
    "is_attacker", "is_defender", "valid_from", "valid_to", "note", "source_url",
)

TEMPLATE_HEADER = ",".join(COLUMNS)
# The example row is commented out on purpose: a template written automatically
# by `champmodel ingest` must not inject a phantom injury into the next run.
# Note the comment marker goes in the first column -- a CSV reader does not
# treat '#' as a comment, parse_rows() does.
TEMPLATE_ROWS = (
    "# One row per unavailable player. Uncomment the example below and edit it.",
    "# minutes_share = the share of this season's team minutes the player has played (0-1).",
    "# is_attacker / is_defender: 1 or 0. A goalkeeper counts as a defender.",
    "# valid_to may be left blank, meaning 'still out'.",
    "#Sheffield Wednesday,Example Striker,INJURY,0.42,1,0,2026-09-01,,hamstring,https://example.invalid/report",
)


class AvailabilityError(ValueError):
    """A row in availability.csv is not usable."""


@dataclass
class AvailabilityReport:
    rows_read: int = 0
    rows_loaded: int = 0
    errors: list[str] = field(default_factory=list)
    missing_file: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors and not self.missing_file


def write_template(path: Path) -> Path:
    """Create a starter availability.csv if there is not one already."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE_HEADER + "\n" + "\n".join(TEMPLATE_ROWS) + "\n", encoding="utf-8")
    log.info("wrote availability template", extra={"path": str(path)})
    return path


def _parse_bool(value: Any, field_name: str, line: int) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"", "0", "false", "no", "n"}:
        return False
    raise AvailabilityError(f"line {line}: {field_name}={value!r} is not a boolean")


def _parse_date(value: Any, field_name: str, line: int, *, required: bool) -> dt.date | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise AvailabilityError(f"line {line}: {field_name} is required")
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        raise AvailabilityError(f"line {line}: {field_name}={text!r} is not YYYY-MM-DD") from exc


def parse_rows(lines: Iterable[dict[str, str]], registry: TeamRegistry,
               report: AvailabilityReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, raw in enumerate(lines, start=2):
        team_raw = (raw.get("team") or "").strip()
        if not team_raw or team_raw.startswith("#"):
            continue
        report.rows_read += 1
        try:
            team_id = registry.team_id(team_raw, SOURCE)
        except UnknownTeamError as exc:
            report.errors.append(f"line {offset}: {exc}")
            continue

        try:
            player = (raw.get("player_name") or "").strip()
            if not player:
                raise AvailabilityError(f"line {offset}: player_name is required")

            reason = (raw.get("reason") or "").strip().upper()
            if reason not in VALID_REASONS:
                raise AvailabilityError(
                    f"line {offset}: reason={reason!r} must be one of {sorted(VALID_REASONS)}"
                )

            share_text = (raw.get("minutes_share") or "").strip()
            try:
                share = float(share_text)
            except ValueError as exc:
                raise AvailabilityError(
                    f"line {offset}: minutes_share={share_text!r} is not a number"
                ) from exc
            if not 0.0 <= share <= 1.0:
                # Clipping here would hide the typo; the cap in the model is a
                # safety net, not a licence to accept nonsense.
                raise AvailabilityError(
                    f"line {offset}: minutes_share={share} is outside [0, 1]"
                )

            valid_from = _parse_date(raw.get("valid_from"), "valid_from", offset, required=True)
            valid_to = _parse_date(raw.get("valid_to"), "valid_to", offset, required=False)
            if valid_to is not None and valid_from is not None and valid_to < valid_from:
                raise AvailabilityError(f"line {offset}: valid_to is before valid_from")

            rows.append({
                "team_id": team_id,
                "player_name": player,
                "reason": reason,
                "minutes_share": round(share, 4),
                "is_attacker": _parse_bool(raw.get("is_attacker"), "is_attacker", offset),
                "is_defender": _parse_bool(raw.get("is_defender"), "is_defender", offset),
                "valid_from": valid_from,
                "valid_to": valid_to,
                "note": (raw.get("note") or "").strip() or None,
                "source_url": (raw.get("source_url") or "").strip() or None,
            })
        except AvailabilityError as exc:
            report.errors.append(str(exc))
    return rows


def load_csv(conn: sa.Connection, registry: TeamRegistry, path: Path,
             *, strict: bool = False) -> AvailabilityReport:
    """Load availability.csv into the override table."""
    report = AvailabilityReport()
    if not path.exists():
        report.missing_file = True
        log.warning("no availability file", extra={"path": str(path)})
        return report

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("team", "player_name", "reason", "minutes_share", "valid_from")
                   if c not in (reader.fieldnames or [])]
        if missing:
            report.errors.append(f"availability.csv is missing column(s): {missing}")
            return report
        rows = parse_rows(reader, registry, report)

    if report.errors and strict:
        raise AvailabilityError("; ".join(report.errors))

    if rows:
        upsert(conn, availability_override, rows,
               ["team_id", "player_name", "valid_from"])
        report.rows_loaded = len(rows)
    log.info("availability loaded",
             extra={"read": report.rows_read, "loaded": report.rows_loaded,
                    "errors": len(report.errors)})
    return report


def active_overrides(conn: sa.Connection, day: dt.date) -> list[dict[str, Any]]:
    """Every override in force on ``day``."""
    stmt = sa.select(availability_override).where(
        sa.and_(
            availability_override.c.valid_from <= day,
            sa.or_(availability_override.c.valid_to.is_(None),
                   availability_override.c.valid_to >= day),
        )
    )
    return [dict(row) for row in conn.execute(stmt).mappings().all()]


def load_derived(conn: sa.Connection, registry: TeamRegistry, derived: Sequence[dict[str, Any]],
                 day: dt.date, valid_days: int = 7) -> int:
    """Write Phase 5 layer-2 (FBref-derived) suspension risk as overrides.

    Manual rows win: a derived row for a player already entered by hand for the
    same window is skipped rather than overwriting the human's judgement.
    """
    if not derived:
        return 0
    existing = {
        (row["team_id"], row["player_name"].lower())
        for row in active_overrides(conn, day)
    }
    rows: list[dict[str, Any]] = []
    for item in derived:
        try:
            team_id = registry.team_id(str(item["team"]), "fbref")
        except UnknownTeamError:
            continue
        player = str(item["player"])
        if (team_id, player.lower()) in existing:
            continue
        rows.append({
            "team_id": team_id,
            "player_name": player,
            "reason": str(item.get("reason", "DOUBT")).upper(),
            "minutes_share": round(float(item.get("minutes_share", 0.0)), 4),
            "is_attacker": bool(item.get("is_attacker", False)),
            "is_defender": bool(item.get("is_defender", False)),
            "valid_from": day,
            "valid_to": day + dt.timedelta(days=valid_days),
            "note": f"derived: {item.get('detail', '')}"[:400],
            "source_url": None,
        })
    if rows:
        upsert(conn, availability_override, rows, ["team_id", "player_name", "valid_from"])
    log.info("derived availability written", extra={"rows": len(rows)})
    return len(rows)
