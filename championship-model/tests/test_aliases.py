"""Phase 9: every team name in every raw CSV must resolve to a canonical club.

This is the check that stops Sheffield Wednesday quietly becoming two
half-strength teams. It scans whatever season files are present in data/raw
(the real ones, if a backfill has run) plus the checked-in fixture, and fails
on any name the registry cannot map.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from champmodel.ingest.footballdata_uk import check_names, read_season_csv
from champmodel.ingest.teams import (ALIASES, CANONICAL_NAMES, TeamRegistry,
                                     UnknownTeamError, is_known, normalise,
                                     resolve_name, unresolved)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _raw_csvs() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("E1*.csv")) + sorted(RAW_DIR.glob("E1_*.csv"))


@pytest.mark.parametrize("path", _raw_csvs(), ids=lambda p: p.name)
def test_every_name_in_every_raw_csv_resolves(path):
    unmapped = check_names(read_season_csv(path))
    assert not unmapped, (
        f"{path.name} contains {len(unmapped)} unmapped team name(s): {unmapped}. "
        f"Add them to ALIASES in champmodel/ingest/teams.py."
    )


@pytest.mark.parametrize("spelling,expected", [
    ("Sheffield Weds", "Sheffield Wednesday"),
    ("Sheff Weds", "Sheffield Wednesday"),
    ("Sheffield Wednesday FC", "Sheffield Wednesday"),
    ("SHEFFIELD WEDNESDAY", "Sheffield Wednesday"),
    ("  sheffield   wednesday  ", "Sheffield Wednesday"),
    ("Nott'm Forest", "Nottingham Forest"),
    ("Nott'ham Forest", "Nottingham Forest"),
    ("Nottingham Forest FC", "Nottingham Forest"),
    ("QPR", "Queens Park Rangers"),
    ("Queens Park Rangers FC", "Queens Park Rangers"),
    ("West Brom", "West Bromwich Albion"),
    ("West Bromwich Albion FC", "West Bromwich Albion"),
    ("Peterboro", "Peterborough United"),
    ("Bournemouth", "AFC Bournemouth"),
    ("AFC Bournemouth", "AFC Bournemouth"),
    ("Brighton", "Brighton & Hove Albion"),
    ("Brighton and Hove Albion", "Brighton & Hove Albion"),
    ("Wolves", "Wolverhampton Wanderers"),
    ("MK Dons", "Milton Keynes Dons"),
    ("Wrexham AFC", "Wrexham"),
    ("Sheff Utd", "Sheffield United"),
])
def test_known_spellings_resolve(spelling, expected):
    assert resolve_name(spelling) == expected


def test_the_three_sheffield_wednesday_spellings_share_one_team_id(engine):
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        ids = {registry.team_id(name) for name
               in ("Sheffield Wednesday", "Sheff Weds", "Sheffield Weds",
                   "Sheffield Wednesday FC")}
    assert len(ids) == 1


def test_unknown_name_is_a_hard_failure_not_a_warning():
    with pytest.raises(UnknownTeamError) as excinfo:
        resolve_name("Real Madrid", "test")
    assert "Real Madrid" in str(excinfo.value)
    assert not is_known("Real Madrid")


def test_unresolved_lists_only_the_bad_names():
    assert unresolved(["Leeds", "QPR", "Barcelona", "Ajax"]) == ["Ajax", "Barcelona"]


def test_no_two_clubs_share_a_normalised_key():
    """A collision would silently merge two clubs; the registry raises on import,
    so reaching this assertion at all means the table is clean."""
    seen: dict[str, str] = {}
    for canonical, variants in ALIASES.items():
        for name in (canonical, *variants):
            key = normalise(name)
            assert seen.setdefault(key, canonical) == canonical
    assert len(CANONICAL_NAMES) == len(set(CANONICAL_NAMES))


def test_registry_round_trips_through_the_database(engine):
    with engine.begin() as conn:
        registry = TeamRegistry.sync(conn)
        reloaded = TeamRegistry.load(conn)
    assert registry.canonical_to_id == reloaded.canonical_to_id
    assert len(registry.canonical_to_id) == len(CANONICAL_NAMES)


def test_syncing_twice_does_not_duplicate(engine):
    from champmodel.db import dim_team, table_count, team_alias
    with engine.begin() as conn:
        TeamRegistry.sync(conn)
        first = (table_count(conn, dim_team), table_count(conn, team_alias))
    with engine.begin() as conn:
        TeamRegistry.sync(conn)
        second = (table_count(conn, dim_team), table_count(conn, team_alias))
    assert first == second
