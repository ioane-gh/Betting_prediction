"""Canonical team registry and name resolution.

This is where pipelines like this one usually break silently: three sources
spell Sheffield Wednesday three different ways and a mis-resolved name quietly
splits one club's history into two half-strength teams. So resolution here is
**strict** -- an unmapped name raises ``UnknownTeamError`` rather than warning.

Two layers do the work:

* ``normalise()`` folds case, accents, punctuation and the FC/AFC noise words,
  which absorbs most variation on its own;
* ``ALIASES`` handles the irregular short forms no normaliser can guess
  ("Nott'm Forest", "QPR", "Peterboro", "Sheff Weds").

The registry covers every club to appear in the Championship in the last
decade plus the Premier League and League One clubs that cross the boundary,
so a promoted or relegated side does not break the first ingest of a season.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import sqlalchemy as sa

from ..db import dim_team, team_alias, upsert
from ..logging_setup import get_logger

log = get_logger(__name__)

# Noise tokens dropped during normalisation.
_NOISE_TOKENS = {"fc", "afc", "cf", "football", "club", "association", "the", "and"}
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


class UnknownTeamError(KeyError):
    """Raised when a source name does not resolve to a canonical team."""

    def __init__(self, name: str, source: str = "") -> None:
        self.name = name
        self.source = source
        suffix = f" (source={source})" if source else ""
        super().__init__(
            f"unmapped team name {name!r}{suffix}. Add it to ALIASES in "
            f"champmodel/ingest/teams.py -- never guess at ingest time."
        )


def normalise(name: str) -> str:
    """Fold a team name to its comparison key."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " ").replace("'", "").replace(".", "")
    text = _PUNCT_RE.sub(" ", text)
    tokens = [t for t in _WS_RE.split(text) if t and t not in _NOISE_TOKENS]
    return " ".join(tokens)


# canonical name -> extra spellings seen in the wild.
# football-data.co.uk uses the short forms; football-data.org uses the long
# official names (which normalise cleanly); FBref sits in between.
ALIASES: dict[str, tuple[str, ...]] = {
    "Aston Villa": (),
    "Barnsley": (),
    "Birmingham City": ("Birmingham",),
    "Blackburn Rovers": ("Blackburn",),
    "Blackpool": (),
    "Bolton Wanderers": ("Bolton",),
    "AFC Bournemouth": ("Bournemouth",),
    "Brentford": (),
    "Brighton & Hove Albion": ("Brighton", "Brighton and Hove Albion"),
    "Bristol City": (),
    "Bristol Rovers": ("Bristol Rvs",),
    "Burnley": (),
    "Burton Albion": ("Burton",),
    "Cardiff City": ("Cardiff",),
    "Charlton Athletic": ("Charlton",),
    "Coventry City": ("Coventry",),
    "Derby County": ("Derby",),
    "Fulham": (),
    "Huddersfield Town": ("Huddersfield",),
    "Hull City": ("Hull",),
    "Ipswich Town": ("Ipswich",),
    "Leeds United": ("Leeds",),
    "Leicester City": ("Leicester",),
    "Luton Town": ("Luton",),
    "Middlesbrough": ("Middlesboro", "Middlesbro"),
    "Millwall": (),
    "Milton Keynes Dons": ("MK Dons", "Milton Keynes"),
    "Newcastle United": ("Newcastle",),
    "Norwich City": ("Norwich",),
    "Nottingham Forest": ("Nott'm Forest", "Nott'ham Forest", "Notts Forest", "Forest"),
    "Oxford United": ("Oxford",),
    "Peterborough United": ("Peterboro", "Peterborough"),
    "Plymouth Argyle": ("Plymouth",),
    "Portsmouth": (),
    "Preston North End": ("Preston",),
    "Queens Park Rangers": ("QPR",),
    "Reading": (),
    "Rotherham United": ("Rotherham",),
    "Sheffield United": ("Sheff United", "Sheffield Utd", "Sheff Utd"),
    "Sheffield Wednesday": ("Sheffield Weds", "Sheff Weds", "Sheffield Wed", "Sheff Wed"),
    "Southampton": (),
    "Stoke City": ("Stoke",),
    "Sunderland": (),
    "Swansea City": ("Swansea",),
    "Watford": (),
    "West Bromwich Albion": ("West Brom", "West Bromwich"),
    "Wigan Athletic": ("Wigan",),
    "Wolverhampton Wanderers": ("Wolves",),
    "Wrexham": (),
    "Wycombe Wanderers": ("Wycombe",),
}

CANONICAL_NAMES: tuple[str, ...] = tuple(sorted(ALIASES))


def _lookup_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for canonical, variants in ALIASES.items():
        for name in (canonical, *variants):
            key = normalise(name)
            existing = table.get(key)
            if existing and existing != canonical:
                raise ValueError(
                    f"alias collision: {name!r} normalises to {key!r}, "
                    f"claimed by both {existing!r} and {canonical!r}"
                )
            table[key] = canonical
    return table


_LOOKUP = _lookup_table()


def resolve_name(name: str, source: str = "") -> str:
    """Map a source spelling to its canonical name, or raise."""
    if name is None:
        raise UnknownTeamError("<null>", source)
    canonical = _LOOKUP.get(normalise(name))
    if canonical is None:
        raise UnknownTeamError(str(name), source)
    return canonical


def is_known(name: str) -> bool:
    return normalise(name) in _LOOKUP


def unresolved(names: Iterable[str]) -> list[str]:
    """The subset of ``names`` that would raise. Used by the alias test."""
    return sorted({str(n) for n in names if not is_known(str(n))})


class TeamRegistry:
    """Reads ``champ.dim_team`` once and hands out team_ids by any spelling."""

    def __init__(self, name_to_id: dict[str, int]) -> None:
        self._by_canonical = name_to_id

    @classmethod
    def sync(cls, conn: sa.Connection) -> "TeamRegistry":
        """Write the registry into dim_team/team_alias, then load the ids."""
        upsert(
            conn,
            dim_team,
            [{"canonical_name": name} for name in CANONICAL_NAMES],
            ["canonical_name"],
            update_cols=[],
        )
        name_to_id = dict(
            conn.execute(sa.select(dim_team.c.canonical_name, dim_team.c.team_id)).all()
        )
        alias_rows = [
            {"alias": alias, "team_id": name_to_id[canonical], "source": "manual"}
            for canonical, variants in ALIASES.items()
            for alias in (canonical, *variants)
        ]
        upsert(conn, team_alias, alias_rows, ["alias"])
        log.info("team registry synced",
                 extra={"teams": len(name_to_id), "aliases": len(alias_rows)})
        return cls(name_to_id)

    @classmethod
    def load(cls, conn: sa.Connection) -> "TeamRegistry":
        rows = conn.execute(sa.select(dim_team.c.canonical_name, dim_team.c.team_id)).all()
        if not rows:
            return cls.sync(conn)
        return cls(dict(rows))

    def team_id(self, name: str, source: str = "") -> int:
        canonical = resolve_name(name, source)
        try:
            return self._by_canonical[canonical]
        except KeyError as exc:  # registry synced before a club was added
            raise UnknownTeamError(name, source) from exc

    @property
    def canonical_to_id(self) -> dict[str, int]:
        return dict(self._by_canonical)

    @property
    def id_to_canonical(self) -> dict[int, str]:
        return {v: k for k, v in self._by_canonical.items()}
