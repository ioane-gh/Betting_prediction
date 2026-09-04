"""Database access: SQLAlchemy Core (no ORM) over pyodbc / SQL Server.

Two things matter here.

1. **One definition of the schema.** ``sql/001_schema.sql`` is the authoritative
   DDL for SQL Server; the ``MetaData`` below mirrors it and is what the Python
   code binds to. ``tests/test_schema_parity.py`` fails if the two drift apart.

2. **Every load is idempotent.** ``upsert()`` merges on a natural key, so
   re-running a day's ingest can never duplicate rows. SQL Server gets a real
   ``MERGE``; SQLite (used by the test suite and for local development) gets
   ``INSERT ... ON CONFLICT DO UPDATE``, which has the same semantics.

SQLite reaches the ``champ`` schema through ``ATTACH DATABASE``, so table
definitions need no dialect-specific casing.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from .config import Config, DbConfig
from .logging_setup import get_logger

log = get_logger(__name__)

SCHEMA = "champ"
metadata = sa.MetaData(schema=SCHEMA)

# SQLite only auto-increments a column declared exactly ``INTEGER PRIMARY KEY``,
# so BIGINT identity keys need a per-dialect variant.
BIGINT_PK = sa.BigInteger().with_variant(sa.Integer, "sqlite")

# --------------------------------------------------------------------------
# Table definitions -- mirror of sql/001_schema.sql
# --------------------------------------------------------------------------
dim_team = sa.Table(
    "dim_team",
    metadata,
    sa.Column("team_id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("canonical_name", sa.Unicode(100), nullable=False, unique=True),
    sa.Column("fbref_id", sa.Unicode(20)),
    sa.Column("fd_org_id", sa.Integer),
)

team_alias = sa.Table(
    "team_alias",
    metadata,
    sa.Column("alias", sa.Unicode(100), primary_key=True),
    sa.Column("team_id", sa.Integer, sa.ForeignKey(f"{SCHEMA}.dim_team.team_id"), nullable=False),
    sa.Column("source", sa.Unicode(30), nullable=False),
)

dim_season = sa.Table(
    "dim_season",
    metadata,
    sa.Column("season_id", sa.Integer, primary_key=True, autoincrement=False),
    sa.Column("label", sa.Unicode(9), nullable=False),
    sa.Column("start_date", sa.Date),
    sa.Column("end_date", sa.Date),
)

fact_match = sa.Table(
    "fact_match",
    metadata,
    sa.Column("match_id", BIGINT_PK, primary_key=True, autoincrement=True),
    sa.Column("season_id", sa.Integer, sa.ForeignKey(f"{SCHEMA}.dim_season.season_id"), nullable=False),
    sa.Column("match_date", sa.Date, nullable=False),
    sa.Column("kickoff_utc", sa.DateTime),
    sa.Column("home_team_id", sa.Integer, sa.ForeignKey(f"{SCHEMA}.dim_team.team_id"), nullable=False),
    sa.Column("away_team_id", sa.Integer, sa.ForeignKey(f"{SCHEMA}.dim_team.team_id"), nullable=False),
    sa.Column("home_goals", sa.SmallInteger),
    sa.Column("away_goals", sa.SmallInteger),
    sa.Column("home_ht", sa.SmallInteger),
    sa.Column("away_ht", sa.SmallInteger),
    sa.Column("status", sa.Unicode(20), nullable=False, server_default="SCHEDULED"),
    sa.Column("source", sa.Unicode(30), nullable=False),
    sa.UniqueConstraint("match_date", "home_team_id", "away_team_id", name="UQ_fact_match_natural"),
    sa.Index("IX_fact_match_date", "match_date"),
    sa.Index("IX_fact_match_teams", "home_team_id", "away_team_id"),
)

fact_match_stats = sa.Table(
    "fact_match_stats",
    metadata,
    sa.Column("match_id", sa.BigInteger, sa.ForeignKey(f"{SCHEMA}.fact_match.match_id"), primary_key=True),
    *(sa.Column(name, sa.SmallInteger) for name in (
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners", "home_yellow", "away_yellow",
        "home_red", "away_red",
    )),
    sa.Column("home_xg", sa.Numeric(5, 2)),
    sa.Column("away_xg", sa.Numeric(5, 2)),
)

market_odds = sa.Table(
    "market_odds",
    metadata,
    sa.Column("match_id", sa.BigInteger, sa.ForeignKey(f"{SCHEMA}.fact_match.match_id"), primary_key=True),
    sa.Column("book", sa.Unicode(20), primary_key=True),
    *(sa.Column(name, sa.Numeric(6, 2)) for name in (
        "odds_over25", "odds_under25", "odds_btts_yes", "odds_btts_no",
        "odds_home", "odds_draw", "odds_away",
    )),
)

availability_override = sa.Table(
    "availability_override",
    metadata,
    sa.Column("override_id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("team_id", sa.Integer, sa.ForeignKey(f"{SCHEMA}.dim_team.team_id"), nullable=False),
    sa.Column("player_name", sa.Unicode(100), nullable=False),
    sa.Column("reason", sa.Unicode(20), nullable=False),
    sa.Column("minutes_share", sa.Numeric(5, 4), nullable=False),
    sa.Column("is_attacker", sa.Boolean, nullable=False, server_default=sa.text("0")),
    sa.Column("is_defender", sa.Boolean, nullable=False, server_default=sa.text("0")),
    sa.Column("valid_from", sa.Date, nullable=False),
    sa.Column("valid_to", sa.Date),
    sa.Column("note", sa.Unicode(400)),
    sa.Column("source_url", sa.Unicode(400)),
    sa.UniqueConstraint("team_id", "player_name", "valid_from", name="UQ_availability_natural"),
)

model_run = sa.Table(
    "model_run",
    metadata,
    sa.Column("run_id", BIGINT_PK, primary_key=True, autoincrement=True),
    sa.Column("run_utc", sa.DateTime, nullable=False, server_default=sa.func.now()),
    sa.Column("model_version", sa.Unicode(20), nullable=False),
    sa.Column("params", sa.UnicodeText),
    sa.Column("train_rows", sa.Integer),
    sa.Column("train_end_date", sa.Date),
)

prediction = sa.Table(
    "prediction",
    metadata,
    sa.Column("prediction_id", BIGINT_PK, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.BigInteger, sa.ForeignKey(f"{SCHEMA}.model_run.run_id"), nullable=False),
    sa.Column("match_id", sa.BigInteger, sa.ForeignKey(f"{SCHEMA}.fact_match.match_id"), nullable=False),
    sa.Column("lambda_home", sa.Numeric(6, 4), nullable=False),
    sa.Column("lambda_away", sa.Numeric(6, 4), nullable=False),
    sa.Column("p_btts", sa.Numeric(6, 5), nullable=False),
    sa.Column("p_over25", sa.Numeric(6, 5), nullable=False),
    sa.Column("p_home", sa.Numeric(6, 5)),
    sa.Column("p_draw", sa.Numeric(6, 5)),
    sa.Column("p_away", sa.Numeric(6, 5)),
    sa.Column("availability_applied", sa.Boolean, nullable=False, server_default=sa.text("0")),
    sa.Column("missing_inputs", sa.Unicode(200)),
    sa.UniqueConstraint("run_id", "match_id", name="UQ_prediction_run_match"),
)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def _attach_sqlite_schema(engine: Engine, url: str) -> None:
    """Give SQLite a ``champ`` schema so one set of table defs serves both."""
    main_path = url.split("sqlite:///", 1)[-1] if "sqlite:///" in url else ""
    if main_path in ("", ":memory:"):
        attach_target = ":memory:"
    else:
        attach_target = str(Path(main_path).with_suffix(".champ.db"))

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:  # pragma: no cover - driver hook
        cur = dbapi_conn.cursor()
        cur.execute(f"ATTACH DATABASE '{attach_target}' AS {SCHEMA}")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def make_engine(db: DbConfig | None = None, *, echo: bool = False) -> Engine:
    """Build the engine. ``fast_executemany`` is enabled for pyodbc."""
    db = db or DbConfig.from_env()
    url = db.sqlalchemy_url()

    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["fast_executemany"] = True
        kwargs["pool_pre_ping"] = True

    engine = sa.create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        _attach_sqlite_schema(engine, url)
    log.debug("engine created", extra={"dialect": engine.dialect.name})
    return engine


@contextmanager
def session(engine: Engine) -> Iterator[sa.Connection]:
    """A transactional connection; commits on clean exit, rolls back on error."""
    with engine.begin() as conn:
        yield conn


def is_mssql(conn_or_engine: sa.Connection | Engine) -> bool:
    return conn_or_engine.dialect.name.startswith("mssql")


# --------------------------------------------------------------------------
# DDL
# --------------------------------------------------------------------------
_GO_SPLIT = re.compile(r"^\s*GO\s*;?\s*$", re.IGNORECASE | re.MULTILINE)


def split_batches(script: str) -> list[str]:
    """Split a T-SQL script on its GO batch separators."""
    return [part.strip() for part in _GO_SPLIT.split(script) if part.strip()]


def apply_sql_script(engine: Engine, path: Path) -> int:
    """Run a .sql file, batch by batch. SQL Server only."""
    batches = split_batches(path.read_text(encoding="utf-8"))
    with engine.begin() as conn:
        for batch in batches:
            conn.execute(sa.text(batch))
    log.info("applied sql script", extra={"path": str(path), "batches": len(batches)})
    return len(batches)


def create_schema(engine: Engine, sql_dir: Path | None = None) -> None:
    """Create the ``champ`` schema and its objects.

    On SQL Server the authored DDL runs verbatim (constraints, indexes,
    views). On any other dialect the mirrored ``MetaData`` is created instead,
    which is what the tests use.
    """
    if is_mssql(engine):
        sql_dir = sql_dir or Path(__file__).resolve().parents[2] / "sql"
        apply_sql_script(engine, sql_dir / "001_schema.sql")
        apply_sql_script(engine, sql_dir / "002_views.sql")
        return

    if engine.dialect.name != "sqlite":
        with engine.begin() as conn:
            conn.execute(sa.schema.CreateSchema(SCHEMA, if_not_exists=True))
    metadata.create_all(engine)
    log.info("created schema from metadata", extra={"dialect": engine.dialect.name})


def drop_all(engine: Engine) -> None:
    """Drop every mirrored table. Used by the test suite."""
    metadata.drop_all(engine)


# --------------------------------------------------------------------------
# Idempotent upsert
# --------------------------------------------------------------------------
def _qualified(table: sa.Table, conn: sa.Connection) -> str:
    return f"{table.schema}.{table.name}" if table.schema else table.name


def _merge_sql(table: sa.Table, key_cols: Sequence[str], insert_cols: Sequence[str],
               update_cols: Sequence[str]) -> str:
    """T-SQL MERGE on the natural key."""
    target = _qualified_name(table)
    src_select = ", ".join(f":{c} AS {c}" for c in insert_cols)
    on_clause = " AND ".join(f"T.{c} = S.{c}" for c in key_cols)
    set_clause = ", ".join(f"T.{c} = S.{c}" for c in update_cols)
    cols = ", ".join(insert_cols)
    vals = ", ".join(f"S.{c}" for c in insert_cols)
    update_block = f"WHEN MATCHED THEN UPDATE SET {set_clause}\n" if update_cols else ""
    return (
        f"MERGE {target} WITH (HOLDLOCK) AS T\n"
        f"USING (SELECT {src_select}) AS S\n"
        f"ON ({on_clause})\n"
        f"{update_block}"
        f"WHEN NOT MATCHED BY TARGET THEN INSERT ({cols}) VALUES ({vals});"
    )


def _qualified_name(table: sa.Table) -> str:
    return f"{table.schema}.{table.name}" if table.schema else table.name


def upsert(
    conn: sa.Connection,
    table: sa.Table,
    rows: Iterable[dict[str, Any]],
    key_cols: Sequence[str],
    update_cols: Sequence[str] | None = None,
    *,
    chunk_size: int = 1000,
) -> int:
    """Merge ``rows`` into ``table`` on ``key_cols``. Returns rows submitted.

    Running the same ingest twice must leave row counts unchanged, so every
    load in this project goes through here rather than a bare INSERT.
    """
    rows = [r for r in rows if r]
    if not rows:
        return 0

    insert_cols = list(rows[0].keys())
    for row in rows:
        if list(row.keys()) != insert_cols:
            raise ValueError("all rows passed to upsert() must share the same columns")
    missing = [c for c in key_cols if c not in insert_cols]
    if missing:
        raise ValueError(f"key columns absent from payload: {missing}")

    if update_cols is None:
        update_cols = [c for c in insert_cols if c not in key_cols]
    else:
        update_cols = [c for c in update_cols if c in insert_cols and c not in key_cols]

    if is_mssql(conn):
        stmt: Any = sa.text(_merge_sql(table, key_cols, insert_cols, update_cols))
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            try:
                conn.execute(stmt, chunk)
            except Exception:
                # fast_executemany can fail to infer a type when the leading
                # row has NULLs; fall back to one round trip per row.
                log.warning("executemany merge failed, retrying row by row",
                            extra={"table": table.name, "rows": len(chunk)})
                for row in chunk:
                    conn.execute(stmt, row)
    else:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        base = sqlite_insert(table)
        if update_cols:
            stmt = base.on_conflict_do_update(
                index_elements=list(key_cols),
                set_={c: getattr(base.excluded, c) for c in update_cols},
            )
        else:
            stmt = base.on_conflict_do_nothing(index_elements=list(key_cols))
        for start in range(0, len(rows), chunk_size):
            conn.execute(stmt, rows[start:start + chunk_size])

    return len(rows)


def scalar(conn: sa.Connection, statement: Any, params: dict[str, Any] | None = None) -> Any:
    return conn.execute(statement if not isinstance(statement, str) else sa.text(statement),
                        params or {}).scalar()


def table_count(conn: sa.Connection, table: sa.Table) -> int:
    return int(conn.execute(sa.select(sa.func.count()).select_from(table)).scalar_one())


def init_db(cfg: Config | None = None) -> Engine:
    """Create the engine and make sure the schema exists."""
    cfg = cfg or Config.load()
    engine = make_engine(cfg.db)
    create_schema(engine)
    return engine
