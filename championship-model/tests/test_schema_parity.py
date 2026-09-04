"""sql/001_schema.sql and champmodel.db.metadata must describe the same tables.

The .sql file is the authoritative DDL for SQL Server; the SQLAlchemy metadata
is what the Python code binds to and what the test suite creates on SQLite.
Nothing enforces that they agree, so this test does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champmodel import db as dbm

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "001_schema.sql"

_CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+champ\.(?P<name>\w+)\s*\((?P<body>.*?)\n\);",
    re.IGNORECASE | re.DOTALL,
)
_CONSTRAINT_START = re.compile(r"^\s*(CONSTRAINT|PRIMARY\s+KEY|UNIQUE|CHECK|FOREIGN\s+KEY)\b",
                               re.IGNORECASE)


def _columns_from_ddl(body: str) -> set[str]:
    columns: set[str] = set()
    depth = 0
    current: list[str] = []
    # Split the table body on commas that are not inside parentheses.
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            current, chunk = [], "".join(current)
            _maybe_add(chunk, columns)
            continue
        current.append(char)
    _maybe_add("".join(current), columns)
    return columns


def _maybe_add(chunk: str, columns: set[str]) -> None:
    chunk = chunk.strip()
    if not chunk or _CONSTRAINT_START.match(chunk):
        return
    name = chunk.split()[0].strip("[]")
    if name.startswith("--") or name.startswith("/*"):
        return
    columns.add(name.lower())


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", text)


def _ddl_tables() -> dict[str, set[str]]:
    text = _strip_comments(SQL_PATH.read_text(encoding="utf-8"))
    return {m.group("name").lower(): _columns_from_ddl(m.group("body")) for m in _CREATE_RE.finditer(text)}


def test_ddl_parses():
    tables = _ddl_tables()
    assert len(tables) == len(dbm.metadata.sorted_tables), (
        f"parsed {sorted(tables)} from DDL, metadata has "
        f"{sorted(t.name for t in dbm.metadata.sorted_tables)}"
    )


@pytest.mark.parametrize("table", sorted(t.name for t in dbm.metadata.sorted_tables))
def test_columns_match(table):
    ddl = _ddl_tables()
    assert table in ddl, f"{table} defined in metadata but not in 001_schema.sql"
    meta_cols = {c.name.lower() for c in dbm.metadata.tables[f"{dbm.SCHEMA}.{table}"].columns}
    assert meta_cols == ddl[table], (
        f"{table}: only in metadata {sorted(meta_cols - ddl[table])}, "
        f"only in DDL {sorted(ddl[table] - meta_cols)}"
    )
