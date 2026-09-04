"""Structured logging.

Every record carries the ``run_id`` of the model run in progress, so lines
from a daily prediction can be tied back to the ``champ.model_run`` row that
produced them. JSON to file, human-readable to the console.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RUN_ID_KEY = "run_id"
_reserved = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class RunIdFilter(logging.Filter):
    """Stamps the current run_id onto every record."""

    def __init__(self) -> None:
        super().__init__()
        self.run_id: Any = None

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, _RUN_ID_KEY):
            setattr(record, _RUN_ID_KEY, self.run_id)
        return True


_run_filter = RunIdFilter()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, _RUN_ID_KEY, None),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _reserved and key != _RUN_ID_KEY:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Idempotent root-logger setup: JSON file handler + console handler."""
    root = logging.getLogger()
    if getattr(root, "_champmodel_configured", False):
        root.setLevel(level.upper())
        return

    root.setLevel(level.upper())
    root.addFilter(_run_filter)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    console.addFilter(_run_filter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        file_handler = logging.FileHandler(log_dir / f"champmodel-{stamp}.jsonl", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(_run_filter)
        root.addHandler(file_handler)

    # soccerdata and urllib3 are chatty at DEBUG.
    for noisy in ("urllib3", "soccerdata", "requests"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, root.level))

    root._champmodel_configured = True  # type: ignore[attr-defined]


def set_run_id(run_id: Any) -> None:
    """Bind a run_id to all subsequent log records in this process."""
    _run_filter.run_id = run_id


def get_run_id() -> Any:
    return _run_filter.run_id


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


os.environ.setdefault("PYTHONUNBUFFERED", "1")
