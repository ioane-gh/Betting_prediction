"""Configuration, loaded from the environment (and a .env file if present).

Every knob the pipeline has lives here so that a run can be reproduced from
the ``params`` JSON stored on ``champ.model_run``.
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class DbConfig:
    """SQL Server connection parts. ``url`` overrides every other field."""

    driver: str = "ODBC Driver 18 for SQL Server"
    host: str = "localhost"
    port: int = 1433
    name: str = "champ"
    user: str = ""
    password: str = ""
    trust_cert: bool = True
    trusted_connection: bool = False
    url: str = ""

    @classmethod
    def from_env(cls) -> "DbConfig":
        return cls(
            driver=_env("CHAMP_DB_DRIVER", "ODBC Driver 18 for SQL Server"),
            host=_env("CHAMP_DB_HOST", "localhost"),
            port=_env_int("CHAMP_DB_PORT", 1433),
            name=_env("CHAMP_DB_NAME", "champ"),
            user=_env("CHAMP_DB_USER"),
            password=_env("CHAMP_DB_PASSWORD"),
            trust_cert=_env_bool("CHAMP_DB_TRUST_CERT", True),
            trusted_connection=_env_bool("CHAMP_DB_TRUSTED_CONNECTION", False),
            url=_env("CHAMP_DB_URL"),
        )

    def sqlalchemy_url(self) -> str:
        """Build a ``mssql+pyodbc`` URL, unless CHAMP_DB_URL was given."""
        if self.url:
            return self.url
        odbc_parts = [
            f"DRIVER={{{self.driver}}}",
            f"SERVER={self.host},{self.port}",
            f"DATABASE={self.name}",
        ]
        if self.trusted_connection:
            odbc_parts.append("Trusted_Connection=yes")
        else:
            odbc_parts.append(f"UID={self.user}")
            odbc_parts.append(f"PWD={self.password}")
        if self.trust_cert:
            odbc_parts.append("TrustServerCertificate=yes")
        odbc = ";".join(odbc_parts)
        return "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc)

    def redacted_url(self) -> str:
        """The connection URL with the password blanked, safe for logs."""
        if self.url:
            return self.url
        return (
            f"mssql+pyodbc://{self.user}:***@{self.host}:{self.port}/{self.name}"
            f"?driver={urllib.parse.quote_plus(self.driver)}"
        )


@dataclass(frozen=True)
class ModelParams:
    """Hyperparameters of the Dixon-Coles fit and the adjustments on top of it.

    These are the objects Phase 7 grid-searches over, so they are kept
    separate from I/O configuration and are JSON-serialisable on their own.
    """

    decay_half_life_days: float = 180.0
    max_goals: int = 10
    min_weighted_matches: float = 20.0
    shrinkage_prior: float = 0.30
    availability_k: float = 0.5
    availability_cap: float = 0.25
    availability_other_weight: float = 0.3
    enable_h2h_adjustment: bool = False
    h2h_weight: float = 0.05
    h2h_window: int = 10

    @classmethod
    def from_env(cls) -> "ModelParams":
        return cls(
            decay_half_life_days=_env_float("CHAMP_DECAY_HALF_LIFE_DAYS", 180.0),
            max_goals=_env_int("CHAMP_MAX_GOALS", 10),
            min_weighted_matches=_env_float("CHAMP_MIN_WEIGHTED_MATCHES", 20.0),
            shrinkage_prior=_env_float("CHAMP_SHRINKAGE_PRIOR", 0.30),
            availability_k=_env_float("CHAMP_AVAILABILITY_K", 0.5),
            availability_cap=_env_float("CHAMP_AVAILABILITY_CAP", 0.25),
            availability_other_weight=_env_float("CHAMP_AVAILABILITY_OTHER_WEIGHT", 0.3),
            enable_h2h_adjustment=_env_bool("CHAMP_ENABLE_H2H_ADJUSTMENT", False),
            h2h_weight=_env_float("CHAMP_H2H_WEIGHT", 0.05),
            h2h_window=_env_int("CHAMP_H2H_WINDOW", 10),
        )

    def replace(self, **kwargs: Any) -> "ModelParams":
        known = {f.name for f in fields(self)}
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(f"unknown model parameter(s): {sorted(unknown)}")
        return ModelParams(**{**asdict(self), **kwargs})

    @property
    def decay_xi(self) -> float:
        """Decay rate per day implied by the half-life. 0 disables decay."""
        import math

        if self.decay_half_life_days <= 0:
            return 0.0
        return math.log(2.0) / float(self.decay_half_life_days)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Config:
    db: DbConfig = field(default_factory=DbConfig)
    model: ModelParams = field(default_factory=ModelParams)
    football_data_org_key: str = ""
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    backfill_seasons: int = 10
    enable_fbref: bool = False
    fbref_crawl_delay: float = 3.0
    refit_after_days: int = 7
    log_level: str = "INFO"

    @classmethod
    def load(cls, env_file: str | os.PathLike[str] | None = None) -> "Config":
        """Read .env (if present) and then the process environment."""
        if env_file is not None:
            load_dotenv(env_file, override=False)
        else:
            # usecwd=True: search from where the command was run, not from the
            # installed package directory.
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found, override=False)
        return cls(
            db=DbConfig.from_env(),
            model=ModelParams.from_env(),
            football_data_org_key=_env("FOOTBALL_DATA_ORG_KEY"),
            data_dir=Path(_env("CHAMP_DATA_DIR", "./data")),
            output_dir=Path(_env("CHAMP_OUTPUT_DIR", "./output")),
            backfill_seasons=_env_int("CHAMP_BACKFILL_SEASONS", 10),
            enable_fbref=_env_bool("CHAMP_ENABLE_FBREF", False),
            fbref_crawl_delay=_env_float("CHAMP_FBREF_CRAWL_DELAY", 3.0),
            refit_after_days=_env_int("CHAMP_REFIT_AFTER_DAYS", 7),
            log_level=_env("CHAMP_LOG_LEVEL", "INFO"),
        )

    # -- derived paths -----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def availability_csv(self) -> Path:
        return self.data_dir / "availability.csv"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def log_dir(self) -> Path:
        return Path("./logs")

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.cache_dir,
            self.artifact_dir,
            self.output_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
