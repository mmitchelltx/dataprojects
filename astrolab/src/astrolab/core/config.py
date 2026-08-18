"""Configuration: a run is fully specified by its config file.

The contract this module enforces is the one Prime Directive 4 depends on: if a number
influences a result and is not in the config or the environment lockfile, that is a defect.
Priors, seeds, algorithm choices, tolerances -- all of it is declared here rather than buried
in a function default, because a default that lives in code cannot be swept, cannot be
recorded, and cannot be reviewed alongside the result it produced.

Validation happens at load, before any compute is spent. A typo in a target name should cost
a second, not an hour of downloading followed by a confusing traceback.

Extension: each science pillar adds its own optional section as a nested model. Phase 1 defines
the sections needed for retrieval; ``extra="forbid"`` means an unrecognised key is an error
rather than a silently ignored instruction, which matters when a config is the record of what
was run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AstrolabConfig",
    "CacheConfig",
    "ConfigError",
    "QueryConfig",
    "RunConfig",
    "TargetConfig",
]


class ConfigError(ValueError):
    """A configuration file was invalid, unreadable, or internally inconsistent."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_Base):
    """Identity and reproducibility settings for the run as a whole."""

    name: str = Field(
        description="Short identifier; names the output directory and appears in the manifest."
    )
    description: str = Field(default="", description="Free text explaining the run's purpose.")
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Master random seed. Every stochastic stage derives its own seed from this, so a "
            "run is reproducible end to end from a single number. Recorded in the manifest."
        ),
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Root directory for products, reports, and the run manifest.",
    )

    @field_validator("name")
    @classmethod
    def _name_is_path_safe(cls, v: str) -> str:
        if not v or any(c in v for c in '/\\:*?"<>| '):
            raise ValueError(
                f"run.name must be a non-empty path-safe token (no spaces or separators), "
                f"got {v!r}"
            )
        return v


class TargetConfig(_Base):
    """What to observe.

    Either a resolvable name or explicit coordinates. Both is allowed and is in fact the
    safer choice: the coordinates are then a check on the resolver rather than a duplicate,
    and a name resolving to the wrong object is a real and recurring failure mode.
    """

    name: str | None = Field(
        default=None, description="Target name to resolve (SIMBAD/MAST name resolution)."
    )
    ra_deg: float | None = Field(
        default=None, ge=0.0, lt=360.0, description="ICRS right ascension in degrees."
    )
    dec_deg: float | None = Field(
        default=None, ge=-90.0, le=90.0, description="ICRS declination in degrees."
    )
    search_radius_arcsec: float = Field(
        default=30.0,
        gt=0.0,
        le=3600.0,
        description=(
            "Cone-search radius. The default is deliberately tight: a wide radius silently "
            "pulls in neighbours, and blended neighbours are the leading cause of false "
            "transit signals."
        ),
    )

    @model_validator(mode="after")
    def _needs_name_or_coords(self) -> TargetConfig:
        has_coords = self.ra_deg is not None and self.dec_deg is not None
        if self.name is None and not has_coords:
            raise ValueError(
                "target requires either 'name' or both 'ra_deg' and 'dec_deg'"
            )
        if (self.ra_deg is None) != (self.dec_deg is None):
            raise ValueError("target coordinates require both 'ra_deg' and 'dec_deg'")
        return self

    @property
    def label(self) -> str:
        """Best available human label for logs and filenames."""
        if self.name:
            return self.name
        return f"({self.ra_deg:.6f},{self.dec_deg:+.6f})"


class QueryConfig(_Base):
    """Which archive products to retrieve.

    Kept deliberately explicit -- mission, product level, and author are all stated rather
    than inferred. "Whatever the archive gave me" is not a reproducible specification: MAST
    hosts several independent pipelines' light curves for the same TESS target, and they do
    not give identical answers.
    """

    archive: Literal["mast"] = Field(
        default="mast", description="Archive to query. Only MAST is implemented in Phase 1."
    )
    mission: Literal["TESS", "Kepler", "K2", "JWST", "HST"] = Field(
        description="Mission whose data products are wanted."
    )
    product: Literal["lightcurve", "targetpixel", "spectrum", "image", "any"] = Field(
        default="any", description="Kind of data product."
    )
    author: str | None = Field(
        default=None,
        description=(
            "Producing pipeline, e.g. 'SPOC' or 'QLP' for TESS. Leave unset to accept any, "
            "but note that mixing authors in one analysis mixes systematics treatments."
        ),
    )
    sequence: list[int] | None = Field(
        default=None,
        description="Sector / quarter / campaign numbers. Unset means all available.",
    )
    exptime_seconds: int | None = Field(
        default=None, gt=0, description="Required cadence, e.g. 120 for TESS 2-minute data."
    )
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Maximum products to retrieve. A guard against an over-broad query.",
    )
    include_proprietary: bool = Field(
        default=False,
        description=(
            "Whether to attempt retrieval of exclusive-access data, which requires "
            "credentials. Default false: this project's designed posture is public data only, "
            "and an unexpected auth prompt in a pipeline is worse than an explicit refusal."
        ),
    )


class CacheConfig(_Base):
    """Content-addressed query cache settings."""

    directory: Path = Field(
        default=Path.home() / ".astrolab" / "cache",
        description="Root of the local content-addressed cache.",
    )
    enabled: bool = Field(default=True, description="Set false to force live queries.")
    max_age_days: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Refuse cache entries older than this. Unset means entries never expire, which is "
            "the reproducibility-preserving default: an archive reprocessing its data should "
            "be a deliberate, reviewed refresh, not a silent change under a running analysis."
        ),
    )


class AstrolabConfig(_Base):
    """Root configuration object. One of these fully specifies a run."""

    run: RunConfig
    target: TargetConfig
    query: QueryConfig
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # -- loading ------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> AstrolabConfig:
        """Load and validate a YAML config.

        Raises
        ------
        ConfigError
            On a missing file, malformed YAML, or a validation failure. The underlying
            pydantic message is preserved -- it names the offending field and is more useful
            than anything this layer could substitute.
        """
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"{p}: malformed YAML: {exc}") from exc
        if raw is None:
            raise ConfigError(f"{p}: file is empty")
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: top level must be a mapping, got {type(raw).__name__}")
        try:
            return cls.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(f"{p}: invalid configuration:\n{exc}") from exc

    # -- provenance support --------------------------------------------------------------

    def resolved_dict(self) -> dict[str, Any]:
        """The fully resolved configuration, defaults included, JSON-safe.

        This is what goes into the run manifest. Recording the *resolved* config rather than
        the file as written is the point: a default that changes between versions would
        otherwise silently change a result while the config file on disk looked untouched.
        """
        return json.loads(self.model_dump_json())

    def content_hash(self) -> str:
        """Stable hash of the resolved configuration.

        Sorted keys and a canonical separator make this independent of field ordering, so two
        configs that mean the same thing hash the same regardless of how they were written.
        """
        canonical = json.dumps(self.resolved_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_yaml(self, path: str | Path) -> None:
        """Write the resolved config, so a run can be replayed from its own record."""
        Path(path).write_text(
            yaml.safe_dump(self.resolved_dict(), sort_keys=False, default_flow_style=False)
        )
