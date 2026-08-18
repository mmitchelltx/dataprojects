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
    "DetrendConfig",
    "FitConfig",
    "QueryConfig",
    "RunConfig",
    "SearchConfig",
    "SourceConfig",
    "TargetConfig",
    "TransitConfig",
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
                f"run.name must be a non-empty path-safe token (no spaces or separators), got {v!r}"
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
            raise ValueError("target requires either 'name' or both 'ra_deg' and 'dec_deg'")
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


class SourceConfig(_Base):
    """Where the light curve comes from.

    Separate from :class:`QueryConfig` because a data *source* and an archive *query* are
    different things. Making the source an explicit, named choice is what lets a run be
    reproduced without guessing whether a file on disk was the input or an intermediate.
    """

    kind: Literal["bundled_validation", "local_csv", "mast"] = Field(
        description=(
            "'bundled_validation' uses the packaged K2-3 benchmark light curve, whose "
            "provenance limitations are recorded in validation/data/SOURCE.md. 'local_csv' "
            "reads a two-column file. 'mast' retrieves from the archive."
        )
    )
    path: Path | None = Field(
        default=None, description="For 'local_csv': path to a time,flux file."
    )
    variant: Literal["raw", "detrended"] = Field(
        default="raw",
        description=(
            "For 'bundled_validation'. Use 'raw': the 'detrended' variant was processed by an "
            "undocumented external procedure, so a result computed from it is not reproducible."
        ),
    )
    mission: Literal["K2", "Kepler", "TESS"] = Field(default="K2")
    time_system: Literal["BKJD", "BTJD", "BJD"] = Field(default="BKJD")

    @model_validator(mode="after")
    def _local_csv_needs_a_path(self) -> SourceConfig:
        if self.kind == "local_csv" and self.path is None:
            raise ValueError("source.kind='local_csv' requires 'path'")
        return self


class DetrendConfig(_Base):
    """Search-stage detrending. See docs/decisions/0003-two-stage-detrending.md."""

    method: str = Field(default="biweight", description="A wotan method name.")
    window_durations: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Filter window in units of the expected transit duration. Below about 2 the filter "
            "eats the transit; measured on the benchmark, 3 costs 0.2% of depth and 2 costs 95%."
        ),
    )


class SearchConfig(_Base):
    """Periodogram search settings."""

    min_period_days: float = Field(default=2.0, gt=0.0)
    max_period_days: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Defaults to a third of the baseline: a longer 'period' cannot show enough "
            "transits to be measured."
        ),
    )
    sde_threshold: float = Field(default=8.0, gt=0.0)
    max_candidates: int = Field(default=3, ge=1, le=10)
    cross_check_bls: bool = Field(default=True)


class FitConfig(_Base):
    """Transit fitting settings. Priors are declared here, never defaulted in code."""

    enabled: bool = Field(default=True)
    n_live: int = Field(default=300, ge=50, description="Nested sampling live points.")
    window_durations: float = Field(default=3.0, gt=0.0)
    supersample: int = Field(
        default=7,
        ge=1,
        description="Model evaluations per exposure; 1 disables the finite-exposure correction.",
    )
    sampler: str = Field(
        default="rwalk",
        description=(
            "dynesty sampling method. 'rwalk' rather than the default: the transit posterior is "
            "narrow and curved, and uniform sampling did not converge on the benchmark."
        ),
    )
    max_candidates_to_fit: int = Field(default=1, ge=0)
    prior_rp: tuple[float, float] = Field(default=(0.005, 0.3))
    prior_a_rs: tuple[float, float] = Field(default=(2.0, 200.0))
    prior_b: tuple[float, float] = Field(default=(0.0, 1.0))
    prior_log_jitter: tuple[float, float] = Field(default=(-14.0, -6.0))
    prior_period_fraction: float = Field(default=0.002, gt=0.0)
    prior_t0_offset_days: float = Field(default=0.05, gt=0.0)


class TransitConfig(_Base):
    """The exoplanet transit pipeline."""

    expected_duration_hours: float = Field(
        gt=0.0,
        description=(
            "Expected transit duration. Required, with no default: the detrending window is "
            "derived from it, and a window chosen without reference to the signal is the "
            "depth-bias failure mode ADR-0003 exists to avoid."
        ),
    )
    detrend: DetrendConfig = Field(default_factory=DetrendConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    fit: FitConfig = Field(default_factory=FitConfig)
    catalogue_target: str | None = Field(
        default=None,
        description="Key into the known-ephemeris table for the vetting cross-match.",
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
    query: QueryConfig | None = Field(
        default=None, description="Archive query. Required for `astrolab query`."
    )
    source: SourceConfig | None = Field(
        default=None, description="Light-curve source. Required for `astrolab run`."
    )
    transit: TransitConfig | None = Field(
        default=None, description="Transit pipeline settings. Required for `astrolab run`."
    )
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="after")
    def _needs_something_to_do(self) -> AstrolabConfig:
        if self.query is None and self.source is None:
            raise ValueError(
                "config must define either 'query' (for `astrolab query`) or 'source' "
                "(for `astrolab run`)"
            )
        if self.source is not None and self.transit is None:
            raise ValueError("a config with 'source' must also define 'transit'")
        return self

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
        resolved: dict[str, Any] = json.loads(self.model_dump_json())
        return resolved

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
