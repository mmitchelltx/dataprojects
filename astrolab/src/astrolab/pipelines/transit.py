"""The exoplanet transit pipeline: raw light curve to vetted, fitted candidates.

Composes the stages in the order ADR-0003 specifies, and threads three things through all of
them: the data, the provenance history, and the quality flags. The last of these is what makes
the final report honest -- a reservation raised at ingestion (this file's chain of custody does
not reach the archive; these uncertainties are estimated) is still attached to the fitted
depth at the end.

Stages
------
1. **Ingest** a light curve from a declared source, with its provenance.
2. **Detrend** for the search, with a window derived from the expected transit duration.
3. **Search** with TLS, cross-checked against BLS, iterating to find multiple planets.
4. **Vet** every candidate through the false-positive gauntlet.
5. **Fit** the survivors with a physical model to get posteriors and evidence.
6. **Report** everything, including what could not be tested.

The pipeline never decides that something is a planet. It produces candidates with
dispositions and a list of the questions it could not answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u

from astrolab.core.config import AstrolabConfig
from astrolab.core.lightcurve import LightCurve
from astrolab.core.logging import get_logger
from astrolab.core.provenance import RunManifest
from astrolab.core.quality import QualityReport, Severity
from astrolab.science.exoplanets.detrend import DetrendResult, detrend
from astrolab.science.exoplanets.fit import TransitFit, TransitPriors, fit_transit
from astrolab.science.exoplanets.search import Candidate, SearchResult, search_transits
from astrolab.science.exoplanets.vetting import VettingReport, vet_candidate

__all__ = ["TransitPipelineResult", "load_source", "run_transit_pipeline"]

log = get_logger(__name__)


@dataclass
class TransitPipelineResult:
    """Everything one pipeline run produced."""

    config: AstrolabConfig
    manifest: RunManifest
    raw: LightCurve
    detrended: LightCurve
    detrend_result: DetrendResult
    search: SearchResult
    vetting: list[VettingReport] = field(default_factory=list)
    fits: dict[int, TransitFit] = field(default_factory=dict)
    quality: QualityReport = field(default_factory=QualityReport)

    @property
    def candidates(self) -> list[Candidate]:
        return self.search.candidates

    def summary(self) -> dict[str, Any]:
        return {
            "run": self.config.run.name,
            "target": self.config.target.label,
            "lightcurve": self.detrended.summary(),
            "detrend": self.detrend_result.summary(),
            "search": self.search.summary(),
            "vetting": [v.summary() for v in self.vetting],
            "fits": {str(i): f.summary() for i, f in self.fits.items()},
            "quality": self.quality.summary_line(),
            "quality_flags": self.quality.to_list(),
        }


def load_source(config: AstrolabConfig) -> LightCurve:
    """Load the light curve declared by ``config.source``."""
    source = config.source
    if source is None:
        raise ValueError("config has no 'source' section; nothing to load")

    if source.kind == "bundled_validation":
        from astrolab.instruments.k2 import load_validation_lightcurve

        return load_validation_lightcurve(source.variant)

    if source.kind == "local_csv":
        from astrolab.instruments.k2 import load_k2_csv

        assert source.path is not None
        return load_k2_csv(
            source.path,
            target=config.target.label,
            time_system=source.time_system,
            source_note="local file declared in config",
            trusted_provenance=False,
        )

    raise NotImplementedError(
        f"source.kind={source.kind!r} is not implemented. Archive retrieval is written but "
        f"unverified in this environment; see docs/phase-1-status.md."
    )


def run_transit_pipeline(
    config: AstrolabConfig,
    *,
    output_dir: Path | None = None,
) -> TransitPipelineResult:
    """Run the full transit pipeline described by ``config``."""
    if config.transit is None:
        raise ValueError("config has no 'transit' section")
    transit_cfg = config.transit

    manifest = RunManifest(
        run_name=config.run.name,
        config=config.resolved_dict(),
        config_hash=config.content_hash(),
    )
    manifest.record_seed("master", config.run.seed)

    run_dir = (output_dir or config.run.output_dir) / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. ingest ---------------------------------------------------------------------
    raw = load_source(config)
    manifest.record_output(run_dir, kind="source", n_points=len(raw), **{"summary": raw.summary()})

    # -- 2. detrend --------------------------------------------------------------------
    duration = transit_cfg.expected_duration_hours * u.hour
    detrend_result = detrend(
        raw,
        expected_duration=duration,
        window_durations=transit_cfg.detrend.window_durations,
        method=transit_cfg.detrend.method,
    )
    working = detrend_result.lightcurve

    # -- 3. search ---------------------------------------------------------------------
    max_period = (
        transit_cfg.search.max_period_days * u.day
        if transit_cfg.search.max_period_days is not None
        else None
    )
    search = search_transits(
        working,
        min_period=transit_cfg.search.min_period_days * u.day,
        max_period=max_period,
        sde_threshold=transit_cfg.search.sde_threshold,
        max_candidates=transit_cfg.search.max_candidates,
        cross_check_bls=transit_cfg.search.cross_check_bls,
    )

    quality = QualityReport()
    quality.extend(search.quality)

    if not search.candidates:
        quality.add(
            "no_candidates",
            Severity.INFO,
            f"No signal exceeded the detection threshold of SDE > "
            f"{transit_cfg.search.sde_threshold}. That is a result, not a failure.",
            sde_threshold=transit_cfg.search.sde_threshold,
        )

    # -- 4. vet ------------------------------------------------------------------------
    vetting = [
        vet_candidate(working, candidate, target=transit_cfg.catalogue_target)
        for candidate in search.candidates
    ]
    for report in vetting:
        quality.extend(report.quality)

    # -- 5. fit ------------------------------------------------------------------------
    fits: dict[int, TransitFit] = {}
    if transit_cfg.fit.enabled:
        priors = TransitPriors(
            t0_offset=(
                -transit_cfg.fit.prior_t0_offset_days,
                transit_cfg.fit.prior_t0_offset_days,
            ),
            period_fraction=transit_cfg.fit.prior_period_fraction,
            rp=transit_cfg.fit.prior_rp,
            a_rs=transit_cfg.fit.prior_a_rs,
            b=transit_cfg.fit.prior_b,
            log_jitter=transit_cfg.fit.prior_log_jitter,
        )
        for index, (candidate, report) in enumerate(zip(search.candidates, vetting, strict=True)):
            if index >= transit_cfg.fit.max_candidates_to_fit:
                break
            if report.disposition == "FALSE_POSITIVE":
                log.info(
                    "pipeline.fit.skipped",
                    reason="candidate failed vetting",
                    period=round(float(candidate.period.value), 5),
                )
                continue
            # Seeds are derived from the master seed and recorded individually, so the run is
            # reproducible from one number but each stage's actual seed is auditable.
            seed = config.run.seed + index
            manifest.record_seed(f"fit_candidate_{index}", seed)
            fits[index] = fit_transit(
                working,
                candidate,
                priors=priors,
                window_durations=transit_cfg.fit.window_durations,
                n_live=transit_cfg.fit.n_live,
                seed=seed,
                supersample=transit_cfg.fit.supersample,
                sample=transit_cfg.fit.sampler,
            )
            quality.extend(fits[index].quality)

    result = TransitPipelineResult(
        config=config,
        manifest=manifest,
        raw=raw,
        detrended=working,
        detrend_result=detrend_result,
        search=search,
        vetting=vetting,
        fits=fits,
        quality=quality,
    )

    manifest.quality.extend(quality)
    manifest.finish("completed")

    log.info(
        "pipeline.transit.done",
        n_candidates=len(search.candidates),
        n_fitted=len(fits),
        dispositions=[v.disposition for v in vetting],
        quality=quality.summary_line(),
    )
    return result
