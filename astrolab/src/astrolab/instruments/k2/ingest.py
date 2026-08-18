"""K2 light-curve ingestion.

K2 was the repurposed Kepler mission: after two reaction wheels failed, the spacecraft
observed fields along the ecliptic in ~80-day campaigns, holding pointing against solar
radiation pressure by balancing on the remaining wheels. That balancing act is the defining
systematic of K2 data -- the spacecraft rolls slowly, targets drift across the detector, and
the flux picks up a sawtooth correlated with the ~6-hour thruster firing cycle. Correcting it
is what pipelines like K2SFF, EVEREST, and K2SC exist to do.

That matters for anything downstream: K2 photometry carries pointing-driven systematics on
timescales of hours, which is the same timescale as a planetary transit. This is why the
detrending choice in ADR-0003 is a real decision and not a formality.

Time system
-----------
K2 publishes BKJD = BJD_TDB - 2454833.0, inherited from Kepler. The offset is handled by
:class:`~astrolab.core.lightcurve.LightCurve`'s required ``epoch_ref``, so it cannot be lost.

References
----------
Howell et al. 2014, PASP 126, 398. doi:10.1086/676406 -- the K2 mission.
Vanderburg & Johnson 2014, PASP 126, 948. doi:10.1086/678764 -- K2SFF roll correction.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.time import Time

from astrolab.core.lightcurve import TIME_REFERENCES, LightCurve
from astrolab.core.logging import get_logger
from astrolab.core.quality import Flag, Severity

__all__ = [
    "K2_CAMPAIGN_1_SPAN",
    "THIRD_PARTY_MIRROR",
    "load_k2_csv",
    "load_validation_lightcurve",
]

log = get_logger(__name__)

#: Raised on any product whose chain of custody does not reach the archive.
THIRD_PARTY_MIRROR = Flag("third_party_mirror")

#: K2 Campaign 1 ran 2014-05-30 to 2014-08-21, in BKJD. Used as a sanity check on ingestion:
#: data claiming to be Campaign 1 but falling outside this window is mislabelled.
K2_CAMPAIGN_1_SPAN: tuple[float, float] = (1975.0, 2065.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_k2_csv(
    path: str | Path,
    *,
    target: str,
    time_system: str = "BKJD",
    flux_err: float | None = None,
    source_note: str = "",
    trusted_provenance: bool = False,
) -> LightCurve:
    """Load a two-column (time, normalised flux) K2 light curve.

    Parameters
    ----------
    path
        CSV with columns ``time, flux`` and no header.
    target
        Target designation, recorded in metadata.
    time_system
        Mission time system; must be a key of
        :data:`~astrolab.core.lightcurve.TIME_REFERENCES`.
    flux_err
        Per-point uncertainty. If omitted, it is **estimated** from the robust point-to-point
        scatter and the product is flagged: an estimated uncertainty is not a measured one,
        and a fit that treats it as measured will report over-confident parameters.
    source_note
        Human description of where the file came from, recorded in provenance.
    trusted_provenance
        Whether the file's chain of custody reaches the archive. Default False, which raises
        :data:`THIRD_PARTY_MIRROR` at ``CAUTION``. Set True only for products retrieved
        through :mod:`astrolab.archives`, where the retrieval is itself recorded.

    Returns
    -------
    LightCurve
        With provenance recording the file's checksum, so a later run can tell whether it is
        reading the same bytes.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"K2 light curve not found: {p}")
    if time_system not in TIME_REFERENCES:
        raise ValueError(f"unknown time system {time_system!r}; known: {sorted(TIME_REFERENCES)}")

    raw = np.loadtxt(p, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"{p}: expected two columns (time, flux), got shape {raw.shape}")
    time, flux = raw[:, 0], raw[:, 1]

    order = np.argsort(time)
    resorted = not np.array_equal(order, np.arange(len(time)))
    time, flux = time[order], flux[order]

    quality = _new_quality()

    if flux_err is None:
        # Robust scatter: MAD scaled to a Gaussian-equivalent sigma. Insensitive to the
        # transits we are looking for, unlike a standard deviation.
        estimated = float(1.4826 * np.median(np.abs(flux - np.median(flux))))
        flux_err_array = np.full_like(flux, estimated)
        quality.add(
            "estimated_uncertainties",
            Severity.CAUTION,
            "Per-point uncertainties were estimated from the light curve's own scatter, not "
            "read from the data product. They are therefore correlated with the data and "
            "cannot detect a mis-scaled noise model; parameter uncertainties from a fit using "
            "them are approximate.",
            method="1.4826 * MAD",
            estimated_sigma_ppm=estimated * 1e6,
        )
    else:
        flux_err_array = np.full_like(flux, float(flux_err))

    if not trusted_provenance:
        quality.add(
            THIRD_PARTY_MIRROR,
            Severity.CAUTION,
            "This light curve did not come through an archive query, so its chain of custody "
            "does not reach the data provider and it cannot be verified against the original "
            "product. Results are usable for development and regression testing, but are not "
            "a validated benchmark until re-run on the archive product.",
            path=str(p),
            note=source_note,
        )

    lc = LightCurve(
        time=time * u.day,
        flux=flux * u.dimensionless_unscaled,
        flux_err=flux_err_array * u.dimensionless_unscaled,
        epoch_ref=Time(TIME_REFERENCES[time_system], format="jd", scale="tdb"),
        meta={
            "target": target,
            "mission": "K2",
            "time_system": time_system,
            "cadence_class": "long",
        },
        quality=quality,
        provenance={
            "source": {
                "kind": "local_file",
                "path": str(p),
                "sha256": _sha256(p),
                "note": source_note,
                "trusted": trusted_provenance,
            },
            "history": [{"operation": "ingest", "n_points": len(time)}],
        },
    )

    if resorted:
        lc.provenance["history"].append({"operation": "sort_by_time"})

    _check_cadence(lc)
    log.info(
        "instrument.k2.ingested",
        target=target,
        n_points=len(lc),
        baseline_days=round(float(lc.baseline.value), 3),
        cadence_min=round(float(lc.cadence.to(u.min).value), 3),
        scatter_ppm=round(float(lc.scatter.value * 1e6), 1),
        trusted_provenance=trusted_provenance,
    )
    return lc


def _new_quality() -> Any:
    from astrolab.core.quality import QualityReport

    return QualityReport()


def _check_cadence(lc: LightCurve) -> None:
    """Flag a cadence that cannot resolve a typical transit.

    K2 long cadence is 29.4 minutes. A small-planet transit lasts a few hours, so long cadence
    samples ingress and egress with only a handful of points -- which smears the transit shape
    and biases the inferred duration and impact parameter. This is a real limitation of the
    data, so the pipeline states it rather than letting a confident-looking duration imply a
    precision the sampling does not support.
    """
    cadence_min = float(lc.cadence.to(u.min).value)
    if cadence_min > 20.0:
        lc.quality.add(
            "undersampled_cadence",
            Severity.CAUTION,
            f"Cadence is {cadence_min:.1f} min. Transit ingress and egress last minutes to "
            f"tens of minutes, so the transit shape is smeared by finite exposure time. Fits "
            f"must integrate the model over the exposure (supersampling) or the duration and "
            f"impact parameter will be biased.",
            cadence_minutes=cadence_min,
        )


def load_validation_lightcurve(variant: str = "raw") -> LightCurve:
    """Load the bundled K2-3 validation light curve.

    Real K2 Campaign 1 photometry of EPIC 201367065 (K2-3), which hosts three known
    transiting planets. See ``src/astrolab/validation/data/SOURCE.md`` for the full provenance
    and, importantly, for what this data does *not* license us to claim.

    Parameters
    ----------
    variant
        ``"raw"`` retains stellar variability and is the honest input to the pipeline.
        ``"detrended"`` was detrended by the upstream author using an undocumented procedure;
        it is provided only for cross-checking our own detrending, never as a pipeline input,
        because a result computed from someone else's undocumented preprocessing is not
        reproducible.
    """
    if variant not in {"raw", "detrended"}:
        raise ValueError(f"variant must be 'raw' or 'detrended', got {variant!r}")

    filename = f"EPIC201367065_k2c1_{variant}.csv"
    data_dir = resources.files("astrolab.validation") / "data"
    with resources.as_file(data_dir / filename) as path:
        lc = load_k2_csv(
            path,
            target="K2-3 (EPIC 201367065)",
            time_system="BKJD",
            source_note=(
                "hippke/tls test fixtures (MIT); underlying K2 data are NASA public domain. "
                "Not the original MAST FITS product -- see validation/data/SOURCE.md."
            ),
            trusted_provenance=False,
        )

    start = float(lc.time[0].value)
    if not K2_CAMPAIGN_1_SPAN[0] <= start <= K2_CAMPAIGN_1_SPAN[1]:
        raise ValueError(
            f"bundled validation data starts at BKJD {start:.2f}, outside K2 Campaign 1 "
            f"{K2_CAMPAIGN_1_SPAN}. The file is not what it claims to be."
        )

    lc.meta["campaign"] = 1
    lc.meta["variant"] = variant
    return lc
