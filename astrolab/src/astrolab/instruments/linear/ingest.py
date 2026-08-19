"""LINEAR survey light-curve ingestion.

LINEAR (Lincoln Near-Earth Asteroid Research) was an asteroid survey whose by-product was a
long-baseline photometric record of the sky, later mined for variable stars. Its data shape is
the opposite of a space mission's, and everything downstream has to cope with that:

- **Sparse and irregular.** A few hundred points spread over years, with a mean spacing of days.
- **Structured gaps.** Observations cluster within nights, nights within observing seasons.
  This imprints a window function with strong spikes at 1 day and 1 year, which is what makes
  aliasing the central difficulty of ground-based period finding.
- **Unfiltered magnitudes.** A broad, non-standard passband, so colours and absolute
  calibration are not available from these data alone.
- **Real measured uncertainties**, unlike a light curve whose errors have to be estimated from
  its own scatter.

Time system is MJD (Modified Julian Date = JD - 2400000.5), stored as UTC. That is a different
convention again from Kepler's BKJD and TESS's BTJD, which is precisely why
:class:`~astrolab.core.lightcurve.LightCurve` requires an explicit reference epoch.

References
----------
Sesar et al. 2011, AJ 142, 190. doi:10.1088/0004-6256/142/6/190 -- LINEAR photometric recalibration.
Palaversa et al. 2013, AJ 146, 101. doi:10.1088/0004-6256/146/4/101 -- LINEAR variable
    star catalogue.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.time import Time

from astrolab.core.lightcurve import LightCurve
from astrolab.core.logging import get_logger
from astrolab.core.quality import QualityReport, Severity
from astrolab.instruments.k2.ingest import THIRD_PARTY_MIRROR

__all__ = ["load_linear_csv", "load_validation_variable"]

log = get_logger(__name__)

#: Objects bundled for validation. See ``validation/data/SOURCE.md``.
VALIDATION_OBJECTS = (11375941, 14752041)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_linear_csv(
    path: str | Path,
    *,
    object_id: str,
    source_note: str = "",
    trusted_provenance: bool = False,
) -> LightCurve:
    """Load a LINEAR ``t,mag,magerr`` CSV into a magnitude-space light curve.

    Parameters
    ----------
    path
        CSV with a header row and columns ``t`` (MJD), ``mag``, ``magerr``.
    object_id
        LINEAR object identifier, recorded in metadata.
    trusted_provenance
        Whether the file's chain of custody reaches the survey archive. Default False.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"LINEAR light curve not found: {p}")

    raw = np.loadtxt(p, delimiter=",", skiprows=1)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{p}: expected three columns (t, mag, magerr), got shape {raw.shape}")

    order = np.argsort(raw[:, 0])
    time, mag, mag_err = raw[order, 0], raw[order, 1], raw[order, 2]

    if np.any(mag_err <= 0):
        raise ValueError(f"{p}: non-positive magnitude uncertainties present")

    quality = QualityReport()
    if not trusted_provenance:
        quality.add(
            THIRD_PARTY_MIRROR,
            Severity.CAUTION,
            "This light curve did not come through a survey archive, so its chain of custody "
            "does not reach the data provider and it cannot be verified against the original "
            "product. Usable for development and regression testing; not a validated benchmark.",
            path=str(p),
            note=source_note,
        )

    lc = LightCurve(
        time=time * u.day,
        flux=mag * u.mag,
        flux_err=mag_err * u.mag,
        # MJD = JD - 2400000.5. Stated explicitly rather than assumed: LINEAR uses MJD where
        # Kepler uses BKJD and TESS uses BTJD, and the offsets differ by millions of days.
        epoch_ref=Time(0.0, format="mjd", scale="utc"),
        meta={
            "target": f"LINEAR {object_id}",
            "object_id": str(object_id),
            "mission": "LINEAR",
            "time_system": "MJD",
            "band": "unfiltered",
            "photometry_system": "magnitude",
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

    _flag_sampling(lc)
    log.info(
        "instrument.linear.ingested",
        object_id=str(object_id),
        n_points=len(lc),
        baseline_days=round(float(lc.baseline.value), 1),
        mean_spacing_days=round(float(np.mean(np.diff(time))), 3),
        scatter_mag=round(float(lc.scatter.value), 4),
    )
    return lc


def _flag_sampling(lc: LightCurve) -> None:
    """Flag the sampling properties that make period-finding hazardous.

    Not a complaint about the data -- this is simply what ground-based survey photometry looks
    like. But a period derived from it needs alias analysis, and stating that up front is
    better than leaving a reader to assume a clean periodogram peak is the answer.
    """
    time = lc.time.value
    spacing = np.diff(time)
    duty = len(lc) / float(lc.baseline.value)
    if duty < 1.0:
        lc.quality.add(
            "sparse_sampling",
            Severity.CAUTION,
            f"Only {len(lc)} observations across {lc.baseline.value:.0f} days "
            f"({duty:.2f} points per day, mean spacing {np.mean(spacing):.1f} d). Sparse "
            f"irregular sampling produces a window function with strong spikes, so periodogram "
            f"aliases are a real hazard: the tallest peak is not automatically the true period. "
            f"Alias analysis is required before quoting one.",
            n_points=len(lc),
            baseline_days=float(lc.baseline.value),
            mean_spacing_days=float(np.mean(spacing)),
        )


def load_validation_variable(object_id: int = 11375941) -> LightCurve:
    """Load a bundled LINEAR validation light curve.

    See ``src/astrolab/validation/data/SOURCE.md`` for provenance and its limitations.
    """
    if object_id not in VALIDATION_OBJECTS:
        raise ValueError(
            f"no bundled LINEAR light curve for object {object_id}; "
            f"available: {list(VALIDATION_OBJECTS)}"
        )
    filename = f"LINEAR_{object_id}.csv"
    data_dir = resources.files("astrolab.validation") / "data"
    with resources.as_file(data_dir / filename) as path:
        return load_linear_csv(
            path,
            object_id=str(object_id),
            source_note=(
                "jakevdp/PracticalLombScargle figure data (BSD-3); underlying LINEAR survey "
                "photometry is public. Not the original survey product -- see "
                "validation/data/SOURCE.md."
            ),
            trusted_provenance=False,
        )
