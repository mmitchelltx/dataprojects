"""The light curve data product.

A light curve is the substrate of both the exoplanet and variable-star pillars, so it lives in
``core`` and knows nothing about either.

Time handling deserves explanation, because it is where light-curve code most often goes
quietly wrong. Missions publish times in mission-specific offset scales: K2 and Kepler use
BKJD (BJD_TDB - 2454833), TESS uses BTJD (BJD_TDB - 2457000). A bare array of "times" is
therefore ambiguous by millions of days, and mixing two missions' conventions in one analysis
produces an ephemeris that is wrong in a way no plot will reveal.

This class stores time as a :class:`~astropy.units.Quantity` of days **since an explicitly
recorded reference epoch**, which is an :class:`~astropy.time.Time`. Relative times stay fast
for fitting; the reference makes them unambiguous; and :meth:`absolute_time` converts to a
real BJD_TDB whenever an absolute answer is needed. The offset can never be lost, because it
is a required field rather than a convention.

Flux may be stored in one of two systems, and the distinction is not cosmetic:

- **Normalised flux**, dimensionless, 1.0 = the out-of-transit level. Storing ppm invites a
  factor-of-10^6 error; see :mod:`astrolab.core.units`.
- **Magnitudes**, the logarithmic scale ground-based surveys publish. Brighter is *smaller*,
  and a magnitude difference is a flux *ratio*, so an "amplitude" in magnitudes is not an
  amplitude in flux and the mean of magnitudes is not the magnitude of the mean flux.

Both are supported because both are what real data arrives as: a K2 transit comes as
normalised flux, a LINEAR RR Lyrae comes as magnitudes. Code that cares must branch on
:attr:`LightCurve.is_magnitude` rather than assume, and :meth:`LightCurve.to_relative_flux`
converts when a flux-space operation is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity

from astrolab.core.quality import QualityReport
from astrolab.core.units import UnitBoundaryError, ensure_quantity

__all__ = ["TIME_REFERENCES", "LightCurve"]

#: Mission time-system reference epochs, as BJD_TDB offsets.
#:
#: BKJD = BJD_TDB - 2454833.0 (Kepler and K2); BTJD = BJD_TDB - 2457000.0 (TESS).
#: These are the offsets published in the mission data products' headers.
TIME_REFERENCES: dict[str, float] = {
    "BKJD": 2454833.0,
    "BTJD": 2457000.0,
    "BJD": 0.0,
}


@dataclass
class LightCurve:
    """Normalised photometric time series with units, provenance, and quality flags.

    Attributes
    ----------
    time
        Days since :attr:`epoch_ref`. Relative, so fitting stays numerically well conditioned:
        absolute BJD values near 2.45e6 lose float precision where it matters for ingress and
        egress timing.
    flux
        Either dimensionless normalised flux (1.0 = the star's baseline level) or magnitudes.
        See :attr:`is_magnitude`; the two are not interchangeable.
    flux_err
        Per-point uncertainty, same units as ``flux``. Required: Prime Directive 2 means a
        light curve without uncertainties is not a measurement.
    epoch_ref
        Absolute time corresponding to ``time = 0``. Makes the offset convention explicit and
        non-losable.
    meta
        Target, mission, cadence, producing pipeline, and anything else describing the product.
    quality
        Reservations accumulated by the stages that produced this product.
    provenance
        Lineage: where the data came from and what has been done to it.
    """

    time: Quantity
    flux: Quantity
    flux_err: Quantity
    epoch_ref: Time
    meta: dict[str, Any] = field(default_factory=dict)
    quality: QualityReport = field(default_factory=QualityReport)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = _require(self.time, u.day, "time")
        self.flux = _require_photometry(self.flux, "flux")
        self.flux_err = _require_photometry(self.flux_err, "flux_err")
        if self.flux.unit.physical_type != self.flux_err.unit.physical_type:
            raise u.UnitConversionError(
                f"flux is in {self.flux.unit} but flux_err is in {self.flux_err.unit}; "
                f"an uncertainty must be in the same system as the value it qualifies"
            )

        n = self.time.size
        if not (self.flux.size == n and self.flux_err.size == n):
            raise ValueError(
                f"time, flux, and flux_err must be the same length; got "
                f"{n}, {self.flux.size}, {self.flux_err.size}"
            )
        if n < 2:
            raise ValueError("a light curve needs at least 2 points")
        if not isinstance(self.epoch_ref, Time):
            raise TypeError(
                "epoch_ref must be an astropy Time. Without it, 'time' is ambiguous by "
                "millions of days -- which mission offset was it?"
            )
        if np.any(self.flux_err.value < 0):
            raise ValueError("flux_err contains negative values")
        if not np.all(np.diff(self.time.value) >= 0):
            raise ValueError(
                "time must be monotonically non-decreasing; sort the light curve before "
                "constructing it, so downstream code can rely on ordering"
            )

    # -- basic properties ---------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def n_points(self) -> int:
        return len(self)

    @property
    def baseline(self) -> Quantity:
        """Total time span."""
        return (self.time[-1] - self.time[0]).to(u.day)

    @property
    def cadence(self) -> Quantity:
        """Median sampling interval.

        The median, not the mean: data gaps would drag a mean upward and misreport the actual
        sampling, which matters when deciding whether a transit is resolved.
        """
        if len(self) < 2:
            return 0.0 * u.day
        return float(np.median(np.diff(self.time.value))) * u.day

    @property
    def scatter(self) -> Quantity:
        """Robust point-to-point scatter (MAD-scaled), a noise estimate insensitive to transits.

        The 1.4826 factor converts median absolute deviation to a Gaussian-equivalent sigma.
        A plain standard deviation would be inflated by the very signals we are looking for.
        """
        resid = self.flux.value - float(np.median(self.flux.value))
        return float(1.4826 * np.median(np.abs(resid))) * u.dimensionless_unscaled

    def absolute_time(self) -> Time:
        """Convert to absolute BJD_TDB. The unambiguous form, for ephemerides and reports."""
        return self.epoch_ref + self.time

    @property
    def is_magnitude(self) -> bool:
        """Whether the photometry is in magnitudes rather than normalised flux."""
        return bool(self.flux.unit.is_equivalent(u.mag))

    def to_relative_flux(self) -> Self:
        """Convert magnitudes to flux relative to the median, leaving flux curves unchanged.

        Uses ``f/f0 = 10 ** (-0.4 * (m - m0))`` with ``m0`` the median magnitude, so the result
        is normalised to 1.0 at the star's median brightness.

        Uncertainties are propagated through the derivative of that relation
        (``sigma_f/f = 0.4 ln(10) sigma_m``, about ``0.921 sigma_m``) rather than converted as
        though they were magnitudes themselves -- an error bar is a local slope, not a value on
        the scale.
        """
        if not self.is_magnitude:
            return self
        mag = self.flux.to_value(u.mag)
        mag_err = self.flux_err.to_value(u.mag)
        m0 = float(np.median(mag))
        rel = 10.0 ** (-0.4 * (mag - m0))
        rel_err = rel * 0.4 * np.log(10.0) * mag_err
        return self._derived(
            flux=rel * u.dimensionless_unscaled,
            flux_err=rel_err * u.dimensionless_unscaled,
            step={
                "operation": "magnitude_to_relative_flux",
                "reference_magnitude": m0,
                "note": "flux normalised to the median magnitude; errors propagated by slope",
            },
        )

    @property
    def time_system(self) -> str:
        """Name of the mission time system, if known ('BKJD', 'BTJD', ...)."""
        return str(self.meta.get("time_system", "unknown"))

    # -- transformations ------------------------------------------------------------------

    def normalise(self) -> Self:
        """Divide by the median flux, returning a new light curve.

        Records the divisor in provenance: a normalisation is a transformation of the data and
        must be recoverable, not silently baked in.

        Refuses on magnitudes, where dividing is meaningless -- magnitudes are logarithmic, so
        the corresponding operation is subtraction. Call :meth:`to_relative_flux` first if a
        flux-space normalisation is what is wanted.
        """
        if self.is_magnitude:
            raise TypeError(
                "cannot divide magnitudes by their median: the magnitude scale is "
                "logarithmic, so normalisation means subtracting a reference magnitude, not "
                "dividing. Use to_relative_flux() to move to flux space first."
            )
        median = float(np.median(self.flux.value))
        if median == 0.0:
            raise ValueError("cannot normalise a light curve whose median flux is zero")
        return self._derived(
            flux=self.flux / median,
            flux_err=self.flux_err / median,
            step={"operation": "normalise", "divisor": median},
        )

    def mask(self, keep: np.ndarray, *, reason: str) -> Self:
        """Keep only the points where ``keep`` is True.

        ``reason`` is required and recorded. Removing data without stating why is how a
        selection effect gets built into a result invisibly.
        """
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (len(self),):
            raise ValueError(f"mask must have shape ({len(self)},), got {keep.shape}")
        n_removed = int((~keep).sum())
        if n_removed == len(self):
            raise ValueError(f"mask would remove every point (reason: {reason})")
        return self._derived(
            time=self.time[keep],
            flux=self.flux[keep],
            flux_err=self.flux_err[keep],
            step={
                "operation": "mask",
                "reason": reason,
                "n_removed": n_removed,
                "n_remaining": int(keep.sum()),
            },
        )

    def remove_nans(self) -> Self:
        """Drop non-finite points, which mission pipelines leave in for cadence continuity."""
        good = (
            np.isfinite(self.time.value)
            & np.isfinite(self.flux.value)
            & np.isfinite(self.flux_err.value)
        )
        if good.all():
            return self
        return self.mask(good, reason="non-finite time, flux, or flux_err")

    def fold(self, period: Quantity, epoch: Quantity) -> Quantity:
        """Phase-fold, returning phase in days within [-P/2, +P/2).

        Parameters
        ----------
        period
            Orbital period.
        epoch
            Reference mid-transit time, in the same relative system as :attr:`time`.
        """
        p = ensure_quantity(period, u.day, name="period")
        e = ensure_quantity(epoch, u.day, name="epoch")
        assert p is not None and e is not None
        if p.value <= 0:
            raise ValueError(f"period must be positive, got {p}")
        phase = (self.time - e + 0.5 * p) % p - 0.5 * p
        return phase.to(u.day)

    def window(self, centre: Quantity, half_width: Quantity, *, reason: str) -> Self:
        """Extract a time window around ``centre``.

        Used by the measurement stage to pull the raw data around a detected transit, which is
        what makes the expensive joint fit affordable (ADR-0003).
        """
        c = ensure_quantity(centre, u.day, name="centre")
        h = ensure_quantity(half_width, u.day, name="half_width")
        assert c is not None and h is not None
        keep = np.abs(self.time.value - c.value) <= h.value
        if not keep.any():
            raise ValueError(
                f"window at {c} +/- {h} contains no data; the light curve spans "
                f"{self.time[0]} to {self.time[-1]}"
            )
        return self.mask(keep, reason=reason)

    # -- internals -------------------------------------------------------------------------

    def _derived(self, *, step: dict[str, Any], **changes: Any) -> Self:
        """Build a derived light curve, appending ``step`` to the provenance history.

        Every transformation goes through here, so the history is complete by construction
        rather than by the author of each method remembering to append to it.
        """
        provenance = dict(self.provenance)
        history = list(provenance.get("history", []))
        history.append(step)
        provenance["history"] = history

        quality = QualityReport()
        quality.extend(self.quality)

        return type(self)(
            time=changes.get("time", self.time),
            flux=changes.get("flux", self.flux),
            flux_err=changes.get("flux_err", self.flux_err),
            epoch_ref=self.epoch_ref,
            meta=dict(self.meta),
            quality=quality,
            provenance=provenance,
        )

    def summary(self) -> dict[str, Any]:
        """Serialisable description, for manifests and report headers."""
        return {
            "target": self.meta.get("target"),
            "mission": self.meta.get("mission"),
            "time_system": self.time_system,
            "epoch_ref_jd": float(self.epoch_ref.jd),
            "n_points": len(self),
            "baseline_days": float(self.baseline.value),
            "cadence_minutes": float(self.cadence.to(u.min).value),
            "photometry_system": "magnitude" if self.is_magnitude else "relative_flux",
            "scatter": float(self.scatter.value),
            "scatter_unit": "mag" if self.is_magnitude else "fraction",
            "scatter_ppm": (None if self.is_magnitude else float(self.scatter.value * 1e6)),
            "time_start": float(self.time[0].value),
            "time_end": float(self.time[-1].value),
            "quality": self.quality.summary_line(),
            "n_provenance_steps": len(self.provenance.get("history", [])),
        }

    def __repr__(self) -> str:
        # ppm is meaningless on a logarithmic scale, so report magnitudes as magnitudes.
        scatter = (
            f"{self.scatter.value:.3f} mag"
            if self.is_magnitude
            else f"{self.scatter.value * 1e6:.0f} ppm"
        )
        return (
            f"<LightCurve {self.meta.get('target', 'unknown')} "
            f"({self.meta.get('mission', '?')}): {len(self)} pts, "
            f"{self.baseline.value:.1f} d, median spacing "
            f"{self.cadence.to(u.min).value:.1f} min, scatter {scatter}>"
        )


def _require(value: Any, unit: u.UnitBase, name: str) -> Quantity:
    q = ensure_quantity(value, unit, name=f"LightCurve.{name}")
    assert q is not None
    return np.atleast_1d(q)


def _require_photometry(value: Any, name: str) -> Quantity:
    """Accept either dimensionless flux or magnitudes, and nothing else.

    Kept permissive between exactly two systems rather than one, because both are what real
    surveys publish -- but not permissive about anything else, since a photometric series in
    janskys or counts has a different meaning again and should be converted deliberately.
    """
    if not isinstance(value, Quantity):
        raise UnitBoundaryError(
            f"LightCurve.{name} must be an astropy Quantity in normalised flux "
            f"(dimensionless) or magnitudes, got a bare {type(value).__name__}"
        )
    if value.unit.is_equivalent(u.mag) or value.unit.is_equivalent(u.dimensionless_unscaled):
        return np.atleast_1d(value)
    raise UnitBoundaryError(
        f"LightCurve.{name} has units {value.unit}; expected normalised flux "
        f"(dimensionless) or magnitudes"
    )
