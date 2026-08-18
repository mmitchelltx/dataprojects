"""Detrending: removing stellar variability and systematics without eating the transit.

This is where transit photometry is most easily and most invisibly gotten wrong, so the
reasoning is spelled out. See ``docs/decisions/0003-two-stage-detrending.md`` for the decision
this module implements.

The problem
-----------
A raw light curve mixes three things: the transit you want (a 0.1-1% dip lasting hours),
genuine stellar variability (spots rotating in and out of view, on timescales of days), and
instrumental systematics (for K2, a sawtooth from the ~6-hour thruster firings that hold the
spacecraft's roll). Removing the unwanted parts requires a filter, and a filter does not know
that the transit is signal rather than trend.

Run it too aggressively and it absorbs part of the transit into the baseline, so the measured
depth comes out shallow. Depth maps directly to planet radius, so a biased depth means a
biased planet -- and because the bias is systematic it does not average away over many
planets, it propagates into any population-level conclusion built on top.

The window length is therefore the parameter that matters most, and this module refuses to
guess it: it must be derived from the expected transit duration. The rule of thumb, and the
default here, is a window of about three transit durations. Wider preserves the transit but
leaves stellar variability behind; narrower removes variability but starts eating the signal.

Why the biweight
----------------
Tukey's biweight is a robust M-estimator: points far from the local centre get zero weight, so
an in-transit point contributes little to the baseline estimated around it. That is exactly
the property wanted here, and it is why `wotan`'s benchmarking finds it preserves transit
depth better than a running mean or median. It is not magic -- with a short enough window it
will still eat the transit, which is what :func:`estimate_depth_bias` is for.

References
----------
Hippke et al. 2019, AJ 158, 143. doi:10.3847/1538-3881/ab3984 -- wotan; benchmarks of
    detrending methods against transit-depth preservation.
Vanderburg & Johnson 2014, PASP 126, 948. doi:10.1086/678764 -- K2 roll systematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from astrolab.core.lightcurve import LightCurve
from astrolab.core.logging import get_logger
from astrolab.core.quality import Severity
from astrolab.core.units import ensure_quantity

__all__ = ["DetrendResult", "detrend", "estimate_depth_bias"]

log = get_logger(__name__)

#: Window length as a multiple of transit duration. Three is the common choice and the wotan
#: paper's recommendation; it is exposed rather than hardcoded because it is a real knob whose
#: effect on completeness must be measured by injection-recovery, not assumed benign.
DEFAULT_WINDOW_DURATIONS = 3.0


@dataclass
class DetrendResult:
    """A detrended light curve and the trend that was removed.

    The trend is kept, not discarded. Being able to plot what was subtracted is how you catch
    a filter that ate the signal, and a detrending you cannot inspect is one you cannot defend.
    """

    lightcurve: LightCurve
    trend: np.ndarray
    method: str
    window_length: Quantity
    n_masked: int

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "window_length_days": float(self.window_length.value),
            "n_masked": self.n_masked,
            "scatter_before_ppm": float(
                self.lightcurve.provenance.get("detrend", {}).get("scatter_before_ppm", np.nan)
            ),
            "scatter_after_ppm": float(self.lightcurve.scatter.value * 1e6),
        }


def detrend(
    lc: LightCurve,
    *,
    expected_duration: Quantity,
    window_durations: float = DEFAULT_WINDOW_DURATIONS,
    method: str = "biweight",
    rescale_estimated_uncertainties: bool = True,
) -> DetrendResult:
    """Detrend a light curve for the *search* stage.

    This is stage one of the two-stage strategy: fast, transit-preserving, and slightly
    depth-biased by construction. Do not measure a depth from its output -- the measurement
    stage re-extracts the raw data around each candidate and fits the transit jointly with a
    noise model, precisely so the depth is not the product of a filter.

    Parameters
    ----------
    lc
        Input light curve, normalised.
    expected_duration
        Expected transit duration. Sets the window length; there is deliberately no default,
        because a window chosen without reference to the signal is the depth-bias failure
        mode this module exists to avoid.
    window_durations
        Window length in units of ``expected_duration``.
    method
        A `wotan` method name. ``"biweight"`` is the default for the reason given in the
        module docstring.
    rescale_estimated_uncertainties
        Whether to re-derive per-point uncertainties from the *detrended* residual scatter when
        the input's uncertainties were themselves estimated rather than measured.

        This matters more than it sounds. An uncertainty estimated from a raw light curve's
        scatter is measuring stellar variability, not photometric precision: for the bundled
        K2-3 data that is 3052 ppm against a true point-to-point scatter of 51 ppm, a factor of
        60. Carrying it through detrending leaves every downstream likelihood nearly flat, and
        a transit fit then reports a real planet as weak evidence with a depth uncertainty
        larger than the depth. Uncertainties that were measured by the mission pipeline are
        never touched.

    Returns
    -------
    DetrendResult
    """
    duration = ensure_quantity(expected_duration, u.day, name="expected_duration")
    assert duration is not None
    if duration.value <= 0:
        raise ValueError(f"expected_duration must be positive, got {duration}")
    if window_durations <= 0:
        raise ValueError(f"window_durations must be positive, got {window_durations}")

    window = (window_durations * duration).to(u.day)

    if window.value >= float(lc.baseline.value):
        raise ValueError(
            f"detrending window ({window:.3f}) is at least as long as the light curve "
            f"({lc.baseline:.3f}); there is no trend to fit"
        )

    from wotan import flatten

    scatter_before = float(lc.scatter.value * 1e6)
    time = lc.time.value
    flux = lc.flux.value

    flat, trend = flatten(
        time,
        flux,
        method=method,
        window_length=float(window.value),
        return_trend=True,
    )

    good = np.isfinite(flat) & np.isfinite(trend)
    n_masked = int((~good).sum())

    # Scaling by the trend keeps the uncertainties relative to the same normalisation as the
    # flux, which is what a subsequent chi-square needs.
    flux_err = lc.flux_err.value / np.where(trend > 0, trend, np.nan)

    residual_scatter = float(1.4826 * np.nanmedian(np.abs(flat - np.nanmedian(flat))))
    rescaled_from: float | None = None
    if rescale_estimated_uncertainties and lc.quality.has("estimated_uncertainties"):
        rescaled_from = float(np.nanmedian(flux_err))
        flux_err = np.full_like(flux_err, residual_scatter)

    detrended = LightCurve(
        time=lc.time,
        flux=flat * u.dimensionless_unscaled,
        flux_err=flux_err * u.dimensionless_unscaled,
        epoch_ref=lc.epoch_ref,
        meta=dict(lc.meta),
        provenance=dict(lc.provenance),
    )
    detrended.quality.extend(lc.quality)
    if rescaled_from is not None:
        detrended.quality.add(
            "uncertainties_rescaled",
            Severity.INFO,
            f"Per-point uncertainties were re-estimated from the detrended residual scatter "
            f"({residual_scatter * 1e6:.1f} ppm), replacing an estimate made from the raw light "
            f"curve ({rescaled_from * 1e6:.1f} ppm) that was dominated by stellar variability "
            f"rather than photometric precision. They remain estimated, not measured.",
            before_ppm=rescaled_from * 1e6,
            after_ppm=residual_scatter * 1e6,
        )

    history = list(detrended.provenance.get("history", []))
    history.append(
        {
            "operation": "detrend",
            "method": method,
            "window_length_days": float(window.value),
            "window_durations": window_durations,
            "expected_duration_days": float(duration.value),
            "n_masked": n_masked,
            "scatter_before_ppm": scatter_before,
            "uncertainty_rescaled_from_ppm": (
                None if rescaled_from is None else rescaled_from * 1e6
            ),
            "scatter_after_ppm": float(
                1.4826 * np.nanmedian(np.abs(flat - np.nanmedian(flat))) * 1e6
            ),
        }
    )
    detrended.provenance["history"] = history
    detrended.provenance["detrend"] = history[-1]

    if n_masked:
        detrended = detrended.remove_nans()

    _flag_aggressive_window(detrended, window, duration)

    log.info(
        "science.detrend.done",
        method=method,
        window_days=round(float(window.value), 4),
        scatter_before_ppm=round(scatter_before, 1),
        scatter_after_ppm=round(float(detrended.scatter.value * 1e6), 1),
        n_masked=n_masked,
    )

    return DetrendResult(
        lightcurve=detrended,
        trend=trend,
        method=method,
        window_length=window,
        n_masked=n_masked,
    )


def _flag_aggressive_window(lc: LightCurve, window: Quantity, duration: Quantity) -> None:
    """Flag a window short enough to bite into the transit."""
    ratio = float(window.value / duration.value)
    if ratio < 2.0:
        lc.quality.add(
            "aggressive_detrending",
            Severity.CAUTION,
            f"Detrending window is only {ratio:.1f} transit durations. Below about 2, the "
            f"filter begins to absorb the transit itself and measured depths are biased "
            f"shallow. Depths from this light curve should not be quoted; re-fit the raw "
            f"data jointly with a noise model instead.",
            window_durations=ratio,
        )


def estimate_depth_bias(
    lc: LightCurve,
    *,
    period: Quantity,
    epoch: Quantity,
    duration: Quantity,
    depth: float,
    window_durations: float = DEFAULT_WINDOW_DURATIONS,
    method: str = "biweight",
) -> float:
    """Measure how much transit depth this detrending configuration destroys.

    Injects a box transit of known depth, detrends the injected light curve, and compares the
    depth recovered afterwards against the depth recoverable from the same injected series
    *before* detrending.

    The comparison is deliberately **differential**, and that detail is load-bearing. Measuring
    the recovered depth against the injected value alone would be corrupted by any real transit
    already present at the chosen phase -- inject on top of a genuine planet and you recover
    roughly twice the depth, which reads as a detrending that *creates* signal. Taking the
    ratio of after-detrending to before-detrending cancels whatever was already there, so the
    result isolates the filter's effect no matter where the injection lands.

    A returned value of 0.95 means the filter removes 5% of the depth, which propagates to
    roughly 2.5% in planet radius (radius goes as the square root of depth).

    This turns the depth bias from a known-unknown into a measured number for the systematics
    budget, and it is why injection machinery is not optional.

    Returns
    -------
    float
        Recovered depth fraction. 1.0 is perfect preservation; below 1.0 the filter is eating
        the transit. Values slightly above 1.0 are noise, not gain.
    """
    p = ensure_quantity(period, u.day, name="period")
    e = ensure_quantity(epoch, u.day, name="epoch")
    d = ensure_quantity(duration, u.day, name="duration")
    assert p is not None and e is not None and d is not None
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")

    def _phase(times: np.ndarray) -> np.ndarray:
        return ((times - e.value + 0.5 * p.value) % p.value) - 0.5 * p.value

    def _measure(times: np.ndarray, flux: np.ndarray) -> float:
        """Box depth: out-of-transit level minus in-transit level."""
        ph = _phase(times)
        # Sample the core of the transit and a clean baseline outside it, leaving a guard band
        # so partially-in-transit points bias neither estimate.
        inside = np.abs(ph) < 0.4 * d.value
        outside = np.abs(ph) > 1.0 * d.value
        if not inside.any() or not outside.any():
            raise ValueError(
                "cannot measure depth: injected ephemeris leaves no in- or out-of-transit points"
            )
        return float(np.median(flux[outside]) - np.median(flux[inside]))

    time = lc.time.value
    in_transit = np.abs(_phase(time)) < 0.5 * d.value
    if not in_transit.any():
        raise ValueError("injected ephemeris places no points in transit")

    injected_flux = lc.flux.value.copy()
    injected_flux[in_transit] -= depth

    depth_before = _measure(time, injected_flux)

    injected = LightCurve(
        time=lc.time,
        flux=injected_flux * u.dimensionless_unscaled,
        flux_err=lc.flux_err,
        epoch_ref=lc.epoch_ref,
        meta=dict(lc.meta),
    )
    result = detrend(
        injected,
        expected_duration=d,
        window_durations=window_durations,
        method=method,
    )
    out = result.lightcurve
    depth_after = _measure(out.time.value, out.flux.value)

    if depth_before <= 0:
        raise ValueError(
            "the injected transit is not detectable even before detrending; increase the "
            "injected depth or check the ephemeris"
        )
    return depth_after / depth_before
