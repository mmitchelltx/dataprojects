"""Transit search: periodogram detection of periodic dips.

Two algorithms are run, deliberately.

**BLS** (Box Least Squares) fits a rectangular dip. It is fast, extremely well understood, and
the baseline every transit survey has used for two decades. Its weakness is that a real
transit is not a box -- limb darkening makes it round-bottomed and ingress/egress are gradual
-- so BLS loses sensitivity on shallow signals.

**TLS** (Transit Least Squares) fits a limb-darkened, physically shaped template instead. The
TLS paper reports roughly 10% better detection efficiency for small planets, which is exactly
the regime where a search is hard.

Running both is cheap and buys a genuine consistency check: two algorithms with different
templates agreeing on a period is meaningful evidence, and disagreeing is a finding worth
investigating rather than a number to average.

The look-elsewhere problem
--------------------------
A periodogram search over a fine grid tries an enormous number of hypotheses. A peak at
"3 sigma" in a search over 10^5 independent trials is not a 3-sigma result -- you expect
roughly 100 such peaks from noise alone. Any significance quoted from a blind search must be
corrected for the number of trials, and this module refuses to report an uncorrected one:
:class:`Candidate` carries both the raw SDE and the trial count, and
:meth:`SearchResult.significance_note` states the correction explicitly.

SDE (Signal Detection Efficiency) is the periodogram peak height in units of the periodogram's
own scatter, which already absorbs much of the trial factor empirically. The conventional
threshold for TLS is SDE > 7-9; this module defaults to 8 and records the value used.

References
----------
Kovacs, Zucker & Mazeh 2002, A&A 391, 369. doi:10.1051/0004-6361:20020802 -- BLS.
Hippke & Heller 2019, A&A 623, A39. doi:10.1051/0004-6361/201834672 -- TLS.
Ofir 2014, A&A 561, A138. doi:10.1051/0004-6361/201220860 -- optimal period sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from astrolab.core.lightcurve import LightCurve
from astrolab.core.logging import get_logger
from astrolab.core.quality import QualityReport, Severity
from astrolab.core.units import ensure_quantity

__all__ = ["Candidate", "SearchResult", "search_bls", "search_transits"]

log = get_logger(__name__)

#: Conventional TLS detection threshold. Below this, a peak is not treated as a candidate.
DEFAULT_SDE_THRESHOLD = 8.0


@dataclass
class Candidate:
    """A periodic dip that crossed the detection threshold.

    A candidate is emphatically **not** a planet. It is a signal that survived a periodogram
    and now owes the vetting gauntlet an explanation. Naming the class ``Candidate`` rather
    than ``Planet`` is not pedantry -- most of these are eclipsing binaries, systematics, or
    the blended light of a neighbouring star.
    """

    period: Quantity
    epoch: Quantity
    duration: Quantity
    depth: float
    sde: float
    method: str
    n_transits: int
    snr: float = float("nan")
    fap: float | None = None
    n_trials: int = 0
    quality: QualityReport = field(default_factory=QualityReport)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def depth_ppm(self) -> float:
        return self.depth * 1e6

    def summary(self) -> dict[str, Any]:
        return {
            "period_days": float(self.period.value),
            "epoch_days": float(self.epoch.value),
            "duration_hours": float(self.duration.to(u.hour).value),
            "depth_ppm": self.depth_ppm,
            "sde": self.sde,
            "snr": self.snr,
            "fap": self.fap,
            "method": self.method,
            "n_transits": self.n_transits,
            "n_trials": self.n_trials,
            "quality": self.quality.summary_line(),
        }

    def __repr__(self) -> str:
        return (
            f"<Candidate P={self.period.value:.5f} d, depth={self.depth_ppm:.0f} ppm, "
            f"SDE={self.sde:.1f}, {self.n_transits} transits ({self.method})>"
        )


@dataclass
class SearchResult:
    """All candidates found, with the search configuration that found them."""

    candidates: list[Candidate]
    periods: np.ndarray
    power: np.ndarray
    method: str
    sde_threshold: float
    n_trials: int
    quality: QualityReport = field(default_factory=QualityReport)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def significance_note(self) -> str:
        """State the trial factor explicitly, so no significance is quoted bare."""
        return (
            f"Search examined {self.n_trials} period trials. Reported SDE values are peak "
            f"heights in units of the periodogram's own scatter, which absorbs much of the "
            f"look-elsewhere penalty empirically; they are NOT Gaussian sigmas and must not "
            f"be converted to a p-value as though they were. Detection threshold used: "
            f"SDE > {self.sde_threshold}."
        )

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_candidates": len(self.candidates),
            "sde_threshold": self.sde_threshold,
            "n_trials": self.n_trials,
            "significance_note": self.significance_note(),
            "candidates": [c.summary() for c in self.candidates],
        }


def _mask_transits(
    lc: LightCurve, candidate: Candidate, *, width_factor: float = 1.5
) -> LightCurve:
    """Remove the in-transit points of a detected candidate.

    Needed for an iterative multi-planet search: the strongest signal must be taken out before
    the next can be seen. ``width_factor`` > 1 removes a little more than the nominal duration,
    because a slightly wrong ephemeris would otherwise leave transit wings behind to be
    rediscovered as a spurious harmonic.
    """
    phase = lc.fold(candidate.period, candidate.epoch).value
    half = 0.5 * width_factor * candidate.duration.to(u.day).value
    keep = np.abs(phase) > half
    return lc.mask(
        keep,
        reason=(
            f"masked transits of candidate P={candidate.period.value:.5f} d for iterative search"
        ),
    )


def search_bls(
    lc: LightCurve,
    *,
    min_period: Quantity,
    max_period: Quantity,
    durations: Quantity | None = None,
    n_periods: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run a Box Least Squares periodogram. Returns ``(periods, power, best_params)``.

    Used as the independent cross-check on TLS rather than as the primary detector.
    """
    from astropy.timeseries import BoxLeastSquares

    pmin = ensure_quantity(min_period, u.day, name="min_period")
    pmax = ensure_quantity(max_period, u.day, name="max_period")
    assert pmin is not None and pmax is not None
    if pmin.value >= pmax.value:
        raise ValueError(f"min_period ({pmin}) must be less than max_period ({pmax})")

    if durations is None:
        durations = np.array([0.02, 0.05, 0.1, 0.15]) * u.day

    bls = BoxLeastSquares(lc.time.value, lc.flux.value, dy=lc.flux_err.value)
    periods = np.linspace(pmin.value, pmax.value, n_periods)
    result = bls.power(periods, np.atleast_1d(durations.to(u.day).value))

    i = int(np.argmax(result.power))
    best = {
        "period": float(result.period[i]),
        "epoch": float(result.transit_time[i]),
        "duration": float(result.duration[i]),
        "depth": float(result.depth[i]),
        "power": float(result.power[i]),
    }
    return np.asarray(result.period), np.asarray(result.power), best


def search_transits(
    lc: LightCurve,
    *,
    min_period: Quantity | None = None,
    max_period: Quantity | None = None,
    sde_threshold: float = DEFAULT_SDE_THRESHOLD,
    max_candidates: int = 3,
    cross_check_bls: bool = True,
    **tls_kwargs: Any,
) -> SearchResult:
    """Search for transits with TLS, iterating to find multiple planets.

    Each iteration takes the strongest surviving signal, records it as a candidate, masks its
    transits, and searches again. The loop stops when a peak falls below ``sde_threshold`` or
    ``max_candidates`` is reached.

    Parameters
    ----------
    lc
        A *detrended* light curve (see :mod:`astrolab.science.exoplanets.detrend`).
    min_period, max_period
        Period search bounds. Defaults: 0.5 d, and a third of the baseline. The upper bound
        matters -- a "period" longer than half the baseline can show at most two transits, and
        one transit is not a period at all.
    sde_threshold
        Detection threshold in SDE.
    max_candidates
        Stop after this many detections.
    cross_check_bls
        Also run BLS and record whether it agrees on the strongest period.

    Returns
    -------
    SearchResult
    """
    from transitleastsquares import transitleastsquares

    baseline = float(lc.baseline.value)
    # Explicit None checks: `or` on a Quantity raises, because the truthiness of a physical
    # quantity is genuinely ambiguous (is 0 K false?).
    checked_min = ensure_quantity(min_period, u.day, name="min_period", allow_none=True)
    checked_max = ensure_quantity(max_period, u.day, name="max_period", allow_none=True)
    pmin = checked_min if checked_min is not None else 0.5 * u.day
    # Default upper bound is a third of the baseline: a "period" longer than half the span can
    # show at most two transits, and one transit is not a period at all.
    pmax = checked_max if checked_max is not None else (baseline / 3.0) * u.day
    if pmin.value >= pmax.value:
        raise ValueError(f"min_period ({pmin}) must be less than max_period ({pmax})")

    quality = QualityReport()
    quality.extend(lc.quality)

    candidates: list[Candidate] = []
    working = lc
    first_periods: np.ndarray = np.array([])
    first_power: np.ndarray = np.array([])
    total_trials = 0

    for iteration in range(max_candidates):
        model = transitleastsquares(working.time.value, working.flux.value)
        res = model.power(
            period_min=float(pmin.value),
            period_max=float(pmax.value),
            show_progress_bar=False,
            **tls_kwargs,
        )

        if iteration == 0:
            first_periods = np.asarray(res.periods)
            first_power = np.asarray(res.power)
        total_trials += int(np.size(res.periods))

        sde = float(res.SDE)
        log.info(
            "science.search.iteration",
            iteration=iteration,
            sde=round(sde, 2),
            period=round(float(res.period), 6),
            threshold=sde_threshold,
        )

        if not np.isfinite(sde) or sde < sde_threshold:
            break

        n_transits = int(res.distinct_transit_count)
        candidate = Candidate(
            period=float(res.period) * u.day,
            epoch=float(res.T0) * u.day,
            duration=float(res.duration) * u.day,
            depth=float(1.0 - res.depth),
            sde=sde,
            method="TLS",
            n_transits=n_transits,
            snr=float(res.snr),
            fap=float(res.FAP) if res.FAP is not None else None,
            n_trials=int(np.size(res.periods)),
            extras={
                "odd_even_mismatch_sigma": float(res.odd_even_mismatch),
                "transit_count": int(res.transit_count),
                "rp_rs": float(res.rp_rs),
            },
        )
        _flag_candidate(candidate, working)
        candidates.append(candidate)

        try:
            working = _mask_transits(working, candidate)
        except ValueError as exc:
            quality.add(
                "search_terminated",
                Severity.INFO,
                f"Iterative search stopped after {len(candidates)} candidate(s): {exc}",
            )
            break

    result = SearchResult(
        candidates=candidates,
        periods=first_periods,
        power=first_power,
        method="TLS",
        sde_threshold=sde_threshold,
        n_trials=total_trials,
        quality=quality,
    )

    if cross_check_bls and candidates:
        _cross_check_bls(lc, result, pmin, pmax)

    return result


def _flag_candidate(candidate: Candidate, lc: LightCurve) -> None:
    """Attach the reservations a candidate's own numbers imply."""
    if candidate.n_transits < 2:
        candidate.quality.add(
            "single_event",
            Severity.UNRELIABLE,
            "Only one transit is present in the data. A single event does not determine a "
            "period: the reported value is the search grid's best guess, not a measurement. "
            "It requires another epoch before it means anything.",
            n_transits=candidate.n_transits,
        )
    elif candidate.n_transits < 3:
        candidate.quality.add(
            "few_transits",
            Severity.CAUTION,
            f"Only {candidate.n_transits} transits observed. The period is determined by two "
            f"epochs alone, so an alias at an integer fraction cannot be excluded.",
            n_transits=candidate.n_transits,
        )

    cadence_h = float(lc.cadence.to(u.hour).value)
    duration_h = float(candidate.duration.to(u.hour).value)
    if duration_h < 3 * cadence_h:
        candidate.quality.add(
            "undersampled_transit",
            Severity.CAUTION,
            f"The transit ({duration_h:.2f} h) spans only {duration_h / cadence_h:.1f} "
            f"cadences. Its shape is not resolved, so duration, impact parameter, and any "
            f"limb-darkening inference from it are unreliable.",
            duration_hours=duration_h,
            cadence_hours=cadence_h,
        )

    depth_snr = candidate.depth / max(float(lc.scatter.value), 1e-12)
    if depth_snr < 1.0:
        candidate.quality.add(
            "shallow_relative_to_noise",
            Severity.CAUTION,
            f"Per-point depth-to-scatter is {depth_snr:.2f}; the signal is visible only when "
            f"folded. Systematics that fold coherently could mimic it.",
            depth_over_scatter=depth_snr,
        )


def _cross_check_bls(lc: LightCurve, result: SearchResult, pmin: Quantity, pmax: Quantity) -> None:
    """Compare the strongest TLS period against an independent BLS search.

    Agreement between two different templates is real evidence. Disagreement is a finding, so
    it is flagged rather than quietly dropped.
    """
    best = result.best
    if best is None:
        return
    try:
        _, _, bls_best = search_bls(lc, min_period=pmin, max_period=pmax)
    except Exception as exc:  # a failed cross-check must not kill the search
        result.quality.add(
            "cross_check_failed", Severity.INFO, f"BLS cross-check did not run: {exc}"
        )
        return

    tls_p = float(best.period.value)
    bls_p = bls_best["period"]
    ratio = bls_p / tls_p
    # Accept the fundamental or a low-order harmonic: BLS commonly locks onto 2P or P/2.
    harmonics = np.array([0.5, 1.0, 2.0])
    agrees = bool(np.min(np.abs(ratio - harmonics)) < 0.01)

    best.extras["bls_period"] = bls_p
    best.extras["bls_agrees"] = agrees
    best.extras["bls_tls_ratio"] = ratio

    if not agrees:
        result.quality.add(
            "algorithm_disagreement",
            Severity.CAUTION,
            f"BLS found P={bls_p:.5f} d where TLS found P={tls_p:.5f} d (ratio {ratio:.3f}), "
            f"which is not the fundamental or a low-order harmonic. Two templates disagreeing "
            f"on the period usually means the signal is not a clean periodic transit.",
            tls_period=tls_p,
            bls_period=bls_p,
        )
    log.info(
        "science.search.bls_cross_check",
        tls_period=round(tls_p, 6),
        bls_period=round(bls_p, 6),
        agrees=agrees,
    )
