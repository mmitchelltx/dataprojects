"""Period finding for variable stars.

The central difficulty of ground-based period finding is not detecting a signal -- with a few
hundred points and a large amplitude, that is easy. It is deciding **which** of several equally
plausible periods is the real one.

Why aliasing dominates
----------------------
An unevenly sampled time series can be written as the true signal multiplied by a sampling
function. In frequency space, multiplication becomes convolution: the periodogram is the true
spectrum convolved with the spectral window of the observing times. Ground-based data are taken
at night, from one site, in seasons, so the window has large spikes at 1 cycle/day and 1
cycle/year. Every true frequency ``f`` therefore appears again at ``f +/- n`` cycles/day, and
those alias peaks can be *taller* than the true one.

This is why :func:`window_function` is not an optional diagnostic here. Reporting a periodogram
peak without checking the window is how a survey ends up with a catalogue full of periods that
are all one day aliases of each other.

What this module does about it
------------------------------
1. **Lomb-Scargle** with a properly normalised periodogram and analytic false-alarm
   probabilities (Baluev's upper bound), which account for the number of independent
   frequencies searched -- the look-elsewhere correction, without which a "5 sigma" peak in a
   10^5-frequency search is meaningless.
2. **The spectral window**, computed as the periodogram of a constant signal at the same
   observation times, so its spikes are exactly the aliasing structure the data impose.
3. **Explicit alias enumeration** for every peak, at ``f +/- n`` cycles/day and the harmonic
   ratios (``2f``, ``f/2``) that catch a doubled or halved period.
4. **Alias resolution by model comparison**: fit a truncated Fourier series at each candidate
   and compare by BIC. A wrong alias phases the data incoherently, so it needs more harmonics
   or fits worse; the right one produces a tight phased curve.
5. **Phase dispersion minimisation** as an independent, non-sinusoidal cross-check. RR Lyrae
   light curves are strongly non-sinusoidal, and PDM makes no assumption of shape at all.

References
----------
Lomb 1976, Ap&SS 39, 447. doi:10.1007/BF00648343
Scargle 1982, ApJ 263, 835. doi:10.1086/160554
VanderPlas 2018, ApJS 236, 16. doi:10.3847/1538-4365/aab766 -- understanding Lomb-Scargle;
    the window-function argument above follows this paper.
Baluev 2008, MNRAS 385, 1279. doi:10.1111/j.1365-2966.2008.12689.x -- analytic FAP bound.
Stellingwerf 1978, ApJ 224, 953. doi:10.1086/156444 -- phase dispersion minimisation.
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

__all__ = [
    "SIDEREAL_DAY_FREQUENCY",
    "HarmonicFit",
    "PeriodCandidate",
    "PeriodogramResult",
    "canonical_fundamental",
    "find_period",
    "harmonic_fit",
    "lomb_scargle",
    "odd_harmonic_fraction",
    "phase_dispersion",
    "window_function",
]

log = get_logger(__name__)

#: Frequencies (cycles/day) whose window-function spikes generate aliases. One solar day is the
#: dominant one for any night-time survey; the sidereal day matters when observations track the
#: stars rather than the Sun, and one year encodes the observing season.
ALIAS_FREQUENCIES: tuple[float, ...] = (1.0, 0.99726957, 1.0 / 365.25)
SIDEREAL_DAY_FREQUENCY = 0.99726957


@dataclass
class HarmonicFit:
    """A truncated Fourier series fitted at a fixed period.

    The Fourier amplitude and phase ratios are the standard shape descriptors for pulsating
    variables: ``R21 = A2/A1`` and ``phi21 = phi2 - 2*phi1`` separate RRab from RRc from
    Cepheids far better than amplitude alone, because they encode light-curve *shape* rather
    than size.
    """

    period: float
    n_terms: int
    amplitudes: np.ndarray
    phases: np.ndarray
    offset: float
    chi2: float
    n_data: int
    bic: float

    @property
    def r21(self) -> float:
        """Second-to-first harmonic amplitude ratio; NaN if only one term was fitted."""
        if self.n_terms < 2 or self.amplitudes[0] == 0:
            return float("nan")
        return float(self.amplitudes[1] / self.amplitudes[0])

    @property
    def phi21(self) -> float:
        """Phase difference ``phi2 - 2*phi1``, wrapped to [0, 2pi)."""
        if self.n_terms < 2:
            return float("nan")
        return float((self.phases[1] - 2.0 * self.phases[0]) % (2.0 * np.pi))

    @property
    def amplitude(self) -> float:
        """Peak-to-peak amplitude of the fitted model."""
        phase = np.linspace(0.0, 1.0, 512, endpoint=False)
        model = np.full_like(phase, self.offset)
        for k in range(self.n_terms):
            model += self.amplitudes[k] * np.cos(2.0 * np.pi * (k + 1) * phase + self.phases[k])
        return float(model.max() - model.min())

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "n_terms": self.n_terms,
            "amplitude": self.amplitude,
            "r21": self.r21,
            "phi21": self.phi21,
            "chi2": self.chi2,
            "reduced_chi2": self.chi2 / max(self.n_data - 2 * self.n_terms - 1, 1),
            "bic": self.bic,
        }


@dataclass
class PeriodCandidate:
    """One candidate period, with the aliases it must be distinguished from."""

    period: Quantity
    power: float
    fap: float
    method: str
    aliases: list[dict[str, Any]] = field(default_factory=list)
    harmonic: HarmonicFit | None = None
    quality: QualityReport = field(default_factory=QualityReport)
    doubling_note: str = ""
    odd_harmonic_fraction: float = float("nan")

    @property
    def frequency(self) -> float:
        return 1.0 / float(self.period.to(u.day).value)

    def summary(self) -> dict[str, Any]:
        return {
            "period_days": float(self.period.to(u.day).value),
            "period_hours": float(self.period.to(u.hour).value),
            "power": self.power,
            "fap": self.fap,
            "method": self.method,
            "n_aliases_tested": len(self.aliases),
            "aliases": self.aliases,
            "harmonic": None if self.harmonic is None else self.harmonic.to_dict(),
            "odd_harmonic_fraction": self.odd_harmonic_fraction,
            "doubling_note": self.doubling_note,
            "quality": self.quality.summary_line(),
        }

    def __repr__(self) -> str:
        return (
            f"<PeriodCandidate P={self.period.to(u.day).value:.6f} d "
            f"({self.period.to(u.hour).value:.4f} h), power={self.power:.3f}, "
            f"FAP={self.fap:.2e}>"
        )


@dataclass
class PeriodogramResult:
    """A periodogram, its window function, and the candidates drawn from it."""

    frequency: np.ndarray
    power: np.ndarray
    window_frequency: np.ndarray
    window_power: np.ndarray
    candidates: list[PeriodCandidate]
    n_independent_frequencies: int
    quality: QualityReport = field(default_factory=QualityReport)

    @property
    def best(self) -> PeriodCandidate | None:
        return self.candidates[0] if self.candidates else None

    def significance_note(self) -> str:
        return (
            f"False-alarm probabilities use Baluev's analytic upper bound, which accounts for "
            f"the approximately {self.n_independent_frequencies} independent frequencies in "
            f"this search. A raw periodogram power must not be converted to a significance "
            f"without that correction."
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_candidates": len(self.candidates),
            "n_independent_frequencies": self.n_independent_frequencies,
            "significance_note": self.significance_note(),
            "candidates": [c.summary() for c in self.candidates],
            "quality": self.quality.summary_line(),
        }


def _photometry(lc: LightCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time, value, uncertainty as bare arrays.

    Lomb-Scargle is a least-squares fit of a sinusoid, so it is indifferent to whether the
    ordinate is magnitudes or flux -- the periodogram peaks in the same place either way.
    Magnitudes are used directly for variable stars, which is standard practice and keeps the
    Fourier amplitudes in the units the literature quotes.
    """
    return lc.time.value, lc.flux.value, lc.flux_err.value


def lomb_scargle(
    lc: LightCurve,
    *,
    min_period: Quantity,
    max_period: Quantity,
    samples_per_peak: int = 10,
    n_terms: int = 1,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Compute a Lomb-Scargle periodogram. Returns ``(frequency, power, model)``.

    Parameters
    ----------
    samples_per_peak
        Frequency oversampling. Below about 5 the grid can straddle a narrow peak and miss it;
        the cost of oversampling is linear and small.
    n_terms
        Harmonics in the model. One is the classical periodogram; more increases sensitivity to
        non-sinusoidal light curves such as RR Lyrae, at the cost of a higher false-alarm rate.
    """
    from astropy.timeseries import LombScargle

    pmin = ensure_quantity(min_period, u.day, name="min_period")
    pmax = ensure_quantity(max_period, u.day, name="max_period")
    assert pmin is not None and pmax is not None
    if not 0 < pmin.value < pmax.value:
        raise ValueError(f"need 0 < min_period < max_period, got {pmin} and {pmax}")

    time, value, error = _photometry(lc)
    model = LombScargle(time, value, error, nterms=n_terms)
    frequency, power = model.autopower(
        minimum_frequency=1.0 / pmax.value,
        maximum_frequency=1.0 / pmin.value,
        samples_per_peak=samples_per_peak,
    )
    return np.asarray(frequency), np.asarray(power), model


def window_function(
    lc: LightCurve,
    *,
    min_period: Quantity,
    max_period: Quantity,
    samples_per_peak: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spectral window of the observation times.

    The window is the periodogram of a *constant* signal sampled at the same times: it contains
    no astrophysics, only the structure the observing schedule imposes. Its peaks are exactly
    the frequencies at which the data will manufacture aliases.

    A large spike at 1 cycle/day is the signature of a single-site, night-time survey, and it
    means every candidate frequency has serious competition at ``f +/- 1``.

    The constant signal is given uniform uncertainties deliberately. Using the real per-point
    errors would weight the window by photometric quality, but the window is a property of
    *when* observations happened, not how good they were.
    """
    from astropy.timeseries import LombScargle

    pmin = ensure_quantity(min_period, u.day, name="min_period")
    pmax = ensure_quantity(max_period, u.day, name="max_period")
    assert pmin is not None and pmax is not None

    time = lc.time.value
    ones = np.ones_like(time)
    # center_data=False and fit_mean=False: subtracting the mean of a constant would leave
    # nothing to transform. This is the standard construction of the spectral window.
    model = LombScargle(time, ones, 1.0, fit_mean=False, center_data=False)
    frequency, power = model.autopower(
        minimum_frequency=1.0 / pmax.value,
        maximum_frequency=1.0 / pmin.value,
        samples_per_peak=samples_per_peak,
    )
    return np.asarray(frequency), np.asarray(power)


def harmonic_fit(lc: LightCurve, period: float, n_terms: int = 4) -> HarmonicFit:
    """Fit a truncated Fourier series at a fixed period by linear least squares.

    The model is ``m(t) = c + sum_k A_k cos(2 pi k t / P + phi_k)``. It is linear in the
    sine/cosine coefficients at fixed period, so this is an exact least-squares solve rather
    than an optimisation -- no starting guess, no local minima.

    The BIC is what makes alias resolution possible: a wrong period phases the data
    incoherently and fits worse at the same model complexity.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if n_terms < 1:
        raise ValueError(f"n_terms must be at least 1, got {n_terms}")

    time, value, error = _photometry(lc)
    weights = 1.0 / error
    phase = 2.0 * np.pi * time / period

    columns = [np.ones_like(time)]
    for k in range(1, n_terms + 1):
        columns.append(np.cos(k * phase))
        columns.append(np.sin(k * phase))
    design = np.vstack(columns).T

    coeffs, *_ = np.linalg.lstsq(design * weights[:, None], value * weights, rcond=None)
    model = design @ coeffs
    chi2 = float(np.sum(((value - model) / error) ** 2))

    offset = float(coeffs[0])
    amplitudes = np.empty(n_terms)
    phases = np.empty(n_terms)
    for k in range(n_terms):
        a, b = coeffs[1 + 2 * k], coeffs[2 + 2 * k]
        amplitudes[k] = float(np.hypot(a, b))
        phases[k] = float(np.arctan2(-b, a))

    n_params = 2 * n_terms + 1
    n = len(time)
    bic = chi2 + n_params * np.log(n)

    return HarmonicFit(
        period=float(period),
        n_terms=n_terms,
        amplitudes=amplitudes,
        phases=phases,
        offset=offset,
        chi2=chi2,
        n_data=n,
        bic=float(bic),
    )


def phase_dispersion(lc: LightCurve, period: float, n_bins: int = 10) -> float:
    """Stellingwerf's PDM statistic, theta, at a fixed period.

    Bins the phased data and compares the within-bin variance against the overall variance.
    ``theta`` near 0 means the folded curve is tight (a good period); ``theta`` near 1 means
    folding achieved nothing.

    Unlike Lomb-Scargle this assumes nothing about the light-curve *shape*, which matters for
    RR Lyrae and Cepheids: their sawtooth profiles are poorly matched by a single sinusoid, so
    a method that only asks "does folding reduce scatter" is a genuinely independent check.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    time, value, _ = _photometry(lc)
    total_variance = float(np.var(value, ddof=1))
    if total_variance == 0:
        return 1.0

    phase = (time / period) % 1.0
    bins = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)

    numerator = 0.0
    dof = 0
    for b in range(n_bins):
        sel = bins == b
        count = int(sel.sum())
        if count > 1:
            numerator += (count - 1) * float(np.var(value[sel], ddof=1))
            dof += count - 1
    if dof == 0:
        return 1.0
    return float((numerator / dof) / total_variance)


def _alias_frequencies(frequency: float, max_order: int = 2) -> list[dict[str, Any]]:
    """Enumerate the offset aliases a peak could be confused with.

    Only ``f +/- n * f_window`` entries appear here. Harmonic confusion (2f, f/2) is
    deliberately *excluded* and handled by :func:`_period_doubling_test`, because BIC cannot
    arbitrate it -- see that function for why.
    """
    out: list[dict[str, Any]] = []
    for base in ALIAS_FREQUENCIES:
        for order in range(1, max_order + 1):
            for sign in (+1, -1):
                candidate = frequency + sign * order * base
                if candidate > 0:
                    out.append(
                        {
                            "frequency": candidate,
                            "period": 1.0 / candidate,
                            "relation": f"f {'+' if sign > 0 else '-'} {order}x{base:.6f} c/d",
                        }
                    )
    return out


def odd_harmonic_fraction(fit: HarmonicFit) -> float:
    """Fraction of fitted harmonic power carried by odd harmonics (k = 1, 3, 5 ...).

    The diagnostic for period doubling. If a light curve of true period ``P`` is fitted at
    ``2P``, the true fundamental lands on harmonic ``k=2`` and every *odd* harmonic of the
    doubled model describes nothing real. So a near-zero odd fraction means the doubling is
    spurious; a substantial one means alternating cycles genuinely differ, which is exactly
    what an eclipsing binary with unequal primary and secondary minima looks like.
    """
    power = fit.amplitudes**2
    total = float(power.sum())
    if total <= 0:
        return 0.0
    odd = float(power[0::2].sum())  # k = 1, 3, 5, ... are indices 0, 2, 4, ...
    return odd / total


def canonical_fundamental(
    lc: LightCurve,
    period: float,
    *,
    n_terms: int = 4,
    threshold: float = 0.05,
    max_steps: int = 3,
) -> tuple[float, float, str]:
    """Reduce a trial period to the fundamental it is a harmonic of.

    Returns ``(fundamental, odd_fraction, note)``.

    **Why this cannot be a BIC comparison.** Fitting at ``2P`` with the same number of
    harmonics is strictly more flexible than fitting at ``P``: the even harmonics reproduce
    whatever the ``P`` model could, and the odd harmonics are free to absorb noise, at an
    identical parameter count. BIC therefore prefers the doubled period essentially always.
    Two earlier versions of this module were caught by that -- the first doubled every period
    directly, and the second let doubling back in through an offset alias that happened to land
    on ``2P`` (for a signal at 2 cycles/day, ``f - 1 c/d`` *is* the doubled period). So the
    decision is made on physics, and it is applied to every trial rather than only at the end.

    The physical test asks whether the odd harmonics at a given trial period carry real power.
    If they do not, successive cycles are identical, the trial is a doubled period, and the
    fundamental is half of it. If they do, alternating cycles genuinely differ -- the signature
    of an eclipsing binary with unequal primary and secondary minima -- and the trial stands.

    The upward case is checked first: if the odd harmonics at ``2P`` are significant, then
    ``2P`` is the true period and ``P`` was a halved one, which is exactly how an eclipsing
    binary is missed when only its two minima per cycle are seen.
    """
    baseline = float(lc.baseline.value)
    terms = max(n_terms, 4)

    doubled = 2.0 * period
    if doubled < baseline:
        odd_at_double = odd_harmonic_fraction(harmonic_fit(lc, doubled, n_terms=terms))
        if odd_at_double >= threshold:
            return (
                doubled,
                odd_at_double,
                f"odd harmonics carry {odd_at_double:.1%} of the power at twice this period, "
                f"so alternating cycles genuinely differ (as for an eclipsing binary with "
                f"unequal minima); the doubled period is the fundamental",
            )

    current = period
    odd_fraction = float("nan")
    steps = 0
    # Floor for halving. Deliberately the *smallest* sampling interval, not the median.
    # Unevenly sampled data have no simple Nyquist limit: a period far shorter than the mean
    # or median spacing is recoverable, because the irregular sampling breaks the degeneracy
    # that aliases a uniformly sampled series (VanderPlas 2018, section 7). Using the median
    # spacing as a floor silently forbids exactly the short periods this survey is best at --
    # it blocked the LINEAR benchmark's own 2.58-hour period against a 7-day mean spacing.
    spacing = np.diff(np.sort(lc.time.value))
    positive = spacing[spacing > 0]
    min_period = 2.0 * float(positive.min()) if positive.size else 0.0
    while steps < max_steps:
        odd_fraction = odd_harmonic_fraction(harmonic_fit(lc, current, n_terms=terms))
        if odd_fraction >= threshold:
            break
        half = current / 2.0
        if half < min_period:
            break
        current = half
        steps += 1

    if steps == 0:
        note = (
            f"odd harmonics carry {odd_fraction:.1%} of the power at this period, so it is the "
            f"fundamental rather than a multiple of one"
        )
    else:
        note = (
            f"reduced by a factor {2**steps} from {period:.6f} d: the odd harmonics there were "
            f"negligible, meaning successive cycles were identical and the trial was a "
            f"multiple of the true period"
        )
    return current, odd_fraction, note


def find_period(
    lc: LightCurve,
    *,
    min_period: Quantity | None = None,
    max_period: Quantity | None = None,
    n_candidates: int = 3,
    n_terms: int = 1,
    harmonics_for_resolution: int = 4,
    fap_threshold: float = 1e-3,
    samples_per_peak: int = 10,
) -> PeriodogramResult:
    """Find candidate periods, enumerate their aliases, and resolve between them.

    The returned candidates are ordered by periodogram power, but the *best* candidate has been
    alias-resolved: for each peak, the aliases are enumerated and each is fitted with a
    truncated Fourier series, and the one with the lowest BIC wins. That is a real decision made
    on evidence, not a preference for the tallest peak.

    Parameters
    ----------
    min_period, max_period
        Search bounds. Defaults: twice the median sampling interval (below which nothing is
        resolvable) and half the baseline (above which a "period" is barely sampled twice).
    fap_threshold
        Peaks with a false-alarm probability above this are not reported as candidates.
    """
    time = lc.time.value
    baseline = float(lc.baseline.value)

    checked_min = ensure_quantity(min_period, u.day, name="min_period", allow_none=True)
    checked_max = ensure_quantity(max_period, u.day, name="max_period", allow_none=True)
    pmin = (
        checked_min
        if checked_min is not None
        else max(2.0 * float(np.median(np.diff(np.sort(time)))), 1e-3) * u.day
    )
    pmax = checked_max if checked_max is not None else (baseline / 2.0) * u.day

    frequency, power, model = lomb_scargle(
        lc,
        min_period=pmin,
        max_period=pmax,
        samples_per_peak=samples_per_peak,
        n_terms=n_terms,
    )
    window_freq, window_power = window_function(
        lc, min_period=pmin, max_period=pmax, samples_per_peak=samples_per_peak
    )

    quality = QualityReport()
    quality.extend(lc.quality)

    # Roughly the number of independent frequencies: baseline times the frequency range.
    n_independent = max(int(baseline * (frequency.max() - frequency.min())), 1)

    _flag_window(quality, window_freq, window_power)

    candidates: list[PeriodCandidate] = []
    used = np.zeros_like(frequency, dtype=bool)
    for _ in range(n_candidates):
        masked = np.where(used, -np.inf, power)
        idx = int(np.argmax(masked))
        if not np.isfinite(masked[idx]):
            break
        peak_freq = float(frequency[idx])
        peak_power = float(power[idx])

        try:
            fap = float(model.false_alarm_probability(peak_power, method="baluev"))
        except Exception:
            # Multi-term models have no analytic FAP; say so rather than invent one.
            fap = float("nan")

        if np.isfinite(fap) and fap > fap_threshold:
            break

        candidate = _resolve_aliases(lc, peak_freq, peak_power, fap, harmonics_for_resolution)
        candidates.append(candidate)

        # Suppress this peak and its immediate neighbourhood before looking for the next.
        used |= np.abs(frequency - peak_freq) < 3.0 / baseline

    result = PeriodogramResult(
        frequency=frequency,
        power=power,
        window_frequency=window_freq,
        window_power=window_power,
        candidates=candidates,
        n_independent_frequencies=n_independent,
        quality=quality,
    )

    log.info(
        "science.variables.period.done",
        n_candidates=len(candidates),
        best_period=(None if not candidates else round(float(candidates[0].period.value), 6)),
        n_independent_frequencies=n_independent,
    )
    return result


def _flag_window(quality: QualityReport, window_freq: np.ndarray, window_power: np.ndarray) -> None:
    """Raise a flag when the sampling imprints strong aliasing structure."""
    for base in (1.0, SIDEREAL_DAY_FREQUENCY):
        near = np.abs(window_freq - base) < 0.02
        if not near.any():
            continue
        strength = float(window_power[near].max())
        if strength > 0.2:
            quality.add(
                "strong_window_alias",
                Severity.CAUTION,
                f"The spectral window has a peak of power {strength:.2f} at {base:.4f} "
                f"cycles/day. Every candidate frequency therefore has a strong alias at "
                f"f +/- {base:.4f} c/d, and the tallest periodogram peak is not automatically "
                f"the true period. Aliases have been enumerated and compared by BIC; treat the "
                f"result as a decision between competing periods, not a direct measurement.",
                alias_frequency=base,
                window_power=strength,
            )
            return


def _resolve_aliases(
    lc: LightCurve, peak_freq: float, peak_power: float, fap: float, n_terms: int
) -> PeriodCandidate:
    """Resolve a periodogram peak against its aliases.

    Every trial -- the peak and each offset alias -- is first reduced to its fundamental by
    :func:`canonical_fundamental`, because a BIC comparison between harmonically related
    periods is not a fair one (see that function). Only after that canonicalisation, with
    duplicates merged, are the survivors compared by BIC, where they are genuinely comparable:
    a wrong offset alias phases the data incoherently and fits worse.
    """
    trials = [{"frequency": peak_freq, "period": 1.0 / peak_freq, "relation": "periodogram peak"}]
    trials.extend(_alias_frequencies(peak_freq))

    baseline = float(lc.baseline.value)
    evaluated: list[dict[str, Any]] = []
    seen: dict[float, dict[str, Any]] = {}

    for trial in trials:
        raw_period = float(trial["period"])
        if raw_period <= 0 or raw_period > baseline:
            continue
        period, odd_fraction, note = canonical_fundamental(lc, raw_period, n_terms=n_terms)

        # Merge trials that canonicalise to the same fundamental, keeping the first relation
        # that reached it. Rounding is relative, so it scales across the period range.
        key = round(period, 9)
        duplicate = next((k for k in seen if abs(k - period) < 1e-6 * max(period, 1e-6)), None)
        if duplicate is not None:
            seen[duplicate].setdefault("also_reached_by", []).append(trial["relation"])
            continue

        fit = harmonic_fit(lc, period, n_terms=n_terms)
        record = {
            "period": period,
            "frequency": 1.0 / period,
            "relation": trial["relation"],
            "raw_period": raw_period,
            "canonicalised": abs(period - raw_period) > 1e-9,
            "canonical_note": note,
            "odd_harmonic_fraction": odd_fraction,
            "bic": fit.bic,
            "chi2": fit.chi2,
            "pdm_theta": phase_dispersion(lc, period),
        }
        seen[key] = record
        evaluated.append(record)

    evaluated.sort(key=lambda r: r["bic"])
    winner = evaluated[0]
    best_period = float(winner["period"])

    quality = QualityReport()
    if winner["relation"] != "periodogram peak":
        peak_record = next((r for r in evaluated if r["relation"] == "periodogram peak"), None)
        peak_bic = peak_record["bic"] if peak_record else float("nan")
        quality.add(
            "alias_preferred_over_peak",
            Severity.CAUTION,
            f"The tallest periodogram peak was at P={1.0 / peak_freq:.6f} d, but the alias at "
            f"P={best_period:.6f} d ({winner['relation']}) fits better by BIC "
            f"({winner['bic']:.1f} against {peak_bic:.1f}). The reported period is the result "
            f"of a model comparison, not the raw peak.",
            peak_period=1.0 / peak_freq,
            chosen_period=best_period,
            relation=winner["relation"],
        )

    if winner["canonicalised"]:
        quality.add(
            "period_canonicalised",
            Severity.INFO,
            f"Reported period adjusted from the trial value {winner['raw_period']:.6f} d to "
            f"{best_period:.6f} d: {winner['canonical_note']}.",
            odd_harmonic_fraction=winner["odd_harmonic_fraction"],
        )

    runner_up = evaluated[1] if len(evaluated) > 1 else None
    if runner_up is not None and (runner_up["bic"] - winner["bic"]) < 10.0:
        quality.add(
            "alias_ambiguous",
            Severity.CAUTION,
            f"The best period (P={best_period:.6f} d) is preferred over its closest competitor "
            f"(P={runner_up['period']:.6f} d, {runner_up['relation']}) by only "
            f"{runner_up['bic'] - winner['bic']:.1f} in BIC. That is not decisive; more data "
            f"or a second band would settle it.",
            delta_bic=runner_up["bic"] - winner["bic"],
            competitor_period=runner_up["period"],
        )

    return PeriodCandidate(
        period=best_period * u.day,
        power=peak_power,
        fap=fap,
        method="lomb-scargle, harmonic canonicalisation, BIC alias resolution",
        aliases=evaluated,
        harmonic=harmonic_fit(lc, best_period, n_terms=max(n_terms, 4)),
        quality=quality,
        doubling_note=str(winner["canonical_note"]),
        odd_harmonic_fraction=float(winner["odd_harmonic_fraction"]),
    )
