"""Transit fitting: from a candidate to a posterior.

This is stage two of ADR-0003. The search stage produced a period and a rough depth from a
filtered light curve; those numbers are good enough to *find* a planet and not good enough to
*report* one. Here the data around each transit is re-extracted and fitted with a physical
model, so the depth is the product of a model rather than of a filter.

Parameterisation, and why each choice
-------------------------------------
Sampling efficiency depends almost entirely on parameterising away degeneracies and hard
boundaries, so the sampled parameters are not the ones reported.

- **Impact parameter b rather than inclination.** Inclination is bounded and its useful range
  shrinks as ``a/R*`` grows, so a uniform prior on inclination is not uniform in anything
  physical. ``b`` is bounded, interpretable, and uniform over transit geometries.
- **Kipping's q1, q2 rather than quadratic limb-darkening u1, u2.** The physical constraints on
  ``(u1, u2)`` -- positive intensity everywhere, monotonic profile -- carve an awkward triangle
  out of the plane, so a square prior wastes most proposals on unphysical models. Kipping
  (2013) maps the physical region onto the unit square exactly.
- **A jitter term added in quadrature.** The bundled light curve's per-point uncertainties are
  *estimated from its own scatter* rather than measured, and detrending leaves correlated
  residuals. Fixing the noise to a possibly-wrong value produces parameter uncertainties that
  are confidently wrong; letting the data set the noise level costs one parameter and is
  honest.
- **Finite exposure time is integrated over.** K2's 29.4-minute cadence smears a 2.4-hour
  transit noticeably. Evaluating the model at cadence midpoints rather than integrating over
  them biases duration and impact parameter. ``batman``'s supersampling handles it.

Nested sampling, not MCMC
-------------------------
``dynesty`` returns the Bayesian evidence alongside the posterior, which is what the
model-comparison requirement needs and what an MCMC chain does not readily give. The cost is
more likelihood evaluations, affordable precisely because the fit runs on a few transit
durations of data rather than a whole campaign.

References
----------
Mandel & Agol 2002, ApJ 580, L171. doi:10.1086/345520 -- analytic transit light curves.
Kreidberg 2015, PASP 127, 1161. doi:10.1086/683602 -- batman.
Kipping 2013, MNRAS 435, 2152. doi:10.1093/mnras/stt1435 -- q1/q2 limb-darkening sampling.
Speagle 2020, MNRAS 493, 3132. doi:10.1093/mnras/staa278 -- dynesty.
Eastman, Gaudi & Agol 2013, PASP 125, 83. doi:10.1086/669497 -- parameter degeneracies.
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
from astrolab.core.uncertainty import JointPosterior, Measurement
from astrolab.science.exoplanets.search import Candidate

__all__ = ["PARAM_NAMES", "TransitFit", "TransitPriors", "fit_transit", "transit_model"]

log = get_logger(__name__)

#: Sampled parameter names, in the order the unit hypercube is mapped.
PARAM_NAMES = ("t0_offset", "period", "rp", "a_rs", "b", "q1", "q2", "log_jitter")


@dataclass
class TransitPriors:
    """Prior ranges, declared explicitly rather than defaulted inside a function.

    Every entry is a uniform range on the sampled parameter. Uniform is chosen for
    transparency, not because it is uninformative -- a uniform prior on ``a/R*`` is not uniform
    in stellar density. The prior sensitivity check exists because of that.
    """

    t0_offset: tuple[float, float] = (-0.05, 0.05)
    """Days around the search epoch. Narrow, because the search already localised the transit."""

    period_fraction: float = 0.002
    """Fractional half-width of the period prior around the search value."""

    rp: tuple[float, float] = (0.005, 0.3)
    """Planet-to-star radius ratio; the upper bound excludes stellar companions."""

    a_rs: tuple[float, float] = (2.0, 200.0)
    """Scaled semi-major axis; the lower bound is roughly the Roche limit."""

    b: tuple[float, float] = (0.0, 1.0)
    """Impact parameter. Grazing geometries above 1 are excluded by default."""

    q1: tuple[float, float] = (0.0, 1.0)
    q2: tuple[float, float] = (0.0, 1.0)
    """Kipping (2013) coordinates; uniform here is uniform over physical limb-darkening laws."""

    log_jitter: tuple[float, float] = (-14.0, -6.0)
    """Natural log of extra white noise added in quadrature to the quoted uncertainties."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "t0_offset_days": list(self.t0_offset),
            "period_fraction": self.period_fraction,
            "rp": list(self.rp),
            "a_rs": list(self.a_rs),
            "b": list(self.b),
            "q1": list(self.q1),
            "q2": list(self.q2),
            "log_jitter": list(self.log_jitter),
        }


def _q_to_u(q1: float, q2: float) -> tuple[float, float]:
    """Kipping (2013) q1,q2 -> quadratic limb-darkening u1,u2."""
    sqrt_q1 = np.sqrt(q1)
    return 2.0 * sqrt_q1 * q2, sqrt_q1 * (1.0 - 2.0 * q2)


def transit_model(
    time: np.ndarray,
    *,
    t0: float,
    period: float,
    rp: float,
    a_rs: float,
    b: float,
    u1: float,
    u2: float,
    exposure_days: float,
    supersample: int,
) -> np.ndarray:
    """Evaluate a Mandel-Agol transit, integrated over the exposure.

    Assumes a circular orbit. Eccentricity is weakly constrained by a single transit's shape --
    it trades off against ``a/R*`` and stellar density -- so fitting it would add a parameter
    the data cannot inform. That assumption is recorded in the fit's metadata rather than left
    implicit.
    """
    import batman

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = a_rs
    # Guard the domain: b can exceed a_rs during sampling, which would make arccos invalid.
    params.inc = float(np.degrees(np.arccos(np.clip(b / a_rs, -1.0, 1.0))))
    params.ecc = 0.0
    params.w = 90.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"

    model = batman.TransitModel(
        params, time, supersample_factor=supersample, exp_time=exposure_days
    )
    return np.asarray(model.light_curve(params))


@dataclass
class TransitFit:
    """Posterior and diagnostics from a transit fit."""

    posterior: JointPosterior
    candidate: Candidate
    log_evidence: float
    log_evidence_err: float
    log_evidence_flat: float
    n_data: int
    priors: TransitPriors
    seed: int
    quality: QualityReport = field(default_factory=QualityReport)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def period(self) -> Measurement:
        return self.posterior.marginal("period").to(u.day)

    @property
    def radius_ratio(self) -> Measurement:
        return self.posterior.marginal("rp")

    @property
    def impact_parameter(self) -> Measurement:
        return self.posterior.marginal("b")

    @property
    def depth(self) -> Measurement:
        """Transit depth as ``(Rp/R*)^2``, derived inside the joint posterior.

        Derived via :meth:`JointPosterior.derive` rather than by squaring a summary, so it
        inherits the correlation structure instead of assuming independence.
        """
        return self.posterior.derive("depth", lambda rp: rp**2, "rp")

    @property
    def duration(self) -> Measurement:
        """Total (first-to-fourth contact) duration for a circular orbit.

        ``T14 = (P/pi) * asin( sqrt((1+k)^2 - b^2) / a )``, evaluated per posterior draw so the
        strong b / a_rs / duration degeneracy is carried rather than discarded.
        """

        def _t14(period: Quantity, rp: Quantity, a_rs: Quantity, b: Quantity) -> Quantity:
            arg = (1.0 + rp.value) ** 2 - b.value**2
            # Non-transiting draws give a negative argument and contribute no duration.
            arg = np.where(arg > 0, arg, np.nan)
            ratio = np.clip(np.sqrt(arg) / a_rs.value, -1.0, 1.0)
            return ((period.value / np.pi) * np.arcsin(ratio)) * u.day

        return self.posterior.derive("duration", _t14, "period", "rp", "a_rs", "b")

    @property
    def log_bayes_factor(self) -> float:
        """ln(evidence for a transit) - ln(evidence for a flat line).

        Positive favours the transit. On the usual reading, above about 5 is strong and above
        10 decisive -- but this compares a transit against a *flat line*, not against the
        eclipsing-binary or systematics hypotheses. Only the vetting gauntlet addresses those,
        so a large Bayes factor here says nothing about them.
        """
        return self.log_evidence - self.log_evidence_flat

    def summary(self) -> dict[str, Any]:
        return {
            "period": self.period.summary(),
            "radius_ratio": self.radius_ratio.summary(),
            "depth": self.depth.summary(),
            "impact_parameter": self.impact_parameter.summary(),
            "duration": self.duration.summary(),
            "log_evidence": self.log_evidence,
            "log_evidence_err": self.log_evidence_err,
            "log_evidence_flat": self.log_evidence_flat,
            "log_bayes_factor": self.log_bayes_factor,
            "n_data": self.n_data,
            "n_samples": self.posterior.n_samples,
            "seed": self.seed,
            "priors": self.priors.to_dict(),
            "correlations": {
                "rp_vs_b": self.posterior.correlation("rp", "b"),
                "rp_vs_a_rs": self.posterior.correlation("rp", "a_rs"),
            },
            "quality": self.quality.summary_line(),
            "metadata": self.metadata,
        }


def _prior_transform(cube: np.ndarray, priors: TransitPriors, period_centre: float) -> np.ndarray:
    """Map the unit hypercube onto the parameter ranges."""
    out = np.empty_like(cube)
    half = priors.period_fraction * period_centre
    ranges = [
        priors.t0_offset,
        (period_centre - half, period_centre + half),
        priors.rp,
        priors.a_rs,
        priors.b,
        priors.q1,
        priors.q2,
        priors.log_jitter,
    ]
    for i, (lo, hi) in enumerate(ranges):
        out[i] = lo + cube[i] * (hi - lo)
    return out


def fit_transit(
    lc: LightCurve,
    candidate: Candidate,
    *,
    priors: TransitPriors | None = None,
    window_durations: float = 3.0,
    n_live: int = 400,
    seed: int = 0,
    supersample: int = 7,
    sample: str = "rwalk",
    max_call: int | None = None,
) -> TransitFit:
    """Fit a transit model to the data around a candidate's transits.

    Only data within ``window_durations`` of each predicted transit is used. That is what makes
    nested sampling affordable at laptop scale (ADR-0003), and it limits the leverage of
    far-from-transit systematics on the fit.

    Parameters
    ----------
    lc
        Light curve to fit.
    candidate
        Provides the starting ephemeris.
    priors
        Prior ranges; recorded in the result and in the manifest.
    n_live
        Live points. More gives a better-characterised posterior and evidence, at linear cost.
    seed
        Recorded, and sufficient to reproduce the run.
    supersample
        Model evaluations per exposure, for the finite-integration correction.
    sample
        dynesty sampling method. ``"rwalk"`` rather than dynesty's default uniform sampling,
        which is a compute decision with a measurable justification: on this problem uniform
        sampling failed to finish in ten minutes, while ``rwalk`` converged in about thirty
        seconds to the same answer. The transit posterior is narrow and strongly curved -- the
        radius-ratio / impact-parameter / duration degeneracy is exactly the kind of shape that
        makes ellipsoidal rejection sampling inefficient -- and dynesty itself warns about it.
        Under the laptop-scale assumption this is the difference between a usable stage and an
        unusable one.
    """
    import dynesty

    priors = priors or TransitPriors()
    quality = QualityReport()
    quality.extend(lc.quality)
    quality.extend(candidate.quality)

    period0 = float(candidate.period.to(u.day).value)
    epoch0 = float(candidate.epoch.to(u.day).value)
    duration0 = float(candidate.duration.to(u.day).value)
    exposure = float(lc.cadence.to(u.day).value)

    phase = lc.time.value - epoch0
    phase -= np.round(phase / period0) * period0
    near = np.abs(phase) < window_durations * duration0
    if int(near.sum()) < 20:
        raise ValueError(
            f"only {int(near.sum())} points within {window_durations} durations of transit; "
            f"too few to fit. Widen the window or check the ephemeris."
        )

    time = lc.time.value[near]
    flux = lc.flux.value[near]
    ferr = lc.flux_err.value[near]

    def log_likelihood(theta: np.ndarray) -> float:
        t0_off, period, rp, a_rs, b, q1, q2, log_jitter = theta
        if b > 1.0 + rp:  # no transit occurs for this geometry
            return -1e100
        u1, u2 = _q_to_u(q1, q2)
        model = transit_model(
            time,
            t0=epoch0 + t0_off,
            period=period,
            rp=rp,
            a_rs=a_rs,
            b=b,
            u1=u1,
            u2=u2,
            exposure_days=exposure,
            supersample=supersample,
        )
        var = ferr**2 + np.exp(2.0 * log_jitter)
        resid = flux - model
        return float(-0.5 * np.sum(resid**2 / var + np.log(2.0 * np.pi * var)))

    sampler = dynesty.NestedSampler(
        log_likelihood,
        lambda cube: _prior_transform(cube, priors, period0),
        ndim=len(PARAM_NAMES),
        nlive=n_live,
        sample=sample,
        bound="multi",
        rstate=np.random.default_rng(seed),
    )
    run_kwargs: dict[str, Any] = {"print_progress": False}
    if max_call is not None:
        run_kwargs["maxcall"] = max_call
    sampler.run_nested(**run_kwargs)
    results = sampler.results

    samples = results.samples_equal(rstate=np.random.default_rng(seed + 1))
    log_z = float(results.logz[-1])
    log_z_err = float(results.logzerr[-1])
    log_z_flat = _flat_line_evidence(flux, ferr, priors, seed=seed, n_live=n_live)

    posterior = JointPosterior(
        samples={
            "t0": u.Quantity(epoch0 + samples[:, 0], u.day),
            "period": u.Quantity(samples[:, 1], u.day),
            "rp": u.Quantity(samples[:, 2]),
            "a_rs": u.Quantity(samples[:, 3]),
            "b": u.Quantity(samples[:, 4]),
            "q1": u.Quantity(samples[:, 5]),
            "q2": u.Quantity(samples[:, 6]),
            "jitter": u.Quantity(np.exp(samples[:, 7])),
        },
        metadata={
            "sampler": "dynesty.NestedSampler",
            "n_live": n_live,
            "log_evidence": log_z,
            "log_evidence_err": log_z_err,
            "seed": seed,
            "sampling_efficiency": float(results.eff),
        },
    )

    fit = TransitFit(
        posterior=posterior,
        candidate=candidate,
        log_evidence=log_z,
        log_evidence_err=log_z_err,
        log_evidence_flat=log_z_flat,
        n_data=int(near.sum()),
        priors=priors,
        seed=seed,
        quality=quality,
        metadata={
            "window_durations": window_durations,
            "supersample": supersample,
            "exposure_days": exposure,
            "sampler_method": sample,
            "eccentricity": "fixed to 0 (circular); a single transit cannot constrain it",
            "limb_darkening": "quadratic, sampled in Kipping (2013) q1/q2",
        },
    )

    _flag_fit(fit, samples)
    log.info(
        "science.fit.done",
        period=round(float(fit.period.value.value), 6),
        rp=round(float(fit.radius_ratio.value.value), 5),
        depth_ppm=round(float(fit.depth.value.value) * 1e6, 1),
        log_bayes_factor=round(fit.log_bayes_factor, 1),
        n_data=fit.n_data,
    )
    return fit


def _flat_line_evidence(
    flux: np.ndarray,
    ferr: np.ndarray,
    priors: TransitPriors,
    *,
    seed: int,
    n_live: int,
) -> float:
    """Evidence for a constant-flux model with a jitter term, for the Bayes factor.

    Same data, same noise treatment, analytic prior volume -- so the comparison is like for
    like rather than a comparison of two differently-set-up problems.
    """
    import dynesty

    lo_j, hi_j = priors.log_jitter
    median = float(np.median(flux))

    def log_likelihood(theta: np.ndarray) -> float:
        offset, log_jitter = theta
        var = ferr**2 + np.exp(2.0 * log_jitter)
        return float(-0.5 * np.sum((flux - offset) ** 2 / var + np.log(2.0 * np.pi * var)))

    def prior_transform(cube: np.ndarray) -> np.ndarray:
        out = np.empty_like(cube)
        out[0] = median - 0.01 + cube[0] * 0.02
        out[1] = lo_j + cube[1] * (hi_j - lo_j)
        return out

    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        ndim=2,
        nlive=n_live,
        rstate=np.random.default_rng(seed + 99),
    )
    sampler.run_nested(print_progress=False)
    return float(sampler.results.logz[-1])


def _flag_fit(fit: TransitFit, samples: np.ndarray) -> None:
    """Attach the reservations the posterior itself implies."""
    for i, name in enumerate(PARAM_NAMES):
        if name in {"t0_offset", "period"}:
            continue
        bounds = getattr(fit.priors, name, None)
        if not isinstance(bounds, tuple):
            continue
        lo, hi = bounds
        span = hi - lo
        if span <= 0:
            continue
        column = samples[:, i]
        near_low = float(np.mean(column < lo + 0.02 * span))
        near_high = float(np.mean(column > hi - 0.02 * span))
        if max(near_low, near_high) > 0.3:
            fit.quality.add(
                "prior_dominated",
                Severity.CAUTION,
                f"The posterior for {name!r} piles up against its prior boundary "
                f"({near_low:.0%} at the lower edge, {near_high:.0%} at the upper). The "
                f"reported value is set by the prior range as much as by the data.",
                parameter=name,
                fraction_low=near_low,
                fraction_high=near_high,
            )

    if fit.log_bayes_factor < 5.0:
        fit.quality.add(
            "weak_evidence",
            Severity.UNRELIABLE,
            f"ln(Bayes factor) against a flat line is only {fit.log_bayes_factor:.1f}. The data "
            f"do not clearly prefer a transit model over no transit at all.",
            log_bayes_factor=fit.log_bayes_factor,
        )

    depth = fit.depth
    depth_value = float(depth.value.value)
    if depth_value > 0 and float(depth.uncertainty.value) / depth_value > 0.5:
        fit.quality.add(
            "poorly_constrained_depth",
            Severity.CAUTION,
            "The depth uncertainty exceeds half its value; the radius ratio is barely measured.",
            relative_uncertainty=float(depth.uncertainty.value) / depth_value,
        )
