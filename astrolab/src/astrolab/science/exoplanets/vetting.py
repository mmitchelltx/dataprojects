"""The false-positive gauntlet.

Prime directive 3: nothing is a detection until it survives vetting. This module is that
gauntlet. Its job is to *fail* candidates, and a vetting module that passes everything is
broken regardless of how clean its code is.

Most periodic dips are not planets. The dominant impostors, and the test that catches each:

**Eclipsing binaries (EBs).** Two stars orbiting each other produce far deeper eclipses than a
planet, but a *grazing* EB, or one diluted by a bright neighbour, can mimic a shallow transit.
Two signatures betray them:

- *Odd/even depth difference.* An EB's primary and secondary eclipses generally differ in
  depth. If the true period is twice the detected one, alternate "transits" are really
  secondaries, so odd- and even-numbered events have different depths. A planet's transits are
  all the same depth. :func:`check_odd_even`.
- *Secondary eclipse.* A companion bright enough to be a star produces a detectable dip at
  phase 0.5 when it passes behind the primary. A planet's secondary is orders of magnitude
  shallower and usually undetectable in optical photometry. :func:`check_secondary_eclipse`.

**Blended background EBs.** A deep eclipse on a faint neighbouring star, diluted by the target's
light, looks like a shallow transit on the target. Catching this properly needs pixel-level
data -- centroid shifts and difference imaging -- which a light curve alone cannot provide.
:func:`check_centroid_shift` therefore reports that it *cannot* run rather than passing
silently, which is directive 5 applied to vetting: an untested hypothesis must not be recorded
as a passed test.

**Systematics.** Instrumental artefacts that fold coherently. :func:`check_transit_consistency`
looks for events whose depths disagree beyond their noise, which is characteristic of
systematics rather than of an astrophysical signal.

**Known objects.** The most common outcome of any search is rediscovering something already
catalogued. :func:`check_known_ephemeris` compares against known ephemerides -- and, when no
catalogue is reachable, says so.

References
----------
Batalha et al. 2013, ApJS 204, 24. doi:10.1088/0067-0049/204/2/24 -- Kepler vetting practice.
Coughlin et al. 2016, ApJS 224, 12. doi:10.3847/0067-0049/224/1/12 -- the Robovetter.
Twicken et al. 2018, PASP 130, 064502. doi:10.1088/1538-3873/aab694 -- TPS/DV diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u

from astrolab.core.lightcurve import LightCurve
from astrolab.core.logging import get_logger
from astrolab.core.quality import QualityReport, Severity
from astrolab.science.exoplanets.search import Candidate

__all__ = [
    "KNOWN_EPHEMERIDES",
    "VettingReport",
    "VettingTest",
    "vet_candidate",
]

log = get_logger(__name__)

#: Locally recorded ephemerides of known systems, used when no catalogue service is reachable.
#:
#: This is deliberately tiny and deliberately explicit. It is *not* a substitute for querying
#: the NASA Exoplanet Archive, VSX, and SIMBAD, which is what a real novelty claim requires;
#: it exists so the cross-match step does something real in an offline environment while
#: reporting honestly that its coverage is a handful of systems rather than a catalogue.
KNOWN_EPHEMERIDES: dict[str, list[dict[str, Any]]] = {
    "K2-3": [
        {"name": "K2-3 b", "period_days": 10.05403, "source": "Crossfield et al. 2015"},
        {"name": "K2-3 c", "period_days": 24.6454, "source": "Crossfield et al. 2015"},
        {"name": "K2-3 d", "period_days": 44.5565, "source": "Crossfield et al. 2015"},
    ]
}


@dataclass
class VettingTest:
    """One test's verdict.

    ``passed=None`` means the test could not be run. That is a distinct outcome from passing,
    and conflating the two is how an unchecked hypothesis becomes an assumed-clean one.
    """

    name: str
    passed: bool | None
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed is None:
            return "NOT RUN"
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "detail": self.detail,
            "metrics": self.metrics,
        }


@dataclass
class VettingReport:
    """The full gauntlet's verdict on one candidate."""

    candidate: Candidate
    tests: list[VettingTest] = field(default_factory=list)
    quality: QualityReport = field(default_factory=QualityReport)

    def add(self, test: VettingTest) -> None:
        self.tests.append(test)

    @property
    def failures(self) -> list[VettingTest]:
        return [t for t in self.tests if t.passed is False]

    @property
    def not_run(self) -> list[VettingTest]:
        return [t for t in self.tests if t.passed is None]

    @property
    def disposition(self) -> str:
        """Overall verdict.

        Deliberately coarse, and deliberately without a "CONFIRMED PLANET" value. This tool
        produces candidates; a human decides. The best available verdict is
        ``PLANET_CANDIDATE``, which means "nothing here killed it", not "this is a planet".

        The catalogue match is checked *before* the astrophysical tests, because the two mean
        different things. A candidate matching a known planet is a **recovery** -- the outcome
        that demonstrates the pipeline works -- and calling it a false positive would be both
        wrong and demoralising. Only an unmatched candidate that fails an astrophysical test is
        a false positive.
        """
        known = [t for t in self.tests if t.name == "known_ephemeris"]
        if any(t.passed is False for t in known):
            return "KNOWN_OBJECT"
        if self.failures:
            return "FALSE_POSITIVE"
        if self.not_run:
            return "PLANET_CANDIDATE_INCOMPLETE_VETTING"
        return "PLANET_CANDIDATE"

    def summary(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "candidate": self.candidate.summary(),
            "n_passed": sum(1 for t in self.tests if t.passed is True),
            "n_failed": len(self.failures),
            "n_not_run": len(self.not_run),
            "tests": [t.to_dict() for t in self.tests],
            "quality": self.quality.summary_line(),
        }

    def describe(self) -> str:
        lines = [f"Disposition: {self.disposition}", f"Candidate: {self.candidate!r}", ""]
        for t in self.tests:
            lines.append(f"  [{t.status:>7}] {t.name}: {t.detail}")
        return "\n".join(lines)


# -- individual tests -------------------------------------------------------------------


def _transit_masks(
    lc: LightCurve, candidate: Candidate, *, core: float = 0.4, baseline: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (epoch index, in-transit core mask, out-of-transit baseline mask)."""
    period = candidate.period.to(u.day).value
    epoch = candidate.epoch.to(u.day).value
    duration = candidate.duration.to(u.day).value
    time = lc.time.value

    cycle = np.round((time - epoch) / period).astype(int)
    phase = (time - epoch) - cycle * period
    in_core = np.abs(phase) < core * duration
    in_baseline = (np.abs(phase) > baseline * duration) & (np.abs(phase) < 3.0 * duration)
    return cycle, in_core, in_baseline


def check_odd_even(lc: LightCurve, candidate: Candidate, *, threshold: float = 3.0) -> VettingTest:
    """Compare odd- and even-numbered transit depths.

    If the true period is twice the detected one, the "transits" alternate between an EB's
    primary and secondary eclipses, which generally differ in depth. A significant odd/even
    difference is therefore strong evidence for an EB at twice the period.
    """
    cycle, in_core, in_baseline = _transit_masks(lc, candidate)
    flux = lc.flux.value

    if not in_baseline.any():
        return VettingTest(
            "odd_even",
            None,
            "no out-of-transit baseline points available to measure depths against",
        )
    base = float(np.median(flux[in_baseline]))

    depths: dict[int, list[float]] = {0: [], 1: []}
    for parity in (0, 1):
        sel = in_core & (np.abs(cycle) % 2 == parity)
        if sel.any():
            depths[parity] = list(base - flux[sel])

    if len(depths[0]) < 3 or len(depths[1]) < 3:
        return VettingTest(
            "odd_even",
            None,
            f"too few points in one parity (odd={len(depths[1])}, even={len(depths[0])}) "
            f"to compare depths; needs more transits",
            {"n_even": len(depths[0]), "n_odd": len(depths[1])},
        )

    even, odd = np.array(depths[0]), np.array(depths[1])
    d_even, d_odd = float(np.mean(even)), float(np.mean(odd))
    # Standard error of each mean, combined in quadrature.
    se = float(np.hypot(np.std(even) / np.sqrt(len(even)), np.std(odd) / np.sqrt(len(odd))))
    sigma = abs(d_even - d_odd) / se if se > 0 else 0.0

    passed = sigma < threshold
    return VettingTest(
        "odd_even",
        passed,
        (
            f"odd/even depth difference is {sigma:.1f} sigma "
            f"(even {d_even * 1e6:.0f} ppm, odd {d_odd * 1e6:.0f} ppm); "
            + (
                "consistent with a planet"
                if passed
                else "suggests an eclipsing binary at twice this period"
            )
        ),
        {
            "sigma": sigma,
            "depth_even_ppm": d_even * 1e6,
            "depth_odd_ppm": d_odd * 1e6,
            "threshold": threshold,
        },
    )


def check_secondary_eclipse(
    lc: LightCurve, candidate: Candidate, *, threshold: float = 3.0
) -> VettingTest:
    """Look for a secondary eclipse at phase 0.5.

    A stellar companion produces a detectable secondary. A planet's is orders of magnitude
    shallower and generally invisible in optical photometry at this precision, so a
    significant secondary argues for an EB.

    Assumes a circular orbit, where the secondary sits at phase 0.5. An eccentric orbit shifts
    it, so a non-detection here does not exclude an eccentric EB -- which is why this is one
    test among several rather than the deciding one.
    """
    period = candidate.period.to(u.day).value
    duration = candidate.duration.to(u.day).value
    phase = lc.fold(candidate.period, candidate.epoch).value

    at_secondary = np.abs(np.abs(phase) - 0.5 * period) < 0.4 * duration
    baseline = (np.abs(phase) > 1.5 * duration) & (
        np.abs(np.abs(phase) - 0.5 * period) > 1.5 * duration
    )

    if at_secondary.sum() < 3 or baseline.sum() < 10:
        return VettingTest(
            "secondary_eclipse",
            None,
            f"insufficient coverage at phase 0.5 ({at_secondary.sum()} points) to test",
            {"n_at_secondary": int(at_secondary.sum())},
        )

    flux = lc.flux.value
    base = float(np.median(flux[baseline]))
    depth = base - float(np.mean(flux[at_secondary]))
    se = float(np.std(flux[baseline]) / np.sqrt(at_secondary.sum()))
    sigma = depth / se if se > 0 else 0.0

    passed = sigma < threshold
    return VettingTest(
        "secondary_eclipse",
        passed,
        (
            f"secondary depth {depth * 1e6:.0f} ppm at {sigma:.1f} sigma; "
            + (
                "no significant secondary, consistent with a planet"
                if passed
                else "significant secondary eclipse suggests a stellar companion"
            )
            + " (assumes a circular orbit; an eccentric secondary would sit elsewhere)"
        ),
        {"depth_ppm": depth * 1e6, "sigma": sigma, "threshold": threshold},
    )


def check_transit_consistency(
    lc: LightCurve,
    candidate: Candidate,
    *,
    threshold: float = 4.0,
    n_null: int = 200,
    seed: int = 0,
) -> VettingTest:
    """Check that individual transit depths agree, against a null calibrated by injection.

    An astrophysical transit repeats identically; systematics that fold coherently generally do
    not. Excess scatter between individual event depths therefore argues against an
    astrophysical origin. The whole difficulty is defining "excess", and it took three wrong
    answers to get there -- each of which flagged genuine planets as systematics.

    **Wrong 1: the analytic expectation.** Photometric noise over the square root of points per
    event. At K2's 29.4-minute cadence a 2.4-hour transit gets a handful of samples, and
    successive transits are sampled at *different orbital phases*, so each event averages a
    different part of the transit profile. That produces real depth-to-depth scatter with
    nothing to do with systematics.

    **Wrong 2: an empirical null at off-transit epochs.** Captures the true cadence and residual
    systematics, but measures the scatter of *nothing*: sham epochs contain no signal, so their
    scatter is pure photometric noise, while the real events carry a transit whose
    sampling-phase variation is several times larger. Different quantities, compared.

    **Wrong 3: injecting a box.** A box has a flat bottom, so every event samples the same
    constant depth and the null still shows no sampling variation.

    **What is done.** The template injected at each sham epoch is the candidate's *own
    phase-folded profile*, taken straight from the data. It carries the true transit shape --
    limb darkening, ingress and egress slopes, finite-exposure smearing -- by construction,
    with no model assumed. Injected into a copy of the light curve with the real transits
    removed, it reproduces everything the real measurement is subject to, for a signal known to
    repeat perfectly.

    The lesson generalises: a null hypothesis must contain the signal being tested, with its
    real shape, not merely the noise around it.
    """
    _, _, in_baseline = _transit_masks(lc, candidate)
    if not in_baseline.any():
        return VettingTest("transit_consistency", None, "no baseline points available")

    flux = lc.flux.value
    time = lc.time.value
    base = float(np.median(flux[in_baseline]))
    period = candidate.period.to(u.day).value
    duration = candidate.duration.to(u.day).value
    epoch = candidate.epoch.to(u.day).value

    def event_depths(values: np.ndarray, ref_epoch: float) -> np.ndarray:
        cyc = np.round((time - ref_epoch) / period).astype(int)
        phase = (time - ref_epoch) - cyc * period
        core = np.abs(phase) < 0.4 * duration
        out = []
        for c in np.unique(cyc[core]):
            sel = core & (cyc == c)
            if sel.sum() >= 2:
                out.append(base - float(np.mean(values[sel])))
        return np.asarray(out)

    depths = event_depths(flux, epoch)
    if len(depths) < 3:
        return VettingTest(
            "transit_consistency",
            None,
            f"only {len(depths)} well-sampled transits; needs at least 3 to test scatter",
            {"n_events": len(depths)},
        )
    observed_scatter = float(np.std(depths, ddof=1))

    # Injection template: the candidate's own folded profile, so the injected signal has the
    # real transit's shape rather than an assumed one.
    real_phase = (time - epoch) - np.round((time - epoch) / period).astype(int) * period
    support = 2.0 * duration
    near = np.abs(real_phase) < support
    if int(near.sum()) < 8:
        return VettingTest(
            "transit_consistency",
            None,
            "too few points near transit to build a folded template for the null",
            {"n_near": int(near.sum())},
        )
    order = np.argsort(real_phase[near])
    template_phase = real_phase[near][order]
    template_delta = flux[near][order] - base  # negative in transit

    def template_at(phases: np.ndarray) -> np.ndarray:
        values = np.interp(phases, template_phase, template_delta, left=0.0, right=0.0)
        return np.where(np.abs(phases) < support, values, 0.0)

    # Remove the real transits so an injected sham event is never contaminated by a real one.
    cleaned = flux.copy()
    cleaned[np.abs(real_phase) < support] = base

    min_offset = 3.0 * duration
    if period <= 2.0 * min_offset:
        return VettingTest(
            "transit_consistency",
            None,
            "period is too short relative to the transit duration to place sham epochs clear "
            "of the real transits",
            {"period_days": period, "duration_days": duration},
        )

    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_null):
        sham_epoch = epoch + rng.uniform(min_offset, period - min_offset)
        sham_phase = (time - sham_epoch) - np.round((time - sham_epoch) / period).astype(
            int
        ) * period
        sham = event_depths(cleaned + template_at(sham_phase), sham_epoch)
        if len(sham) >= 3:
            null.append(float(np.std(sham, ddof=1)))

    if len(null) < 20:
        return VettingTest(
            "transit_consistency",
            None,
            "could not build an injection-calibrated null (too few usable sham epochs); the "
            "depth-scatter test is not calibrated for this light curve",
            {"n_null": len(null)},
        )

    null_arr = np.asarray(null)
    null_median = float(np.median(null_arr))
    null_sigma = float(1.4826 * np.median(np.abs(null_arr - null_median)))
    z = (observed_scatter - null_median) / null_sigma if null_sigma > 0 else 0.0

    passed = bool(z < threshold)
    return VettingTest(
        "transit_consistency",
        passed,
        (
            f"depth scatter across {len(depths)} transits is {observed_scatter * 1e6:.0f} ppm, "
            f"{z:+.1f} sigma against a null of {null_median * 1e6:.0f} ppm from {len(null)} "
            f"sham epochs injected with the candidate's own folded profile; "
            + (
                "consistent with a repeating astrophysical signal"
                if passed
                else "excessive, suggesting systematics rather than a coherent signal"
            )
        ),
        {
            "n_events": len(depths),
            "observed_scatter_ppm": observed_scatter * 1e6,
            "null_median_ppm": null_median * 1e6,
            "null_sigma_ppm": null_sigma * 1e6,
            "z": z,
            "threshold": threshold,
            "n_null_samples": len(null),
        },
    )


def check_centroid_shift(lc: LightCurve, candidate: Candidate) -> VettingTest:
    """Test whether the flux dip comes from the target rather than a blended neighbour.

    Always reports NOT RUN from a light curve alone, and that is the correct answer rather than
    a limitation to apologise for. Detecting a blended background eclipsing binary requires
    pixel-level data: the photometric centroid shifts towards the true source during an event,
    and difference imaging localises it. Neither is available here.

    Recording this as NOT RUN rather than omitting it keeps the gap visible in every report.
    A vetting summary that silently lacks the blend test reads as though blending was excluded.
    """
    return VettingTest(
        "centroid_shift",
        None,
        "requires target pixel files or difference imaging; a light curve cannot localise the "
        "source of a dip, so a blended background eclipsing binary is NOT excluded by this "
        "vetting run",
        {"requires": "target pixel data"},
    )


def check_known_ephemeris(
    candidate: Candidate,
    *,
    target: str | None = None,
    catalogue: dict[str, list[dict[str, Any]]] | None = None,
    tolerance_fraction: float = 0.01,
    catalogue_is_authoritative: bool = False,
) -> VettingTest:
    """Match against known ephemerides before any novelty is even considered.

    The most likely explanation for any new detection is that it is already known. This test
    "fails" a candidate that matches a catalogued object -- which is a *good* outcome, because
    recovering a known planet is how you demonstrate the pipeline works.

    Harmonics are checked too: a search can lock onto twice or half the true period, and such a
    match is still a match, not a discovery.

    Parameters
    ----------
    catalogue_is_authoritative
        Whether the catalogue is a real, complete cross-match (NASA Exoplanet Archive, VSX,
        SIMBAD). When False -- the default -- a *non*-match is reported as inconclusive rather
        than as evidence of novelty, because absence from a three-entry local table means
        nothing at all.
    """
    catalogue = catalogue if catalogue is not None else KNOWN_EPHEMERIDES
    entries = catalogue.get(target or "", []) if target else []

    period = candidate.period.to(u.day).value
    for entry in entries:
        known = float(entry["period_days"])
        for factor, label in ((1.0, "match"), (2.0, "2x harmonic"), (0.5, "1/2 harmonic")):
            if abs(period - known * factor) / (known * factor) < tolerance_fraction:
                return VettingTest(
                    "known_ephemeris",
                    False,
                    f"matches known object {entry['name']} "
                    f"(P={known} d, {label}) from {entry['source']}. This is a recovery, "
                    f"not a discovery.",
                    {
                        "matched": entry["name"],
                        "known_period": known,
                        "relation": label,
                        "source": entry["source"],
                    },
                )

    if not catalogue_is_authoritative:
        return VettingTest(
            "known_ephemeris",
            None,
            "no match in the local ephemeris table, but that table covers a handful of systems "
            "and is not a catalogue. A novelty claim requires cross-matching the NASA Exoplanet "
            "Archive, VSX, SIMBAD, and the relevant survey catalogues, none of which were "
            "reachable. Novelty is NOT established.",
            {"n_entries_checked": len(entries), "authoritative": False},
        )

    return VettingTest(
        "known_ephemeris",
        True,
        "no match against the catalogues checked",
        {"n_entries_checked": len(entries), "authoritative": True},
    )


# -- the gauntlet -----------------------------------------------------------------------


def vet_candidate(
    lc: LightCurve,
    candidate: Candidate,
    *,
    target: str | None = None,
    catalogue: dict[str, list[dict[str, Any]]] | None = None,
    catalogue_is_authoritative: bool = False,
) -> VettingReport:
    """Run the full gauntlet on one candidate.

    Parameters
    ----------
    lc
        The detrended light curve the candidate was found in.
    candidate
        The candidate to test.
    target
        Target name, for the ephemeris cross-match.

    Returns
    -------
    VettingReport
        Whose :attr:`~VettingReport.disposition` is never better than
        ``PLANET_CANDIDATE``. This tool does not confirm planets.
    """
    report = VettingReport(candidate=candidate)
    report.quality.extend(lc.quality)
    report.quality.extend(candidate.quality)

    report.add(check_odd_even(lc, candidate))
    report.add(check_secondary_eclipse(lc, candidate))
    report.add(check_transit_consistency(lc, candidate))
    report.add(check_centroid_shift(lc, candidate))
    report.add(
        check_known_ephemeris(
            candidate,
            target=target,
            catalogue=catalogue,
            catalogue_is_authoritative=catalogue_is_authoritative,
        )
    )

    if report.failures:
        report.quality.add(
            "vetting_failure",
            Severity.UNRELIABLE,
            "Candidate failed "
            + ", ".join(t.name for t in report.failures)
            + ". It is not a planet candidate.",
            failed_tests=[t.name for t in report.failures],
        )
    if report.not_run:
        report.quality.add(
            "incomplete_vetting",
            Severity.CAUTION,
            "Vetting is incomplete: "
            + ", ".join(t.name for t in report.not_run)
            + " could not be run, so the corresponding false-positive scenarios are not "
            "excluded.",
            not_run=[t.name for t in report.not_run],
        )

    log.info(
        "science.vetting.done",
        period=round(float(candidate.period.value), 5),
        disposition=report.disposition,
        n_failed=len(report.failures),
        n_not_run=len(report.not_run),
    )
    return report
