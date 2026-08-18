"""The false-positive gauntlet.

A vetting module that passes everything is broken, so these tests do two things: confirm real
planets survive, and confirm injected impostors do not.

Synthetic eclipsing-binary signals are injected into the *real* K2 light curve to build the
impostors. That is injection-recovery, explicitly permitted by ADR-0004: the injection is the
point of the measurement, it goes into real noise, and no result here is presented as an
observation.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from astrolab.core.lightcurve import LightCurve
from astrolab.instruments.k2 import load_validation_lightcurve
from astrolab.science.exoplanets.detrend import detrend
from astrolab.science.exoplanets.search import Candidate
from astrolab.science.exoplanets.vetting import (
    KNOWN_EPHEMERIDES,
    check_centroid_shift,
    check_known_ephemeris,
    check_odd_even,
    check_secondary_eclipse,
    check_transit_consistency,
    vet_candidate,
)

PERIOD = 6.0
EPOCH = 1980.0
DURATION = 0.12  # d
DEPTH = 0.004


@pytest.fixture(scope="module")
def base_lc() -> LightCurve:
    """Real K2 photometry, detrended, with the real transits left in place."""
    lc = load_validation_lightcurve("raw")
    return detrend(lc, expected_duration=2.4 * u.hour).lightcurve


def candidate(period: float = PERIOD, depth: float = DEPTH) -> Candidate:
    return Candidate(
        period=period * u.day,
        epoch=EPOCH * u.day,
        duration=DURATION * u.day,
        depth=depth,
        sde=20.0,
        method="test",
        n_transits=int(80.0 / period),
    )


def inject(lc: LightCurve, *, odd_depth: float, even_depth: float, secondary: float = 0.0):
    """Inject a signal whose odd and even events may differ -- the EB signature."""
    time = lc.time.value
    flux = lc.flux.value.copy()
    cycle = np.round((time - EPOCH) / PERIOD).astype(int)
    phase = (time - EPOCH) - cycle * PERIOD

    in_transit = np.abs(phase) < 0.5 * DURATION
    depths = np.where(np.abs(cycle) % 2 == 0, even_depth, odd_depth)
    flux[in_transit] -= depths[in_transit]

    if secondary > 0:
        at_secondary = np.abs(np.abs(phase) - 0.5 * PERIOD) < 0.5 * DURATION
        flux[at_secondary] -= secondary

    return LightCurve(
        time=lc.time,
        flux=flux * u.dimensionless_unscaled,
        flux_err=lc.flux_err,
        epoch_ref=lc.epoch_ref,
        meta=dict(lc.meta),
    )


class TestOddEven:
    def test_passes_a_planet_like_signal(self, base_lc: LightCurve) -> None:
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        test = check_odd_even(lc, candidate())
        assert test.passed is True, test.detail

    def test_catches_an_eclipsing_binary(self, base_lc: LightCurve) -> None:
        """Alternating depths mean the true period is twice the detected one."""
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH * 0.5)
        test = check_odd_even(lc, candidate())
        assert test.passed is False
        assert "eclipsing binary" in test.detail
        assert test.metrics["sigma"] > 3.0

    def test_reports_not_run_when_it_cannot_test(self, base_lc: LightCurve) -> None:
        """A period longer than the baseline leaves one parity empty: NOT RUN, not PASS.

        The distinction matters. With no odd-numbered events there is nothing to compare, and
        reporting PASS would record an eclipsing-binary scenario as excluded when it was never
        tested.
        """
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        test = check_odd_even(lc, candidate(period=200.0))
        assert test.passed is None
        assert min(test.metrics["n_even"], test.metrics["n_odd"]) < 3


class TestSecondaryEclipse:
    def test_passes_without_a_secondary(self, base_lc: LightCurve) -> None:
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        assert check_secondary_eclipse(lc, candidate()).passed is True

    def test_catches_a_stellar_secondary(self, base_lc: LightCurve) -> None:
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH, secondary=DEPTH * 0.4)
        test = check_secondary_eclipse(lc, candidate())
        assert test.passed is False
        assert "stellar companion" in test.detail

    def test_states_its_circular_orbit_assumption(self, base_lc: LightCurve) -> None:
        """A non-detection does not exclude an eccentric EB, and the output must say so."""
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        assert "circular orbit" in check_secondary_eclipse(lc, candidate()).detail


class TestTransitConsistency:
    def test_passes_a_repeating_signal(self, base_lc: LightCurve) -> None:
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        test = check_transit_consistency(lc, candidate())
        assert test.passed is True, test.detail

    def test_box_injection_yields_a_noise_limited_null(self, base_lc: LightCurve) -> None:
        """A box transit is the degenerate case, and it is worth pinning down.

        Every point in a box's core sits at the same depth, so every event measures the same
        value and the event-to-event scatter is pure photometric noise -- independent of how
        deep the box is. Both the observed scatter and the injected null therefore collapse to
        the noise level, and the test passes trivially.

        That is correct behaviour, but it also means a box injection *cannot* exercise the
        sampling-phase effect this test exists to calibrate: that effect needs a shaped,
        limb-darkened transit. The real regression guard lives in
        ``test_transit_pipeline.py``, where the check runs against the genuine K2-3 b transits.
        """
        shallow = check_transit_consistency(
            inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH), candidate(depth=DEPTH)
        )
        deep = check_transit_consistency(
            inject(base_lc, odd_depth=4 * DEPTH, even_depth=4 * DEPTH),
            candidate(depth=4 * DEPTH),
        )
        assert shallow.passed is True and deep.passed is True
        assert deep.metrics["null_median_ppm"] == pytest.approx(
            shallow.metrics["null_median_ppm"], rel=0.05
        )

    def test_reports_not_run_with_too_few_events(self, base_lc: LightCurve) -> None:
        test = check_transit_consistency(base_lc, candidate(period=45.0))
        assert test.passed is None


class TestCentroidShift:
    def test_always_reports_not_run_from_a_light_curve(self, base_lc: LightCurve) -> None:
        """Recording NOT RUN keeps the gap visible; omitting it would imply blends were
        excluded."""
        test = check_centroid_shift(base_lc, candidate())
        assert test.passed is None
        assert "NOT excluded" in test.detail


class TestKnownEphemeris:
    def test_matches_a_known_planet(self) -> None:
        test = check_known_ephemeris(candidate(period=10.05403), target="K2-3")
        assert test.passed is False
        assert test.metrics["matched"] == "K2-3 b"
        assert "recovery, not a discovery" in test.detail

    def test_matches_a_half_harmonic(self) -> None:
        """The search can lock onto half the true period; that is still a known object."""
        test = check_known_ephemeris(candidate(period=44.5565 / 2), target="K2-3")
        assert test.passed is False
        assert test.metrics["relation"] == "1/2 harmonic"

    def test_non_match_is_inconclusive_not_novel(self) -> None:
        """Absence from a three-entry table is not evidence of novelty."""
        test = check_known_ephemeris(candidate(period=3.14159), target="K2-3")
        assert test.passed is None
        assert "Novelty is NOT established" in test.detail

    def test_authoritative_catalogue_may_report_a_clean_pass(self) -> None:
        test = check_known_ephemeris(
            candidate(period=3.14159), target="K2-3", catalogue_is_authoritative=True
        )
        assert test.passed is True

    def test_unknown_target_still_refuses_to_claim_novelty(self) -> None:
        assert check_known_ephemeris(candidate(), target="NOT-IN-TABLE").passed is None

    def test_catalogue_entries_carry_sources(self) -> None:
        for entries in KNOWN_EPHEMERIDES.values():
            for entry in entries:
                assert entry["source"]


class TestDisposition:
    def test_known_object_outranks_test_failures(self, base_lc: LightCurve) -> None:
        """Recovering a known planet is success, not a false positive."""
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        report = vet_candidate(lc, candidate(period=10.05403), target="K2-3")
        assert report.disposition == "KNOWN_OBJECT"

    def test_failed_astrophysical_test_gives_false_positive(self, base_lc: LightCurve) -> None:
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH * 0.4, secondary=DEPTH * 0.4)
        report = vet_candidate(lc, candidate(), target="NOT-IN-TABLE")
        assert report.disposition == "FALSE_POSITIVE"
        assert len(report.failures) >= 1

    def test_never_returns_a_confirmed_planet(self, base_lc: LightCurve) -> None:
        """The tool produces candidates. A human confirms planets."""
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        report = vet_candidate(lc, candidate(), target="NOT-IN-TABLE")
        assert "CONFIRMED" not in report.disposition
        assert report.disposition.startswith("PLANET_CANDIDATE")

    def test_incomplete_vetting_is_marked(self, base_lc: LightCurve) -> None:
        """Centroid vetting can never run here, so no candidate gets a clean bill of health."""
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        report = vet_candidate(lc, candidate(), target="NOT-IN-TABLE")
        assert report.disposition == "PLANET_CANDIDATE_INCOMPLETE_VETTING"
        assert report.quality.has("incomplete_vetting")

    def test_report_serialises_and_describes(self, base_lc: LightCurve) -> None:
        lc = inject(base_lc, odd_depth=DEPTH, even_depth=DEPTH)
        report = vet_candidate(lc, candidate(), target="K2-3")
        summary = report.summary()
        assert summary["disposition"] == report.disposition
        assert len(summary["tests"]) == 5
        assert "Disposition:" in report.describe()

    def test_upstream_quality_flags_propagate(self, base_lc: LightCurve) -> None:
        report = vet_candidate(base_lc, candidate(), target="K2-3")
        assert report.quality.has("third_party_mirror")
