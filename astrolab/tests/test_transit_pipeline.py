"""End-to-end transit pipeline on real K2 photometry, including the golden-target benchmark.

These tests run on genuine K2 Campaign 1 observations of K2-3 (EPIC 201367065). See
``src/astrolab/validation/data/SOURCE.md`` for the provenance and its limitation: this is a
*regression* benchmark against a fixed real input, not a validated one, because the file's
chain of custody does not reach MAST.

The full TLS search is marked ``slow``. Everything else runs in seconds.
"""

from __future__ import annotations

import pytest
from astropy import units as u

from astrolab.core.quality import Severity
from astrolab.instruments.k2 import THIRD_PARTY_MIRROR, load_validation_lightcurve
from astrolab.science.exoplanets.detrend import detrend, estimate_depth_bias
from astrolab.science.exoplanets.search import search_transits
from astrolab.validation.targets import GOLDEN_TARGETS

K2_3_B_PERIOD = 10.05403  # d, Crossfield et al. 2015
TRANSIT_DURATION = 2.4 * u.hour


@pytest.fixture(scope="module")
def raw_lc():
    return load_validation_lightcurve("raw")


@pytest.fixture(scope="module")
def detrended_lc(raw_lc):
    return detrend(raw_lc, expected_duration=TRANSIT_DURATION).lightcurve


class TestIngestion:
    def test_matches_k2_campaign_1(self, raw_lc) -> None:
        """Independent checks that this is what it claims to be."""
        assert len(raw_lc) == 3632
        assert raw_lc.cadence.to(u.min).value == pytest.approx(29.4, abs=0.1)
        assert raw_lc.baseline.value == pytest.approx(80.07, abs=0.1)
        assert raw_lc.absolute_time()[0].iso.startswith("2014-06-01")

    def test_untrusted_provenance_is_flagged(self, raw_lc) -> None:
        """The weak chain of custody must travel with the data, not sit in a README."""
        assert raw_lc.quality.has(THIRD_PARTY_MIRROR)
        flag = next(f for f in raw_lc.quality.flags if f.flag == THIRD_PARTY_MIRROR)
        assert flag.severity is Severity.CAUTION

    def test_long_cadence_is_flagged(self, raw_lc) -> None:
        assert raw_lc.quality.has("undersampled_cadence")

    def test_estimated_uncertainties_are_flagged(self, raw_lc) -> None:
        """An estimated error bar is not a measured one, and a fit must know the difference."""
        assert raw_lc.quality.has("estimated_uncertainties")

    def test_checksum_recorded(self, raw_lc) -> None:
        assert len(raw_lc.provenance["source"]["sha256"]) == 64

    def test_detrended_variant_available_for_cross_check(self) -> None:
        assert len(load_validation_lightcurve("detrended")) == 3632

    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(ValueError, match="raw' or 'detrended"):
            load_validation_lightcurve("whatever")


class TestDetrending:
    def test_removes_stellar_variability(self, raw_lc, detrended_lc) -> None:
        """3000 ppm of real stellar variability down to tens of ppm."""
        assert raw_lc.scatter.value * 1e6 > 1000
        assert detrended_lc.scatter.value * 1e6 < 100

    def test_agrees_with_an_independent_detrending(self, detrended_lc) -> None:
        """Cross-check against the upstream author's independently detrended series.

        Not a tight equality: different methods legitimately differ. But an order-of-magnitude
        disagreement would mean one of the two is wrong.
        """
        upstream = load_validation_lightcurve("detrended")
        ours = detrended_lc.scatter.value
        theirs = upstream.scatter.value
        assert 0.5 < ours / theirs < 2.0

    def test_preserves_transit_depth_at_the_default_window(self, detrended_lc) -> None:
        """ADR-0003's 3-durations default, measured rather than assumed."""
        recovered = estimate_depth_bias(
            detrended_lc,
            period=K2_3_B_PERIOD * u.day,
            epoch=1983.0 * u.day,
            duration=TRANSIT_DURATION,
            depth=0.00122,
            window_durations=3.0,
        )
        assert recovered == pytest.approx(1.0, abs=0.05)

    def test_short_window_destroys_the_transit(self, detrended_lc) -> None:
        """The failure mode the default exists to avoid, demonstrated rather than asserted."""
        recovered = estimate_depth_bias(
            detrended_lc,
            period=K2_3_B_PERIOD * u.day,
            epoch=1983.0 * u.day,
            duration=TRANSIT_DURATION,
            depth=0.00122,
            window_durations=1.0,
        )
        assert recovered < 0.5

    def test_aggressive_window_is_flagged(self, raw_lc) -> None:
        result = detrend(raw_lc, expected_duration=TRANSIT_DURATION, window_durations=1.5)
        assert result.lightcurve.quality.has("aggressive_detrending")

    def test_window_longer_than_baseline_rejected(self, raw_lc) -> None:
        with pytest.raises(ValueError, match="no trend to fit"):
            detrend(raw_lc, expected_duration=100 * u.day)

    def test_requires_positive_duration(self, raw_lc) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            detrend(raw_lc, expected_duration=-1 * u.hour)

    def test_records_its_configuration_in_provenance(self, detrended_lc) -> None:
        step = detrended_lc.provenance["detrend"]
        assert step["method"] == "biweight"
        assert step["window_durations"] == 3.0
        assert step["scatter_before_ppm"] > step["scatter_after_ppm"]


@pytest.mark.slow
class TestGoldenTargetSearch:
    """The Phase 2 benchmark: recover published planets from real photometry."""

    @pytest.fixture(scope="class")
    def search(self, detrended_lc):
        return search_transits(
            detrended_lc,
            min_period=2.0 * u.day,
            max_period=30.0 * u.day,
            max_candidates=3,
        )

    def test_recovers_k2_3_b(self, search) -> None:
        golden = GOLDEN_TARGETS["K2-3"].values["period_b"]
        periods = [c.period.value for c in search.candidates]
        best = min(periods, key=lambda p: abs(p - golden.expected))
        passed, _ = golden.check(best)
        assert passed, golden.describe(best)

    def test_recovers_k2_3_c(self, search) -> None:
        golden = GOLDEN_TARGETS["K2-3"].values["period_c"]
        periods = [c.period.value for c in search.candidates]
        best = min(periods, key=lambda p: abs(p - golden.expected))
        passed, _ = golden.check(best)
        assert passed, golden.describe(best)

    def test_strongest_candidate_is_the_strongest_planet(self, search) -> None:
        assert search.best is not None
        # Tolerance is grid-limited, as for the golden values; see validation/targets.py.
        assert search.best.period.value == pytest.approx(K2_3_B_PERIOD, abs=0.005)
        assert search.best.sde > 20

    def test_bls_independently_agrees_on_the_period(self, search) -> None:
        """Two different templates agreeing is real evidence; it is checked, not assumed."""
        assert search.best is not None
        assert search.best.extras["bls_agrees"] is True

    def test_reports_the_trial_factor(self, search) -> None:
        """A 3-sigma peak in a 10^4-trial search is not 3 sigma, and the output says so."""
        assert search.n_trials > 1000
        assert "look-elsewhere" in search.significance_note()

    def test_marginal_candidates_are_flagged_not_promoted(self, search) -> None:
        """The pipeline emits candidates, never discoveries.

        The third detection in this system has no published counterpart and rests on two
        events; it must carry a reservation rather than appear as a clean result.
        """
        unpublished = [
            c
            for c in search.candidates
            if abs(c.period.value - 10.05403) > 0.01 and abs(c.period.value - 24.6454) > 0.01
        ]
        assert all(len(c.quality) > 0 for c in unpublished), (
            "a candidate with no published counterpart carried no quality reservation"
        )

    def test_real_planets_survive_the_depth_consistency_check(self, search, detrended_lc) -> None:
        """Regression guard: genuine planets must not be flagged as systematics.

        Earlier versions of check_transit_consistency failed K2-3 b at +9 sigma because the
        null contained only photometric noise, while the real events carry a limb-darkened
        transit whose sampling-phase variation at 29-minute cadence is several times larger.
        This is the test that catches that class of mistake, because only a real shaped transit
        exercises it -- a box injection cannot.
        """
        from astrolab.science.exoplanets.vetting import check_transit_consistency

        assert search.best is not None
        result = check_transit_consistency(detrended_lc, search.best)
        assert result.passed is True, result.detail
        # The null must sit well above a noise-only estimate, or it has stopped containing
        # the signal.
        assert result.metrics["null_median_ppm"] > detrended_lc.scatter.value * 1e6 * 0.5

    def test_recovered_planets_are_labelled_known_objects(self, search, detrended_lc) -> None:
        """Recovering a catalogued planet is success, and must not read as a false positive."""
        from astrolab.science.exoplanets.vetting import vet_candidate

        assert search.best is not None
        report = vet_candidate(detrended_lc, search.best, target="K2-3")
        assert report.disposition == "KNOWN_OBJECT"

    def test_quality_flags_propagate_from_ingestion(self, search) -> None:
        """The provenance caveat must reach the search result, not stop at the light curve."""
        assert search.quality.has(THIRD_PARTY_MIRROR)


class TestSearchValidation:
    def test_rejects_inverted_period_range(self, detrended_lc) -> None:
        with pytest.raises(ValueError, match="must be less than"):
            search_transits(detrended_lc, min_period=30 * u.day, max_period=2 * u.day)


class TestGoldenTargetMetadata:
    def test_every_value_carries_a_citation_and_a_rationale(self) -> None:
        """A tolerance chosen to make a test pass certifies nothing."""
        for target in GOLDEN_TARGETS.values():
            for value in target.values.values():
                assert value.doi
                assert value.source
                assert value.last_verified
                assert len(value.tolerance_rationale) > 50

    def test_provenance_caveat_is_recorded(self) -> None:
        assert "not a validated one" in GOLDEN_TARGETS["K2-3"].provenance_caveat

    def test_check_reports_deviation(self) -> None:
        golden = GOLDEN_TARGETS["K2-3"].values["period_b"]
        passed, dev = golden.check(golden.expected + 0.5 * golden.tolerance)
        assert passed and dev == pytest.approx(0.5 * golden.tolerance)
        passed, _ = golden.check(golden.expected + 2 * golden.tolerance)
        assert not passed
