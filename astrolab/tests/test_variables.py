"""Variable-star period finding, alias resolution, and features, on real LINEAR photometry.

The data are 280 points over 5.4 years at a 7-day mean spacing -- ground-based, sparse, and
heavily aliased. That is the regime these tests exist to cover, and it is deliberately unlike
the uniform space-based sampling of the transit benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from astrolab.core.lightcurve import LightCurve
from astrolab.core.units import UnitBoundaryError
from astrolab.instruments.linear import load_validation_variable
from astrolab.science.variables.features import extract_features
from astrolab.science.variables.period import (
    find_period,
    harmonic_fit,
    lomb_scargle,
    odd_harmonic_fraction,
    phase_dispersion,
    window_function,
)
from astrolab.validation.targets import GOLDEN_TARGETS


@pytest.fixture(scope="module")
def rrl() -> LightCurve:
    return load_validation_variable(11375941)


@pytest.fixture(scope="module")
def binary() -> LightCurve:
    return load_validation_variable(14752041)


def synthetic(
    period: float,
    *,
    n: int = 400,
    amplitude: float = 0.3,
    seed: int = 0,
    doubled: bool = False,
    span: float = 200.0,
) -> LightCurve:
    """A synthetic light curve, used only to test that the algorithms recover a known input.

    Labelled synthetic throughout and never presented as an observation; this is the
    recover-a-known-answer case that ADR-0004 permits.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    phase = 2 * np.pi * t / period
    signal = amplitude * np.sin(phase)
    if doubled:
        # Alternating deep/shallow minima: the eclipsing-binary signature that makes the
        # doubled period the true one.
        signal += 0.5 * amplitude * np.sin(phase / 2.0)
    err = np.full(n, 0.01)
    return LightCurve(
        time=t * u.day,
        flux=(15.0 + signal + rng.normal(0, 0.01, n)) * u.mag,
        flux_err=err * u.mag,
        epoch_ref=Time(0.0, format="mjd", scale="utc"),
        meta={"target": "synthetic", "mission": "simulated"},
    )


class TestIngestion:
    def test_reads_real_linear_photometry(self, rrl: LightCurve) -> None:
        assert len(rrl) == 280
        assert rrl.is_magnitude
        assert rrl.baseline.value == pytest.approx(1961.8, abs=1.0)
        assert rrl.absolute_time()[0].iso.startswith("2003-01")

    def test_flags_sparse_sampling(self, rrl: LightCurve) -> None:
        """Sparse ground-based sampling makes aliasing a hazard; the data must say so."""
        assert rrl.quality.has("sparse_sampling")
        assert rrl.quality.has("third_party_mirror")

    def test_uses_measured_uncertainties(self, rrl: LightCurve) -> None:
        """Unlike the K2 benchmark, LINEAR publishes real errors, so none are estimated."""
        assert not rrl.quality.has("estimated_uncertainties")
        assert np.all(rrl.flux_err.value > 0)

    def test_unknown_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="no bundled LINEAR"):
            load_validation_variable(999)


class TestMagnitudeHandling:
    def test_magnitudes_convert_to_relative_flux(self, rrl: LightCurve) -> None:
        flux = rrl.to_relative_flux()
        assert not flux.is_magnitude
        # Not exact: with an even number of points the median is the mean of the two central
        # magnitudes, and a nonlinear transform of a mean is not the mean of the transform.
        assert float(np.median(flux.flux.value)) == pytest.approx(1.0, abs=1e-5)

    def test_conversion_propagates_errors_by_slope(self, rrl: LightCurve) -> None:
        """An error bar is a local slope, not a value on the magnitude scale."""
        flux = rrl.to_relative_flux()
        expected = flux.flux.value * 0.4 * np.log(10.0) * rrl.flux_err.value
        assert np.allclose(flux.flux_err.value, expected)

    def test_normalise_refuses_on_magnitudes(self, rrl: LightCurve) -> None:
        with pytest.raises(TypeError, match="logarithmic"):
            rrl.normalise()

    def test_rejects_photometry_in_other_units(self) -> None:
        with pytest.raises(UnitBoundaryError, match="normalised flux"):
            LightCurve(
                time=np.arange(10.0) * u.day,
                flux=np.ones(10) * u.Jy,
                flux_err=np.ones(10) * u.Jy,
                epoch_ref=Time(0.0, format="mjd"),
            )


class TestWindowFunction:
    def test_detects_the_one_day_alias_structure(self, rrl: LightCurve) -> None:
        """The central diagnostic for ground-based data: a night-time survey aliases at 1 c/d."""
        freq, power = window_function(rrl, min_period=0.05 * u.day, max_period=2.0 * u.day)
        near_one = np.abs(freq - 1.0) < 0.02
        assert near_one.any()
        assert power[near_one].max() > 0.5

    def test_window_flag_is_raised(self, rrl: LightCurve) -> None:
        result = find_period(rrl, min_period=0.05 * u.day, max_period=2.0 * u.day)
        assert result.quality.has("strong_window_alias")


class TestHarmonicFit:
    def test_recovers_a_known_amplitude(self) -> None:
        lc = synthetic(0.5, amplitude=0.3, seed=1)
        fit = harmonic_fit(lc, 0.5, n_terms=1)
        assert fit.amplitudes[0] == pytest.approx(0.3, rel=0.05)

    def test_wrong_period_fits_worse(self) -> None:
        """The property alias resolution rests on."""
        lc = synthetic(0.5, seed=2)
        good = harmonic_fit(lc, 0.5, n_terms=3)
        bad = harmonic_fit(lc, 0.37, n_terms=3)
        assert good.bic < bad.bic

    def test_rejects_invalid_inputs(self) -> None:
        lc = synthetic(0.5)
        with pytest.raises(ValueError, match="period must be positive"):
            harmonic_fit(lc, 0.0)
        with pytest.raises(ValueError, match="n_terms"):
            harmonic_fit(lc, 0.5, n_terms=0)


class TestPhaseDispersion:
    def test_correct_period_gives_low_theta(self) -> None:
        lc = synthetic(0.5, seed=3)
        assert phase_dispersion(lc, 0.5) < 0.2

    def test_wrong_period_gives_theta_near_one(self) -> None:
        lc = synthetic(0.5, seed=3)
        assert phase_dispersion(lc, 0.317) > 0.6


class TestPeriodDoubling:
    def test_odd_harmonic_fraction_is_small_for_a_pure_signal(self) -> None:
        """Folding a single-period signal at twice its period leaves odd harmonics empty."""
        lc = synthetic(0.5, seed=4)
        fit = harmonic_fit(lc, 1.0, n_terms=4)
        assert odd_harmonic_fraction(fit) < 0.05

    def test_does_not_double_a_simple_pulsator(self) -> None:
        """Regression guard for two related bugs, both of which doubled real periods.

        Fitting at 2P with the same harmonic count is strictly more flexible than fitting at P
        -- the even harmonics reproduce the P model and the odd ones absorb noise, at identical
        parameter count -- so BIC can never penalise doubling. Only the odd-harmonic test can.

        The period here is 0.5 d deliberately: at f = 2 cycles/day the offset alias
        ``f - 1 c/d`` *is* the doubled period, so doubling can re-enter through the alias route
        even once the direct test is in place. Every trial must be canonicalised, not just the
        winner.
        """
        lc = synthetic(0.5, seed=5)
        result = find_period(lc, min_period=0.1 * u.day, max_period=2.0 * u.day)
        assert result.best is not None
        assert float(result.best.period.value) == pytest.approx(0.5, rel=0.01)

    def test_doubles_when_alternating_cycles_differ(self) -> None:
        """The eclipsing-binary case, where the doubled period really is correct."""
        lc = synthetic(0.5, doubled=True, seed=6)
        fit = harmonic_fit(lc, 1.0, n_terms=4)
        assert odd_harmonic_fraction(fit) > 0.05


class TestGoldenPeriod:
    def test_recovers_the_published_period(self, rrl: LightCurve) -> None:
        golden = GOLDEN_TARGETS["LINEAR-11375941"].values["period"]
        result = find_period(rrl, min_period=0.05 * u.day, max_period=2.0 * u.day)
        assert result.best is not None
        measured = float(result.best.period.to(u.day).value)
        passed, _ = golden.check(measured)
        assert passed, golden.describe(measured)

    def test_detection_is_overwhelmingly_significant(self, rrl: LightCurve) -> None:
        result = find_period(rrl, min_period=0.05 * u.day, max_period=2.0 * u.day)
        assert result.best is not None
        assert result.best.fap < 1e-20

    def test_reports_the_trial_factor(self, rrl: LightCurve) -> None:
        """A raw power must not become a significance without the look-elsewhere correction."""
        result = find_period(rrl, min_period=0.05 * u.day, max_period=2.0 * u.day)
        assert result.n_independent_frequencies > 1000
        assert "independent frequencies" in result.significance_note()

    def test_aliases_are_enumerated_and_compared(self, rrl: LightCurve) -> None:
        result = find_period(rrl, min_period=0.05 * u.day, max_period=2.0 * u.day)
        assert result.best is not None
        relations = {a["relation"] for a in result.best.aliases}
        assert "periodogram peak" in relations
        assert any("1.000000 c/d" in r for r in relations)
        assert all("bic" in a and "pdm_theta" in a for a in result.best.aliases)

    def test_pulsator_shape_is_physical(self, rrl: LightCurve) -> None:
        """R21 below 1 means the fundamental dominates, as it should for a pulsator.

        R21 above 1 at the reported period would mean the second harmonic is stronger than the
        first, which is the signature of a period that has been wrongly doubled.
        """
        result = find_period(rrl, min_period=0.05 * u.day, max_period=2.0 * u.day)
        assert result.best is not None and result.best.harmonic is not None
        assert 0.0 < result.best.harmonic.r21 < 1.0


class TestEclipsingBinary:
    def test_identifies_alternating_minima_in_real_data(self, binary: LightCurve) -> None:
        """LINEAR 14752041: the doubling test fires on real data, for a stated physical reason."""
        result = find_period(binary, min_period=0.05 * u.day, max_period=2.0 * u.day)
        assert result.best is not None
        assert result.best.quality.has("period_canonicalised")
        assert result.best.odd_harmonic_fraction > 0.05
        # Two minima per cycle put most of the power in the second harmonic.
        assert result.best.harmonic is not None
        assert result.best.harmonic.r21 > 1.0


class TestFeatures:
    def test_separates_the_two_real_objects(self, rrl: LightCurve, binary: LightCurve) -> None:
        """Independent corroboration of the two classifications, from shape statistics alone.

        An eclipsing binary sits near maximum light most of the time and drops briefly, giving
        positive skew and few points beyond one sigma. A pulsator varies smoothly and gives the
        opposite.
        """
        a = extract_features(rrl)
        b = extract_features(binary)
        assert a.skew < 0 < b.skew
        assert b.beyond_1_std < a.beyond_1_std

    def test_detects_variability_against_the_error_bars(self, rrl: LightCurve) -> None:
        assert extract_features(rrl).reduced_chi2_constant > 5.0

    def test_records_the_photometric_system(self, rrl: LightCurve) -> None:
        """An amplitude in magnitudes is not an amplitude in flux; features must say which."""
        assert extract_features(rrl).to_dict()["photometry_system"] == "magnitude"
        flux_features = extract_features(rrl.to_relative_flux())
        assert flux_features.to_dict()["photometry_system"] == "relative_flux"

    def test_needs_enough_points(self) -> None:
        lc = synthetic(0.5, n=2, span=1.0)
        with pytest.raises(ValueError, match="at least 3 points"):
            extract_features(lc)


class TestLombScargleValidation:
    def test_rejects_inverted_period_range(self, rrl: LightCurve) -> None:
        with pytest.raises(ValueError, match="min_period < max_period"):
            lomb_scargle(rrl, min_period=2.0 * u.day, max_period=0.1 * u.day)
