"""Transit fitting: model correctness, posterior behaviour, and the fitted-period benchmark.

The full nested-sampling fit is marked ``slow`` (about a minute). The model and
parameterisation tests are fast and run always.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from astrolab.instruments.k2 import load_validation_lightcurve
from astrolab.science.exoplanets.detrend import detrend
from astrolab.science.exoplanets.fit import (
    TransitPriors,
    _q_to_u,
    fit_transit,
    transit_model,
)
from astrolab.science.exoplanets.search import Candidate
from astrolab.validation.targets import GOLDEN_TARGETS

TRANSIT_DURATION = 2.4 * u.hour


class TestLimbDarkeningParameterisation:
    def test_kipping_maps_into_the_physical_region(self) -> None:
        """Uniform q1,q2 must give limb-darkening laws that are physically allowed.

        The constraints are u1+u2 < 1 (positive intensity at the limb) and u1 > 0
        (intensity decreasing outward). A square prior on u1,u2 would violate both.
        """
        rng = np.random.default_rng(0)
        q1, q2 = rng.uniform(0, 1, 5000), rng.uniform(0, 1, 5000)
        u1 = 2.0 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1.0 - 2.0 * q2)
        assert np.all(u1 + u2 <= 1.0 + 1e-12)
        assert np.all(u1 >= -1e-12)

    def test_scalar_conversion_matches_the_published_transformation(self) -> None:
        u1, u2 = _q_to_u(0.49, 0.25)
        assert u1 == pytest.approx(2 * 0.7 * 0.25)
        assert u2 == pytest.approx(0.7 * (1 - 0.5))


class TestTransitModel:
    def test_depth_exceeds_the_geometric_value_from_limb_darkening(self) -> None:
        """A limb-darkened star is brighter at the centre, so a central transit is deeper."""
        t = np.linspace(-0.2, 0.2, 400)
        f = transit_model(
            t,
            t0=0.0,
            period=10.0,
            rp=0.035,
            a_rs=30.0,
            b=0.0,
            u1=0.4,
            u2=0.2,
            exposure_days=0.0,
            supersample=1,
        )
        measured = 1.0 - f.min()
        assert measured > 0.035**2
        assert measured < 2.0 * 0.035**2

    def test_no_transit_when_impact_parameter_is_too_large(self) -> None:
        t = np.linspace(-0.2, 0.2, 200)
        f = transit_model(
            t,
            t0=0.0,
            period=10.0,
            rp=0.035,
            a_rs=30.0,
            b=1.5,
            u1=0.4,
            u2=0.2,
            exposure_days=0.0,
            supersample=1,
        )
        assert f.min() == pytest.approx(1.0, abs=1e-9)

    def test_long_exposure_smears_the_transit(self) -> None:
        """The reason supersampling is not optional at 29-minute cadence.

        Integrating over a long exposure rounds the profile and makes the minimum shallower;
        ignoring it biases duration and impact parameter.
        """
        t = np.linspace(-0.2, 0.2, 200)
        kw = dict(t0=0.0, period=10.0, rp=0.035, a_rs=30.0, b=0.0, u1=0.4, u2=0.2)
        sharp = transit_model(t, exposure_days=0.0, supersample=1, **kw)
        smeared = transit_model(t, exposure_days=0.0204, supersample=7, **kw)
        assert smeared.min() > sharp.min()

    def test_deeper_planet_gives_deeper_transit(self) -> None:
        t = np.linspace(-0.2, 0.2, 200)
        kw = dict(
            t0=0.0, period=10.0, a_rs=30.0, b=0.0, u1=0.4, u2=0.2, exposure_days=0.0, supersample=1
        )
        assert transit_model(t, rp=0.07, **kw).min() < transit_model(t, rp=0.035, **kw).min()


class TestPriors:
    def test_serialises_for_the_manifest(self) -> None:
        d = TransitPriors().to_dict()
        assert set(d) >= {"rp", "a_rs", "b", "q1", "q2", "log_jitter"}
        assert d["rp"] == [0.005, 0.3]


class TestFitValidation:
    def test_refuses_when_too_few_points_near_transit(self) -> None:
        lc = detrend(
            load_validation_lightcurve("raw"), expected_duration=TRANSIT_DURATION
        ).lightcurve
        bad = Candidate(
            period=10.05 * u.day,
            epoch=1980.4 * u.day,
            duration=1e-5 * u.day,
            depth=0.001,
            sde=20.0,
            method="test",
            n_transits=8,
        )
        with pytest.raises(ValueError, match="too few to fit"):
            fit_transit(lc, bad)


@pytest.mark.slow
class TestFittedGoldenTarget:
    """Fit K2-3 b and compare against the published period."""

    @pytest.fixture(scope="class")
    def fit(self):
        lc = load_validation_lightcurve("raw")
        det = detrend(lc, expected_duration=TRANSIT_DURATION).lightcurve
        candidate = Candidate(
            period=10.052832 * u.day,
            epoch=1980.4270 * u.day,
            duration=0.1069 * u.day,
            depth=0.0014,
            sde=15.8,
            method="TLS",
            n_transits=8,
        )
        return fit_transit(det, candidate, n_live=300, seed=20260818)

    def test_period_matches_the_published_value(self, fit) -> None:
        golden = GOLDEN_TARGETS["K2-3"].values["fitted_period_b"]
        measured = float(fit.period.value.value)
        passed, _ = golden.check(measured)
        assert passed, golden.describe(measured)

    def test_radius_ratio_is_physically_sensible(self, fit) -> None:
        """K2-3 b is a ~2 Earth-radius planet on an M dwarf: Rp/R* of a few percent."""
        rp = float(fit.radius_ratio.value.value)
        assert 0.02 < rp < 0.05

    def test_duration_matches_the_search(self, fit) -> None:
        """An independent check: the fitted T14 should agree with the periodogram duration."""
        assert float(fit.duration.to(u.hour).value.value) == pytest.approx(2.5, abs=0.5)

    def test_evidence_decisively_favours_a_transit(self, fit) -> None:
        assert fit.log_bayes_factor > 10.0

    def test_recovers_the_radius_ratio_impact_parameter_degeneracy(self, fit) -> None:
        """The correlation ADR-0002 exists to preserve, measured on real data.

        A larger planet crossing near the limb mimics a smaller one crossing the centre, so
        these two parameters trade off. If this correlation vanished, the sampler would not be
        exploring the degeneracy and the marginal error bars would be too small.
        """
        assert fit.posterior.correlation("rp", "b") > 0.3

    def test_depth_is_derived_within_the_joint_posterior(self, fit) -> None:
        """Depth must carry the posterior identity, so it can be combined with its siblings."""
        assert fit.depth.posterior_id == fit.posterior.posterior_id
        assert float(fit.depth.value.value) == pytest.approx(
            float(fit.radius_ratio.value.value) ** 2, rel=0.05
        )

    def test_every_reported_quantity_has_an_uncertainty(self, fit) -> None:
        """Prime directive 2, checked mechanically."""
        for measurement in (fit.period, fit.radius_ratio, fit.depth, fit.impact_parameter):
            assert measurement.samples.size > 100
            assert float(measurement.uncertainty.value) > 0.0

    def test_upstream_quality_flags_reach_the_fit(self, fit) -> None:
        assert fit.quality.has("third_party_mirror")
        assert fit.quality.has("estimated_uncertainties")

    def test_summary_records_priors_and_seed(self, fit) -> None:
        """A fit that cannot be reproduced from its own record is not a result."""
        s = fit.summary()
        assert s["seed"] == 20260818
        assert s["priors"]["rp"] == [0.005, 0.3]
        assert s["metadata"]["sampler_method"] == "rwalk"
        assert "circular" in s["metadata"]["eccentricity"]


class TestGoldenTargetMetadata:
    def test_every_value_carries_a_citation_and_a_rationale(self) -> None:
        for target in GOLDEN_TARGETS.values():
            for value in target.values.values():
                assert value.doi and value.source and value.last_verified
                assert len(value.tolerance_rationale) > 50

    def test_unverified_values_are_labelled_as_such(self) -> None:
        """A number transcribed from memory must never masquerade as literature agreement."""
        for target in GOLDEN_TARGETS.values():
            for value in target.values.values():
                assert value.verification in {"sourced", "unverified"}
                if value.verification == "unverified":
                    assert value.note, f"{value.name} is unverified but explains nothing"
                    assert "UNVERIFIED" in value.describe(value.expected)
