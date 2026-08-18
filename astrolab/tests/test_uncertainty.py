"""Uncertainty representation: correlation preservation and the independence guard."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from astrolab.core.uncertainty import (
    IndependenceError,
    JointPosterior,
    Measurement,
    combine_independent,
    from_gaussian,
)


@pytest.fixture
def correlated_posterior(rng: np.random.Generator) -> JointPosterior:
    """A posterior with the radius-ratio / impact-parameter degeneracy built in.

    This degeneracy is real: a larger planet crossing near the stellar limb produces nearly
    the same light curve as a smaller planet crossing the centre, so the two parameters trade
    off against each other in any transit fit.
    """
    n = 4000
    b = rng.uniform(0.0, 0.9, n)
    ror = 0.097 + 0.004 * b + rng.normal(0.0, 0.0005, n)
    return JointPosterior(
        {
            "ror": u.Quantity(ror),
            "b": u.Quantity(b),
            "rstar": u.Quantity(rng.normal(1.24, 0.04, n), u.Rsun),
        }
    )


class TestMeasurementConstruction:
    def test_rejects_bare_arrays(self) -> None:
        with pytest.raises(TypeError, match="must be an astropy Quantity"):
            Measurement(samples=np.zeros(10), name="x")  # type: ignore[arg-type]

    def test_rejects_scalar(self) -> None:
        """A single number is not a measurement -- it has no uncertainty."""
        with pytest.raises(ValueError, match="1-D array"):
            Measurement(samples=u.Quantity(1.0), name="x")

    def test_rejects_too_few_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            Measurement(samples=u.Quantity([1.0]), name="x")

    def test_rejects_incompatible_systematic_units(self) -> None:
        with pytest.raises(u.UnitConversionError):
            Measurement(
                samples=u.Quantity(np.zeros(10), u.day),
                name="x",
                systematic=u.Quantity(np.zeros(10), u.m),
            )


class TestSummaries:
    def test_recovers_a_known_gaussian(self, rng: np.random.Generator) -> None:
        m = Measurement(samples=u.Quantity(rng.normal(5.0, 0.5, 200_000), u.day), name="p")
        assert m.value.value == pytest.approx(5.0, abs=0.01)
        assert m.uncertainty.value == pytest.approx(0.5, rel=0.02)

    def test_interval_is_equal_tailed(self, rng: np.random.Generator) -> None:
        m = Measurement(samples=u.Quantity(rng.normal(0.0, 1.0, 200_000)), name="x")
        lo, hi = m.interval(0.6827)
        assert lo.value == pytest.approx(-1.0, abs=0.02)
        assert hi.value == pytest.approx(1.0, abs=0.02)

    def test_asymmetry_detected(self, rng: np.random.Generator) -> None:
        """A boundary-limited posterior, like eccentricity piling up at zero."""
        skewed = np.abs(rng.normal(0.0, 0.1, 50_000))
        m = Measurement(samples=u.Quantity(skewed), name="ecc")
        assert m.asymmetry > 0.1

    def test_symmetric_posterior_has_low_asymmetry(self, rng: np.random.Generator) -> None:
        m = Measurement(samples=u.Quantity(rng.normal(0.0, 1.0, 50_000)), name="x")
        assert m.asymmetry < 0.02

    def test_invalid_credible_level_rejected(self, rng: np.random.Generator) -> None:
        m = Measurement(samples=u.Quantity(rng.normal(0, 1, 100)), name="x")
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="level must be"):
                m.interval(bad)


class TestSystematics:
    def test_components_stay_separate_and_combine_in_quadrature(
        self, rng: np.random.Generator
    ) -> None:
        m = Measurement(samples=u.Quantity(rng.normal(1.0, 0.3, 100_000)), name="x")
        m = m.with_systematic(u.Quantity(rng.normal(0.0, 0.4, 100_000)))
        assert m.uncertainty.value == pytest.approx(0.3, rel=0.03)
        assert m.systematic_uncertainty.value == pytest.approx(0.4, rel=0.03)
        assert m.total_uncertainty.value == pytest.approx(np.hypot(0.3, 0.4), rel=0.03)

    def test_absent_systematic_is_zero_not_missing(self, rng: np.random.Generator) -> None:
        m = Measurement(samples=u.Quantity(rng.normal(0, 1, 100), u.day), name="x")
        assert m.systematic_uncertainty.value == 0.0
        assert m.systematic_uncertainty.unit == u.day


class TestCorrelationPreservation:
    def test_derive_preserves_correlation(self, correlated_posterior: JointPosterior) -> None:
        """The whole point of ADR-0002.

        A quantity derived jointly must differ from the same quantity computed under a false
        independence assumption whenever the inputs are correlated. If these agreed, the
        sample representation would be buying nothing.
        """
        post = correlated_posterior
        assert post.correlation("ror", "b") > 0.5

        joint = post.derive("sum", lambda ror, b: ror + b, "ror", "b")
        wrong = combine_independent(
            "sum_wrong",
            [post.marginal("ror"), post.marginal("b")],
            lambda a, b: a + b,
            seed=1,
        )
        # Positive correlation inflates the variance of a sum; assuming independence
        # understates it. This is the error the type exists to prevent.
        assert joint.uncertainty.value > wrong.uncertainty.value

    def test_derive_requires_units(self, correlated_posterior: JointPosterior) -> None:
        with pytest.raises(TypeError, match="must return a Quantity"):
            correlated_posterior.derive("bad", lambda ror: ror.value, "ror")

    def test_derive_rejects_unknown_parameter(self, correlated_posterior: JointPosterior) -> None:
        with pytest.raises(KeyError):
            correlated_posterior.derive("x", lambda a: a, "not_a_parameter")

    def test_marginals_share_posterior_id(self, correlated_posterior: JointPosterior) -> None:
        a = correlated_posterior.marginal("ror")
        b = correlated_posterior.marginal("b")
        assert a.posterior_id == b.posterior_id is not None

    def test_unequal_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="same number of draws"):
            JointPosterior({"a": u.Quantity(np.zeros(10)), "b": u.Quantity(np.zeros(11))})


class TestIndependenceGuard:
    def test_cross_posterior_arithmetic_refused(
        self, correlated_posterior: JointPosterior, rng: np.random.Generator
    ) -> None:
        """Pairing samples from unrelated posteriors would invent a correlation."""
        other = JointPosterior({"x": u.Quantity(rng.normal(0, 1, 4000))})
        with pytest.raises(IndependenceError, match="different posteriors"):
            _ = correlated_posterior.marginal("ror") + other.marginal("x")

    def test_untagged_measurement_refused(self, rng: np.random.Generator) -> None:
        a = Measurement(samples=u.Quantity(rng.normal(0, 1, 100)), name="a")
        b = Measurement(samples=u.Quantity(rng.normal(0, 1, 100)), name="b")
        with pytest.raises(IndependenceError, match="no posterior id"):
            _ = a + b

    def test_same_posterior_arithmetic_allowed(self, correlated_posterior: JointPosterior) -> None:
        total = correlated_posterior.marginal("ror") + correlated_posterior.marginal("b")
        assert total.samples.size == correlated_posterior.n_samples

    def test_scalar_arithmetic_always_allowed(self, correlated_posterior: JointPosterior) -> None:
        doubled = correlated_posterior.marginal("ror") * 2.0
        assert doubled.value.value == pytest.approx(
            2.0 * correlated_posterior.marginal("ror").value.value, rel=1e-9
        )

    def test_combine_independent_records_the_assumption(self, rng: np.random.Generator) -> None:
        a = from_gaussian("a", 1.0 * u.day, 0.1 * u.day, seed=1)
        b = from_gaussian("b", 2.0 * u.day, 0.2 * u.day, seed=2)
        c = combine_independent("sum", [a, b], lambda x, y: x + y, seed=3)
        assert "independence assumption" in c.description
        assert c.value.value == pytest.approx(3.0, abs=0.02)


class TestFromGaussian:
    def test_reproduces_quoted_value(self) -> None:
        m = from_gaussian("P", 0.94145 * u.day, 1e-5 * u.day, n_samples=100_000, seed=7)
        assert m.value.value == pytest.approx(0.94145, abs=1e-7)
        assert m.uncertainty.value == pytest.approx(1e-5, rel=0.02)

    def test_seeded_and_reproducible(self) -> None:
        a = from_gaussian("x", 1.0 * u.day, 0.1 * u.day, seed=42)
        b = from_gaussian("x", 1.0 * u.day, 0.1 * u.day, seed=42)
        assert np.array_equal(a.samples.value, b.samples.value)

    def test_rejects_negative_sigma(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            from_gaussian("x", 1.0 * u.day, -0.1 * u.day)

    def test_rejects_bare_floats(self) -> None:
        with pytest.raises(TypeError):
            from_gaussian("x", 1.0, 0.1)  # type: ignore[arg-type]


class TestNumericalInvariants:
    """Property-based checks on the summary statistics."""

    @given(
        loc=st.floats(-1e3, 1e3),
        scale=st.floats(1e-3, 1e2),
        seed=st.integers(0, 2**16),
    )
    @settings(max_examples=50, deadline=None)
    def test_median_lies_inside_the_credible_interval(
        self, loc: float, scale: float, seed: int
    ) -> None:
        m = from_gaussian("x", loc * u.day, scale * u.day, n_samples=2000, seed=seed)
        lo, hi = m.interval()
        assert lo.value <= m.value.value <= hi.value

    @given(
        loc=st.floats(-1e3, 1e3),
        scale=st.floats(1e-3, 1e2),
        seed=st.integers(0, 2**16),
    )
    @settings(max_examples=50, deadline=None)
    def test_wider_level_gives_wider_interval(self, loc: float, scale: float, seed: int) -> None:
        m = from_gaussian("x", loc * u.day, scale * u.day, n_samples=2000, seed=seed)
        lo68, hi68 = m.interval(0.68)
        lo95, hi95 = m.interval(0.95)
        assert lo95.value <= lo68.value
        assert hi95.value >= hi68.value

    @given(scale=st.floats(1e-3, 1e2), seed=st.integers(0, 2**16))
    @settings(max_examples=30, deadline=None)
    def test_unit_conversion_preserves_relative_uncertainty(self, scale: float, seed: int) -> None:
        m = from_gaussian("x", 10.0 * u.day, scale * u.day, n_samples=2000, seed=seed)
        converted = m.to(u.hour)
        assert converted.uncertainty.to(u.day).value == pytest.approx(m.uncertainty.value, rel=1e-9)
