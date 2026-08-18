"""Unit boundary enforcement."""

from __future__ import annotations

import pytest
from astropy import units as u

from astrolab.core.units import (
    CANONICAL_UNITS,
    UnitBoundaryError,
    as_float,
    ensure_quantity,
    is_dimensionless_fraction,
)


class TestEnsureQuantity:
    def test_converts_compatible_units(self) -> None:
        got = ensure_quantity(3.0 * u.hour, u.day, name="duration")
        assert got is not None
        assert got.unit == u.day
        assert got.value == pytest.approx(0.125)

    @pytest.mark.parametrize("bad", [3.5, 0, [1.0, 2.0], "3.5"])
    def test_rejects_values_without_units(self, bad: object) -> None:
        """A bare number is never assumed to be in the target unit.

        Assuming is exactly the behaviour that lets a ppm/fraction confusion through.
        """
        with pytest.raises(UnitBoundaryError, match="must be an astropy Quantity"):
            ensure_quantity(bad, u.day, name="period")

    def test_rejects_incompatible_units(self) -> None:
        with pytest.raises(UnitBoundaryError, match="not convertible"):
            ensure_quantity(3.5 * u.m, u.day, name="period")

    def test_none_rejected_unless_allowed(self) -> None:
        with pytest.raises(UnitBoundaryError, match="required"):
            ensure_quantity(None, u.day, name="period")
        assert ensure_quantity(None, u.day, name="period", allow_none=True) is None

    def test_error_names_the_offending_parameter(self) -> None:
        with pytest.raises(UnitBoundaryError, match="'transit_depth'"):
            ensure_quantity(0.01, u.dimensionless_unscaled, name="transit_depth")


class TestAsFloat:
    def test_requires_explicit_unit_and_converts(self) -> None:
        assert as_float(120.0 * u.s, u.min, name="cadence") == pytest.approx(2.0)

    def test_still_rejects_bare_floats(self) -> None:
        with pytest.raises(UnitBoundaryError):
            as_float(120.0, u.s, name="cadence")  # type: ignore[arg-type]


class TestCanonicalUnits:
    def test_depth_is_dimensionless_not_ppm(self) -> None:
        """Depth is stored as a fraction. Storing ppm invites a factor-of-10^6 error."""
        assert CANONICAL_UNITS["depth"] == u.dimensionless_unscaled

    def test_ppm_is_convertible_to_the_canonical_fraction(self) -> None:
        depth_ppm = 9440 * u.def_unit("ppm", 1e-6 * u.dimensionless_unscaled)
        converted = ensure_quantity(depth_ppm, CANONICAL_UNITS["depth"], name="depth")
        assert converted is not None
        assert converted.value == pytest.approx(0.00944)

    def test_is_dimensionless_fraction(self) -> None:
        assert is_dimensionless_fraction(0.01 * u.dimensionless_unscaled)
        assert not is_dimensionless_fraction(0.01 * u.day)
