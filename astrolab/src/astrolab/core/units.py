"""Unit conventions and boundary validation.

Prime directive: bare floats crossing a module boundary are a defect, not a style issue.
The failure modes this prevents are real and expensive in time-series astronomy:

- Days-since-epoch confused with absolute BJD, shifting an ephemeris by 2.457 million days.
- Transit depth in ppm confused with fractional depth, a factor of 10^6.
- Arcseconds confused with degrees in a centroid test, a factor of 3600.

Each of those has cost someone a paper. :func:`ensure_quantity` makes them loud and immediate
instead of silent and downstream.

References
----------
astropy.units : Astropy Collaboration et al. 2022, ApJ 935, 167. doi:10.3847/1538-4357/ac7c74
"""

from __future__ import annotations

from typing import Any

from astropy import units as u
from astropy.units import Quantity, UnitBase, UnitConversionError

__all__ = [
    "CANONICAL_UNITS",
    "UnitBoundaryError",
    "as_float",
    "ensure_quantity",
    "is_dimensionless_fraction",
]


class UnitBoundaryError(TypeError):
    """A value crossed a module boundary without the units it needs.

    Deliberately a ``TypeError``: passing a bare float where a physical quantity is required
    is a type error, and treating it as one means it surfaces in the same places other type
    errors do rather than being caught by a broad ``except ValueError``.
    """


#: The unit each named quantity is normalised to internally. Reporting code may convert for
#: display, but everything stored or compared inside the pipeline uses these, so that two
#: values of the same named quantity are always directly comparable.
CANONICAL_UNITS: dict[str, UnitBase] = {
    # Time. Absolute epochs are astropy Time objects, not Quantities; these are durations
    # and periods, where a bare "days" number is the classic ambiguity.
    "period": u.day,
    "duration": u.hour,
    "epoch_offset": u.day,
    # Photometry. Transit depth is kept dimensionless (fractional), never ppm, because ppm
    # is a display convention and storing it invites the 10^6 error.
    "depth": u.dimensionless_unscaled,
    "flux_relative": u.dimensionless_unscaled,
    "magnitude": u.mag,
    # Geometry.
    "radius_ratio": u.dimensionless_unscaled,
    "semimajor_over_rstar": u.dimensionless_unscaled,
    "impact_parameter": u.dimensionless_unscaled,
    "inclination": u.deg,
    # Astrometry. Residuals are reported in arcsec by convention in solar-system work.
    "angle": u.deg,
    "separation": u.arcsec,
    "proper_motion": u.mas / u.yr,
    # Stellar / planetary.
    "stellar_radius": u.Rsun,
    "planet_radius": u.Rearth,
    "stellar_mass": u.Msun,
    "temperature": u.K,
}


def ensure_quantity(
    value: Any,
    unit: UnitBase | str,
    *,
    name: str,
    allow_none: bool = False,
) -> Quantity | None:
    """Validate that ``value`` carries units and convert it to ``unit``.

    Call this on every public function argument that represents a physical quantity.

    Parameters
    ----------
    value
        The value to validate. Must be an :class:`~astropy.units.Quantity`; a bare float or
        array is rejected rather than silently assumed to be in ``unit``. Assuming is exactly
        the behaviour that lets a ppm/fraction confusion through.
    unit
        Target unit. ``value`` must be convertible to it.
    name
        Parameter name, used in the error message so the failure names the culprit.
    allow_none
        Whether ``None`` is an acceptable value (for genuinely optional quantities).

    Returns
    -------
    Quantity or None
        ``value`` converted to ``unit``.

    Raises
    ------
    UnitBoundaryError
        If ``value`` is not a Quantity, or is not convertible to ``unit``.

    Examples
    --------
    >>> from astropy import units as u
    >>> ensure_quantity(3.5 * u.hour, u.day, name="duration")  # doctest: +ELLIPSIS
    <Quantity 0.1458... d>
    >>> ensure_quantity(3.5, u.day, name="period")
    Traceback (most recent call last):
        ...
    astrolab.core.units.UnitBoundaryError: 'period' must be an astropy Quantity in units ...
    """
    if value is None:
        if allow_none:
            return None
        raise UnitBoundaryError(f"{name!r} is required but was None")

    target = u.Unit(unit) if isinstance(unit, str) else unit

    if not isinstance(value, Quantity):
        raise UnitBoundaryError(
            f"{name!r} must be an astropy Quantity in units convertible to {target}, "
            f"got a bare {type(value).__name__}. Attach units explicitly at the call site "
            f"rather than relying on an assumed convention."
        )

    try:
        return value.to(target)
    except UnitConversionError as exc:
        raise UnitBoundaryError(
            f"{name!r} has units {value.unit}, which are not convertible to {target}"
        ) from exc


def as_float(value: Quantity, unit: UnitBase | str, *, name: str) -> float:
    """Convert a Quantity to a bare float in a stated unit, at an external boundary.

    Use this only where a bare number genuinely must leave the system -- writing to a file
    format without unit support, or calling a third-party function that takes floats. The
    explicit ``unit`` argument forces the caller to state the convention being assumed, which
    is the whole point: the assumption becomes visible in the diff.
    """
    converted = ensure_quantity(value, unit, name=name)
    assert converted is not None  # allow_none defaults to False
    return float(converted.value)


def is_dimensionless_fraction(value: Quantity) -> bool:
    """Whether ``value`` is a dimensionless fraction (as opposed to ppm, percent, or a unit).

    Used to guard the depth/ppm confusion at the point where depths are stored.
    """
    return bool(value.unit.is_equivalent(u.dimensionless_unscaled))
