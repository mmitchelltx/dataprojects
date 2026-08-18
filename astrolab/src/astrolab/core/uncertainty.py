"""Uncertainty as posterior samples.

See ``docs/decisions/0002-uncertainty-representation.md`` for why samples, and not
value-plus-sigma, are the canonical internal representation.

The short version: transit parameters are neither Gaussian nor independent. Radius ratio and
impact parameter are strongly correlated -- a large planet crossing near the stellar limb makes
nearly the same light curve as a smaller planet crossing the centre. Eccentricity piles up
against a hard boundary at zero. Summarising to value +/- sigma discards the covariance, and
propagating the summaries then produces an answer that is not conservative but simply wrong, in
a direction you cannot predict.

Samples fix this for free: correlations survive because the samples survive, and any derived
quantity is computed element-wise across the joint draw.

The one hazard samples introduce is combining two of them that did *not* come from the same
posterior. Element-wise arithmetic on independent sample sets invents a correlation that does
not exist -- it is silently wrong in the other direction. This module makes that an error:
every :class:`Measurement` records which posterior it came from, and arithmetic across
different posteriors is refused until the caller says explicitly what they mean.

References
----------
Hogg, Bovy & Lang 2010, arXiv:1008.4686 -- on fitting models to data and reporting the result.
Eastman, Gaudi & Agol 2013, PASP 125, 83. doi:10.1086/669497 -- transit parameter correlations.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u
from astropy.units import Quantity

__all__ = [
    "IndependenceError",
    "JointPosterior",
    "Measurement",
    "combine_independent",
    "from_gaussian",
]


class IndependenceError(ValueError):
    """Element-wise arithmetic was attempted across two unrelated posteriors.

    Doing so would fabricate a correlation between quantities that were never sampled
    jointly. Use :func:`combine_independent` if the quantities really are independent, or
    derive both from the same :class:`JointPosterior` if they are not.
    """


def _percentile(samples: Quantity, q: float) -> Quantity:
    """Percentile of a Quantity array, preserving units."""
    return Quantity(np.percentile(samples.value, q), samples.unit)


@dataclass(frozen=True)
class Measurement:
    """A scalar physical quantity represented by its posterior samples.

    Attributes
    ----------
    samples
        1-D :class:`~astropy.units.Quantity` of posterior draws. This is the value; everything
        else is derived from it.
    name
        Human-readable name, used in reports and error messages.
    posterior_id
        Identifier of the joint posterior these samples were drawn from. Two Measurements
        sharing an id have row-aligned samples and may be combined element-wise. Measurements
        with different ids may not. A ``None`` id means "not from a joint fit" and is treated
        as incompatible with everything except itself.
    systematic
        Optional samples of an *additive systematic* offset in the same units, carried
        separately from the statistical samples. Kept separate all the way to reporting
        because collapsing them early destroys information the reader needs: a result that is
        systematics-dominated demands a different response than one that is statistics-limited.
    description
        Optional note on provenance or method, surfaced in reports.
    """

    samples: Quantity
    name: str
    posterior_id: str | None = None
    systematic: Quantity | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.samples, Quantity):
            raise TypeError(
                f"Measurement {self.name!r}: samples must be an astropy Quantity, "
                f"got {type(self.samples).__name__}"
            )
        if self.samples.isscalar or self.samples.ndim != 1:
            raise ValueError(
                f"Measurement {self.name!r}: samples must be a 1-D array of posterior draws, "
                f"got shape {getattr(self.samples, 'shape', ())}. A single number is not a "
                f"measurement -- it has no uncertainty."
            )
        if self.samples.size < 2:
            raise ValueError(
                f"Measurement {self.name!r}: need at least 2 samples to express an uncertainty"
            )
        if self.systematic is not None:
            if not isinstance(self.systematic, Quantity):
                raise TypeError(f"Measurement {self.name!r}: systematic must be a Quantity")
            if not self.systematic.unit.is_equivalent(self.samples.unit):
                raise u.UnitConversionError(
                    f"Measurement {self.name!r}: systematic units {self.systematic.unit} are "
                    f"not compatible with sample units {self.samples.unit}"
                )

    # -- summaries (derived views, never stored) --------------------------------------

    @property
    def value(self) -> Quantity:
        """Posterior median.

        The median rather than the mean, because it is invariant under monotonic
        reparameterisation -- the median of ``log P`` is the log of the median of ``P``, which
        is not true of the mean. Transit work reparameterises constantly.
        """
        return _percentile(self.samples, 50.0)

    def interval(self, level: float = 0.6827) -> tuple[Quantity, Quantity]:
        """Equal-tailed credible interval at ``level`` (default 1-sigma equivalent).

        Equal-tailed rather than highest-density: it is reparameterisation-invariant and it is
        what the 16th/84th percentile convention in the literature means, so results are
        directly comparable to published values.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = 100.0 * (1.0 - level) / 2.0
        return _percentile(self.samples, tail), _percentile(self.samples, 100.0 - tail)

    @property
    def uncertainty(self) -> Quantity:
        """Symmetric statistical uncertainty: half the 68.27% equal-tailed interval width.

        Provided for comparison against published symmetric error bars. When the posterior is
        materially asymmetric this is lossy, which is why :meth:`interval` exists and why
        reports use it; check :attr:`asymmetry` before quoting this number.
        """
        lo, hi = self.interval()
        return (hi - lo) / 2.0

    @property
    def uncertainty_asymmetric(self) -> tuple[Quantity, Quantity]:
        """Lower and upper 1-sigma-equivalent distances from the median (both positive)."""
        lo, hi = self.interval()
        return self.value - lo, hi - self.value

    @property
    def asymmetry(self) -> float:
        """Fractional asymmetry of the 68% interval; 0 is symmetric.

        A value above roughly 0.1 means quoting a single symmetric error bar is misleading and
        the report should carry the asymmetric interval instead.
        """
        minus, plus = self.uncertainty_asymmetric
        total = (plus + minus).value
        if total == 0.0:
            return 0.0
        return float(abs(plus.value - minus.value) / total)

    @property
    def systematic_uncertainty(self) -> Quantity:
        """Symmetric systematic uncertainty, or zero if none was assigned."""
        if self.systematic is None:
            return Quantity(0.0, self.samples.unit)
        lo, hi = _percentile(self.systematic, 15.865), _percentile(self.systematic, 84.135)
        return (hi - lo) / 2.0

    @property
    def total_uncertainty(self) -> Quantity:
        """Statistical and systematic combined in quadrature.

        Reports must show the components as well; this is for the single headline number only.
        """
        stat = self.uncertainty
        syst = self.systematic_uncertainty.to(stat.unit)
        return Quantity(np.hypot(stat.value, syst.value), stat.unit)

    # -- combination -------------------------------------------------------------------

    def _check_compatible(self, other: Measurement) -> None:
        if self.posterior_id is None or other.posterior_id is None:
            raise IndependenceError(
                f"Cannot combine {self.name!r} and {other.name!r} element-wise: at least one "
                f"has no posterior id, so their samples are not known to be row-aligned. "
                f"Use combine_independent() if they are genuinely independent."
            )
        if self.posterior_id != other.posterior_id:
            raise IndependenceError(
                f"Cannot combine {self.name!r} and {other.name!r} element-wise: they come from "
                f"different posteriors ({self.posterior_id} vs {other.posterior_id}), so "
                f"pairing their samples would invent a correlation that was never measured. "
                f"Use combine_independent() if they are genuinely independent, or derive both "
                f"from one JointPosterior if they are not."
            )
        if self.samples.size != other.samples.size:
            raise IndependenceError(
                f"{self.name!r} and {other.name!r} claim the same posterior but have "
                f"{self.samples.size} and {other.samples.size} samples"
            )

    def _binary(self, other: Measurement | Quantity, op: Any, symbol: str) -> Measurement:
        if isinstance(other, Measurement):
            self._check_compatible(other)
            return Measurement(
                samples=op(self.samples, other.samples),
                name=f"({self.name} {symbol} {other.name})",
                posterior_id=self.posterior_id,
            )
        if isinstance(other, Quantity | int | float):
            return Measurement(
                samples=op(self.samples, other),
                name=f"({self.name} {symbol} {other})",
                posterior_id=self.posterior_id,
            )
        return NotImplemented  # type: ignore[unreachable]

    def __add__(self, other: Measurement | Quantity) -> Measurement:
        return self._binary(other, lambda a, b: a + b, "+")

    def __sub__(self, other: Measurement | Quantity) -> Measurement:
        return self._binary(other, lambda a, b: a - b, "-")

    def __mul__(self, other: Measurement | Quantity) -> Measurement:
        return self._binary(other, lambda a, b: a * b, "*")

    def __truediv__(self, other: Measurement | Quantity) -> Measurement:
        return self._binary(other, lambda a, b: a / b, "/")

    def to(self, unit: Any) -> Measurement:
        """Convert to different units, preserving posterior identity."""
        return Measurement(
            samples=self.samples.to(unit),
            name=self.name,
            posterior_id=self.posterior_id,
            systematic=None if self.systematic is None else self.systematic.to(unit),
            description=self.description,
        )

    def with_systematic(self, systematic: Quantity, *, note: str = "") -> Measurement:
        """Attach or replace the systematic component, returning a new Measurement."""
        return Measurement(
            samples=self.samples,
            name=self.name,
            posterior_id=self.posterior_id,
            systematic=systematic,
            description=(self.description + " " + note).strip(),
        )

    # -- reporting ---------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Serialisable summary for reports and provenance records."""
        lo, hi = self.interval()
        minus, plus = self.uncertainty_asymmetric
        out: dict[str, Any] = {
            "name": self.name,
            "unit": str(self.samples.unit),
            "median": float(self.value.value),
            "interval_68_low": float(lo.value),
            "interval_68_high": float(hi.value),
            "sigma_minus": float(minus.value),
            "sigma_plus": float(plus.value),
            "asymmetry": self.asymmetry,
            "n_samples": int(self.samples.size),
            "stat_uncertainty": float(self.uncertainty.value),
            "syst_uncertainty": float(self.systematic_uncertainty.to(self.samples.unit).value),
            "total_uncertainty": float(self.total_uncertainty.value),
            "posterior_id": self.posterior_id,
        }
        if self.description:
            out["description"] = self.description
        return out

    def __repr__(self) -> str:
        minus, plus = self.uncertainty_asymmetric
        syst = self.systematic_uncertainty
        syst_str = f" (sys +/-{syst.value:.4g})" if syst.value != 0.0 else ""
        return (
            f"<Measurement {self.name}: {self.value.value:.6g} "
            f"+{plus.value:.3g}/-{minus.value:.3g}{syst_str} {self.samples.unit}, "
            f"n={self.samples.size}>"
        )


@dataclass(frozen=True)
class JointPosterior:
    """A set of parameters sampled jointly, so their correlations are represented.

    This is what an MCMC or nested-sampling run produces, and keeping it intact rather than
    immediately marginalising is what lets derived quantities inherit the right uncertainty.

    Parameters
    ----------
    samples
        Mapping of parameter name to a 1-D Quantity of draws. All must have the same length,
        and row *i* of every array must belong to the same posterior draw.
    metadata
        Sampler diagnostics, evidence, seed -- anything that belongs with the chain.
    """

    samples: Mapping[str, Quantity]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    posterior_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("JointPosterior requires at least one parameter")
        sizes = {name: q.size for name, q in self.samples.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(
                f"All parameters must have the same number of draws to be row-aligned; got "
                f"{sizes}. Unequal lengths mean these did not come from one posterior."
            )
        for name, q in self.samples.items():
            if not isinstance(q, Quantity):
                raise TypeError(f"Parameter {name!r} must be a Quantity, got {type(q).__name__}")

    @property
    def n_samples(self) -> int:
        return int(next(iter(self.samples.values())).size)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[str]:
        return iter(self.samples)

    def __contains__(self, name: object) -> bool:
        return name in self.samples

    def marginal(self, name: str) -> Measurement:
        """Extract one parameter as a :class:`Measurement`, tagged with this posterior's id."""
        if name not in self.samples:
            raise KeyError(
                f"{name!r} is not in this posterior; available: {sorted(self.samples)}"
            )
        return Measurement(
            samples=self.samples[name], name=name, posterior_id=self.posterior_id
        )

    def derive(
        self, name: str, func: Callable[..., Quantity], *parameters: str
    ) -> Measurement:
        """Compute a derived quantity element-wise across the joint draws.

        This is the correlation-preserving operation. ``func`` is applied row by row (in
        practice, vectorised over the sample arrays), so a quantity like planet radius derived
        from radius ratio and stellar radius inherits their covariance instead of assuming
        independence.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy import units as u
        >>> rng = np.random.default_rng(0)
        >>> post = JointPosterior({
        ...     "ror": u.Quantity(rng.normal(0.1, 0.001, 500)),
        ...     "rstar": u.Quantity(rng.normal(1.0, 0.05, 500), u.Rsun),
        ... })
        >>> rp = post.derive("rp", lambda ror, rstar: (ror * rstar).to(u.Rearth),
        ...                  "ror", "rstar")
        >>> rp.name
        'rp'
        """
        missing = [p for p in parameters if p not in self.samples]
        if missing:
            raise KeyError(f"Parameters not in posterior: {missing}")
        result = func(*(self.samples[p] for p in parameters))
        if not isinstance(result, Quantity):
            raise TypeError(
                f"derive({name!r}) must return a Quantity; got {type(result).__name__}. "
                f"Attach units inside the function."
            )
        return Measurement(samples=result, name=name, posterior_id=self.posterior_id)

    def correlation(self, a: str, b: str) -> float:
        """Pearson correlation between two parameters -- a diagnostic, not a summary.

        Useful for spotting the degeneracies that make marginal error bars misleading, such as
        radius ratio against impact parameter.
        """
        for p in (a, b):
            if p not in self.samples:
                raise KeyError(f"{p!r} is not in this posterior")
        return float(np.corrcoef(self.samples[a].value, self.samples[b].value)[0, 1])


def from_gaussian(
    name: str,
    value: Quantity,
    sigma: Quantity,
    *,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> Measurement:
    """Build a Measurement by sampling a Gaussian, for genuinely Gaussian inputs.

    Use this for a literature value quoted as ``x +/- sigma``, or a calibration offset with a
    known symmetric error. It exists so that everything in the pipeline has one representation
    and no call site needs to branch on "is this samples or is this value-plus-sigma".

    The seed is recorded by the caller's provenance manifest; pass one for reproducibility.
    """
    if not isinstance(value, Quantity) or not isinstance(sigma, Quantity):
        raise TypeError(f"{name}: value and sigma must both be Quantities")
    if not sigma.unit.is_equivalent(value.unit):
        raise u.UnitConversionError(
            f"{name}: sigma units {sigma.unit} are not compatible with value units {value.unit}"
        )
    if float(sigma.value) < 0.0:
        raise ValueError(f"{name}: sigma must be non-negative, got {sigma}")
    rng = np.random.default_rng(seed)
    draws = rng.normal(
        float(value.value), float(sigma.to(value.unit).value), size=n_samples
    )
    return Measurement(
        samples=Quantity(draws, value.unit),
        name=name,
        posterior_id=None,
        description="sampled from a quoted Gaussian",
    )


def combine_independent(
    name: str,
    measurements: list[Measurement],
    func: Callable[..., Quantity],
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> Measurement:
    """Combine Measurements from *different* posteriors, assuming independence.

    The independence assumption is the caller's to justify, and making it a distinct function
    with an explicit name is the point: the assumption appears in the code review rather than
    hiding inside an operator. Samples are drawn independently (with replacement) from each
    input, which is correct only if the inputs really are uncorrelated.

    Parameters
    ----------
    n_samples
        Size of the output. Defaults to the smallest input's sample count, since resampling
        beyond that adds no information.
    seed
        Recorded in provenance by the caller.
    """
    if not measurements:
        raise ValueError("combine_independent requires at least one Measurement")
    size = n_samples if n_samples is not None else min(m.samples.size for m in measurements)
    rng = np.random.default_rng(seed)
    drawn = [
        m.samples[rng.integers(0, m.samples.size, size=size)] for m in measurements
    ]
    result = func(*drawn)
    if not isinstance(result, Quantity):
        raise TypeError(f"combine_independent({name!r}) must return a Quantity")
    return Measurement(
        samples=result,
        name=name,
        posterior_id=None,
        description=(
            "combined under an explicit independence assumption from: "
            + ", ".join(m.name for m in measurements)
        ),
    )
