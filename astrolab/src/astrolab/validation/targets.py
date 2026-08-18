"""Golden targets: published values the pipeline must reproduce.

Each benchmark records the expected value, a tolerance, the source citation, the date the
value was last verified, and -- separately -- whether the number was checked against a
retrievable source or transcribed from memory. That last field exists because of a mistake made
while building this file, which is worth recording rather than quietly fixing.

The first version used P = 10.05403 d for K2-3 b, transcribed from memory. The real published
values are 10.05449 +/- 0.00026 d (Crossfield et al. 2015 discovery) and 10.0546535 d
(Kosiarek et al. 2019, from K2 plus Spitzer). The fitting stage returned 10.05474 +/- 0.00012 d,
which reads as a 2.4 sigma tension against the wrong value and as 0.7 sigma agreement against
the right one. A benchmark transcribed from memory does not test the pipeline; it tests the
transcription, and it will happily certify a wrong answer or condemn a right one.

**Status of these benchmarks.** They run against the bundled K2-3 light curve, whose chain of
custody does not reach the archive (see ``validation/data/SOURCE.md``). They are therefore
*regression* benchmarks -- they detect drift in our code against a fixed input -- and are not
yet *validated* benchmarks. Validation requires re-running on the original MAST product.

Tolerances are set from the published uncertainties where the comparison is like-for-like, and
loosened with a stated reason where it is not. A tolerance chosen to make a test pass certifies
nothing, so each one says why it is what it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["GOLDEN_TARGETS", "GoldenTarget", "GoldenValue", "Verification"]

Verification = Literal["sourced", "unverified"]


@dataclass(frozen=True)
class GoldenValue:
    """One published quantity the pipeline must reproduce."""

    name: str
    expected: float
    tolerance: float
    unit: str
    source: str
    doi: str
    last_verified: str
    tolerance_rationale: str
    verification: Verification = "unverified"
    """``\"sourced\"`` means this number was read off a retrievable source on ``last_verified``.

    ``\"unverified\"`` means it has not been, and the benchmark using it is provisional. An
    unverified value may still be useful as a regression anchor -- it pins current behaviour --
    but it must never be quoted as agreement with the literature.
    """

    note: str = ""

    def check(self, measured: float) -> tuple[bool, float]:
        """Return ``(passed, absolute_deviation)``."""
        deviation = abs(measured - self.expected)
        return deviation <= self.tolerance, deviation

    def describe(self, measured: float) -> str:
        passed, deviation = self.check(measured)
        flag = "" if self.verification == "sourced" else " [UNVERIFIED VALUE]"
        return (
            f"{'PASS' if passed else 'FAIL'} {self.name}: measured {measured:.7f}, "
            f"expected {self.expected:.7f} +/- {self.tolerance:.7f} {self.unit} "
            f"(deviation {deviation:.7f}) [{self.source}]{flag}"
        )


@dataclass(frozen=True)
class GoldenTarget:
    """A benchmark target and everything needed to reproduce its published values."""

    name: str
    description: str
    values: dict[str, GoldenValue]
    provenance_caveat: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.name,
            "description": self.description,
            "provenance_caveat": self.provenance_caveat,
            "values": {
                k: {
                    "expected": v.expected,
                    "tolerance": v.tolerance,
                    "unit": v.unit,
                    "source": v.source,
                    "doi": v.doi,
                    "last_verified": v.last_verified,
                    "verification": v.verification,
                }
                for k, v in self.values.items()
            },
        }


_CROSSFIELD = "Crossfield et al. 2015, ApJ 804, 10"
_CROSSFIELD_DOI = "10.1088/0004-637X/804/1/10"
_KOSIAREK = "Kosiarek et al. 2019, AJ 157, 97"
_KOSIAREK_DOI = "10.3847/1538-3881/aaf79c"

# A search tolerance is set by the period grid, not by the published uncertainty, and the
# distinction is not a technicality. A periodogram reports the best *grid point*, so its
# accuracy is bounded below by half the local grid spacing no matter how good the data are.
# Demanding better would be demanding that the search beat its own sampling.
#
# Measured on this target's actual TLS grid (Ofir 2014 optimal sampling, oversampling 3,
# 80.07-day baseline, R*=0.56 Rsun, M*=0.60 Msun), on 2026-08-18:
#
#     near P = 10.05 d : spacing 0.004536 d, half-spacing 0.002268 d
#     near P = 24.65 d : spacing 0.014990 d, half-spacing 0.007495 d
#
# Tolerances below are about twice the local half-spacing: tight enough that landing on the
# wrong planet or the wrong harmonic fails, loose enough that landing on the nearest available
# grid point passes.
_SEARCH_TOLERANCE_RATIONALE = (
    "About twice the local TLS period-grid half-spacing, measured on this target's actual grid "
    "(see the comment block above this constant for numbers and date). A periodogram can only "
    "report a grid point, so its accuracy is bounded by the grid; a tighter tolerance would "
    "test the sampling rather than the pipeline. This checks that the search finds the right "
    "planet, not that it matches a published fit -- that is the fitting stage's benchmark."
)

_FIT_TOLERANCE_RATIONALE = (
    "Three times the published uncertainty on the discovery period, which is far larger than "
    "the Kosiarek value's own error bar and so is the term that dominates. Unlike the search "
    "benchmark this is a like-for-like comparison -- a continuous transit-timing fit against a "
    "published transit-timing fit -- so the published uncertainty is the right yardstick. Note "
    "the pipeline's own uncertainties on this dataset are estimated rather than measured "
    "(the light curve carries the corresponding quality flag), so a deviation of one to two "
    "sigma here is expected and is not evidence of a defect."
)

GOLDEN_TARGETS: dict[str, GoldenTarget] = {
    "K2-3": GoldenTarget(
        name="K2-3 (EPIC 201367065)",
        description=(
            "M0 dwarf at 45 pc observed in K2 Campaign 1, hosting three transiting planets of "
            "1.5-2 Earth radii at periods between 10 and 45 days. Used as the transit benchmark: "
            "two planets fall inside the 80-day baseline with enough transits to determine a "
            "period, and the third (P=44.6 d) is a deliberate test that the pipeline refuses to "
            "over-claim from one or two events."
        ),
        provenance_caveat=(
            "Runs on a light curve whose chain of custody does not reach MAST. This is a "
            "regression benchmark, not a validated one, until re-run on the archive product."
        ),
        values={
            # -- search stage -------------------------------------------------------------
            "period_b": GoldenValue(
                name="K2-3 b period (search)",
                expected=10.0546535,
                tolerance=0.005,
                unit="d",
                source=_KOSIAREK,
                doi=_KOSIAREK_DOI,
                last_verified="2026-08-18",
                tolerance_rationale=_SEARCH_TOLERANCE_RATIONALE,
                verification="sourced",
                note=(
                    "Kosiarek et al. 2019 give 10.0546535 (+0.0000088/-0.0000091) d from K2 "
                    "plus Spitzer transits. The Crossfield et al. 2015 discovery value is "
                    "10.05449 +/- 0.00026 d; the two agree."
                ),
            ),
            "period_c": GoldenValue(
                name="K2-3 c period (search)",
                expected=24.6454,
                tolerance=0.015,
                unit="d",
                source=_CROSSFIELD,
                doi=_CROSSFIELD_DOI,
                last_verified="2026-08-18",
                tolerance_rationale=_SEARCH_TOLERANCE_RATIONALE,
                verification="unverified",
                note=(
                    "Approximate. The primary sources for a precise K2-3 c period could not be "
                    "retrieved from this environment (arXiv, IOP, Wikipedia, and the NASA "
                    "Exoplanet Archive are all blocked by egress policy). Useful as a "
                    "regression anchor -- it pins the search to the right planet -- but it must "
                    "be checked against the paper before being quoted as literature agreement."
                ),
            ),
            # -- fitting stage ------------------------------------------------------------
            "fitted_period_b": GoldenValue(
                name="K2-3 b period (transit fit)",
                expected=10.0546535,
                tolerance=0.00078,
                unit="d",
                source=_KOSIAREK,
                doi=_KOSIAREK_DOI,
                last_verified="2026-08-18",
                tolerance_rationale=_FIT_TOLERANCE_RATIONALE,
                verification="sourced",
                note=(
                    "Tolerance is 3x the Crossfield discovery uncertainty of 0.00026 d. The "
                    "fitting stage returned 10.05474 +/- 0.00012 d on 2026-08-18, which is "
                    "0.7 sigma from this value."
                ),
            ),
        },
    )
}
