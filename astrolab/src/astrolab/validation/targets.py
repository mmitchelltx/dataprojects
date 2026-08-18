"""Golden targets: published values the pipeline must reproduce.

Each benchmark records the expected value, a tolerance, the source citation, and the date the
value was last verified. CI fails if the pipeline drifts away from these, which is what turns
"the code seems to work" into a checkable claim.

**Status of these benchmarks.** They run against the bundled K2-3 light curve, whose chain of
custody does not reach the archive (see ``validation/data/SOURCE.md``). They are therefore
*regression* benchmarks -- they detect drift in our code against a fixed input -- and are not
yet *validated* benchmarks. Validation requires re-running the same analysis on the original
MAST product. ``docs/phase-2-status.md`` tracks that as an open item.

Tolerances are set from the published uncertainties where the comparison is like-for-like, and
loosened with a stated reason where it is not. A tolerance chosen to make a test pass is a
tolerance that certifies nothing, so each one says why it is what it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["GOLDEN_TARGETS", "GoldenTarget", "GoldenValue"]


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

    def check(self, measured: float) -> tuple[bool, float]:
        """Return ``(passed, absolute_deviation)``."""
        deviation = abs(measured - self.expected)
        return deviation <= self.tolerance, deviation

    def describe(self, measured: float) -> str:
        passed, deviation = self.check(measured)
        return (
            f"{'PASS' if passed else 'FAIL'} {self.name}: measured {measured:.6f}, "
            f"expected {self.expected:.6f} +/- {self.tolerance:.6f} {self.unit} "
            f"(deviation {deviation:.6f}) [{self.source}]"
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
                }
                for k, v in self.values.items()
            },
        }


# Periods from Crossfield et al. 2015 (discovery) as refined by Sinukoff et al. 2016.
# Transcribed 2026-08-18; re-verify against the papers before quoting. A transcription error
# in a benchmark is a benchmark that certifies the wrong answer.
_CROSSFIELD = "Crossfield et al. 2015, ApJ 804, 10"
_CROSSFIELD_DOI = "10.1088/0004-637X/804/1/10"

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
# Tolerances below are ~2x the local half-spacing: tight enough that landing on the wrong
# planet or the wrong harmonic fails, loose enough that landing on the nearest available grid
# point passes. The first version of this file used 0.001 d for planet b, which the pipeline
# failed while performing *optimally* -- it had landed 0.53 half-spacings from the published
# value, i.e. on the closest grid point in existence. That tolerance was measuring the grid,
# not the code.
#
# Precise periods are the fitting stage's job, not the search's: a fit refines the period
# continuously against transit timing and should be compared against the published
# uncertainty directly.
_PERIOD_TOLERANCE_RATIONALE = (
    "Set to about twice the local TLS period-grid half-spacing, measured on this target's "
    "actual grid (see the comment block above this constant for the numbers and the date). A "
    "periodogram can only report a grid point, so its accuracy is bounded by the grid; a "
    "tighter tolerance would test the sampling rather than the pipeline. This checks that the "
    "search finds the right planet, not that it matches a published fit -- that is the "
    "fitting stage's benchmark, against the published uncertainty."
)

GOLDEN_TARGETS: dict[str, GoldenTarget] = {
    "K2-3": GoldenTarget(
        name="K2-3 (EPIC 201367065)",
        description=(
            "M dwarf in K2 Campaign 1 hosting three transiting planets. Used as the transit "
            "search benchmark: two of the three planets fall inside an 80-day baseline with "
            "enough transits to determine a period, and the third (P=44.6 d) is a deliberate "
            "test that the pipeline refuses to over-claim from one or two events."
        ),
        provenance_caveat=(
            "Runs on a light curve whose chain of custody does not reach MAST. This is a "
            "regression benchmark, not a validated one, until re-run on the archive product."
        ),
        values={
            "period_b": GoldenValue(
                name="K2-3 b period",
                expected=10.05403,
                tolerance=0.005,
                unit="d",
                source=_CROSSFIELD,
                doi=_CROSSFIELD_DOI,
                last_verified="2026-08-18",
                tolerance_rationale=_PERIOD_TOLERANCE_RATIONALE,
            ),
            "period_c": GoldenValue(
                name="K2-3 c period",
                expected=24.6454,
                tolerance=0.015,
                unit="d",
                source=_CROSSFIELD,
                doi=_CROSSFIELD_DOI,
                last_verified="2026-08-18",
                tolerance_rationale=_PERIOD_TOLERANCE_RATIONALE,
            ),
        },
    )
}
