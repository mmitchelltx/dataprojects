"""Quality flags -- the machinery behind "say when you don't know".

Prime directive 5 requires the pipeline to flag rather than guess when a step exceeds what the
data can support. That only works if "I cannot answer this" is a first-class output that
travels with the result, rather than a warning someone missed in a log.

A flagged result is still written. Suppressing it would hide the diagnosis; the point is that
the flag is attached to the number, visible in the product metadata, the report, and the CLI
exit status, so a downstream consumer cannot use the number without meeting the caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = ["Flag", "QualityFlag", "QualityReport", "Severity"]


class Severity(IntEnum):
    """How badly a flag should affect trust in the result.

    Ordered so that ``max()`` over a set of flags gives the overall severity.
    """

    INFO = 10
    """Worth recording; does not undermine the result."""

    CAUTION = 20
    """The result is usable but a stated caveat must travel with it."""

    UNRELIABLE = 30
    """The data do not support the requested measurement. The number should not be quoted."""


class Flag(str):
    """A flag identifier. Subclasses ``str`` so it serialises transparently."""

    __slots__ = ()


# Well-known flags. Modules may define their own; these are the cross-cutting ones.
INSUFFICIENT_SNR = Flag("insufficient_snr")
UNDERSAMPLED_CADENCE = Flag("undersampled_cadence")
UNCONVERGED_FIT = Flag("unconverged_fit")
DATA_GAP_AT_EVENT = Flag("data_gap_at_event")
SYSTEMATICS_COMPARABLE_TO_SIGNAL = Flag("systematics_comparable_to_signal")
EXTRAPOLATED_CALIBRATION = Flag("extrapolated_calibration")
TRIAL_FACTOR_UNCORRECTED = Flag("trial_factor_uncorrected")
SINGLE_EVENT = Flag("single_event")


@dataclass(frozen=True)
class QualityFlag:
    """One specific reason to doubt, or to qualify, a result.

    Attributes
    ----------
    flag
        Machine-readable identifier, for filtering and for regression tests.
    severity
        See :class:`Severity`.
    message
        Human-readable explanation. Should say what was measured, what the threshold was, and
        what the consequence is -- enough for a reader to decide whether they care.
    context
        Structured supporting numbers (the measured S/N, the threshold, the gap duration).
        These go into the product metadata so the flag is auditable rather than assertive.
    """

    flag: Flag
    severity: Severity
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag": str(self.flag),
            "severity": self.severity.name,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class QualityReport:
    """The accumulated quality flags for one data product or result.

    Mutable by design: stages append to it as a product moves through the pipeline, so the
    final report carries the full history of every reservation raised along the way.
    """

    flags: list[QualityFlag] = field(default_factory=list)

    def add(
        self,
        flag: Flag | str,
        severity: Severity,
        message: str,
        **context: Any,
    ) -> None:
        """Record a reservation about the result."""
        self.flags.append(
            QualityFlag(flag=Flag(flag), severity=severity, message=message, context=dict(context))
        )

    @property
    def severity(self) -> Severity | None:
        """Worst severity present, or ``None`` if the report is clean."""
        return max((f.severity for f in self.flags), default=None)

    @property
    def is_reliable(self) -> bool:
        """Whether any flag says the data do not support the measurement."""
        return not any(f.severity >= Severity.UNRELIABLE for f in self.flags)

    def has(self, flag: Flag | str) -> bool:
        return any(f.flag == flag for f in self.flags)

    def extend(self, other: QualityReport) -> None:
        """Absorb another report, as when a derived product inherits its inputs' caveats."""
        self.flags.extend(other.flags)

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.flags]

    def summary_line(self) -> str:
        """One-line status suitable for a CLI or a report header."""
        if not self.flags:
            return "clean: no quality reservations"
        worst = self.severity
        assert worst is not None
        counts: dict[str, int] = {}
        for f in self.flags:
            counts[f.severity.name] = counts.get(f.severity.name, 0) + 1
        detail = ", ".join(f"{n} {name.lower()}" for name, n in sorted(counts.items()))
        return f"{worst.name}: {detail}"

    def __len__(self) -> int:
        return len(self.flags)

    def __bool__(self) -> bool:
        """True when flags are present. Note this is the *opposite* of "everything is fine"."""
        return bool(self.flags)
