"""Quality flags: the machinery behind "say when you don't know"."""

from __future__ import annotations

import pytest

from astrolab.core.quality import QualityReport, Severity


class TestSeverity:
    def test_ordering_supports_taking_the_worst(self) -> None:
        assert Severity.INFO < Severity.CAUTION < Severity.UNRELIABLE
        assert max([Severity.INFO, Severity.UNRELIABLE, Severity.CAUTION]) is Severity.UNRELIABLE


class TestQualityReport:
    def test_clean_report(self) -> None:
        report = QualityReport()
        assert report.severity is None
        assert report.is_reliable
        assert not report
        assert "clean" in report.summary_line()

    def test_records_context_for_auditability(self) -> None:
        """A flag asserts something; the context is what makes it checkable."""
        report = QualityReport()
        report.add(
            "insufficient_snr",
            Severity.UNRELIABLE,
            "S/N 2.1 is below the threshold of 7.0",
            snr=2.1,
            threshold=7.0,
        )
        entry = report.to_list()[0]
        assert entry["context"] == {"snr": 2.1, "threshold": 7.0}
        assert entry["severity"] == "UNRELIABLE"

    def test_unreliable_flag_marks_the_result_unusable(self) -> None:
        report = QualityReport()
        report.add("insufficient_snr", Severity.UNRELIABLE, "too faint")
        assert not report.is_reliable

    def test_caution_does_not_make_a_result_unreliable(self) -> None:
        """A caveat is not a disqualification; conflating them would cry wolf."""
        report = QualityReport()
        report.add("single_event", Severity.CAUTION, "only one transit observed")
        assert report.is_reliable
        assert report.severity is Severity.CAUTION

    def test_severity_is_the_worst_present(self) -> None:
        report = QualityReport()
        report.add("a", Severity.INFO, "note")
        report.add("b", Severity.UNRELIABLE, "fatal")
        report.add("c", Severity.CAUTION, "hmm")
        assert report.severity is Severity.UNRELIABLE

    def test_has_finds_a_specific_flag(self) -> None:
        report = QualityReport()
        report.add("data_gap_at_event", Severity.CAUTION, "gap overlaps ingress")
        assert report.has("data_gap_at_event")
        assert not report.has("insufficient_snr")

    def test_derived_products_inherit_their_inputs_caveats(self) -> None:
        """A reservation about an input is a reservation about anything built from it."""
        upstream = QualityReport()
        upstream.add("undersampled_cadence", Severity.CAUTION, "30-min cadence, 1.2 h transit")
        downstream = QualityReport()
        downstream.add("single_event", Severity.CAUTION, "one transit")
        downstream.extend(upstream)
        assert len(downstream) == 2
        assert downstream.has("undersampled_cadence")

    def test_summary_line_counts_by_severity(self) -> None:
        report = QualityReport()
        report.add("a", Severity.CAUTION, "x")
        report.add("b", Severity.CAUTION, "y")
        report.add("c", Severity.UNRELIABLE, "z")
        line = report.summary_line()
        assert line.startswith("UNRELIABLE")
        assert "2 caution" in line

    def test_truthiness_means_flags_present_not_all_clear(self) -> None:
        """Documented explicitly because the polarity is easy to misread at a call site."""
        report = QualityReport()
        assert not report
        report.add("a", Severity.INFO, "note")
        assert report

    @pytest.mark.parametrize("severity", list(Severity))
    def test_all_severities_serialise(self, severity: Severity) -> None:
        report = QualityReport()
        report.add("f", severity, "m")
        assert report.to_list()[0]["severity"] == severity.name
