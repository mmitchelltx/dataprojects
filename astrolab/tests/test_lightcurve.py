"""LightCurve product: units, time handling, and provenance accumulation."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from astrolab.core.lightcurve import TIME_REFERENCES, LightCurve
from astrolab.core.units import UnitBoundaryError


def make_lc(n: int = 100, **kw: object) -> LightCurve:
    t = np.linspace(0.0, 10.0, n)
    return LightCurve(
        time=kw.pop("time", t * u.day),
        flux=kw.pop("flux", np.ones(n) * u.dimensionless_unscaled),
        flux_err=kw.pop("flux_err", np.full(n, 1e-4) * u.dimensionless_unscaled),
        epoch_ref=kw.pop("epoch_ref", Time(TIME_REFERENCES["BKJD"], format="jd", scale="tdb")),
        meta=kw.pop("meta", {"target": "test", "mission": "K2", "time_system": "BKJD"}),
    )


class TestConstruction:
    def test_rejects_bare_arrays(self) -> None:
        with pytest.raises(UnitBoundaryError):
            make_lc(time=np.linspace(0, 10, 100))

    def test_requires_epoch_ref_as_time(self) -> None:
        """Without it, 'time' is ambiguous by millions of days."""
        with pytest.raises(TypeError, match="ambiguous by"):
            make_lc(epoch_ref=2454833.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            make_lc(flux=np.ones(50) * u.dimensionless_unscaled)

    def test_rejects_unsorted_time(self) -> None:
        t = np.linspace(0, 10, 100)[::-1]
        with pytest.raises(ValueError, match="monotonically"):
            make_lc(time=t * u.day)

    def test_rejects_negative_errors(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            make_lc(flux_err=np.full(100, -1e-4) * u.dimensionless_unscaled)


class TestTimeHandling:
    def test_absolute_time_applies_the_mission_offset(self) -> None:
        """BKJD 1977.26 is 2014-06-01, inside K2 Campaign 1 (2014-05-30 to 2014-08-21)."""
        t = np.array([1977.26244947, 1977.3])
        lc = make_lc(
            n=2,
            time=t * u.day,
            flux=np.ones(2) * u.dimensionless_unscaled,
            flux_err=np.full(2, 1e-4) * u.dimensionless_unscaled,
        )
        assert lc.absolute_time()[0].iso.startswith("2014-06-01")

    def test_tess_and_kepler_offsets_differ_by_the_documented_amount(self) -> None:
        assert TIME_REFERENCES["BTJD"] - TIME_REFERENCES["BKJD"] == 2167.0

    def test_cadence_uses_the_median_not_the_mean(self) -> None:
        """A data gap must not be reported as the sampling interval."""
        t = np.concatenate([np.arange(0, 1, 0.02), np.arange(30, 31, 0.02)])
        lc = make_lc(
            n=len(t),
            time=t * u.day,
            flux=np.ones(len(t)) * u.dimensionless_unscaled,
            flux_err=np.full(len(t), 1e-4) * u.dimensionless_unscaled,
        )
        assert lc.cadence.value == pytest.approx(0.02, abs=1e-9)


class TestScatter:
    def test_robust_to_transits(self) -> None:
        """A standard deviation would be inflated by the signal we are looking for."""
        rng = np.random.default_rng(0)
        n = 2000
        flux = 1.0 + rng.normal(0, 1e-4, n)
        flux[500:520] -= 0.01  # a deep transit
        lc = make_lc(
            n=n,
            time=np.linspace(0, 10, n) * u.day,
            flux=flux * u.dimensionless_unscaled,
            flux_err=np.full(n, 1e-4) * u.dimensionless_unscaled,
        )
        assert lc.scatter.value == pytest.approx(1e-4, rel=0.1)
        assert lc.scatter.value < np.std(flux)


class TestTransformations:
    def test_mask_requires_and_records_a_reason(self) -> None:
        lc = make_lc()
        keep = np.ones(len(lc), dtype=bool)
        keep[:10] = False
        out = lc.mask(keep, reason="testing")
        assert len(out) == 90
        assert out.provenance["history"][-1]["reason"] == "testing"

    def test_mask_refuses_to_remove_everything(self) -> None:
        lc = make_lc()
        with pytest.raises(ValueError, match="every point"):
            lc.mask(np.zeros(len(lc), dtype=bool), reason="oops")

    def test_provenance_accumulates_across_transformations(self) -> None:
        """History is complete by construction, not by each method remembering to append."""
        lc = make_lc().normalise()

        def drop_first(curve: LightCurve, reason: str) -> LightCurve:
            keep = np.ones(len(curve), dtype=bool)
            keep[0] = False
            return curve.mask(keep, reason=reason)

        lc = drop_first(drop_first(lc, "a"), "b")
        ops = [h["operation"] for h in lc.provenance["history"]]
        assert ops == ["normalise", "mask", "mask"]
        assert [h.get("reason") for h in lc.provenance["history"][1:]] == ["a", "b"]

    def test_normalise_records_the_divisor(self) -> None:
        lc = make_lc(flux=np.full(100, 2.0) * u.dimensionless_unscaled)
        out = lc.normalise()
        assert out.flux.value[0] == pytest.approx(1.0)
        assert out.provenance["history"][-1]["divisor"] == pytest.approx(2.0)

    def test_fold_returns_phase_within_half_a_period(self) -> None:
        lc = make_lc(
            n=1000,
            time=np.linspace(0, 100, 1000) * u.day,
            flux=np.ones(1000) * u.dimensionless_unscaled,
            flux_err=np.full(1000, 1e-4) * u.dimensionless_unscaled,
        )
        phase = lc.fold(10.0 * u.day, 0.0 * u.day)
        assert np.all(np.abs(phase.value) <= 5.0 + 1e-9)

    def test_fold_rejects_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="period must be positive"):
            make_lc().fold(0.0 * u.day, 0.0 * u.day)

    def test_window_extracts_and_refuses_empty(self) -> None:
        lc = make_lc()
        assert len(lc.window(5.0 * u.day, 1.0 * u.day, reason="around event")) > 0
        with pytest.raises(ValueError, match="contains no data"):
            lc.window(500.0 * u.day, 1.0 * u.day, reason="nowhere")

    def test_quality_flags_survive_transformation(self) -> None:
        from astrolab.core.quality import Severity

        lc = make_lc()
        lc.quality.add("undersampled_cadence", Severity.CAUTION, "long cadence")
        out = lc.normalise()
        assert out.quality.has("undersampled_cadence")
