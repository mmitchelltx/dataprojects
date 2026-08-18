"""End-to-end transit pipeline, report generation, and the `astrolab run` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from astrolab.cli import EXIT_OK, EXIT_USAGE, app
from astrolab.core.config import AstrolabConfig
from astrolab.pipelines.report import write_report
from astrolab.pipelines.transit import load_source, run_transit_pipeline

runner = CliRunner()


def config_dict(tmp_path: Path, **overrides: object) -> dict:
    cfg = {
        "run": {
            "name": "test-transit",
            "seed": 20260818,
            "output_dir": str(tmp_path / "outputs"),
        },
        "target": {"name": "K2-3 (EPIC 201367065)"},
        "source": {"kind": "bundled_validation", "variant": "raw"},
        "transit": {
            "expected_duration_hours": 2.4,
            "catalogue_target": "K2-3",
            "search": {
                "min_period_days": 8.0,
                "max_period_days": 12.0,
                "max_candidates": 1,
                "cross_check_bls": False,
            },
            "fit": {"enabled": False},
        },
    }
    cfg.update(overrides)  # type: ignore[arg-type]
    return cfg


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(config_dict(tmp_path)))
    return p


class TestSourceLoading:
    def test_loads_the_bundled_validation_curve(self, tmp_path: Path) -> None:
        cfg = AstrolabConfig.model_validate(config_dict(tmp_path))
        lc = load_source(cfg)
        assert len(lc) == 3632
        assert lc.quality.has("third_party_mirror")

    def test_unimplemented_source_says_so(self, tmp_path: Path) -> None:
        """A source that cannot be honoured must fail loudly, not silently substitute."""
        raw = config_dict(tmp_path)
        raw["source"] = {"kind": "mast", "mission": "TESS", "time_system": "BTJD"}
        cfg = AstrolabConfig.model_validate(raw)
        with pytest.raises(NotImplementedError, match="unverified in this environment"):
            load_source(cfg)

    def test_local_csv_requires_a_path(self, tmp_path: Path) -> None:
        raw = config_dict(tmp_path)
        raw["source"] = {"kind": "local_csv"}
        with pytest.raises(Exception, match="requires 'path'"):
            AstrolabConfig.model_validate(raw)


class TestConfigShapes:
    def test_config_needs_a_query_or_a_source(self, tmp_path: Path) -> None:
        raw = config_dict(tmp_path)
        del raw["source"]
        del raw["transit"]
        with pytest.raises(Exception, match=r"either 'query'"):
            AstrolabConfig.model_validate(raw)

    def test_source_without_transit_is_rejected(self, tmp_path: Path) -> None:
        raw = config_dict(tmp_path)
        del raw["transit"]
        with pytest.raises(Exception, match="must also define 'transit'"):
            AstrolabConfig.model_validate(raw)

    def test_shipped_pipeline_config_is_valid(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "k2-3-transit.yaml"
        cfg = AstrolabConfig.from_yaml(path)
        assert cfg.transit is not None
        assert cfg.transit.expected_duration_hours == 2.4


@pytest.mark.slow
class TestPipeline:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("pipeline")
        cfg = AstrolabConfig.model_validate(config_dict(tmp))
        return run_transit_pipeline(cfg, output_dir=tmp / "outputs")

    def test_finds_k2_3_b(self, result) -> None:
        assert len(result.candidates) == 1
        assert result.candidates[0].period.value == pytest.approx(10.0546535, abs=0.005)

    def test_labels_it_a_known_object(self, result) -> None:
        assert result.vetting[0].disposition == "KNOWN_OBJECT"

    def test_quality_flags_reach_the_top_level(self, result) -> None:
        """A caveat raised at ingestion must still be attached to the final result."""
        assert result.quality.has("third_party_mirror")
        assert result.quality.has("estimated_uncertainties")

    def test_flags_are_deduplicated(self, result) -> None:
        """Flags propagate along every edge; duplicates would bury the distinct ones."""
        keys = [(f.flag, f.message) for f in result.quality.flags]
        assert len(keys) == len(set(keys))

    def test_summary_is_json_serialisable(self, result) -> None:
        json.dumps(result.summary(), default=str)

    def test_manifest_records_seeds_and_config_hash(self, result) -> None:
        d = result.manifest.to_dict()
        assert d["seeds"]["master"] == 20260818
        assert len(d["config_hash"]) == 64

    def test_report_renders_self_contained_html(self, result, tmp_path: Path) -> None:
        out = write_report(result, tmp_path / "report.html")
        text = out.read_text()
        assert text.startswith("<!doctype html>")
        # Self-contained: no external asset references at all.
        assert 'src="data:image/png;base64,' in text
        assert "http://" not in text and "https://" not in text
        assert "Provenance appendix" in text

    def test_report_puts_caveats_before_results(self, result, tmp_path: Path) -> None:
        """The design commitment: a caveat buried under a number manufactures confidence."""
        text = write_report(result, tmp_path / "report.html").read_text()
        assert text.index("Read this first") < text.index("<h2>Data</h2>")
        assert text.index("Quality reservations") < text.index("<h2>Search</h2>")

    def test_report_states_the_tool_does_not_confirm_planets(self, result, tmp_path: Path) -> None:
        text = write_report(result, tmp_path / "report.html").read_text()
        assert "never confirmed" in text


@pytest.mark.slow
class TestRunCLI:
    def test_runs_and_writes_all_products(self, config_file: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", str(config_file)])
        assert result.exit_code == EXIT_OK, result.output

        run_dir = tmp_path / "outputs" / "test-transit"
        for name in ("report.html", "manifest.json", "summary.json"):
            assert (run_dir / name).is_file(), f"{name} missing"

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        kinds = {o["kind"] for o in manifest["outputs"]}
        assert {"report", "summary"} <= kinds

    def test_rejects_a_query_only_config(self, tmp_path: Path) -> None:
        """`astrolab run` and `astrolab query` need different configs; say which is missing."""
        p = tmp_path / "q.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "run": {"name": "q", "output_dir": str(tmp_path)},
                    "target": {"name": "WASP-18"},
                    "query": {"mission": "TESS"},
                }
            )
        )
        result = runner.invoke(app, ["run", str(p)])
        assert result.exit_code == EXIT_USAGE
        assert "astrolab query" in result.stderr

    def test_no_fit_flag_skips_fitting(self, config_file: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", str(config_file), "--no-fit"])
        assert result.exit_code == EXIT_OK
        summary = json.loads((tmp_path / "outputs" / "test-transit" / "summary.json").read_text())
        assert summary["fits"] == {}
