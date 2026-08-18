"""CLI behaviour, including the meaning of each exit code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from astrolab.cli import (
    EXIT_ARCHIVE,
    EXIT_EMPTY,
    EXIT_OK,
    EXIT_USAGE,
    app,
)

runner = CliRunner()

CONFIG = {
    "run": {"name": "test-run", "seed": 7},
    "target": {"name": "WASP-18", "ra_deg": 24.354292, "dec_deg": -45.677891},
    "query": {"mission": "TESS", "product": "lightcurve", "author": "SPOC"},
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "run.yaml"
    cfg = json.loads(json.dumps(CONFIG))
    cfg["cache"] = {"directory": str(tmp_path / "cache")}
    cfg["run"]["output_dir"] = str(tmp_path / "outputs")
    p.write_text(yaml.safe_dump(cfg))
    return p


class TestVersionAndHelp:
    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == EXIT_OK
        assert result.stdout.strip()

    def test_help_lists_the_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        for command in ("query", "replay", "validate", "cache"):
            assert command in result.stdout


class TestValidate:
    def test_accepts_a_good_config(self, config_file: Path) -> None:
        result = runner.invoke(app, ["validate", str(config_file)])
        assert result.exit_code == EXIT_OK
        assert "config valid" in result.stdout
        assert "will run" in result.stdout

    def test_rejects_a_bad_config_with_usage_exit_code(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"run": {"name": "x"}}))
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == EXIT_USAGE

    def test_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(tmp_path / "nope.yaml")])
        assert result.exit_code == EXIT_USAGE

    def test_shipped_config_validates(self) -> None:
        cfg = Path(__file__).resolve().parents[1] / "configs" / "wasp18b-tess.yaml"
        result = runner.invoke(app, ["validate", str(cfg)])
        assert result.exit_code == EXIT_OK


class TestQueryWritesAManifestEvenOnFailure:
    def test_unreachable_archive_yields_exit_4_and_a_manifest(
        self, config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed run still records what it attempted -- that is when you most need it."""
        import astrolab.archives.mast as mast_mod

        def boom(self: object, spec: object) -> None:
            raise ConnectionError("network unreachable")

        monkeypatch.setattr(mast_mod.MastClient, "_execute", boom)
        result = runner.invoke(app, ["query", str(config_file)])
        assert result.exit_code == EXIT_ARCHIVE

        manifest_path = tmp_path / "outputs" / "test-run" / "manifest.json"
        assert manifest_path.is_file()
        data = json.loads(manifest_path.read_text())
        assert data["status"] == "failed"
        assert data["queries"][0]["error"]
        assert data["config_hash"]

    def test_empty_result_yields_exit_2_and_a_quality_flag(
        self, config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing found is a real answer, and it gets its own exit code."""
        import astrolab.archives.mast as mast_mod
        from astrolab.archives.base import EmptyResultError

        def empty(self: object, spec: object) -> None:
            raise EmptyResultError(spec, "no observations")  # type: ignore[arg-type]

        monkeypatch.setattr(mast_mod.MastClient, "_execute", empty)
        result = runner.invoke(app, ["query", str(config_file)])
        assert result.exit_code == EXIT_EMPTY

        data = json.loads((tmp_path / "outputs" / "test-run" / "manifest.json").read_text())
        assert data["status"] == "empty"
        assert any(f["flag"] == "empty_result" for f in data["quality_flags"])

    def test_successful_query_writes_table_and_manifest(
        self, config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from astropy.table import Table

        import astrolab.archives.mast as mast_mod

        def ok(self: object, spec: object) -> Table:
            return Table({"obsid": ["1", "2"], "dataRights": ["PUBLIC", "PUBLIC"]})

        monkeypatch.setattr(mast_mod.MastClient, "_execute", ok)
        result = runner.invoke(app, ["query", str(config_file)])
        assert result.exit_code == EXIT_OK

        run_dir = tmp_path / "outputs" / "test-run"
        assert (run_dir / "observations.ecsv").is_file()
        data = json.loads((run_dir / "manifest.json").read_text())
        assert data["status"] == "completed"
        assert data["outputs"][0]["n_rows"] == 2
        assert data["seeds"]["master"] == 7


class TestReplay:
    def test_replays_a_recorded_query_exactly(
        self, config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The executable form of the reproducibility claim."""
        from astropy.table import Table

        import astrolab.archives.mast as mast_mod

        monkeypatch.setattr(
            mast_mod.MastClient,
            "_execute",
            lambda self, spec: Table({"obsid": ["1"], "dataRights": ["PUBLIC"]}),
        )
        runner.invoke(app, ["query", str(config_file)])

        manifest = tmp_path / "outputs" / "test-run" / "manifest.json"
        result = runner.invoke(app, ["replay", str(manifest)])
        assert result.exit_code == EXIT_OK
        assert "hash matches" in result.stdout
        assert "reproduce exactly" in result.stdout

    def test_detects_a_tampered_manifest(
        self, config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the recorded spec no longer hashes to the recorded value, say so loudly."""
        from astropy.table import Table

        import astrolab.archives.mast as mast_mod

        monkeypatch.setattr(
            mast_mod.MastClient,
            "_execute",
            lambda self, spec: Table({"obsid": ["1"], "dataRights": ["PUBLIC"]}),
        )
        runner.invoke(app, ["query", str(config_file)])

        manifest = tmp_path / "outputs" / "test-run" / "manifest.json"
        data = json.loads(manifest.read_text())
        data["queries"][0]["spec"]["params"]["target_name"] = "SOMETHING-ELSE"
        manifest.write_text(json.dumps(data))

        result = runner.invoke(app, ["replay", str(manifest)])
        assert result.exit_code == EXIT_ARCHIVE
        # The mismatch is a diagnostic, so it goes to stderr; stdout stays a data channel.
        assert "hash mismatch" in result.stderr

    def test_missing_manifest(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["replay", str(tmp_path / "nope.json")])
        assert result.exit_code == EXIT_USAGE


class TestCacheCommands:
    def test_stats_on_an_empty_cache(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["cache", "stats", "-d", str(tmp_path)])
        assert result.exit_code == EXIT_OK
        assert json.loads(result.stdout)["n_entries"] == 0

    def test_list_reports_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["cache", "list", "-d", str(tmp_path)])
        assert result.exit_code == EXIT_OK
        assert "empty" in result.stdout
