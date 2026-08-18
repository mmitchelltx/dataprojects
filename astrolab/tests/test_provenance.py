"""Run manifests: what they record, and what they admit they cannot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from astrolab.core.provenance import (
    QueryRecord,
    RunManifest,
    environment_record,
    git_state,
)
from astrolab.core.quality import Severity


@pytest.fixture
def manifest() -> RunManifest:
    return RunManifest(run_name="demo", config={"run": {"name": "demo"}}, config_hash="abc123")


class TestGitState:
    def test_reports_unavailable_outside_a_repository(self, tmp_path: Path) -> None:
        """Honesty over silence: an absent field would look clean and be a lie."""
        state = git_state(tmp_path)
        assert state.available is False
        assert state.sha is None
        assert "cannot be identified" in state.note
        assert state.reproducible is False

    def test_detects_a_clean_tree(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

        state = git_state(tmp_path)
        assert state.available is True
        assert state.dirty is False
        assert state.reproducible is True
        assert state.sha is not None and len(state.sha) == 40

    def test_detects_a_dirty_tree(self, tmp_path: Path) -> None:
        """A SHA identifies the code only if the tree matched it."""
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("changed")

        state = git_state(tmp_path)
        assert state.dirty is True
        assert "a.txt" in state.dirty_files
        assert state.reproducible is False
        assert "does not identify the code" in state.note


class TestEnvironmentRecord:
    def test_records_science_relevant_versions(self) -> None:
        env = environment_record()
        assert env.packages["astropy"] != "not installed"
        assert env.packages["numpy"] != "not installed"
        assert env.python_version.startswith("3.")

    def test_missing_package_recorded_explicitly(self) -> None:
        """ "not installed" is information; an absent key is ambiguous."""
        env = environment_record()
        assert all(isinstance(v, str) and v for v in env.packages.values())

    def test_absent_lockfile_recorded_as_none(self, tmp_path: Path) -> None:
        env = environment_record(lockfile=tmp_path / "nonexistent.lock")
        assert env.lockfile_hash is None

    def test_lockfile_hashed_when_present(self, tmp_path: Path) -> None:
        lock = tmp_path / "uv.lock"
        lock.write_text("version = 1\n")
        env = environment_record(lockfile=lock)
        assert env.lockfile_hash is not None and len(env.lockfile_hash) == 64


class TestManifestRecording:
    def test_records_queries_seeds_and_overrides(self, manifest: RunManifest) -> None:
        manifest.record_query(
            QueryRecord(
                archive="mast",
                spec={"archive": "mast", "operation": "obs", "params": {}},
                spec_hash="deadbeef",
                timestamp="2026-08-18T00:00:00+00:00",
                served_from="cache",
                n_results=3,
            )
        )
        manifest.record_seed("detrend", 12345)
        manifest.record_override("allow_unconverged", True)
        d = manifest.to_dict()
        assert d["queries"][0]["served_from"] == "cache"
        assert d["seeds"]["detrend"] == 12345
        assert d["overrides"]["allow_unconverged"] is True

    def test_quality_flags_travel_with_the_manifest(self, manifest: RunManifest) -> None:
        manifest.quality.add(
            "insufficient_snr", Severity.UNRELIABLE, "S/N 2.1 below threshold 7.0", snr=2.1
        )
        d = manifest.to_dict()
        assert d["quality_flags"][0]["severity"] == "UNRELIABLE"
        assert d["quality_flags"][0]["context"]["snr"] == 2.1


class TestReproducibilityAssessment:
    def test_overrides_make_a_run_non_reproducible(self, manifest: RunManifest) -> None:
        """An override is a human bypassing a refusal. Hiding it would defeat the refusal."""
        manifest.record_override("allow_unconverged", True)
        assessment = manifest.reproducibility_assessment()
        assert assessment["fully_reproducible"] is False
        assert any("override" in c for c in assessment["caveats"])

    def test_live_query_is_flagged(self, manifest: RunManifest) -> None:
        manifest.record_query(
            QueryRecord(
                archive="mast",
                spec={},
                spec_hash="x",
                timestamp="t",
                served_from="live",
            )
        )
        assert any("live" in c for c in manifest.reproducibility_assessment()["caveats"])

    def test_cached_query_is_not_flagged_as_live(self, manifest: RunManifest) -> None:
        manifest.record_query(
            QueryRecord(
                archive="mast",
                spec={},
                spec_hash="x",
                timestamp="t",
                served_from="cache",
            )
        )
        assert not any("live" in c for c in manifest.reproducibility_assessment()["caveats"])


class TestSerialisation:
    def test_writes_and_reloads(self, manifest: RunManifest, tmp_path: Path) -> None:
        manifest.finish("completed")
        path = manifest.write(tmp_path / "sub" / "manifest.json")
        assert path.is_file()
        data = RunManifest.load(path)
        assert data["run_name"] == "demo"
        assert data["status"] == "completed"
        assert data["schema_version"] == 1

    def test_output_is_valid_json(self, manifest: RunManifest, tmp_path: Path) -> None:
        path = manifest.write(tmp_path / "manifest.json")
        json.loads(path.read_text())

    def test_manifest_written_even_for_a_failed_run(
        self, manifest: RunManifest, tmp_path: Path
    ) -> None:
        """A manifest for a crashed run is exactly when you most want to know what it did."""
        manifest.finish("failed", error="archive unreachable")
        data = RunManifest.load(manifest.write(tmp_path / "m.json"))
        assert data["status"] == "failed"
        assert data["error"] == "archive unreachable"
