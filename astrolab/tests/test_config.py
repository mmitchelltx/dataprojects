"""Config validation and hash stability."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from astrolab.core.config import AstrolabConfig, ConfigError

MINIMAL = {
    "run": {"name": "demo"},
    "target": {"name": "WASP-18"},
    "query": {"mission": "TESS"},
}


def write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


class TestLoading:
    def test_minimal_config_loads(self, tmp_path: Path) -> None:
        cfg = AstrolabConfig.from_yaml(write(tmp_path, MINIMAL))
        assert cfg.run.name == "demo"
        assert cfg.query.mission == "TESS"
        assert cfg.query.product == "any"  # default applied

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            AstrolabConfig.from_yaml(tmp_path / "nope.yaml")

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ConfigError, match="empty"):
            AstrolabConfig.from_yaml(p)

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("run: [unclosed\n")
        with pytest.raises(ConfigError, match="malformed YAML"):
            AstrolabConfig.from_yaml(p)

    def test_non_mapping_top_level(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- a\n- b\n")
        with pytest.raises(ConfigError, match="must be a mapping"):
            AstrolabConfig.from_yaml(p)


class TestValidation:
    def test_unknown_key_is_an_error(self, tmp_path: Path) -> None:
        """An unrecognised key is a typo'd instruction, not something to ignore.

        Silently dropping it would mean the config file no longer describes the run.
        """
        data = {**MINIMAL, "query": {"mission": "TESS", "athor": "SPOC"}}
        with pytest.raises(ConfigError, match="athor"):
            AstrolabConfig.from_yaml(write(tmp_path, data))

    def test_target_needs_name_or_coordinates(self, tmp_path: Path) -> None:
        data = {**MINIMAL, "target": {"search_radius_arcsec": 10.0}}
        with pytest.raises(ConfigError, match=r"requires either"):
            AstrolabConfig.from_yaml(write(tmp_path, data))

    def test_half_specified_coordinates_rejected(self, tmp_path: Path) -> None:
        data = {**MINIMAL, "target": {"ra_deg": 24.35}}
        with pytest.raises(ConfigError):
            AstrolabConfig.from_yaml(write(tmp_path, data))

    def test_out_of_range_declination(self, tmp_path: Path) -> None:
        data = {**MINIMAL, "target": {"ra_deg": 24.0, "dec_deg": -120.0}}
        with pytest.raises(ConfigError):
            AstrolabConfig.from_yaml(write(tmp_path, data))

    def test_unknown_mission_rejected(self, tmp_path: Path) -> None:
        data = {**MINIMAL, "query": {"mission": "HUBBLE"}}
        with pytest.raises(ConfigError):
            AstrolabConfig.from_yaml(write(tmp_path, data))

    def test_run_name_must_be_path_safe(self, tmp_path: Path) -> None:
        data = {**MINIMAL, "run": {"name": "my run/1"}}
        with pytest.raises(ConfigError, match="path-safe"):
            AstrolabConfig.from_yaml(write(tmp_path, data))


class TestHashing:
    def test_hash_is_order_independent(self, tmp_path: Path) -> None:
        a = AstrolabConfig.from_yaml(write(tmp_path, MINIMAL))
        reordered = {
            "query": {"mission": "TESS"},
            "target": {"name": "WASP-18"},
            "run": {"name": "demo"},
        }
        b = AstrolabConfig.from_yaml(write(tmp_path, reordered))
        assert a.content_hash() == b.content_hash()

    def test_hash_changes_with_a_meaningful_field(self, tmp_path: Path) -> None:
        a = AstrolabConfig.from_yaml(write(tmp_path, MINIMAL))
        changed = {**MINIMAL, "query": {"mission": "TESS", "author": "QLP"}}
        b = AstrolabConfig.from_yaml(write(tmp_path, changed))
        assert a.content_hash() != b.content_hash()

    def test_resolved_dict_includes_defaults(self, tmp_path: Path) -> None:
        """Recording resolved values means a changed default cannot silently alter a run."""
        cfg = AstrolabConfig.from_yaml(write(tmp_path, MINIMAL))
        resolved = cfg.resolved_dict()
        assert resolved["query"]["product"] == "any"
        assert resolved["run"]["seed"] == 0
        assert "cache" in resolved

    def test_round_trips_through_yaml(self, tmp_path: Path) -> None:
        cfg = AstrolabConfig.from_yaml(write(tmp_path, MINIMAL))
        out = tmp_path / "resolved.yaml"
        cfg.to_yaml(out)
        again = AstrolabConfig.from_yaml(out)
        assert again.content_hash() == cfg.content_hash()


class TestShippedConfigs:
    def test_repository_configs_are_valid(self) -> None:
        """Every config in configs/ must load. A broken example is worse than none."""
        root = Path(__file__).resolve().parents[1] / "configs"
        files = sorted(root.glob("*.yaml"))
        assert files, "no configs found to validate"
        for f in files:
            AstrolabConfig.from_yaml(f)
