"""Content-addressed cache behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest
from astropy import units as u
from astropy.table import Table

from astrolab.archives.base import QuerySpec
from astrolab.archives.cache import CorruptCacheError, QueryCache


@pytest.fixture
def spec() -> QuerySpec:
    return QuerySpec("mast", "observations", {"target_name": "WASP-18", "mission": "TESS"})


class TestQuerySpecHashing:
    def test_none_values_dropped(self) -> None:
        """``{"author": None}`` and ``{}`` are the same question."""
        a = QuerySpec("mast", "obs", {"t": "X", "author": None})
        b = QuerySpec("mast", "obs", {"t": "X"})
        assert a.content_hash() == b.content_hash()

    def test_key_order_irrelevant(self) -> None:
        a = QuerySpec("mast", "obs", {"a": 1, "b": 2})
        b = QuerySpec("mast", "obs", {"b": 2, "a": 1})
        assert a.content_hash() == b.content_hash()

    def test_operation_participates_in_hash(self) -> None:
        a = QuerySpec("mast", "observations", {"t": "X"})
        b = QuerySpec("mast", "products", {"t": "X"})
        assert a.content_hash() != b.content_hash()

    def test_client_version_participates_in_hash(self) -> None:
        """Changing how a query is built must invalidate old answers."""
        a = QuerySpec("mast", "obs", {"t": "X"}, client_version="1")
        b = QuerySpec("mast", "obs", {"t": "X"}, client_version="2")
        assert a.content_hash() != b.content_hash()

    def test_unserialisable_params_rejected_at_construction(self) -> None:
        with pytest.raises(TypeError, match="JSON-serialisable"):
            QuerySpec("mast", "obs", {"f": {1, 2, 3}})

    def test_round_trips_through_dict(self, spec: QuerySpec) -> None:
        """The identity the manifest-replay guarantee rests on."""
        assert QuerySpec.from_dict(spec.to_dict()).content_hash() == spec.content_hash()

    def test_from_dict_rejects_incomplete_data(self) -> None:
        with pytest.raises(ValueError, match="missing keys"):
            QuerySpec.from_dict({"archive": "mast"})


class TestCacheRoundTrip:
    def test_miss_then_hit(self, tmp_path: Path, spec: QuerySpec, observation_table: Table) -> None:
        cache = QueryCache(tmp_path)
        assert cache.get(spec) is None
        cache.put(spec, observation_table, fetched_at="2026-08-18T00:00:00+00:00")
        got = cache.get(spec)
        assert got is not None
        table, meta = got
        assert len(table) == len(observation_table)
        assert meta["fetched_at"] == "2026-08-18T00:00:00+00:00"

    def test_units_survive_the_round_trip(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        """ECSV rather than CSV precisely so this holds."""
        cache = QueryCache(tmp_path)
        cache.put(spec, observation_table)
        got = cache.get(spec)
        assert got is not None
        assert got[0]["t_exptime"].unit == u.s

    def test_different_spec_does_not_collide(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        cache = QueryCache(tmp_path)
        cache.put(spec, observation_table)
        other = QuerySpec("mast", "observations", {"target_name": "WASP-19"})
        assert cache.get(other) is None


class TestIntegrity:
    def test_tampered_payload_detected(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        """A silent re-fetch would hide a real fault -- a partial write or a bad disk."""
        cache = QueryCache(tmp_path)
        cache.put(spec, observation_table)
        cache.entry_for(spec).result_path.write_text("corrupted")
        with pytest.raises(CorruptCacheError, match="integrity check"):
            cache.get(spec)

    def test_unreadable_metadata_detected(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        cache = QueryCache(tmp_path)
        cache.put(spec, observation_table)
        cache.entry_for(spec).metadata_path.write_text("{not json")
        with pytest.raises(CorruptCacheError):
            cache.get(spec)

    def test_no_partial_entry_left_after_a_failed_write(
        self, tmp_path: Path, spec: QuerySpec
    ) -> None:
        """A crash mid-write must not leave something a later run reads as complete."""
        cache = QueryCache(tmp_path)

        class Exploding(Table):
            def write(self, *args: object, **kwargs: object) -> None:
                raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            cache.put(spec, Exploding({"a": [1, 2]}))
        assert not cache.entry_for(spec).exists
        assert not list(tmp_path.glob("**/.tmp-*"))


class TestExpiryAndDisabling:
    def test_disabled_cache_never_hits_and_never_writes(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        cache = QueryCache(tmp_path, enabled=False)
        meta = cache.put(spec, observation_table)
        assert meta["cached"] is False
        assert cache.get(spec) is None
        assert not cache.entry_for(spec).exists

    def test_entry_older_than_max_age_is_a_miss(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        QueryCache(tmp_path).put(spec, observation_table, fetched_at="2020-01-01T00:00:00+00:00")
        fresh = QueryCache(tmp_path, max_age_days=30.0)
        assert fresh.get(spec) is None
        # Without an age limit the same entry is still served: entries do not expire
        # by default, so refreshing stays a deliberate act.
        assert QueryCache(tmp_path).get(spec) is not None

    def test_format_version_mismatch_is_a_miss_not_a_crash(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        import json

        cache = QueryCache(tmp_path)
        cache.put(spec, observation_table)
        entry = cache.entry_for(spec)
        meta = json.loads(entry.metadata_path.read_text())
        meta["format_version"] = 9999
        entry.metadata_path.write_text(json.dumps(meta))
        assert cache.get(spec) is None


class TestMaintenance:
    def test_invalidate(self, tmp_path: Path, spec: QuerySpec, observation_table: Table) -> None:
        cache = QueryCache(tmp_path)
        cache.put(spec, observation_table)
        assert cache.invalidate(spec) is True
        assert cache.get(spec) is None
        assert cache.invalidate(spec) is False

    def test_stats_and_listing(
        self, tmp_path: Path, spec: QuerySpec, observation_table: Table
    ) -> None:
        cache = QueryCache(tmp_path)
        cache.put(spec, observation_table)
        stats = cache.stats()
        assert stats["n_entries"] == 1
        assert stats["total_bytes"] > 0
        entries = cache.iter_entries()
        assert entries[0]["spec_hash"] == spec.content_hash()
