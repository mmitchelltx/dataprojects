"""MAST client: spec construction, caching behaviour, and data-rights handling.

Query *construction* is pure and fully tested here. Query *execution* against the live
archive is covered only by the ``network``-marked tests at the bottom, which are the only
thing that can confirm the MAST query vocabulary is correct. Nothing in this file asserts a
scientific result; the stub below exercises the caching and provenance layer, which is the
unit under test, and it is never used to stand in for archive data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from astropy.table import Table

from astrolab.archives.base import (
    ArchiveError,
    EmptyResultError,
    ProprietaryDataError,
    QuerySpec,
)
from astrolab.archives.cache import QueryCache
from astrolab.archives.mast import (
    MastClient,
    build_observation_spec,
    build_product_spec,
)


class StubbedClient(MastClient):
    """MastClient with the network call replaced, to test everything around it."""

    def __init__(self, cache: QueryCache, table: Table, **kw: object) -> None:
        super().__init__(cache, **kw)  # type: ignore[arg-type]
        self.table = table
        self.execute_calls = 0

    def _execute(self, spec: QuerySpec) -> Table:
        self.execute_calls += 1
        if len(self.table) == 0:
            raise EmptyResultError(spec, "stub configured to return nothing")
        return self._filter_data_rights(self.table, spec)


class TestSpecConstruction:
    def test_builds_a_complete_spec(self) -> None:
        spec = build_observation_spec(
            target_name="WASP-18",
            mission="TESS",
            product="lightcurve",
            author="SPOC",
            exptime_seconds=120,
        )
        assert spec.archive == "mast"
        assert spec.operation == "observations"
        assert spec.params["obs_collection"] == "TESS"
        assert spec.params["dataproduct_type"] == "timeseries"

    def test_sequence_order_does_not_change_the_question(self) -> None:
        a = build_observation_spec(target_name="X", mission="TESS", sequence=[3, 2])
        b = build_observation_spec(target_name="X", mission="TESS", sequence=[2, 3])
        assert a.content_hash() == b.content_hash()

    def test_product_ids_order_independent(self) -> None:
        a = build_product_spec(["b", "a"], mission="TESS")
        b = build_product_spec(["a", "b"], mission="TESS")
        assert a.content_hash() == b.content_hash()

    def test_empty_product_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_product_spec([], mission="TESS")

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"mission": "TESS"}, "target_name or both"),
            ({"target_name": "X", "mission": "HUBBLE"}, "unknown mission"),
            ({"target_name": "X", "product": "sandwich"}, "unknown product"),
            ({"target_name": "X", "radius_arcsec": 0.0}, "must be positive"),
        ],
    )
    def test_invalid_inputs_rejected_before_any_network_call(
        self, kwargs: dict, match: str
    ) -> None:
        """Validation at construction: a typo should cost a second, not a download."""
        with pytest.raises(ValueError, match=match):
            build_observation_spec(**kwargs)

    def test_coordinates_accepted_without_a_name(self) -> None:
        spec = build_observation_spec(ra_deg=24.35, dec_deg=-45.68, mission="TESS")
        assert spec.params["target_name"] is None
        assert spec.params["ra_deg"] == 24.35


class TestFetchAndCache:
    def test_second_fetch_is_served_from_cache(
        self, tmp_path: Path, observation_table: Table
    ) -> None:
        """The archive must not be hit twice for the same question."""
        client = StubbedClient(QueryCache(tmp_path), observation_table)
        spec = build_observation_spec(target_name="WASP-18", mission="TESS")

        first, rec1 = client.fetch(spec)
        second, rec2 = client.fetch(spec)

        assert client.execute_calls == 1
        assert rec1.served_from == "live"
        assert rec2.served_from == "cache"
        assert len(first.table) == len(second.table)

    def test_cache_record_preserves_original_fetch_time(
        self, tmp_path: Path, observation_table: Table
    ) -> None:
        """A year-old cached result must not look freshly retrieved in the manifest."""
        client = StubbedClient(QueryCache(tmp_path), observation_table)
        spec = build_observation_spec(target_name="WASP-18", mission="TESS")
        _, rec1 = client.fetch(spec)
        _, rec2 = client.fetch(spec)
        assert rec2.cached_at == rec1.timestamp
        assert rec2.timestamp >= rec1.timestamp

    def test_record_carries_a_replayable_spec(
        self, tmp_path: Path, observation_table: Table
    ) -> None:
        client = StubbedClient(QueryCache(tmp_path), observation_table)
        spec = build_observation_spec(target_name="WASP-18", mission="TESS")
        _, record = client.fetch(spec)
        rebuilt = QuerySpec.from_dict(record.spec)
        assert rebuilt.content_hash() == record.spec_hash == spec.content_hash()

    def test_empty_result_raises_and_is_not_cached(self, tmp_path: Path) -> None:
        """Directive 1: nothing matched is said out loud, and nothing is stored."""
        cache = QueryCache(tmp_path)
        client = StubbedClient(cache, Table({"obsid": []}))
        spec = build_observation_spec(target_name="Nonexistent", mission="TESS")
        with pytest.raises(EmptyResultError, match="matched no products"):
            client.fetch(spec)
        assert cache.get(spec) is None

    def test_unsupported_operation_rejected(self, tmp_path: Path) -> None:
        client = MastClient(QueryCache(tmp_path))
        with pytest.raises(ArchiveError, match="unsupported MAST operation"):
            client._execute(QuerySpec("mast", "teleport", {"x": 1}))


class TestDataRights:
    def test_proprietary_rows_dropped_by_default(
        self, tmp_path: Path, observation_table: Table
    ) -> None:
        client = StubbedClient(QueryCache(tmp_path), observation_table)
        result, _ = client.fetch(build_observation_spec(target_name="X", mission="TESS"))
        assert len(result.table) == 2
        assert all(str(v).upper() == "PUBLIC" for v in result.table["dataRights"])

    def test_proprietary_rows_kept_when_requested(
        self, tmp_path: Path, observation_table: Table
    ) -> None:
        client = StubbedClient(QueryCache(tmp_path), observation_table, allow_proprietary=True)
        result, _ = client.fetch(build_observation_spec(target_name="X", mission="TESS"))
        assert len(result.table) == 3

    def test_all_proprietary_raises_a_distinct_error(self, tmp_path: Path) -> None:
        """ "Exists but not yours" and "does not exist" demand different responses."""
        table = Table({"obsid": ["1"], "dataRights": ["EXCLUSIVE_ACCESS"]})
        client = StubbedClient(QueryCache(tmp_path), table)
        with pytest.raises(ProprietaryDataError, match="exclusive access period"):
            client.fetch(build_observation_spec(target_name="X", mission="JWST"))

    def test_missing_data_rights_column_is_not_guessed(self, tmp_path: Path) -> None:
        """Inventing an access determination is worse than admitting we lack one."""
        table = Table({"obsid": ["1", "2"]})
        client = StubbedClient(QueryCache(tmp_path), table)
        result, _ = client.fetch(build_observation_spec(target_name="X", mission="TESS"))
        assert len(result.table) == 2


@pytest.mark.network
class TestLiveArchive:
    """Live MAST retrieval.

    These are the only tests that can confirm the query vocabulary in
    ``PRODUCT_TO_DATAPRODUCT_TYPE`` and the criteria names in ``_execute_observations`` are
    right. They were not runnable in the environment this module was developed in, where MAST
    is blocked by egress policy; run them with ``--run-network`` before relying on retrieval.
    """

    def test_retrieves_public_tess_lightcurves_for_wasp18(self, tmp_path: Path) -> None:
        client = MastClient(QueryCache(tmp_path))
        spec = build_observation_spec(
            target_name="WASP-18",
            ra_deg=24.354292,
            dec_deg=-45.677891,
            radius_arcsec=20.0,
            mission="TESS",
            product="lightcurve",
            author="SPOC",
            exptime_seconds=120,
        )
        result, record = client.fetch(spec)
        assert len(result.table) > 0
        assert record.served_from == "live"
        assert all(str(v).upper() == "PUBLIC" for v in result.table["dataRights"])

    def test_second_call_hits_the_cache(self, tmp_path: Path) -> None:
        client = MastClient(QueryCache(tmp_path))
        spec = build_observation_spec(target_name="WASP-18", mission="TESS", product="lightcurve")
        client.fetch(spec)
        _, record = client.fetch(spec)
        assert record.served_from == "cache"

    def test_nonexistent_target_raises_empty_result(self, tmp_path: Path) -> None:
        client = MastClient(QueryCache(tmp_path))
        spec = build_observation_spec(
            target_name="ASTROLAB-NOT-A-REAL-TARGET-99999", mission="TESS"
        )
        with pytest.raises((EmptyResultError, ArchiveError)):
            client.fetch(spec)

    def test_retrieves_a_public_jwst_product(self, tmp_path: Path) -> None:
        """JWST ERS programs are zero-EAP, so this needs no credentials."""
        client = MastClient(QueryCache(tmp_path))
        spec = build_observation_spec(target_name="WASP-39", mission="JWST", product="spectrum")
        result, _ = client.fetch(spec)
        assert len(result.table) > 0
