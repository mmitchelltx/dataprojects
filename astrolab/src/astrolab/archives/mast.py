"""MAST (Mikulski Archive for Space Telescopes) client.

MAST hosts TESS, Kepler/K2, JWST, and HST holdings. This client wraps
:mod:`astroquery.mast` with three things astroquery does not provide:

1. **Content-addressed caching**, so re-running a pipeline does not re-hammer a shared,
   publicly funded archive.
2. **Provenance records**, so the manifest can state exactly what was asked and whether the
   answer came from cache or from the live service.
3. **An explicit stance on empty and proprietary results.** A query matching nothing raises
   :class:`~astrolab.archives.base.EmptyResultError`; a query matching only exclusive-access
   data raises :class:`~astrolab.archives.base.ProprietaryDataError`. Neither returns an empty
   table for a caller to misread.

Design note: query *construction* is pure and lives in :func:`build_observation_spec`, separate
from query *execution*. That split is what lets the interesting logic be tested without a
network, and what makes the cache key provably a function of the question asked.

Exclusive access periods
------------------------
JWST GO programs default to a 12-month exclusive access period, though ERS, DD, and many GO
programs are zero-EAP and public immediately. TESS and Kepler data are public. This client's
default posture is public data only: ``include_proprietary`` must be set deliberately, and
even then the caller needs their own MAST authentication, which this client does not manage.

References
----------
astroquery : Ginsburg et al. 2019, AJ 157, 98. doi:10.3847/1538-3881/aafc33
MAST API documentation : https://mast.stsci.edu/api/v0/
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from astrolab.archives.base import (
    ArchiveError,
    EmptyResultError,
    ProprietaryDataError,
    QueryResult,
    QuerySpec,
)
from astrolab.archives.cache import QueryCache
from astrolab.core.logging import get_logger
from astrolab.core.provenance import QueryRecord

if TYPE_CHECKING:
    from astropy.table import Table

__all__ = [
    "MISSION_TO_COLLECTION",
    "MastClient",
    "build_observation_spec",
    "build_product_spec",
]

log = get_logger(__name__)

#: Client version. Bumping this invalidates cached entries built by older query logic, which
#: is the correct behaviour: if we changed how a question is asked, old answers are answers to
#: a different question.
CLIENT_VERSION = "1"

#: Mission name to MAST ``obs_collection`` value.
MISSION_TO_COLLECTION: dict[str, str] = {
    "TESS": "TESS",
    "Kepler": "Kepler",
    "K2": "K2",
    "JWST": "JWST",
    "HST": "HST",
}

#: Our product vocabulary to MAST ``dataproduct_type``. ``None`` means "do not constrain".
#:
#: These follow MAST's documented ``dataproduct_type`` vocabulary. They are exercised against
#: the live service only by the network-marked tests, which are the sole way to confirm them;
#: see ``docs/design.md`` on the archive-access limitation in this environment.
PRODUCT_TO_DATAPRODUCT_TYPE: dict[str, str | None] = {
    "lightcurve": "timeseries",
    "targetpixel": "cube",
    "spectrum": "spectrum",
    "image": "image",
    "any": None,
}


def build_observation_spec(
    *,
    target_name: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_arcsec: float = 30.0,
    mission: str = "TESS",
    product: str = "any",
    author: str | None = None,
    sequence: list[int] | None = None,
    exptime_seconds: int | None = None,
) -> QuerySpec:
    """Build a normalised observation-search spec. Pure: no network, no side effects.

    Either ``target_name`` or both coordinates must be given. Providing both is encouraged --
    name resolution failures are common and near-silent, and having coordinates to check
    against turns a wrong-object error into a caught one.

    Raises
    ------
    ValueError
        If the target is underspecified or the mission/product is not recognised. Raised at
        construction time so a typo costs a second rather than a download.
    """
    if target_name is None and (ra_deg is None or dec_deg is None):
        raise ValueError("build_observation_spec requires target_name or both ra_deg and dec_deg")
    if mission not in MISSION_TO_COLLECTION:
        raise ValueError(f"unknown mission {mission!r}; known: {sorted(MISSION_TO_COLLECTION)}")
    if product not in PRODUCT_TO_DATAPRODUCT_TYPE:
        raise ValueError(
            f"unknown product {product!r}; known: {sorted(PRODUCT_TO_DATAPRODUCT_TYPE)}"
        )
    if radius_arcsec <= 0:
        raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")

    params: dict[str, Any] = {
        "target_name": target_name,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "radius_arcsec": round(float(radius_arcsec), 6),
        "obs_collection": MISSION_TO_COLLECTION[mission],
        "mission": mission,
        "dataproduct_type": PRODUCT_TO_DATAPRODUCT_TYPE[product],
        "product": product,
        "author": author,
        # Sorted so that [1, 2] and [2, 1] are the same question and hash identically.
        "sequence": sorted(sequence) if sequence else None,
        "exptime_seconds": exptime_seconds,
    }
    return QuerySpec(
        archive="mast",
        operation="observations",
        params=params,
        client_version=CLIENT_VERSION,
    )


def build_product_spec(obs_ids: list[str], *, mission: str) -> QuerySpec:
    """Build a spec for the product list belonging to a set of observation ids.

    Observation ids are sorted so the spec is order-independent: the same set of observations
    is the same question regardless of the order the search returned them in.
    """
    if not obs_ids:
        raise ValueError("build_product_spec requires at least one observation id")
    return QuerySpec(
        archive="mast",
        operation="products",
        params={"obs_ids": sorted(str(o) for o in obs_ids), "mission": mission},
        client_version=CLIENT_VERSION,
    )


class MastClient:
    """Cache-aware, provenance-recording MAST client.

    Parameters
    ----------
    cache
        Query cache. Required -- there is no uncached mode, because an uncached pipeline is
        one that re-downloads on every run.
    allow_proprietary
        Whether to accept exclusive-access products. Default false. When false and a query
        matches *only* proprietary data, :class:`ProprietaryDataError` is raised so the user
        learns the data exist but are not theirs, rather than being told nothing was found.
    """

    def __init__(
        self,
        cache: QueryCache,
        *,
        allow_proprietary: bool = False,
    ) -> None:
        self.cache = cache
        self.allow_proprietary = allow_proprietary

    # -- public API ---------------------------------------------------------------------

    def fetch(self, spec: QuerySpec) -> tuple[QueryResult, QueryRecord]:
        """Execute a spec, preferring the cache, and return the result with its record.

        The :class:`QueryRecord` is returned alongside the data rather than logged and
        forgotten, so the caller is obliged to put it in the manifest. Provenance that depends
        on the caller remembering to ask for it is provenance that goes missing.
        """
        cached = self.cache.get(spec)
        if cached is not None:
            table, meta = cached
            log.info(
                "archive.query.cache_hit",
                archive=spec.archive,
                operation=spec.operation,
                spec_hash=spec.short_hash,
                n_rows=len(table),
                originally_fetched_at=meta.get("fetched_at"),
            )
            result = QueryResult(
                table=table,
                spec=spec,
                served_from="cache",
                fetched_at=str(meta.get("fetched_at")),
                result_hash=str(meta.get("result_hash")),
            )
            record = QueryRecord(
                archive=spec.archive,
                spec=spec.to_dict(),
                spec_hash=spec.content_hash(),
                timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                served_from="cache",
                n_results=len(table),
                result_hash=result.result_hash,
                cached_at=result.fetched_at,
            )
            return result, record

        log.info(
            "archive.query.live",
            archive=spec.archive,
            operation=spec.operation,
            spec_hash=spec.short_hash,
            params={k: v for k, v in spec.params.items() if v is not None},
        )
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            table = self._execute(spec)
        except (EmptyResultError, ProprietaryDataError):
            raise
        except Exception as exc:
            log.error(
                "archive.query.failed",
                archive=spec.archive,
                spec_hash=spec.short_hash,
                error=str(exc),
            )
            raise ArchiveError(f"MAST query failed for {spec.describe()}: {exc}") from exc

        meta = self.cache.put(spec, table, fetched_at=fetched_at)
        log.info(
            "archive.query.stored",
            spec_hash=spec.short_hash,
            n_rows=len(table),
            cached=meta.get("cached"),
        )
        result = QueryResult(
            table=table,
            spec=spec,
            served_from="live",
            fetched_at=fetched_at,
            result_hash=str(meta.get("result_hash")),
        )
        record = QueryRecord(
            archive=spec.archive,
            spec=spec.to_dict(),
            spec_hash=spec.content_hash(),
            timestamp=fetched_at,
            served_from="live",
            n_results=len(table),
            result_hash=result.result_hash,
        )
        return result, record

    # -- execution ----------------------------------------------------------------------

    def _execute(self, spec: QuerySpec) -> Table:
        """Dispatch a spec to the astroquery call that answers it."""
        if spec.operation == "observations":
            return self._execute_observations(spec)
        if spec.operation == "products":
            return self._execute_products(spec)
        raise ArchiveError(f"unsupported MAST operation: {spec.operation!r}")

    def _execute_observations(self, spec: QuerySpec) -> Table:
        from astroquery.mast import Observations

        p = spec.params
        criteria: dict[str, Any] = {"obs_collection": p["obs_collection"]}
        if p.get("dataproduct_type"):
            criteria["dataproduct_type"] = p["dataproduct_type"]
        if p.get("sequence"):
            criteria["sequence_number"] = p["sequence"]
        if p.get("exptime_seconds"):
            criteria["t_exptime"] = p["exptime_seconds"]
        if p.get("author"):
            criteria["provenance_name"] = p["author"]

        criteria["radius"] = f"{p['radius_arcsec']} arcsec"
        if p.get("target_name"):
            criteria["objectname"] = p["target_name"]
        else:
            criteria["s_ra"] = p["ra_deg"]
            criteria["s_dec"] = p["dec_deg"]

        table = Observations.query_criteria(**criteria)

        if table is None or len(table) == 0:
            raise EmptyResultError(
                spec,
                "no observations matched. Check the target name resolves, and that this "
                "mission actually observed it.",
            )

        return self._filter_data_rights(table, spec)

    def _filter_data_rights(self, table: Table, spec: QuerySpec) -> Table:
        """Drop exclusive-access rows unless proprietary data was explicitly requested.

        MAST reports access status in ``dataRights``. If the column is absent we do not guess:
        the table is returned unfiltered and a warning is logged, because inventing a
        data-rights determination is worse than admitting we could not make one.
        """
        if "dataRights" not in table.colnames:
            log.warning(
                "archive.data_rights.unknown",
                spec_hash=spec.short_hash,
                detail="result has no dataRights column; access status not determined",
            )
            return table

        if self.allow_proprietary:
            return table

        public_mask = [str(v).upper() == "PUBLIC" for v in table["dataRights"]]
        n_public = sum(public_mask)
        if n_public == 0:
            raise ProprietaryDataError(
                f"{len(table)} matching observation(s) exist for {spec.describe()}, but all are "
                f"under an exclusive access period. They are not missing -- they are not yours "
                f"yet. Set query.include_proprietary=true and supply MAST credentials, or wait "
                f"for the exclusive access period to end."
            )
        if n_public < len(table):
            log.info(
                "archive.data_rights.filtered",
                spec_hash=spec.short_hash,
                kept=n_public,
                dropped=len(table) - n_public,
            )
        return table[public_mask]

    def _execute_products(self, spec: QuerySpec) -> Table:
        from astroquery.mast import Observations

        obs_ids = spec.params["obs_ids"]
        products = Observations.get_product_list(obs_ids)
        if products is None or len(products) == 0:
            raise EmptyResultError(
                spec, f"observations {obs_ids[:3]}... exist but expose no data products"
            )
        return products
