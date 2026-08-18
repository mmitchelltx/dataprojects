"""Content-addressed cache for archive queries and downloaded files.

Two requirements shape this design.

**Re-running a pipeline must not re-hammer MAST.** These are shared community resources funded
by everyone's taxes; a pipeline that re-downloads on every invocation is antisocial, and one
that does it inside a parameter sweep is worse.

**A cached answer and a live answer are different facts, and the manifest must say which.**
An archive can reprocess its holdings -- MAST does, routinely. If an analysis silently mixes
data retrieved before and after a reprocessing, the result is irreproducible in a way that is
nearly impossible to diagnose after the fact. So every entry records when it was originally
fetched, and every retrieval reports which path it took.

Layout on disk::

    <root>/<archive>/<hash[:2]>/<hash>/
        entry.json      metadata: spec, timestamps, result hash, client version
        result.ecsv     the result table

ECSV rather than Parquet for query results: it is text (so a cache entry is diffable and
inspectable with ``cat``), and it round-trips astropy units and column metadata, which a bare
CSV would silently discard.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrolab.archives.base import QuerySpec

if TYPE_CHECKING:
    from astropy.table import Table

__all__ = ["CacheEntry", "CorruptCacheError", "QueryCache"]

CACHE_FORMAT_VERSION = 1


class CorruptCacheError(RuntimeError):
    """A cache entry exists but is unreadable or fails its integrity check.

    Raised rather than silently re-fetching, because a corrupt entry means something is wrong
    -- a partial write, a disk problem, a concurrent run -- and silently papering over it hides
    a fault that will recur.
    """


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CacheEntry:
    """One stored query result."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.metadata_path = directory / "entry.json"
        self.result_path = directory / "result.ecsv"

    @property
    def exists(self) -> bool:
        return self.metadata_path.is_file() and self.result_path.is_file()

    def read_metadata(self) -> dict[str, Any]:
        try:
            data: dict[str, Any] = json.loads(self.metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptCacheError(
                f"cache entry metadata unreadable at {self.metadata_path}: {exc}"
            ) from exc
        return data

    def age_days(self) -> float:
        meta = self.read_metadata()
        fetched = datetime.fromisoformat(meta["fetched_at"])
        return (datetime.now(UTC) - fetched).total_seconds() / 86400.0


class QueryCache:
    """Content-addressed store keyed by :meth:`QuerySpec.content_hash`.

    Parameters
    ----------
    root
        Cache root directory. Created on demand.
    enabled
        When false, every lookup misses and nothing is written. Provided so a run can be forced
        live without deleting anything -- destroying a cache to test a code path is a good way
        to lose data you cannot re-fetch.
    max_age_days
        Refuse entries older than this. ``None`` (the default) means entries never expire,
        which is the reproducibility-preserving choice: refreshing should be deliberate.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        enabled: bool = True,
        max_age_days: float | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.enabled = enabled
        self.max_age_days = max_age_days

    # -- addressing ---------------------------------------------------------------------

    def entry_for(self, spec: QuerySpec) -> CacheEntry:
        """Locate the entry for a spec, whether or not it exists yet.

        The two-character shard directory keeps any single directory from accumulating tens of
        thousands of children, which some filesystems handle poorly.
        """
        digest = spec.content_hash()
        return CacheEntry(self.root / spec.archive / digest[:2] / digest)

    # -- reading ------------------------------------------------------------------------

    def get(self, spec: QuerySpec) -> tuple[Table, dict[str, Any]] | None:
        """Return ``(table, metadata)`` for a cached spec, or ``None`` on a miss.

        Raises
        ------
        CorruptCacheError
            If an entry exists but cannot be read or fails its content-hash check.
        """
        if not self.enabled:
            return None
        entry = self.entry_for(spec)
        if not entry.exists:
            return None

        meta = entry.read_metadata()

        if meta.get("format_version") != CACHE_FORMAT_VERSION:
            # A format change means we cannot trust our own reader; treat as a miss rather
            # than risk misinterpreting stored bytes.
            return None

        if self.max_age_days is not None and entry.age_days() > self.max_age_days:
            return None

        raw = entry.result_path.read_bytes()
        actual = _hash_bytes(raw)
        if actual != meta.get("result_hash"):
            raise CorruptCacheError(
                f"cache entry {entry.directory} failed its integrity check "
                f"(stored {meta.get('result_hash')}, computed {actual}). "
                f"Delete the entry to force a re-fetch, but consider why it changed."
            )

        from astropy.table import Table

        table = Table.read(entry.result_path, format="ascii.ecsv")
        return table, meta

    # -- writing ------------------------------------------------------------------------

    def put(
        self,
        spec: QuerySpec,
        table: Table,
        *,
        fetched_at: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Store a query result and return the metadata that was written.

        The write is atomic: the payload goes to a temporary sibling directory and is moved
        into place only once complete. A crash mid-download therefore leaves no half-written
        entry that a later run would happily read as though it were whole.
        """
        meta: dict[str, Any] = {
            "format_version": CACHE_FORMAT_VERSION,
            "spec": spec.to_dict(),
            "spec_hash": spec.content_hash(),
            "fetched_at": fetched_at or _utcnow(),
            "written_at": _utcnow(),
            "n_rows": len(table),
            **extra,
        }

        entry = self.entry_for(spec)
        if not self.enabled:
            # Still report what *would* have been written, so callers have consistent
            # metadata whether or not caching is on.
            meta["result_hash"] = None
            meta["cached"] = False
            return meta

        staging = entry.directory.parent / f".tmp-{spec.short_hash}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            result_path = staging / "result.ecsv"
            table.write(result_path, format="ascii.ecsv", overwrite=True)
            meta["result_hash"] = _hash_bytes(result_path.read_bytes())
            meta["cached"] = True
            (staging / "entry.json").write_text(json.dumps(meta, indent=2, default=str))

            if entry.directory.exists():
                shutil.rmtree(entry.directory)
            entry.directory.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(entry.directory)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return meta

    # -- maintenance --------------------------------------------------------------------

    def invalidate(self, spec: QuerySpec) -> bool:
        """Remove one entry. Returns whether anything was removed."""
        entry = self.entry_for(spec)
        if not entry.directory.exists():
            return False
        shutil.rmtree(entry.directory)
        return True

    def iter_entries(self, archive: str | None = None) -> list[dict[str, Any]]:
        """List stored entries' metadata, for inspection and for cache reports."""
        base = self.root / archive if archive else self.root
        if not base.exists():
            return []
        out: list[dict[str, Any]] = []
        for meta_path in sorted(base.glob("**/entry.json")):
            try:
                out.append(json.loads(meta_path.read_text()))
            except (OSError, json.JSONDecodeError):
                out.append({"error": "unreadable", "path": str(meta_path)})
        return out

    def stats(self) -> dict[str, Any]:
        """Summary of cache contents and size, for the ``astrolab cache`` command."""
        entries = self.iter_entries()
        total_bytes = sum(f.stat().st_size for f in self.root.glob("**/*") if f.is_file())
        return {
            "root": str(self.root),
            "enabled": self.enabled,
            "n_entries": len(entries),
            "total_bytes": total_bytes,
            "max_age_days": self.max_age_days,
        }
