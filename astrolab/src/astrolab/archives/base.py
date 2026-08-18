"""Query specifications and archive-layer contracts.

The central abstraction is :class:`QuerySpec`: a normalised, hashable description of *what was
asked for*, separated from the mechanics of asking. That separation buys three things:

- **Caching** becomes content-addressed and correct, because the hash covers exactly the inputs
  that determine the answer.
- **Provenance** becomes replayable, because the manifest stores the spec rather than a
  transcript of HTTP calls.
- **Testing** becomes possible without a network, because spec construction -- where the
  interesting logic and most of the bugs live -- is pure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ArchiveError",
    "EmptyResultError",
    "ProprietaryDataError",
    "QueryResult",
    "QuerySpec",
]


class ArchiveError(RuntimeError):
    """An archive query failed. Subclasses distinguish *how*, because the responses differ."""


class EmptyResultError(ArchiveError):
    """The query succeeded and matched nothing.

    A distinct exception rather than an empty return value, because Prime Directive 1 requires
    the pipeline to say so and stop. An empty table flowing downstream is how a pipeline ends
    up reporting a confident result computed from nothing.

    This is not necessarily an error in the user's request -- "no TESS data for this target" is
    a legitimate and useful answer. It is an error to *continue* as though data had arrived.
    """

    def __init__(self, spec: QuerySpec, message: str = "") -> None:
        self.spec = spec
        detail = f": {message}" if message else ""
        super().__init__(
            f"{spec.archive} query matched no products{detail}\n"
            f"  spec: {spec.describe()}\n"
            f"  This is a real answer, not a failure. Nothing downstream will run."
        )


class ProprietaryDataError(ArchiveError):
    """Matching products exist but are under an exclusive-access period.

    Distinguished from an empty result because the responses differ completely: an empty result
    means look elsewhere, while this means the data exist and you need credentials or you need
    to wait. Silently returning nothing here would send the user hunting for a target that is
    right there.
    """


@dataclass(frozen=True)
class QuerySpec:
    """A normalised, hashable description of an archive query.

    Parameters
    ----------
    archive
        Archive identifier, e.g. ``"mast"``.
    operation
        What is being asked, e.g. ``"observations"`` or ``"products"``. Part of the hash, so
        two different questions about the same target never collide in the cache.
    params
        Query parameters. Must be JSON-serialisable: anything that cannot be serialised cannot
        be recorded in a manifest, and a query that cannot be recorded cannot be reproduced.
    client_version
        Version of the client code that builds and interprets this spec. Included in the hash
        so that changing how a query is constructed invalidates stale cache entries rather
        than silently reusing results from a different question.
    """

    archive: str
    operation: str
    params: dict[str, Any] = field(default_factory=dict)
    client_version: str = "1"

    def __post_init__(self) -> None:
        try:
            # No ``default=`` fallback: a coercion here would silently accept a value whose
            # JSON form is not stable. A set, for instance, would stringify in whatever order
            # the interpreter happened to iterate it, making the content hash non-deterministic
            # across runs -- which would quietly break both caching and replay.
            json.dumps(self.params, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"QuerySpec params must be JSON-serialisable so the query can be recorded and "
                f"replayed exactly; got {self.params!r}. Convert Paths, sets, and objects to "
                f"plain JSON types at the call site, where the intended representation is known."
            ) from exc

    def canonical(self) -> str:
        """Canonical JSON form: sorted keys, no insignificant whitespace, ``None`` dropped.

        Dropping ``None`` matters: ``{"author": None}`` and ``{}`` express the same query, and
        hashing them differently would cause spurious cache misses and make two identical runs
        look different in their manifests.
        """
        cleaned = {k: v for k, v in sorted(self.params.items()) if v is not None}
        return json.dumps(
            {
                "archive": self.archive,
                "operation": self.operation,
                "client_version": self.client_version,
                "params": cleaned,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def content_hash(self) -> str:
        """SHA-256 of the canonical form. The cache key and the provenance identifier."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.content_hash()[:12]

    def describe(self) -> str:
        """One-line human description, for logs and error messages."""
        cleaned = {k: v for k, v in sorted(self.params.items()) if v is not None}
        inner = ", ".join(f"{k}={v!r}" for k, v in cleaned.items())
        return f"{self.archive}.{self.operation}({inner}) [{self.short_hash}]"

    def to_dict(self) -> dict[str, Any]:
        parsed: dict[str, Any] = json.loads(self.canonical())
        return parsed

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuerySpec:
        """Rebuild a spec from its recorded form, for replaying a run from its manifest.

        Round-trips exactly: ``QuerySpec.from_dict(spec.to_dict()).content_hash()`` equals
        ``spec.content_hash()``. That identity is what makes a manifest a reproduction recipe
        rather than a description, and it is asserted by the test suite.
        """
        missing = {"archive", "operation", "params"} - set(data)
        if missing:
            raise ValueError(f"cannot rebuild QuerySpec, missing keys: {sorted(missing)}")
        return cls(
            archive=str(data["archive"]),
            operation=str(data["operation"]),
            params=dict(data["params"]),
            client_version=str(data.get("client_version", "1")),
        )


@dataclass
class QueryResult:
    """The outcome of a successful archive query.

    Attributes
    ----------
    table
        The result table (an ``astropy.table.Table``). Typed loosely here so this module does
        not force an astropy import on callers that only build specs.
    spec
        The spec that produced it.
    served_from
        ``"cache"`` or ``"live"``.
    fetched_at
        ISO timestamp of the original live fetch -- for a cache hit, this is when the data were
        *originally* retrieved, not when they were read back. Confusing the two would make a
        year-old cached result look fresh in the manifest.
    result_hash
        Content hash of the serialised result, so a cached answer can be verified rather than
        trusted.
    """

    table: Any
    spec: QuerySpec
    served_from: str
    fetched_at: str
    result_hash: str

    def __len__(self) -> int:
        return len(self.table)
