"""Run manifests: the traceable lineage behind every number.

Prime directive 2 says a measurement without a traceable lineage is a bug. This module is what
makes lineage a fact rather than an intention. Every run emits a manifest recording enough to
answer, months later and from the file alone:

- What code produced this? (git SHA, and whether the tree was dirty -- a dirty tree means the
  SHA does not identify the code that ran, and pretending otherwise is worse than admitting it)
- In what environment? (Python version, platform, lockfile hash, versions of the packages whose
  numerics actually affect results)
- From what inputs? (every archive query, its parameters, its content hash, and whether it was
  served from cache or fetched live)
- Under what choices? (the fully resolved config, every random seed, every override flag)
- With what reservations? (quality flags raised anywhere in the run)

The design rule throughout: when something cannot be determined, the manifest records that it
could not be determined. It never records a plausible-looking guess. A manifest that says
"git state unavailable" is useful; one that silently omits the field looks clean and is a lie.
"""

from __future__ import annotations

import getpass
import json
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from astrolab.core.quality import QualityReport

__all__ = [
    "EnvironmentRecord",
    "GitState",
    "QueryRecord",
    "RunManifest",
    "environment_record",
    "git_state",
]

#: Packages whose versions can change a numerical result. Recorded in every manifest.
#: Deliberately not "everything installed" -- that is noise which obscures the few that matter.
SCIENCE_RELEVANT_PACKAGES = (
    "astrolab",
    "astropy",
    "numpy",
    "scipy",
    "astroquery",
    "lightkurve",
    "erfa",
    "pyerfa",
)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class GitState:
    """State of the source repository at run time.

    ``dirty`` is the field that matters most and is the one most often omitted elsewhere. A
    commit SHA identifies the code only if the working tree matched it; if it did not, the run
    is not reproducible from that SHA and the manifest must say so.
    """

    available: bool
    sha: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    dirty_files: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "sha": self.sha,
            "branch": self.branch,
            "dirty": self.dirty,
            "dirty_files": self.dirty_files,
            "note": self.note,
        }

    @property
    def reproducible(self) -> bool:
        """Whether the code that ran can be recovered from this record alone."""
        return self.available and self.dirty is False


def _git(args: list[str], cwd: Path, *, strip: bool = True) -> str | None:
    """Run a git command, returning stdout or ``None`` if git could not answer.

    ``strip`` must be False for ``status --porcelain``: its first column is a space for
    worktree-only modifications, so stripping the whole output shifts every filename by one
    character and silently corrupts the dirty-file list.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() if strip else out.stdout.rstrip("\n")


def git_state(repo_root: str | Path | None = None) -> GitState:
    """Capture the git state of the working tree, honestly.

    Parameters
    ----------
    repo_root
        Directory inside the repository. Defaults to the directory containing this package,
        so the answer describes *the code that is running*, not the caller's cwd.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    sha = _git(["rev-parse", "HEAD"], root)
    if sha is None:
        return GitState(
            available=False,
            note="git unavailable, not a repository, or the command failed; the code that "
            "produced this run cannot be identified from the manifest",
        )
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain"], root, strip=False)
    if status is None:
        return GitState(
            available=True,
            sha=sha,
            branch=branch,
            dirty=None,
            note="could not determine whether the working tree was clean",
        )
    dirty_files = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return GitState(
        available=True,
        sha=sha,
        branch=branch,
        dirty=bool(dirty_files),
        dirty_files=dirty_files[:50],
        note=(
            "working tree had uncommitted changes; this SHA does not identify the code that ran"
            if dirty_files
            else ""
        ),
    )


@dataclass(frozen=True)
class EnvironmentRecord:
    """The software environment a run executed in."""

    python_version: str
    platform: str
    packages: dict[str, str]
    lockfile_hash: str | None
    lockfile_path: str | None
    hostname: str
    username: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": self.packages,
            "lockfile_hash": self.lockfile_hash,
            "lockfile_path": self.lockfile_path,
            "hostname": self.hostname,
            "username": self.username,
        }


def _hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def environment_record(lockfile: str | Path | None = None) -> EnvironmentRecord:
    """Capture the environment. Missing pieces are recorded as ``None``, never invented."""
    packages: dict[str, str] = {}
    for name in SCIENCE_RELEVANT_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"

    lock_path: Path | None = None
    if lockfile is not None:
        candidate = Path(lockfile)
        if candidate.is_file():
            lock_path = candidate
    else:
        # Walk up from this file looking for the project lockfile.
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "uv.lock"
            if candidate.is_file():
                lock_path = candidate
                break

    try:
        username = getpass.getuser()
    except Exception:  # pragma: no cover - depends on host user database
        username = "unknown"

    return EnvironmentRecord(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
        lockfile_hash=_hash_file(lock_path) if lock_path else None,
        lockfile_path=str(lock_path) if lock_path else None,
        hostname=socket.gethostname(),
        username=username,
    )


@dataclass
class QueryRecord:
    """One archive query, recorded in enough detail to replay it exactly.

    ``spec`` is the normalised query specification, not the raw HTTP call: it is what the
    caching layer hashes, and replaying it is what reproduces the retrieval.
    """

    archive: str
    spec: dict[str, Any]
    spec_hash: str
    timestamp: str
    served_from: str
    """``"cache"`` or ``"live"`` -- a cached result and a fresh one are not the same fact."""
    n_results: int | None = None
    result_hash: str | None = None
    cached_at: str | None = None
    """When the cached entry was originally fetched, if served from cache."""
    error: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive": self.archive,
            "spec": self.spec,
            "spec_hash": self.spec_hash,
            "timestamp": self.timestamp,
            "served_from": self.served_from,
            "n_results": self.n_results,
            "result_hash": self.result_hash,
            "cached_at": self.cached_at,
            "error": self.error,
            "notes": self.notes,
        }


@dataclass
class RunManifest:
    """The complete provenance record for one pipeline run.

    Built up as the run proceeds and written at the end (and, ideally, on failure too -- a
    manifest for a run that crashed is exactly when you most want to know what it did).
    """

    run_name: str
    config: dict[str, Any]
    config_hash: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=_utcnow)
    finished_at: str | None = None
    git: GitState = field(default_factory=git_state)
    environment: EnvironmentRecord = field(default_factory=environment_record)
    queries: list[QueryRecord] = field(default_factory=list)
    seeds: dict[str, int] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    quality: QualityReport = field(default_factory=QualityReport)
    status: str = "running"
    error: str | None = None
    schema_version: int = 1

    # -- recording ---------------------------------------------------------------------

    def record_query(self, record: QueryRecord) -> None:
        self.queries.append(record)

    def record_seed(self, stage: str, seed: int) -> None:
        """Record the seed a stochastic stage actually used.

        Derived seeds, not just the master seed: knowing the master seed only reproduces the
        run if the derivation is also stable, and recording the derived value makes the claim
        checkable instead of assumed.
        """
        self.seeds[stage] = seed

    def record_override(self, name: str, value: Any) -> None:
        """Stamp an override flag into the record.

        ``--allow-unconverged`` and its relatives exist so that a human can force a result the
        pipeline would refuse. Making that choice invisible in the output would defeat the
        refusal entirely, so it is recorded here and surfaced in the report.
        """
        self.overrides[name] = value

    def record_output(self, path: str | Path, kind: str, **details: Any) -> None:
        self.outputs.append(
            {
                "path": str(path),
                "kind": kind,
                "written_at": _utcnow(),
                **details,
            }
        )

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self.finished_at = _utcnow()
        self.status = status
        self.error = error

    # -- serialisation ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git": self.git.to_dict(),
            "environment": self.environment.to_dict(),
            "config_hash": self.config_hash,
            "config": self.config,
            "queries": [q.to_dict() for q in self.queries],
            "seeds": self.seeds,
            "overrides": self.overrides,
            "outputs": self.outputs,
            "quality_flags": self.quality.to_list(),
            "reproducibility": self.reproducibility_assessment(),
        }

    def reproducibility_assessment(self) -> dict[str, Any]:
        """A blunt self-assessment of whether this run can be reproduced.

        Included in the manifest because the honest answer is often "no", and a reader
        deserves to be told that directly rather than having to infer it from a dirty flag
        buried three levels down.
        """
        reasons: list[str] = []
        if not self.git.available:
            reasons.append("git state unavailable: the code version is unknown")
        elif self.git.dirty:
            reasons.append("working tree was dirty: the recorded SHA does not identify the code")
        elif self.git.dirty is None:
            reasons.append("could not determine whether the working tree was clean")
        if self.environment.lockfile_hash is None:
            reasons.append("no environment lockfile found: dependency versions are not pinned")
        if self.overrides:
            reasons.append(
                f"overrides were used: {sorted(self.overrides)} -- results reflect a human "
                f"decision to bypass a pipeline refusal"
            )
        live = [q for q in self.queries if q.served_from == "live"]
        if live:
            reasons.append(
                f"{len(live)} query/queries were fetched live; archives can reprocess data, so "
                f"a future re-run may retrieve different bytes unless served from cache"
            )
        return {"fully_reproducible": not reasons, "caveats": reasons}

    def write(self, path: str | Path) -> Path:
        """Write the manifest as indented JSON.

        JSON rather than YAML: manifests are machine-read more often than hand-edited, and
        JSON has exactly one way to represent a string.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False, default=str))
        return p

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load a manifest as a plain dict, for replay and comparison.

        Deliberately returns a dict rather than reconstructing a live ``RunManifest``: an old
        manifest describes a past run and should not be mistaken for a runnable object, and
        the schema will evolve.
        """
        data: dict[str, Any] = json.loads(Path(path).read_text())
        return data
