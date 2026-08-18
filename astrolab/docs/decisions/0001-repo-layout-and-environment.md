# 0001 — Repository layout and environment manager

**Status:** Accepted · **Date:** 2026-08-18

## Context

`astrolab` is being built inside `mmitchelltx/dataprojects`, an existing monorepo that already holds
unrelated projects (`energy_demand/`, `datacenter/`, `predict_energy_output/`). It needs its own
dependency set — a heavy scientific stack that the other projects have no business inheriting.

## Decision

`astrolab/` becomes a top-level project directory in the monorepo, with a `src/astrolab/` layout
inside it and its own `pyproject.toml`, lockfile, `docs/`, `configs/`, and `tests/`.

Environment management via **uv**, Python 3.12.

## Rationale

The `src/` layout prevents the most common packaging failure: tests accidentally importing the local
source directory rather than the installed package, which hides broken packaging until someone else
tries to install it. For a project whose central claim is reproducibility, that failure mode is
disqualifying.

Python 3.12 rather than the 3.11 floor because the astropy ecosystem fully supports it and there is
no reason to start on the oldest supported version.

**uv over pixi:** every dependency currently planned is pip-installable, so conda's main advantage
(binary non-Python dependencies) does not apply. uv is fast, produces a real cross-platform lockfile,
and is trivial in CI. This decision should be revisited if the cosmology pillar is built — `pyccl`
has historically been friendlier under conda — but choosing pixi now to hedge against a phase that
may never be scheduled is speculative complexity.

## Consequences

- The lockfile hash becomes part of every provenance manifest; environment drift is detectable.
- Adding a conda-only dependency later means either finding a wheel or migrating to pixi. Accepted
  as a known cost, flagged in the design doc as R4.
- Other projects in the monorepo are unaffected — no shared dependency surface.
