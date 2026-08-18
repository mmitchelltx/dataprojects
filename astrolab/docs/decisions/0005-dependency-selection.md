# 0005 — Dependency selection and maintenance posture

**Status:** Accepted · **Date:** 2026-08-18

## Context

The brief asks for boring, well-tested community packages over clever custom code, and for honesty
when a library is unmaintained or an approach superseded. It also forbids "it should work"
placeholders. These combine into a posture on dependencies worth stating once.

## Decision

1. **Prefer astropy-ecosystem packages** with active maintenance and broad citation. Full list and
   per-choice rationale in `docs/design.md` §4.
2. **Custom numerics require an equivalence test** against a reference implementation. The one
   substantial piece of custom numerics currently planned is the transit likelihood function
   (see design §6.2), which is tested against `batman` reference outputs and against published
   parameters for the golden target.
3. **Dependencies are adopted when their phase starts, not speculatively.** `petitRADTRANS`,
   `sncosmo`, `pyccl`, `sbpy`, the `jwst` pipeline, and `romancal` are all deferred. Pinning them
   now would inflate the lockfile and the install burden for phases that may never be scheduled.
4. **Known maintenance concerns are documented rather than hidden.**

## Maintenance flags, as of 2026-08-18

- **`transitleastsquares`** — effectively feature-complete rather than actively developed. Accepted:
  the algorithm is well-specified and published, and `astropy`'s BLS runs alongside it as an
  independent check. If the two disagree materially on a target, that disagreement is a finding.
- **`exotic-ld`** — actively maintained, but requires downloading stellar model grids (~GB) on first
  use. Under the laptop-scale assumption this needs deliberate cache handling and cannot be an
  implicit side effect of an import.
- **`wotan`** — stable and purpose-built, with published benchmarks comparing its own detrending
  methods. Low risk.
- **Deferred stacks (R4 in the design doc)** — the `jwst` pipeline carries a live CRDS network
  dependency (reference files are fetched at runtime, so the CRDS context must be pinned and
  recorded in provenance or reductions are not reproducible); `petitRADTRANS` needs GB-scale opacity
  data. Both may prove infeasible at laptop scale, which is a finding to report honestly, not to
  work around with a reduced-fidelity substitute presented as the real thing.

## Consequences

- The lockfile stays comparatively small through Phases 1–3.
- Each future phase begins with a dependency evaluation step rather than inheriting decisions made
  speculatively today.
- Where a package is flagged above, the flag is repeated in the docstring of the module that uses
  it, so it is visible at the point of use and not only in this log.
