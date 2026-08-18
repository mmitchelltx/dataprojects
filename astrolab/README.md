# astrolab

Reproducible, provenance-tracked analysis of data from space-based observatories and public sky
surveys.

**Status: Phase 2 (transit vertical slice) complete on real data; live retrieval unverified.**

`astrolab run configs/k2-3-transit.yaml` ingests real K2 Campaign 1 photometry of K2-3, detrends
it, searches it, vets every candidate, fits the survivor, and writes a report with a provenance
appendix. It recovers K2-3 b at **0.6σ from the published period**, recovers K2-3 c, and
correctly identifies a third detection as K2-3 d at its 1/2 harmonic rather than promoting it.
See `docs/phase-2-status.md`.

**Phase 1 (core + archives) built; live retrieval unverified.** The archive hosts are
blocked by an egress policy in the environment this was developed in, so no query has actually
reached MAST. Everything else — 135 tests, `mypy --strict`, the provenance and replay path — was
run and passes. See `docs/phase-1-status.md` for exactly what is and is not verified, and the
commands to finish the acceptance test on an unrestricted network.

See `docs/design.md` for the architecture and phase plan, `docs/decisions/` for why each
non-obvious choice was made, and `docs/mission-status.md` for the verified state of the missions
this toolkit targets.

This project makes three promises, and the code exists to keep them:

1. Every number carries an uncertainty and a traceable lineage.
2. Nothing is a detection until it survives a documented false-positive gauntlet.
3. Every result regenerates from a config file plus a pinned environment.

Nothing here fabricates data. If an archive query returns nothing, the pipeline says so and stops.
