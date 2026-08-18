# astrolab

Reproducible, provenance-tracked analysis of data from space-based observatories and public sky
surveys.

**Status: Phase 1 (core + archives) in progress.** See `docs/design.md` for the architecture and
phase plan, `docs/decisions/` for why each non-obvious choice was made, and
`docs/mission-status.md` for the verified state of the missions this toolkit targets.

This project makes three promises, and the code exists to keep them:

1. Every number carries an uncertainty and a traceable lineage.
2. Nothing is a detection until it survives a documented false-positive gauntlet.
3. Every result regenerates from a config file plus a pinned environment.

Nothing here fabricates data. If an archive query returns nothing, the pipeline says so and stops.
