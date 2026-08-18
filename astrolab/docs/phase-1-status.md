# Phase 1 status — core + archives

**Date:** 2026-08-18 · **Branch:** `claude/astrolab-design-phase-0-yru0n6`

Phase 1's acceptance criterion was: *"I can pull a TESS light curve and a JWST product from
MAST by target name, and the run manifest fully reproduces the query."*

**That criterion is half met, and the unmet half is not met by this code's fault.** The
manifest-reproduces-the-query half is done and demonstrated. The pull-from-MAST half could not
be executed in the environment this was built in, because MAST is blocked there. Details and
the exact steps to finish it are below.

---

## The blocker

The development environment's egress proxy refuses connections to the archive hosts:

```
$ curl -sS -o /dev/null -w "%{http_code}" https://mast.stsci.edu/api/v0/
curl: (56) CONNECT tunnel failed, response 403
```

The proxy's own status endpoint records the reason:

```json
{"kind": "connect_rejected",
 "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
 "host": "mast.stsci.edu:443"}
```

`archive.stsci.edu` and `exoplanetarchive.ipac.caltech.edu` are blocked identically;
`rubinobservatory.org` and `www.lsst.org` were already noted as blocked in
`mission-status.md`. This is an organization egress policy, and the environment's own
documentation is explicit that such denials must be reported rather than routed around. So
they are reported, and no workaround was attempted.

**What this does not excuse.** Nothing here is stubbed to paper over it. There is no fake MAST
response, no synthesized light curve, and no function returning a plausible number. The live
path is written, typed, and reachable; it is simply unexecuted, and every place that matters
says so.

---

## What was verified by running it

Everything below was executed, not reasoned about.

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest -q` | **135 passed, 4 skipped** (skipped = network-marked) |
| Lint | `ruff check .` | clean |
| Format | `ruff format --check .` | clean |
| Strict types | `mypy` (strict on `core/`, `archives/`) | **no issues, 11 files** |
| Coverage | `pytest --cov` | **89%**; the uncovered remainder is concentrated in the live network calls |
| Config validation | `astrolab validate configs/wasp18b-tess.yaml` | valid, prints the query it will run |
| Failure path | `astrolab query configs/wasp18b-tess.yaml` | **exit 4**, manifest written, caveats printed |
| Manifest replay | `astrolab replay outputs/wasp18b-tess/manifest.json` | **exit 0**, spec reconstructed, hash matches |

The failure path is worth showing, because how a pipeline behaves when it cannot get data is
as much a design commitment as what it does when it can:

```
$ astrolab query configs/wasp18b-tess.yaml
[error] archive.query.failed  spec_hash=98c4b49fdeaf
        error="...Tunnel connection failed: 403 Forbidden"
MAST query failed for mast.observations(author='SPOC', ..., target_name='WASP-18')
The archive could not be reached or refused the query. Nothing was written beyond this manifest.
manifest: outputs/wasp18b-tess/manifest.json
reproducibility caveats:
  - working tree was dirty: the recorded SHA does not identify the code
  - 1 query/queries were fetched live; archives can reprocess data, so a future re-run may
    retrieve different bytes unless served from cache
$ echo $?
4
```

It fails loudly, writes the manifest anyway, and volunteers that the run is not reproducible
and why. Then `replay` reconstructs the attempted query from that manifest and confirms it
hashes to the value recorded at run time — which is the reproducibility claim in executable
form, and it passes.

## Three real bugs the tests caught

Recorded because they are the argument for writing tests alongside code rather than after,
and none would have been found by reading:

1. **`QuerySpec` validation was inert.** It passed `default=str` to `json.dumps`, so a
   non-serialisable param was coerced instead of rejected. A `set` would have stringified in
   whatever order the interpreter happened to iterate it, making the content hash unstable
   between runs — silently breaking both caching and replay, the two things the hash exists
   for.
2. **`git_state` mangled every dirty filename.** It called `.strip()` on the whole
   `git status --porcelain` output, but that format's first column is a space for
   worktree-only modifications, so stripping shifted every filename one character left
   (`astrolab/...` → `strolab/...`). Caught by asserting on a real dirty tree.
3. **The logger wrote to a closed stream.** The structlog factory captured `sys.stderr` at
   configuration time; anything that later replaced the stream left it writing to a dead file.
   This surfaced as tests passing individually and failing together — the signature of shared
   state, and a bug that would have hit any subprocess wrapper too.

## What is built

```
src/astrolab/
  core/     units.py  uncertainty.py  quality.py  config.py  provenance.py  logging.py
  archives/ base.py  cache.py  mast.py
  cli.py
tests/      135 tests across 7 files
configs/    wasp18b-tess.yaml
```

Design decisions and their reasoning are in `docs/decisions/`. The parts worth knowing:

- **`Measurement` carries posterior samples, not value±σ** (ADR-0002). `JointPosterior.derive`
  propagates correlations by construction; cross-posterior arithmetic raises
  `IndependenceError` so element-wise ops cannot invent a correlation that was never sampled.
  `combine_independent` exists so the independence assumption appears in code review instead of
  hiding inside an operator.
- **`EmptyResultError` rather than an empty table**, and `ProprietaryDataError` distinguished
  from it — "exists but not yours" and "does not exist" call for different responses.
- **The cache is integrity-checked and atomic.** A tampered payload raises rather than silently
  re-fetching, because silent recovery hides a fault that will recur.
- **The manifest self-assesses.** `reproducibility_assessment()` states plainly when a run is
  not reproducible, rather than leaving a reader to infer it from a `dirty` flag three levels
  down.

## Finishing the acceptance test

On any machine with normal network access:

```bash
cd astrolab
uv sync --extra dev

# 1. The live acceptance criterion: pull a TESS light curve for WASP-18 by name.
uv run astrolab query configs/wasp18b-tess.yaml        # expect exit 0
uv run astrolab replay outputs/wasp18b-tess/manifest.json
uv run astrolab query configs/wasp18b-tess.yaml        # expect "cache" in the manifest

# 2. The live archive tests, including a public zero-EAP JWST product.
uv run pytest --run-network -m network -v
```

**What to watch for**, since these paths are unexercised: the MAST query vocabulary in
`PRODUCT_TO_DATAPRODUCT_TYPE` and the criteria names in `_execute_observations` (`provenance_name`
for author, `sequence_number` for sector, `t_exptime` for cadence) are taken from MAST's
documented schema but have not been confirmed against the live service. If a query returns
nothing for a target that certainly has data, that mapping is the first place to look — and
`EmptyResultError` will say so rather than quietly returning an empty table.

## Consequences for later phases

- **ADR-0004's pinned-data snapshot cannot be created here.** It requires one genuine retrieval
  from MAST, which is exactly what is blocked. Phase 2's golden-target test therefore cannot be
  built to completion in this environment either. This is the single largest constraint on
  continuing, and it is worth resolving before Phase 2 rather than during it.
- **Risk R6 (archive instability) is unmitigated in practice.** The caching layer that isolates
  it is built and tested; its behaviour against a real archive is not yet observed.
