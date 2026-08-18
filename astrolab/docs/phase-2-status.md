# Phase 2 status — exoplanet transit vertical slice

**Date:** 2026-08-18 · **Branch:** `claude/astrolab-design-phase-0-yru0n6`

Phase 2's acceptance criterion was: *"`astrolab run configs/<known-planet>.yaml` reproduces
published parameters within uncertainties and emits a report with a provenance appendix."*

**Met, on real data, with one substitution and one honest limitation.** The target is K2-3
rather than WASP-18 b, because WASP-18 b needs MAST and MAST is blocked. The limitation is that
the light curve's chain of custody does not reach the archive, which makes these *regression*
benchmarks rather than *validated* ones.

---

## The headline result

```
$ astrolab run configs/k2-3-transit.yaml
3 candidate(s) found
  1. P = 10.05283 d  depth = 1332 ppm  -> KNOWN_OBJECT
  2. P = 24.64641 d  depth =  655 ppm  -> KNOWN_OBJECT
  3. P = 22.28129 d  depth =  382 ppm  -> KNOWN_OBJECT
     fitted P = 10.054719 +0.000114/-0.000126 d, Rp/R* = 0.03539, ln BF = 477.2
report:   outputs/k2-3-transit/report.html
quality: CAUTION: 6 caution, 4 info
$ echo $?
0
```

Against published values:

| Quantity | Pipeline | Published | Agreement |
|---|---|---|---|
| K2-3 b period (fit) | 10.054719 (+0.000114/−0.000126) d | 10.0546535 d (Kosiarek et al. 2019) | **0.6σ** |
| K2-3 b period (search) | 10.05283 d | — | within the periodogram grid half-spacing |
| K2-3 c period (search) | 24.64641 d | 24.6454 d *(unverified)* | 0.004% |
| K2-3 b Rp/R* | 0.03539 ± 0.0005 | ~0.035 implied by published Rp, R* | consistent |
| K2-3 b T14 | 2.46 h | ~2.5 h | consistent |

The third candidate is the interesting one. The search reported it at 22.28 d on two transits;
the cross-match identified it as **K2-3 d (44.5565 d) at its 1/2 harmonic**, which is exactly
the alias TLS itself warns about when transits are missing. It was flagged, not promoted.

## What exists

```
core/         units  uncertainty  quality  config  provenance  logging  lightcurve  plotting
archives/     base  cache  mast
instruments/  k2/
science/      exoplanets/{detrend, search, fit, vetting}
pipelines/    transit  report
validation/   targets  data/{K2-3 light curves, SOURCE.md, checksums}
```

`astrolab run` executes ingest → detrend → search → vet → fit → report, threading provenance and
quality flags through every stage, and emits a self-contained HTML report (383 KB, six figures
inlined, no external references) plus `manifest.json` and `summary.json`.

**239 tests pass; `ruff` and `mypy --strict` are clean.**

## Verified by running it

| Check | Result |
|---|---|
| `pytest -q` | 239 passed, 4 skipped (network) |
| `ruff check` / `format --check` | clean |
| `mypy` (strict on `core/`, `archives/`) | clean, 13 files |
| `astrolab run configs/k2-3-transit.yaml` | exit 0, report + manifest + summary written |
| Golden target: fitted period | 0.6σ from published |
| Report structure | caveats precede every result; provenance appendix present |

## Bugs found by running, not by reading

Recorded because they are the argument for tests alongside code, and because several were
invisible to inspection.

1. **Uncertainties 60× too large.** Ingestion estimated per-point errors from the *raw* light
   curve's scatter — 3052 ppm of stellar variability against a true precision of 51 ppm.
   Carried through detrending, the likelihood went nearly flat and the fit reported a real
   planet as *weak evidence* with a depth uncertainty larger than the depth. `detrend()` now
   re-derives estimated (never measured) uncertainties from the detrended residuals.
2. **The null hypothesis contained no signal.** The depth-consistency vetting test failed K2-3 b
   at +9σ. Three nulls were wrong in turn: the analytic noise/√N expectation ignores that
   successive transits are sampled at different orbital phases at 29-minute cadence; an
   empirical null at off-transit epochs measures the scatter of *nothing*; injecting a box gives
   every event the same depth. Injecting the candidate's own folded profile fixed it — K2-3 b
   now passes at +1.4σ.
3. **A benchmark transcribed from memory.** The golden period was 10.05403 d; the published
   values are 10.05449 ± 0.00026 (Crossfield 2015) and 10.0546535 (Kosiarek 2019). The fit's
   10.05474 reads as 2.4σ tension against the wrong number and 0.6σ agreement against the right
   one. `GoldenValue` now carries a `verification` field, and unverified values say so in their
   output.
4. **Recovery reported as failure.** A catalogue match sets `passed=False` on the ephemeris
   test, which initially propagated into a `FALSE_POSITIVE` disposition and an `UNRELIABLE`
   flag — so recovering three known planets exited non-zero. Catalogue matches are now
   separated from astrophysical failures in both the disposition and the flag.
5. **Duplicate flags burying distinct ones.** Flags propagate along every pipeline edge, so the
   report listed 27 with heavy repetition. `QualityReport.extend` now deduplicates: 10 remain,
   and the repeated `known_object` entries are genuinely different planets.

## The limitation that still stands

**These are regression benchmarks, not validated ones.** The light curve came from the
MIT-licensed `hippke/tls` test fixtures rather than from MAST, because every astronomy archive
is blocked here (`mast.stsci.edu`, `archive.stsci.edu`, `exoplanetarchive.ipac.caltech.edu`,
`heasarc`, IRSA, VizieR, SIMBAD, Gaia, JPL Horizons, MPC, SDSS, ESO — all 403 or unreachable).
The data are genuinely real K2 Campaign 1 observations, independently confirmed by cadence
(29.42 min), baseline (80.07 d), and first-cadence epoch (2014-06-01, inside Campaign 1). But
the FITS headers are gone, so nothing can be checked byte-for-byte against the archive product.

That limitation is enforced in code, not merely documented: ingestion raises
`THIRD_PARTY_MIRROR` at CAUTION severity, it propagates to the fitted depth, and it appears in
the report above the results.

Two published values (K2-3 c and d periods) could not be verified either, since arXiv, IOP,
Wikipedia and the NASA Exoplanet Archive are all blocked. They are marked `unverified`.

## To finish validation

On a machine with archive access:

```bash
uv run pytest --run-network -m network          # confirm the MAST query vocabulary
uv run astrolab query configs/wasp18b-tess.yaml # retrieve real TESS data
# then: pin the retrieved product per ADR-0004 and re-run the benchmarks against it
```

Open items, in priority order:

1. Re-run the K2-3 benchmarks on the MAST product and promote them from regression to
   validated; drop the `THIRD_PARTY_MIRROR` flag when the chain of custody is real.
2. Verify the K2-3 c and d periods against the papers.
3. Add WASP-18 b as a second golden target — a 1% hot-Jupiter transit is a much easier signal,
   and a pipeline tested only on 1200 ppm M-dwarf transits has not been tested across the range.
4. **Injection-recovery for completeness and reliability.** The statistical-rigour requirement
   calls for it on every search algorithm, and it is not yet built. `estimate_depth_bias()` is a
   single-point version of the machinery; the full sweep over period and depth is Phase 2's
   remaining gap.
5. Limb-darkening comparison (`exotic-ld` theoretical vs free fit) as a stated systematic.
