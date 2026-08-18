# Provenance of the bundled validation data

**Read this before treating any result computed from these files as a validated benchmark.**

## What these files are

Real K2 Campaign 1 long-cadence photometry of **EPIC 201367065 = K2-3**, an M dwarf hosting
three known transiting planets (Crossfield et al. 2015, ApJ 804, 10, doi:10.1088/0004-637X/804/1/10).

| File | Contents | SHA-256 (see `checksums.txt`) |
|---|---|---|
| `EPIC201367065_k2c1_raw.csv` | time (BKJD), normalised flux — retains stellar variability | recorded |
| `EPIC201367065_k2c1_detrended.csv` | the same series, detrended by the upstream author | recorded |

3632 cadences, 29.42-minute sampling, 80.07-day baseline, robust scatter ~54 ppm. The first
cadence converts to 2014-06-01 BJD_TDB, consistent with K2 Campaign 1 (2014-05-30 to
2014-08-21). Column 1 is BKJD (BJD_TDB − 2454833.0); column 2 is normalised flux.

## Where they came from, and the limitation that creates

These files were taken from the test fixtures of
[`hippke/tls`](https://github.com/hippke/tls) (the `transitleastsquares` package), MIT
licensed, which redistributes them as test data. The underlying K2 observations are NASA
public-domain data products.

**The chain of custody is incomplete, and that matters.** They are *not* the original MAST
FITS products: the headers are gone, so there is no `OBJECT`, no `CAMPAIGN`, no `DATA_REL`,
and no pipeline version to check. It cannot be verified byte-for-byte against what MAST would
serve, and the upstream detrending applied to the second file is not documented in a form this
project can reproduce.

They were used because MAST, the NASA Exoplanet Archive, IRSA, VizieR, SIMBAD, HEASARC, and
every other astronomy archive are blocked by the egress policy of the environment this project
was developed in (see `docs/phase-1-status.md`). The choice was between real data with a weak
provenance chain and synthesised data with none. Prime Directive 1 makes that choice easy:
**this is real data, honestly labelled, with its weakness stated.**

## What this permits and forbids

**Permitted.** Developing and testing the pipeline. Demonstrating that the search recovers a
known planet from real photometry. Regression testing against a fixed input — the file is
checksummed, so drift in *our* code is detectable even if drift in the upstream file would
not be.

**Forbidden.** Treating agreement with published K2-3 parameters as a *validated* benchmark.
Every product derived from these files carries the `THIRD_PARTY_MIRROR` quality flag at
`CAUTION` severity, and it propagates. A benchmark is only validated once the same analysis is
re-run on the original MAST product; `docs/phase-2-status.md` records that as an open item.

## Published values for comparison

From Crossfield et al. 2015 (discovery) and Sinukoff et al. 2016 (ApJ 827, 78,
doi:10.3847/0004-637X/827/1/78):

| Planet | Period (d) | Notes |
|---|---|---|
| K2-3 b | 10.05403 ± 0.00026 | ~7 transits in this baseline |
| K2-3 c | 24.6454 ± 0.0013 | ~3 transits |
| K2-3 d | 44.5565 ± 0.0021 | ~1–2 transits; expect the pipeline to flag `SINGLE_EVENT` |

Values transcribed on 2026-08-18. Re-verify against the papers before quoting them; a
transcription error in a benchmark is a benchmark that certifies the wrong answer.
