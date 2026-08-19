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

---

# LINEAR variable star photometry

## What these files are

Real photometry from the **LINEAR** survey (Lincoln Near-Earth Asteroid Research), used as the
Phase 3 variable-star benchmark.

| File | Object | N | Baseline | Notes |
|---|---|---|---|---|
| `LINEAR_11375941.csv` | LINEAR 11375941 | 280 | 1962 d | Period ~2.58 h (VanderPlas 2018) |
| `LINEAR_14752041.csv` | LINEAR 14752041 | 253 | 1967 d | Period not independently sourced |

Columns: `t` (MJD), `mag` (unfiltered LINEAR magnitude), `magerr`.

These are a deliberately different regime from the K2 transit data, and that is why they were
chosen. Where K2 gives uniform 29.4-minute sampling over 80 continuous days, LINEAR gives 280
points spread over 5.4 years with a **mean spacing of 7 days** and a strong nightly/annual
observing pattern. Sparse, irregular, ground-based sampling is what makes period-finding hard:
it produces a window function with a large 1-day spike, so aliases at
``f_alias = f_true +/- n`` cycles/day are strong and a naive periodogram peak can easily be the
wrong one. Any period-finding code that only ever ran on space-based data has not been tested.

Unlike the bundled K2 data, these carry **measured per-point uncertainties** from the survey
pipeline, so no error-estimation flag is raised.

## Where they came from, and the limitation that creates

Taken from the figure data of
[`jakevdp/PracticalLombScargle`](https://github.com/jakevdp/PracticalLombScargle) (BSD-3-Clause
for code), the reproduction repository for VanderPlas 2018, "Understanding the Lomb-Scargle
Periodogram", ApJS 236, 16, doi:10.3847/1538-4365/aab766. The underlying LINEAR photometry is
public survey data (Sesar et al. 2011, AJ 142, 190; Palaversa et al. 2013, AJ 146, 101).

**The same limitation as the K2 data applies:** this is a derived extract, not the original
survey data product, so it cannot be verified against the survey archive. Every product carries
the `THIRD_PARTY_MIRROR` flag at CAUTION severity, and it propagates.

## Published value for comparison

VanderPlas 2018 identifies the period of LINEAR object 11375941 as **approximately 2.58 hours**
(section 2, discussing Figures 1 and 2), citing Palaversa et al. 2013 for the object. The paper
quotes two significant figures, which is what the benchmark tolerance reflects -- see
`validation/targets.py`.

No published period was found for LINEAR 14752041 from a retrievable source; it is used only as
a second real light curve for exercising the code, never as a benchmark.
