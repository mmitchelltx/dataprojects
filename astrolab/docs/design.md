# astrolab — design

**Status:** Phase 0, awaiting approval · **Date:** 2026-08-18

---

## 1. What this is

A modular, reproducible Python toolkit for retrieving, calibrating, analyzing, and modeling data
from space-based observatories and public sky surveys. The bar is that an astronomer could use it to
produce a result they would defend in a refereed paper.

That bar mostly translates into three engineering properties, and the architecture exists to serve
them:

1. **Every number is traceable.** A value without an uncertainty and a lineage back to a specific
   archive query, reference file, and code version is a bug, not a result.
2. **Every claim has survived an attempt to kill it.** Detection is a conclusion reached after a
   documented false-positive gauntlet, not a threshold crossing.
3. **Every result regenerates from a config file.** Deterministic paths bit-for-bit, stochastic paths
   statistically equivalent under a recorded seed.

Everything below is downstream of those three.

## 2. Scope decisions (from Phase 0)

| Question | Decision | Consequence for the design |
|---|---|---|
| Data access | **Public only** | No credentialed paths on the critical path. Rubin via broker alert streams only; JWST restricted to zero-EAP programs. CI never needs a secret. |
| Compute | **Unknown → design for laptop-scale** | Chunked/streaming processing, strict cache discipline, small golden targets. Parallelism seams are explicit so scaling up is config, not rewrite. |
| Pillar order | Exoplanets → **variable stars** → reassess | The core is optimized for **time-series photometry**: light curves, periodograms, folding, GP noise models. Both pillars share it almost entirely. |
| Ambition | **Depth-first** | One pillar genuinely defensible beats five demos. Unbuilt pillars are marked unbuilt — no stubs returning plausible numbers. |

The depth-first choice is the most load-bearing one. It means the deliverable at the end is a
*narrow, complete, trustworthy* exoplanet transit pipeline with a second pillar reusing its core —
not a five-pillar framework. Sections 9 and 10 reflect this.

## 3. Architecture

```
astrolab/                      # project root inside the dataprojects monorepo
  src/astrolab/
    core/
      config.py                # pydantic-settings + YAML; a run is fully specified by its config
      provenance.py            # run manifest: git SHA, env hash, query params, seeds, versions
      units.py                 # unit conventions, Quantity helpers, boundary validation
      uncertainty.py           # the Measurement type (see 5.3)
      io/                      # FITS, ASDF, Parquet, HDF5; unit-aware Tables; self-describing products
      wcs.py                   # WCS/gwcs, coordinate frames, epoch propagation
      photometry.py            # aperture + PSF photometry, background estimation
      stats/                   # periodograms, FAP, GP kernels, model comparison, bootstrap
      plotting.py              # one style module; no magic numbers at call sites
    archives/
      mast.py gaia.py irsa.py vizier.py simbad.py horizons.py mpc.py skybot.py exoarchive.py
      brokers/                 # ZTF / Rubin alert stream clients
      cache.py                 # content-addressed; queries idempotent and replayable
    instruments/
      tess/ kepler/ jwst/ roman/ hst/ ztf/ rubin/
                               # per-instrument calibration quirks live HERE, never in science/
    science/
      exoplanets/ variables/ cosmology/ solarsystem/ anomaly/
    pipelines/                 # declarative DAGs composing the above
    cli.py                     # typer
    validation/                # golden-target regression suite
  docs/                        # mkdocs site + ADR log + methods notes
  configs/                     # one YAML per science run, version controlled
  tests/
```

`astrolab` lives as its own top-level project directory in the `dataprojects` monorepo, alongside
`energy_demand/` and the others, with a `src/` layout inside it. Rationale in ADR-0001.

**The three architectural rules, and why each exists:**

- **Science modules never call an archive directly.** They consume calibrated data products.
  This is not tidiness — it is what makes the instrument layer swappable, which is precisely what
  the Roman "simulated now, real later" requirement demands, and what lets the same transit-search
  code run on TESS and Kepler without knowing the difference.
- **Instrument systematics never leak into science code.** A scattered-light correction is a TESS
  fact, not a transit fact. When it lives in `science/`, every future instrument inherits a
  correction that does not apply to it.
- **All quantities carry `astropy.units` at module boundaries.** Bare floats crossing a boundary are
  a code smell that this codebase treats as an error: boundary functions validate. The failure mode
  being prevented is real and expensive — days in vs. days since epoch, ppm vs. fractional depth,
  arcsec vs. degrees.

## 4. Dependency choices

Preference throughout: boring, well-tested, widely-cited astropy-ecosystem packages. Custom numerics
require a test proving equivalence to a reference implementation.

| Need | Choice | Why this one |
|---|---|---|
| Env / deps | **uv**, Python 3.12 | Fast, real lockfile, straightforward CI. Everything needed is pip-installable — see risk R4 for the `pyccl` caveat if cosmology is built later. |
| Core arrays / tables / units | **astropy** | Non-negotiable foundation. `Quantity`, `Table`, `Time`, coordinates, WCS, cosmology. |
| Light curve access | **lightkurve** | The community standard for TESS/Kepler retrieval and basic manipulation. Well-maintained, well-documented. Used for *access*, not as the analysis engine. |
| Archive queries | **astroquery** | Broad, maintained coverage of MAST/Gaia/IRSA/VizieR/SIMBAD/Horizons under one API. Wrapped behind our own caching layer. |
| Detrending | **wotan** (search phase) + **celerite2** (fit phase) | Two different jobs — see 6.1. `wotan` is purpose-built for transit-preserving detrending and benchmarks its own methods. `celerite2` gives O(N) GPs, essential at laptop scale. |
| Transit search | **astropy BLS** + **transitleastsquares** | BLS as the well-understood baseline; TLS uses a limb-darkened template and is materially more sensitive to small planets. Running both is a cheap consistency check. |
| Transit forward model | **batman** | Stable, heavily cited, analytic Mandel-Agol. Does one thing. |
| Inference | **dynesty** | Nested sampling gives the Bayesian evidence directly, which the model-comparison requirement needs. See 6.2 for the alternative considered. |
| Limb darkening | **exotic-ld** | Actively maintained, supports theoretical coefficients from modern stellar grids for the comparison against free-fit values. |
| Period finding | **astropy LombScargle** | Correct FAP treatment, well-documented, the reference implementation everyone checks against. |
| Photometry | **photutils** | Astropy-affiliated, the standard for aperture and PSF photometry. |
| Config | **pydantic-settings** | Validation at load, typed, YAML-backed, good error messages. Catches config errors before compute is spent. |
| CLI | **typer** | Type hints become the interface. Low ceremony. |
| Logging | **structlog** | Structured events, not prose. Every query, reference file, and seed becomes a queryable record. |
| Data versioning | **pooch** | Checksummed fetch-and-cache. Simpler than DVC and sufficient here; DVC revisited only if data volume demands it. |
| Testing | **pytest**, **hypothesis**, **pytest-regressions** | Unit + numerical invariants + golden outputs, respectively. |
| Lint / types | **ruff**, **mypy --strict** on `core/` | Strict typing on the shared foundation where breakage propagates; pragmatic elsewhere. |
| Pipelines | **Snakemake** *(deferred to Phase 3)* | Not adopted until there is enough of a DAG to justify it. Phase 2 is a linear pipeline; a workflow engine before then is ceremony. |

**Things deliberately not chosen yet:** `juliet`, `exoplanet`/PyMC, `petitRADTRANS`, `POSEIDON`,
`sncosmo`, `pyccl`, `sbpy`. Each belongs to a phase not yet approved, and each carries real
installation weight. They get evaluated when their phase starts, not speculatively pinned now.

**Maintenance flags, honestly:** `transitleastsquares` is essentially feature-complete but not
actively developed — acceptable for a well-specified algorithm, and BLS runs alongside as a check.
`exotic-ld` requires downloading stellar grid files (~GB) on first use, which needs handling under
the laptop-scale assumption. Both are noted in ADR-0005.

## 5. Cross-cutting design

### 5.1 Config

One YAML per run, loaded into a validated pydantic model. The config is the complete specification:
target, data source, every algorithm choice, every prior, every seed, every tolerance. If a number
influences a result and is not in the config or the lockfile, that is a defect.

Priors are declared explicitly in the config, never buried in code defaults. This is what makes the
required prior-sensitivity check a config sweep rather than a code change.

### 5.2 Provenance

Every run emits a manifest recording: git SHA and working-tree-dirty flag, environment lockfile
hash, resolved config (post-validation, post-defaults), every archive query with its parameters and
response hash, every reference/calibration file with its version, all random seeds, software
versions of the science-relevant packages, wall-clock timestamps, and any override flags used
(`--allow-unconverged` stamps itself here).

Every data product carries a metadata block naming its inputs, the software that made it, and the
parameters used. Products are self-describing: given a file and no other context, you can determine
what it is and how it came to exist.

### 5.3 Uncertainty representation — the `Measurement` type

**This is a genuine design decision with no obviously right answer**, so the reasoning is explicit.

Three options were considered:

- **Linear error propagation** (`uncertainties` package): cheap and composable, but wrong whenever
  the posterior is asymmetric or correlated — which for transit depth, impact parameter, and
  eccentricity is the normal case, not the exception. Rejected as the primary representation.
- **Summary statistics** (value + σ, or value + asymmetric σ): what papers report, but throws away
  the correlations that matter for downstream propagation. Rp/Rs and impact parameter are strongly
  correlated; propagating them as independent Gaussians misstates the planet radius.
- **Posterior samples as canonical, summaries derived** ← **chosen.**

The canonical internal representation of an uncertain quantity is its posterior samples, carrying
units. Summary statistics (median, 16th/84th percentile) are *derived views* for reporting, computed
at the boundary. Correlations survive because samples survive. Where a quantity genuinely comes from
linear propagation (a calibration offset, say), it is represented as samples drawn from that
distribution, so the type is uniform.

Cost: memory, which matters under the laptop-scale assumption. Mitigation: sample count is a config
parameter, thinning is available, and samples spill to disk (Parquet) rather than staying resident.

Statistical and systematic uncertainties are carried as **separate components**, combined only at
the final reporting step and always reported separately as well. Collapsing them early destroys
information that a reader needs.

### 5.4 Caching

Content-addressed by a hash of the normalized query specification. Re-running a pipeline must not
re-hammer MAST. The cache records what was fetched, when, and from where, so a cached result and a
live result are distinguishable in the provenance record — that distinction matters when an archive
changes underneath you.

Raw data is immutable. Nothing is edited in place, ever. Corrections produce new products.

### 5.5 "I don't know" as a first-class output

Directive 5 requires the pipeline to flag rather than guess. Concretely, this is a `QualityFlag`
mechanism threaded through the products: insufficient S/N for the requested measurement, cadence
undersampling the feature being fitted, unconverged chains, data gaps overlapping the event of
interest, systematics amplitude comparable to the signal. A flagged result still gets written — with
the flag prominent in the product metadata, in the report, and in the CLI exit status. Silence about
uncertainty is the failure mode being designed against.

## 6. Open domain judgment calls

Per the brief, these are laid out rather than silently decided. Each has a recommendation; all are
config-switchable, so a different choice later is a config edit, not a rewrite.

### 6.1 Detrending strategy — recommended: two-stage

Stellar variability and instrumental systematics must be removed, but *how* and *when* changes the
answer. The trap: detrend aggressively before fitting, and the detrending eats part of the transit,
biasing depth downward — which propagates straight into planet radius and then into anything built
on it.

- **Option A — detrend then fit.** Fast, simple, standard in search pipelines. Biases depth. The
  bias is small for deep transits and can be severe for shallow ones.
- **Option B — joint fit of transit + noise model.** GP (celerite2, e.g. a SHO or Matérn-3/2 kernel)
  fitted *simultaneously* with the transit. Unbiased, correctly propagates noise-model uncertainty
  into the transit parameters. Costs much more compute.
- **Option C — two-stage** ← **recommended.** Use `wotan`'s biweight filter for the *search* phase,
  where speed matters and small depth bias is irrelevant to detection; then re-extract the raw light
  curve around the detected events and do a **joint transit + GP fit** for the *measurement* phase,
  where bias matters and only a small span of data is involved.

C is recommended because the two phases have genuinely different requirements, and it keeps the
expensive method confined to the small data span where it earns its cost — which matters directly
under the laptop-scale assumption. The window half-width for the biweight filter should be set from
the expected transit duration (a common choice is ~3× duration) and recorded in the config; too
narrow and the filter eats the transit even in the search phase.

### 6.2 Inference engine — recommended: dynesty over a custom likelihood

- **`juliet`** wraps transit + RV + GP modeling with nested sampling and would save real work. But
  it adds a layer between us and the likelihood, and the likelihood is exactly the object we most
  want to be explicit and inspectable. Maintenance is also less active than the components it wraps.
- **`exoplanet`/PyMC** is powerful and gradient-based (fast for high dimensions) but pulls in a
  heavy stack and makes evidence computation awkward.
- **`batman` + `celerite2` + `dynesty` with our own likelihood** ← **recommended.** Fewer layers,
  every modeling assumption visible in code we own, and nested sampling yields the evidence that the
  model-comparison requirement needs. The custom code is the likelihood function only — thin, and
  testable against `batman` reference outputs directly.

Cost of this choice: we own the likelihood, so we own its bugs. Mitigated by testing against
published parameters for the golden target (§7) and by a `hypothesis` invariant suite.

### 6.3 Limb darkening — recommended: fit free, compare to theory

Fixing coefficients to theoretical values understates the depth uncertainty; fitting them freely
inflates it and can be unconstrained in low-S/N data. The recommendation is to fit with priors
centered on `exotic-ld` theoretical values, then report both the free-fit and fixed-theory results
and their difference as a **systematic** contribution. This makes the choice visible in the output
rather than hidden in a default.

## 7. Validation strategy

The validation suite is what makes the claim "research-grade" checkable rather than asserted. Each
benchmark records: expected value, tolerance, source citation, last-verified date.

**Phase 2 golden target — recommended: WASP-18 b (TOI-185.01).** A hot Jupiter with P ≈ 0.94 d and a
deep transit, so a single TESS sector contains many transits — the whole benchmark is one small
download. Published TESS-derived parameters exist in Shporer et al. 2019 (TESS full orbital phase
curve of WASP-18b), including Rp/Rs = 0.09716 ± 0.00014.

The remaining parameters (period, duration, depth, their uncertainties) will be **extracted from the
paper and cited at implementation time, not guessed now.** A second, shallower target should be
added as well — WASP-18 b is deep enough to be forgiving, and a pipeline that only works on easy
signals has not been tested.

**Phase 3 (variables) golden targets:** known RR Lyrae and Cepheid periods recovered from public
survey photometry, plus reproduction of a published period–luminosity slope. Values sourced and
cited at implementation.

### The CI-versus-real-data problem, and its resolution

There is a genuine tension: Directive 1 forbids fabricated data and fake fixtures, but CI cannot
hammer MAST on every pull request, and archive downtime would make the suite flaky for reasons
unrelated to code quality.

Resolution: **real data, pinned and checksummed.** A small snapshot of genuinely-retrieved archive
data is stored via `pooch` with checksums and full provenance for how it was obtained. CI runs
against that snapshot — real observations, immutable, offline. A separate **scheduled** job re-fetches
live from the archive and compares, so upstream drift is detected deliberately rather than
discovered as a mysterious CI failure. Nothing is synthesized; the fixture is a cached real
observation, and its manifest says so.

## 8. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **Scope.** Five pillars is a multi-year program, not a project. | High | Depth-first decision already taken. Unbuilt pillars stay visibly unbuilt. This is the primary risk and the primary mitigation is saying no. |
| R2 | **Compute budget unknown.** Nested sampling and injection-recovery can each consume a machine for days. | High | Design for laptop-scale; joint GP fits confined to small data spans (§6.1); sample counts and injection grids are config parameters with honest cost documentation. Needs a real answer before Phase 2 fitting work. |
| R3 | **Gaia DR4 lands 2026-12-02**, mid-project, touching three pillars. | Medium | Catalog version is a config parameter from day one, never hardcoded. Golden-target values tied to Gaia will need re-verification in December — this is scheduled, not a surprise. |
| R4 | **Dependency weight** in later phases: the `jwst` pipeline + CRDS network dependency, `petitRADTRANS` opacity data at GB scale, `pyccl` build friction. | Medium | Deferred, not pinned. Each is evaluated at its phase. Under laptop-scale these may prove infeasible locally, which is a finding to report rather than paper over. |
| R5 | **Custom likelihood bugs** (§6.2). | Medium | Golden target must reproduce published parameters within uncertainty, plus `hypothesis` invariants and direct comparison to `batman` reference outputs. If the golden target fails, the pipeline is wrong — that is the point of it. |
| R6 | **Archive instability / API drift** in `astroquery` and MAST. | Medium | Caching layer isolates it; pinned snapshot for CI; scheduled live-fetch job detects drift early (§7). |
| R7 | **Rubin facts rest on secondary sources** — `rubinobservatory.org` and `lsst.org` are egress-blocked from this environment. | Low now, higher if Rubin work starts | Documented in `mission-status.md`. Re-verify from an unrestricted network before building anything Rubin-dependent. |
| R8 | **Overconfident output.** The most dangerous failure is a confident-looking number the data does not support. | High | §5.5 `QualityFlag` mechanism; mandatory injection-recovery; trial-factor correction; convergence diagnostics gating writes. |

## 9. Phase plan (revised for depth-first)

**Phase 0 — Plan.** ← *awaiting your approval.* Mission status verified (`docs/mission-status.md`),
this design doc, ADR log seeded.

**Phase 1 — Core + archives.** `core/` and `archives/` with content-addressed caching, provenance
manifests, unit-aware IO, structured logging, and a working `astrolab query` CLI.
*Done when:* a TESS light curve and a public JWST product can be pulled from MAST by target name,
and the run manifest fully reproduces the query.

**Phase 2 — Exoplanet transit vertical slice.** Raw → calibrated → detrended → searched → fitted →
vetted → report. Includes the vetting gauntlet (odd/even, secondary eclipse, centroid shift,
ephemeris match against known EBs and planets, dilution correction) and injection-recovery for
detection efficiency.
*Done when:* `astrolab run configs/wasp18b.yaml` reproduces published parameters within
uncertainties and emits a report with a provenance appendix.

**Phase 3 — Variable stars.** Reuses the Phase 2 time-series core: Lomb-Scargle with correct FAP and
window-function analysis, phase-dispersion minimization, harmonic fitting, feature extraction,
cross-match against VSX/Gaia/SIMBAD before any novelty claim.
*Done when:* known RR Lyrae/Cepheid periods are recovered and the P–L slope is reproduced.

**Phase 4 — Reassess.** With two pillars genuinely working, decide what is worth building next based
on what has been learned about compute and effort — rather than committing now to phases whose cost
is currently guesswork.

Phases for cosmology, solar system, JWST spectroscopy, Roman, and the anomaly layer remain in the
brief as the long-range vision. They are deliberately **not** scheduled here. Committing to them now
would be the kind of confident-looking number this project is built to avoid.

## 10. Definition of done for the approved scope

- `astrolab run configs/wasp18b.yaml` reproduces published WASP-18 b parameters within stated
  uncertainties, from a config file plus a pinned environment.
- Re-running it produces bit-identical output on deterministic paths and statistically equivalent
  output on seeded stochastic paths.
- Every reported number carries an uncertainty, a provenance record, and a quality flag.
- The variable-star golden targets pass, reusing the same core.
- CI runs lint, strict types on `core/`, unit tests, and the fast validation subset, offline, on real
  pinned data.
- `docs/` builds to a site with a worked example per implemented module, an ADR log explaining why
  each domain decision was made, and honest documentation of what is *not* built.

---

## Questions for you before Phase 1

1. **Compute (R2)** — the largest open risk. Even a rough answer ("16 GB laptop", "64 GB workstation")
   changes how the Phase 2 fitting stage is designed. "Not sure yet" is workable; a number is better.
2. **The three recommendations in §6** — detrending strategy, inference engine, limb darkening.
   Silence reads as agreement, but §6.2 in particular (owning the likelihood vs. using `juliet`) is
   a real fork worth a moment.
3. **Golden target (§7)** — WASP-18 b as primary, plus a shallower second target. Any preference for
   a specific system you already know the literature on?
