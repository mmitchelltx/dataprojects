# Mission status — verification log

**Purpose.** The `astrolab` brief asserted a set of facts about which observatories are producing
data right now. Those assertions drive real architectural decisions (what is a "real data" path vs.
a "simulated now, real later" path, what needs an auth token, what can be a golden-target test), so
they are verified rather than assumed, and re-verified on a schedule.

**All entries below verified on 2026-08-18** unless noted otherwise.

**Verification method and its limits.** Checked via web search against mission sites, STScI/IPAC
documentation, and refereed literature. Two primary sources — `rubinobservatory.org` and
`www.lsst.org` — are **blocked by this environment's network egress proxy**, so every Rubin entry
below is sourced from secondary reporting (NOIRLab, SLAC, community forum, `dp1.lsst.io`) rather
than read directly off the observatory's own pages. Rubin entries are marked accordingly and should
be re-checked from an unrestricted network before anything depends on them.

Confidence key: **[A]** primary source read directly · **[B]** secondary/aggregated source ·
**[C]** inferred, needs confirmation.

---

## Summary: what the brief got right, and what changed

| Brief's claim | Verdict | Note |
|---|---|---|
| JWST fully operational, public after ~12-month proprietary period | **Confirmed** | Cycle 5 science started 2026-07-01 |
| Roman launching ~30 Aug 2026, no science data yet | **Confirmed** | Launch is **12 days from today**; still no science data |
| Rubin 10-yr survey began 30 Jun 2026 | **Confirmed** | — |
| Rubin alert stream is the accessible entry point; full releases need data rights | **Confirmed, and sharper** | Broker list is 7 full-stream, not 4; 2-yr proprietary period then public |
| Rubin solar-system discoveries → MPC as obs code X05 | **Confirmed** | ~12,700 asteroids to date |
| "Always-available workhorses" incl. Gaia | **Confirmed, with a dated caveat** | **Gaia DR4 lands 2026-12-02** — mid-project. See below. |

Two findings change the plan rather than merely confirming it: **Gaia DR4 (Dec 2026)** and the
**DESI DR2 dynamical-dark-energy result**. Both are flagged in the relevant sections.

---

## JWST — operational, primary real-data source **[A/B]**

- Fully operational. **Cycle 5 science start: 2026-07-01.** Cycle 5 TAC met early Feb 2026;
  selections announced Mar 2026.
- Exclusive access period (EAP): GO programs default to **12 months**. Many programs are
  zero-EAP — all ERS, DD, and a substantial fraction of GO — and those are immediately public.
- EAP-protected data requires an STScI MyST account and, for programmatic access, an
  **Auth.MAST token**. Public data needs no credential.
- **Design implication:** the `archives/mast.py` client must handle both the anonymous and
  token-authenticated paths, and must *report* which one a given product came through in the
  provenance record — "was this file public at retrieval time" is a reproducibility fact.
- **Design implication:** golden-target JWST benchmarks must be chosen from **zero-EAP** programs
  (ERS in particular), so CI can fetch them without a secret.

## Roman — pre-launch, simulation-only **[A/B]**

- **Launch: 2026-08-30, 07:26 EDT**, Falcon Heavy, KSC LC-39A. Roughly eight months ahead of the
  original commitment. Commissioning follows launch; **no science data exists**, and none will for
  months after launch.
- Software stack is real and moving: **Roman I-Sim (`romanisim`) v0.12** (2026-01-20) added the new
  `roman_datamodels` L2 schema; **v0.13** (2026-02-11) added PSF-rendering performance work.
  `romancal` converts L1 (uncalibrated, 3D) → L2 (calibrated, 2D) and installs `stcal` +
  `roman_datamodels` alongside.
- **Roman Research Nexus** entered early-access for the community mid-Dec 2025 — cloud platform
  carrying Roman I-Sim datasets (resultants, rate images, a mosaic coadd) that are pipeline-compatible.
- **Design implication:** the brief's "simulated now, real later" instruction is correct and urgent.
  Everything Roman goes through a `simulated/` namespace with simulated-provenance stamped in the
  product metadata, and the source is a config switch. Note the launch date falls inside this
  project's likely lifetime — the interface will be exercised for real.

## Rubin / LSST — surveying; alerts public, pixels gated **[B — primary sources egress-blocked]**

- **LSST began 2026-06-30** (some reporting says first survey night 2026-07-01). Ten-year survey.
- Alert stream is live and is the accessible entry point, at roughly **7 million alerts/night**.
- **Full-stream brokers (7)** — the brief listed 4; the actual set is: **ALeRCE, AMPEL, ANTARES,
  Babamul, Fink, Lasair, Pitt-Google.** Two **down-stream** brokers (SNAPS, POI Broker) take a
  filtered subset via a full-stream partner.
- **Data rights, precisely:** alert packets and the contents of the Prompt Products Database are
  **public**. Images, coadds, and catalogs (DP1, DP2, and the annual data releases) require **data
  rights** — all scientists and students in the US and Chile, plus named members of international
  in-kind teams. Non-alert data products become open to everyone after a **two-year proprietary
  period**.
- DP1 (released 2026) is ~3.5 TB: raw + calibrated single-epoch images, coadds, difference images,
  and detection catalogs from 1792 ComCam exposures over 48 nights of late-2024 commissioning.
  Access is via the Rubin Science Platform, data-rights holders only.
- Solar system: Rubin submits to the MPC under **observatory code X05**, confirmed. ~1,900 asteroids
  from the Apr–May 2025 First Look, ~11,000 more since — ~12,700 total. MPC-submitted observations
  are openly available regardless of data rights.
- **Design implication:** the broker client is the Rubin path that works for anyone. The RSP/pixel
  path must be built so that not having data rights degrades to a clear, actionable error, never a
  silent empty result. **Open question for the user: do you hold Rubin data rights?**

## Gaia — DR3 current, **DR4 on 2026-12-02** **[B]**

- Gaia DR3 remains the current release and the astrometric reference frame in use.
- **Gaia DR4 is scheduled for 2026-12-02** — during this project. DR4 brings the first large
  astrometric exoplanet-candidate sample and substantially expanded variability and epoch data.
- **Design implication, and it is not small:** three pillars touch Gaia — astrometric calibration
  (solar system), variability cross-match (variables), stellar parameters (exoplanet radii). The
  catalog version must be an explicit, recorded config parameter from day one, never a hardcoded
  `gaiadr3.gaia_source` string scattered across modules. A DR3→DR4 migration should be a config
  edit plus a re-run of the golden targets with updated expected values. Budget for the fact that
  **golden-target expected values tied to Gaia will shift in December.**

## Other public workhorses **[B]**

- **TESS**: operating. New TESS Project Office policy (Feb 2026) enabling rapid community follow-up
  of high-energy transients. Formal extension status through the next senior review not confirmed
  here — **[C]**, worth a direct check.
- **Kepler/K2**: archival, stable, complete. Ideal golden targets precisely because nothing moves.
- **ZTF**: ZTF-III runs 2025-01-01 → 2026-12-31. Public releases annually — Jan 2026 (current) and
  Jan 2027. Images on a 60-day sliding window; light curves recomputed every 12 months.
- **Pan-STARRS**: **DR2 (Jan 2019) is still the current release.** No successor announced. Fine as a
  static reference catalog; do not expect new epochs.
- **Euclid**: Q1 released 2025-03-19 (63.1 deg² wide survey, extensive A&A paper series). **Q2 on
  2026-06-24**; **DR1 "Foundation" Nov 2026**; DR1 Complete mid-2027.
- **HST, Spitzer (archival), WISE/NEOWISE, SDSS, DESI**: available as stated in the brief.

## Cosmology data landscape — a live result the brief should absorb **[A/B]**

The brief frames the cosmology pillar around reproducing Pantheon+ ΛCDM Ωm. That remains the right
*first* benchmark — public, stable, well-documented, deterministic enough to test against. But the
field has moved and the module should be built for where it is:

- **DESI DR2 BAO (2025)** combined with CMB and SNe shows a preference for **evolving dark energy**
  (w₀waCDM over ΛCDM). Departures from ΛCDM run **2.6σ–3.9σ** depending on which SN compilation is
  used — Pantheon+, Union3, DES-SN5YR. DES-SN5YR alone favors w > −1 at the >1σ level.
- The spread *across SN compilations* is itself the story: it is a systematics question as much as a
  cosmology question, which is exactly what a systematics-budget-aware pipeline is for.
- **Design implication:** w₀waCDM is not an optional extra model in the cosmology pillar — it is a
  headline case, and the module should support swapping the SN compilation (Pantheon+ / Union3 /
  DES-SN5YR) as a config choice so the compilation-dependence can be *measured*, not inherited.
  This raises the value of the blinding requirement in the brief: this is a live, contested result.

---

## Re-verification policy

This file is a dated snapshot, not a standing truth. It gets re-verified:

- before any phase that depends on a mission's data availability,
- when a golden-target test fails in a way that smells like an upstream release rather than a
  regression,
- **on 2026-08-30** (Roman launch), and **on 2026-12-02** (Gaia DR4).

Each re-verification appends rather than overwrites, so the history of what was believed when stays
readable — which matters when reconstructing why a result computed in September differs from the
same config run in January.

---

## Sources

Accessed 2026-08-18.

- NASA Roman mission page — https://science.nasa.gov/mission/roman-space-telescope/
- NASA Roman blog — https://science.nasa.gov/blogs/roman/2026/06/03/hello-world-nasa-shares-new-home-for-roman-space-telescope-updates/
- SpacePolicyOnline, Roman launch date — https://spacepolicyonline.com/news/nasa-sets-launch-date-for-roman-space-telescope/
- STScI, Roman Research Nexus — https://www.stsci.edu/contents/newsletters/2026-volume-43-issue-01/join-the-roman-research-nexus-plug-into-a-full-suite-of-resources-and-simulated-data
- Roman user docs, Research Nexus — https://roman-docs.stsci.edu/data-handbook/roman-research-nexus
- Simulated data on the RRN (Zenodo) — https://zenodo.org/records/16929956
- JWST Cycle 5 proposal categories — https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-call-for-proposals-for-cycle-5/jwst-proposal-categories
- Accessing JWST data — https://jwst-docs.stsci.edu/accessing-jwst-data
- MAST Primer for JWST — https://outerspace.stsci.edu/spaces/MASTDOCS/pages/153686810/MAST+Primer+for+JWST
- STScI, JWST Cycle 5 peer review results — https://www.stsci.edu/files/live/sites/www/files/home/jwst/science-planning/user-committees/jwst-users-committee/_documents/jwst-cycle5-peer-review-results.pdf
- NOIRLab, Rubin real-time alert system — https://noirlab.edu/public/news/noirlab2605/
- SLAC, Rubin alert stream — https://www6.slac.stanford.edu/news/2026-02-25-nsf-doe-vera-c-rubin-observatory-launches-real-time-discovery-machine-monitoring
- Rubin DP1 documentation — https://dp1.lsst.io/index.html
- Rubin DP1 paper — https://arxiv.org/abs/2603.23786
- Rubin community forum, DP1 — https://community.lsst.org/t/rubin-observatory-data-preview-1/10405
- CNN, Rubin survey start — https://www.cnn.com/2026/07/01/science/rubin-observatory-legacy-survey-space-and-time
- B612, Rubin operations and MPC submissions — https://b612foundation.org/asteroid-day-2026-rubin-officially-begins-operations/
- GeekWire, Rubin 11,000 new asteroids — https://www.geekwire.com/2026/rubin-observatory-11000-new-asteroids/
- ESA Gaia DR4 — https://www.cosmos.esa.int/web/gaia/data-release-4
- ZTF public data releases — https://www.ztf.caltech.edu/ztf-public-releases.html
- Pan-STARRS1 archive — https://outerspace.stsci.edu/display/PANSTARRS/
- Euclid data release timeline — https://euclid.caltech.edu/page/data-release-timeline
- Euclid Q1 overview (A&A) — https://www.aanda.org/articles/aa/full_html/2026/07/aa54610-25/aa54610-25.html
- Evolving dark energy from DESI DR2 BAO + SNe — https://arxiv.org/abs/2508.10514
- Comparing DES-SN5YR and Pantheon+ analyses — https://arxiv.org/pdf/2501.06664
