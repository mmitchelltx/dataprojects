# Phase 3 status — variable stars

**Date:** 2026-08-19 · **Branch:** `claude/astrolab-design-phase-0-yru0n6`

Phase 3's criterion was: *"recover known RR Lyrae/Cepheid periods and reproduce the P–L relation
slope."*

**Period recovery: met, on real data.** **P–L relation: not built, and it cannot be here** — the
reason is data, not effort, and it is stated below rather than worked around.

---

## The headline result

| Object | Pipeline | Published | Agreement |
|---|---|---|---|
| LINEAR 11375941 | **0.107505 d = 2.5801 h** | ~2.58 h (VanderPlas 2018) | within the quoted precision |
| LINEAR 14752041 | 0.729890 d | none retrievable | consistent with a W UMa binary |

FAP ≈ 1e-76 for the first, against ~38,000 independent frequencies searched.

## Why this data was chosen

280 points over 1962 days at a **7-day mean spacing**. That is the point. The K2 transit
benchmark is uniform 29.4-minute sampling over 80 continuous days; this is sparse, irregular,
single-site, ground-based photometry whose spectral window has a **power-0.99 spike at 1
cycle/day**. Every candidate frequency therefore has strong competition at `f ± n` c/d, and the
tallest periodogram peak is routinely not the true period.

A period-finding module that had only ever run on space-based data would look fine and be
untested. This one is exercised where it is actually hard.

It also carries **measured** per-point uncertainties from the survey pipeline, unlike the K2
benchmark whose errors had to be estimated — so no estimation flag is raised, and the two
datasets exercise different paths through the uncertainty handling.

## What was built

```
core/lightcurve.py            magnitudes as well as normalised flux
instruments/linear/           LINEAR ingestion, MJD, sparse-sampling flags
science/variables/period.py   Lomb-Scargle, Baluev FAP, spectral window, alias
                              enumeration, harmonic canonicalisation, BIC resolution, PDM
science/variables/features.py amplitude, MAD, skew, kurtosis, Stetson K,
                              von Neumann eta, reduced chi2, beyond-1-sigma
validation/targets.py         LINEAR 11375941 golden period
```

**268 tests pass; `ruff` and `mypy --strict` clean.**

## Three bugs, all in alias handling, all found by running it

This is the part worth reading, because each was invisible to inspection and each produced
confidently wrong periods.

1. **BIC structurally cannot arbitrate period doubling.** Fitting at `2P` with the same number
   of harmonics is *strictly more flexible* than fitting at `P`: the even harmonics reproduce
   whatever the `P` model could, and the odd harmonics are free to absorb noise — at an
   identical parameter count. So BIC preferred the doubled period for every object it saw. The
   replacement is physical: ask whether the odd harmonics at the doubled period carry real
   power. They do not for a pulsator (2.2% for LINEAR 11375941 → correctly not doubled) and do
   for an eclipsing binary with unequal minima (14.3% for 14752041 → correctly doubled).
2. **Doubling re-entered through the alias route.** For a signal at 2 cycles/day, the offset
   alias `f − 1 c/d` *is* the doubled period, so the fix above never saw it. Every trial is now
   canonicalised to its fundamental *before* any BIC comparison, not just the winner.
3. **The halving floor used the median sampling interval.** That blocked reduction below 0.5 d
   for randomly sampled data. Unevenly sampled series have no simple Nyquist limit — the
   benchmark's own 2.58-hour period is recovered against a 7-day mean spacing — so the floor is
   now the smallest interval, not the median.

## An independent check on the classifications

The two objects were classified by the period machinery (pulsator vs eclipsing binary). The
feature statistics, computed independently, agree:

| Feature | 11375941 | 14752041 | Reading |
|---|---|---|---|
| skew | −0.41 | **+1.34** | positive skew = mostly bright with brief deep dips |
| beyond 1σ | 0.34 | **0.17** | binary sits near maximum light most of the time |
| R21 | 0.33 | **2.94** | R21 > 1 = two minima per cycle |
| reduced χ² vs constant | 15.2 | 110.1 | both unambiguously variable |

## What is deliberately not built

**No classifier.** A supervised model needs labelled light curves from ZTF, Gaia DR3
variability, OGLE, or VSX. None is reachable (every archive is blocked) and none is bundled.
Training on labels invented for the purpose would produce a model that looks like it works and
means nothing — the failure the no-fabrication directive and "ML ranks and triages; physics
fits" both exist to prevent. `features.py` produces the feature vector, documents what each
feature separates, and stops. Classification slots on top unchanged when labelled data exists.

**No period–luminosity relation, and so no distances.** A P–L calibration needs a *sample* of
RR Lyrae or Cepheids with independent distances — Gaia parallaxes, or a cluster at known
distance. Two field variables with no parallaxes and no calibrated photometry cannot produce a
slope. Fitting a line through two points and calling it a P–L relation would be theatre.

**No catalogue cross-match.** VSX, SIMBAD, and Gaia variability are all blocked, so novelty
cannot be assessed for these objects at all. Neither is claimed as new; both are treated as
known objects from the literature.

## To finish Phase 3

1. **Cross-match** against VSX/SIMBAD/Gaia once archives are reachable — required before any
   novelty statement, and currently impossible.
2. **A variable-star pipeline and report**, mirroring `pipelines/transit.py`, so
   `astrolab run` covers this pillar end to end with a provenance appendix. The science is
   built; this is plumbing.
3. **P–L relation** on a real calibrating sample (Gaia DR3 RR Lyrae with parallaxes would do
   it), which also gives the distance-scale machinery the design doc wants.
4. **Multiband periodograms** — the design mentions them, and they are the standard way to
   break exactly the aliases this module currently resolves by model comparison alone.
5. **Injection-recovery for period finding**, matching the gap noted for the transit search:
   completeness as a function of period, amplitude, and number of observations.
