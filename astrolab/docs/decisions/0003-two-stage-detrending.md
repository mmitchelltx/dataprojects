# 0003 — Two-stage detrending: filter to search, joint-fit to measure

**Status:** Proposed (awaiting Phase 0 sign-off) · **Date:** 2026-08-18

## Context — the domain problem

A raw TESS light curve is not a clean measurement of stellar brightness. It contains, mixed
together: the astrophysical signal you want (a transit — a ~0.1–1% dip lasting a few hours),
genuine stellar variability (spots rotating in and out of view, pulsations, flares), and
instrumental systematics (scattered light from Earth and Moon, pointing jitter, thermal drifts,
detector effects).

Removing the unwanted components is called *detrending*, and it is where transit photometry is most
easily gotten wrong. The trap: a filter that removes slow trends does not know that the transit is
signal rather than trend. Run it aggressively and it absorbs part of the transit into the baseline,
making the measured depth shallower than the truth. Depth maps directly to planet radius, so a
biased depth means a biased planet — and the bias is systematic, meaning it does not average away
across many planets. It propagates into any population-level conclusion built on top.

## Options considered

**A — Detrend, then fit the residuals.** Fast and simple, and what most search pipelines do. Biases
the depth. The bias scales with how aggressive the filter is relative to the transit duration, and
it is worst for the shallow transits that matter most.

**B — Joint fit of transit and noise model.** Model the correlated noise with a Gaussian Process
(`celerite2`, e.g. a stochastically-driven harmonic oscillator or Matérn-3/2 kernel) and fit it
*simultaneously* with the transit model. The transit is no longer at the mercy of a preprocessing
step, and — importantly — uncertainty in the noise model propagates properly into the transit
parameter uncertainties rather than being silently ignored. Costs far more compute: a GP likelihood
evaluated inside a nested sampling loop, over a full sector of data, is expensive even at
`celerite2`'s O(N).

**C — Two stages, matched to two different jobs.**

## Decision

Use **C**:

1. **Search phase:** `wotan`'s biweight filter over the full light curve. Speed matters here and a
   small depth bias does not — the only question being asked is "is there a periodic dip." The
   filter window half-width is set from the expected transit duration (a common choice being ~3×
   duration) and is recorded in the config. Too narrow a window eats the transit even at this stage.
2. **Measurement phase:** for each detected candidate, re-extract the *raw* light curve around the
   events and perform a joint transit + GP fit over that limited span.

## Rationale

The two phases have genuinely different requirements, and conflating them forces a bad compromise:
either search slowly or measure sloppily. Splitting them lets each be right.

The compute argument is not incidental — it is what makes B affordable at all. Restricting the
expensive joint fit to a few transit durations of data around each candidate, rather than a full
sector, is what fits the laptop-scale design assumption. This matters more here than it would on a
cluster.

## Consequences

- Two code paths for detrending, which must be kept consistent about what "the light curve" means.
  The raw data is retained (never edited in place) precisely so stage 2 can go back to it.
- The biweight window becomes a config parameter with a real effect on completeness. Its influence
  must be measured by the injection-recovery suite, not assumed benign.
- The GP kernel choice becomes a modeling assumption that needs justification in the run's output —
  a kernel that can mimic a transit is a genuine hazard, and kernel choice should be reported.
- Both stages are config-switchable, so the pure-A and pure-B paths remain available for comparison.
  Being able to *measure* the depth bias that A introduces is itself a useful diagnostic.
