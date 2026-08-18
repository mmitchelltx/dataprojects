# 0002 — Posterior samples as the canonical uncertainty type

**Status:** Accepted · **Date:** 2026-08-18

## Context

Prime Directive 2 requires every number to carry an uncertainty. That leaves open *how* an
uncertainty is represented internally, and the choice determines whether uncertainties stay correct
as they propagate through the pipeline.

The astrophysical reality driving this: transit parameters are neither Gaussian nor independent.
The planet-to-star radius ratio and the impact parameter are strongly correlated — a larger planet
crossing nearer the stellar limb produces nearly the same light curve as a smaller planet crossing
the center. Eccentricity posteriors pile up against a hard boundary at zero. Any representation
that assumes symmetric, independent errors gets these wrong in ways that are invisible until someone
compares against a published value.

## Options considered

**Linear propagation** (the `uncertainties` package). Cheap, composable, and correct only for small,
symmetric, independent errors. Fails on exactly the cases above.

**Summary statistics** (value plus symmetric or asymmetric σ). What papers report, and therefore
tempting. But summaries discard the covariance. Propagating Rp/Rs and impact parameter as
independent Gaussians into a planet radius produces an uncertainty that is simply wrong — not
conservative, wrong in an unpredictable direction.

**Posterior samples**, with summaries derived on demand. Correlations survive because the samples
survive. Asymmetry and boundaries are represented natively because nothing is assumed about shape.

## Decision

The canonical internal representation is **posterior samples carrying `astropy.units`**. Summary
statistics (median, 16th/84th percentile) are derived views computed at reporting boundaries, never
stored as the primary value.

Quantities that genuinely arise from linear propagation — a calibration offset with a known Gaussian
error, say — are represented as samples drawn from that distribution, so the type is uniform across
the codebase and no call site needs to branch on representation.

Statistical and systematic components are carried **separately** and combined only at final
reporting, where both the combination and the components are shown.

## Consequences

- Memory cost, which matters given the laptop-scale design assumption. Mitigated by making sample
  count a config parameter, supporting thinning, and spilling to Parquet rather than holding
  resident.
- Every downstream operation must be sample-aware. This is a real constraint on how `core/` is
  written, and it is deliberate: it makes the correct thing the easy thing.
- Reporting code gets slightly more complex, since it derives rather than reads. Worth it.
