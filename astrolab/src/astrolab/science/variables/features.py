"""Variability features for characterisation and triage.

These are the standard descriptors a variable-star classifier consumes. They are computed here;
the classifier itself is deliberately **not** built, and the reason is worth stating plainly
rather than leaving as an omission.

Why there is no classifier here
-------------------------------
A supervised classifier needs a labelled training set -- ZTF, Gaia DR3 variability, OGLE, or
VSX cross-matched to light curves. None of those catalogues is reachable from this environment,
and none is bundled. Training a classifier on labels invented for the purpose would produce a
model that looks like it works and means nothing, which is precisely what the no-fabrication
directive forbids and what "ML ranks and triages; physics fits" is meant to prevent.

So this module produces the feature vector, documents what each feature is for, and stops
there. When a labelled set is available, classification slots in on top of these features
without any of them changing.

What the features are for
-------------------------
The physically informative ones are about *shape*, not size. Amplitude alone barely separates
classes; the Fourier ratios R21 and phi21 (in :mod:`astrolab.science.variables.period`) do a
great deal of the work, because they encode light-curve asymmetry. RRab stars have sawtooth
profiles with steep rises; RRc are near-sinusoidal; contact binaries have two minima per cycle
and so show strong even harmonics; Cepheids follow their own progression of R21 with period.

References
----------
Stetson 1996, PASP 108, 851. doi:10.1086/133808 -- variability indices J and K.
Richards et al. 2011, ApJ 733, 10. doi:10.1088/0004-637X/733/1/10 -- features for classification.
Nun et al. 2015, arXiv:1506.00010 -- FATS: feature analysis for time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astrolab.core.lightcurve import LightCurve

__all__ = ["VariabilityFeatures", "extract_features"]


@dataclass
class VariabilityFeatures:
    """Shape and scatter descriptors for one light curve."""

    n_points: int
    baseline_days: float
    weighted_mean: float
    amplitude: float
    """Robust peak-to-peak: the 5th-to-95th percentile range, insensitive to single outliers."""

    mad: float
    """Median absolute deviation, scaled to a Gaussian-equivalent sigma."""

    skew: float
    kurtosis: float
    beyond_1_std: float
    """Fraction of points more than one standard deviation from the mean.

    A Gaussian gives about 0.32. Eclipsing binaries sit near maximum light most of the time and
    give a much smaller value; sinusoidal pulsators give a larger one.
    """

    stetson_k: float
    """Stetson's K: a kurtosis-like measure of the magnitude distribution's shape.

    About 0.798 for a Gaussian, lower for sharply peaked distributions.
    """

    von_neumann_eta: float
    """Ratio of mean square successive difference to variance.

    About 2 for uncorrelated noise; much less than 2 indicates smooth, correlated variation --
    i.e. a real signal rather than scatter.
    """

    reduced_chi2_constant: float
    """Chi-square per degree of freedom against a constant model.

    The most direct statement of "is this star variable at all", given measured uncertainties.
    Values far above 1 mean the scatter exceeds what the error bars allow.
    """

    is_magnitude: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "baseline_days": self.baseline_days,
            "weighted_mean": self.weighted_mean,
            "amplitude": self.amplitude,
            "mad": self.mad,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "beyond_1_std": self.beyond_1_std,
            "stetson_k": self.stetson_k,
            "von_neumann_eta": self.von_neumann_eta,
            "reduced_chi2_constant": self.reduced_chi2_constant,
            "photometry_system": "magnitude" if self.is_magnitude else "relative_flux",
        }


def extract_features(lc: LightCurve) -> VariabilityFeatures:
    """Compute the variability feature vector for a light curve.

    Features are computed in whatever photometric system the light curve carries, and the
    system is recorded. That matters: an amplitude of 0.4 in magnitudes is a 45% flux
    variation, so comparing a magnitude-space amplitude against a flux-space one is
    meaningless. Any classifier trained on these features must be trained on features from the
    same system.
    """
    value = lc.flux.value
    error = lc.flux_err.value
    n = len(value)
    if n < 3:
        raise ValueError(f"need at least 3 points to characterise variability, got {n}")

    weights = 1.0 / error**2
    weighted_mean = float(np.sum(value * weights) / np.sum(weights))

    lo, hi = np.percentile(value, [5.0, 95.0])
    amplitude = float(hi - lo)

    centred = value - float(np.mean(value))
    std = float(np.std(value, ddof=1))
    mad = float(1.4826 * np.median(np.abs(value - np.median(value))))

    skew = float(np.mean(centred**3) / std**3) if std > 0 else 0.0
    kurtosis = float(np.mean(centred**4) / std**4 - 3.0) if std > 0 else 0.0
    beyond = float(np.mean(np.abs(centred) > std)) if std > 0 else 0.0

    # Stetson K, on the normalised residuals from the weighted mean.
    delta = np.sqrt(n / (n - 1.0)) * (value - weighted_mean) / error
    denom = float(np.sqrt(np.sum(delta**2)))
    stetson_k = float(np.sum(np.abs(delta)) / (np.sqrt(n) * denom)) if denom > 0 else 0.0

    order = np.argsort(lc.time.value)
    ordered = value[order]
    successive = float(np.sum(np.diff(ordered) ** 2) / (n - 1.0))
    eta = float(successive / std**2) if std > 0 else 0.0

    chi2 = float(np.sum(((value - weighted_mean) / error) ** 2))
    reduced_chi2 = chi2 / (n - 1.0)

    return VariabilityFeatures(
        n_points=n,
        baseline_days=float(lc.baseline.value),
        weighted_mean=weighted_mean,
        amplitude=amplitude,
        mad=mad,
        skew=skew,
        kurtosis=kurtosis,
        beyond_1_std=beyond,
        stetson_k=stetson_k,
        von_neumann_eta=eta,
        reduced_chi2_constant=reduced_chi2,
        is_magnitude=lc.is_magnitude,
    )
