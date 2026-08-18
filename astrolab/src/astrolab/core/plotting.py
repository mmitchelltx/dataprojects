"""Publication-quality figures, in one style module.

One place for the style so figures are consistent and call sites carry no magic numbers. Every
function returns a matplotlib Figure; nothing here writes files or calls ``show``, so plotting
stays composable and testable.

Figures are diagnostics first and decoration never. Each one exists to let a reader check a
specific claim: that the detrending removed variability without eating the transit, that the
periodogram peak is isolated rather than one of a forest, that the folded transit looks like a
transit rather than a V-shaped eclipse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from astropy import units as u

from astrolab.core.lightcurve import LightCurve

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from astrolab.science.exoplanets.fit import TransitFit
    from astrolab.science.exoplanets.search import Candidate, SearchResult

__all__ = [
    "STYLE",
    "fold_figure",
    "lightcurve_figure",
    "periodogram_figure",
    "posterior_figure",
]

#: Shared style. Colour-blind-safe qualitative colours, muted so data reads before decoration.
STYLE: dict[str, Any] = {
    "figure.figsize": (9.0, 4.0),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "legend.frameon": False,
}

DATA_COLOR = "#4c72b0"
MODEL_COLOR = "#c44e52"
TREND_COLOR = "#dd8452"
BINNED_COLOR = "#000000"


def _figure(nrows: int = 1, **kw: Any) -> tuple[Figure, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # cast: matplotlib types rc_context's argument as a TypedDict of every valid rcParam
    # key, which a plain dict literal cannot satisfy. The keys here are all real rcParams.
    with plt.rc_context(cast(Any, STYLE)):
        fig, axes = plt.subplots(nrows, 1, **kw)
    return fig, axes


def lightcurve_figure(
    raw: LightCurve, detrended: LightCurve, trend: np.ndarray | None = None
) -> Figure:
    """Raw light curve with the fitted trend, and the detrended residual below.

    The diagnostic: if the trend follows the transits rather than passing over them, the
    detrending is eating the signal.
    """
    fig, (ax0, ax1) = _figure(2, figsize=(9.0, 5.5), sharex=True)

    ax0.plot(raw.time.value, raw.flux.value, ".", ms=1.5, color=DATA_COLOR, label="raw")
    if trend is not None:
        ax0.plot(raw.time.value, trend, "-", lw=1.0, color=TREND_COLOR, label="trend")
    ax0.set_ylabel("normalised flux")
    ax0.legend(loc="lower left", ncols=2)
    ax0.set_title(
        f"{raw.meta.get('target', 'target')} ({raw.meta.get('mission', '?')}) - "
        f"{len(raw)} points, {raw.baseline.value:.1f} d"
    )

    ax1.plot(detrended.time.value, detrended.flux.value, ".", ms=1.5, color=DATA_COLOR)
    ax1.set_ylabel("detrended")
    ax1.set_xlabel(f"time ({detrended.time_system}, days)")
    fig.tight_layout()
    return fig


def periodogram_figure(search: SearchResult) -> Figure:
    """Search periodogram with detected candidates marked.

    The diagnostic: an isolated peak well above a flat forest is a signal; a peak barely above
    a comb of similar peaks is a trial-factor problem.
    """
    fig, ax = _figure()
    if search.periods.size:
        ax.plot(search.periods, search.power, "-", lw=0.7, color=DATA_COLOR)
    for candidate in search.candidates:
        ax.axvline(
            float(candidate.period.value),
            color=MODEL_COLOR,
            ls="--",
            lw=1.0,
            alpha=0.8,
        )
        ax.annotate(
            f"{candidate.period.value:.4f} d\nSDE {candidate.sde:.0f}",
            (float(candidate.period.value), ax.get_ylim()[1]),
            textcoords="offset points",
            xytext=(3, -22),
            fontsize=7,
            color=MODEL_COLOR,
        )
    ax.axhline(search.sde_threshold, color="k", ls=":", lw=0.8)
    ax.set_xlabel("period (d)")
    ax.set_ylabel("SDE")
    ax.set_title(f"{search.method} periodogram - {search.n_trials} trials")
    fig.tight_layout()
    return fig


def _bin_phase(
    phase: np.ndarray, flux: np.ndarray, n_bins: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(phase.min(), phase.max(), n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    centres, means = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 2:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(float(np.mean(flux[sel])))
    return np.asarray(centres), np.asarray(means)


def fold_figure(lc: LightCurve, candidate: Candidate, fit: TransitFit | None = None) -> Figure:
    """Phase-folded transit with binned points and, if available, the fitted model.

    The diagnostic: a planet transit has a flat-ish bottom and finite ingress; a V-shape with
    no flat bottom suggests a grazing geometry or an eclipsing binary.
    """
    fig, ax = _figure()
    phase = lc.fold(candidate.period, candidate.epoch).to(u.hour).value
    width = 4.0 * float(candidate.duration.to(u.hour).value)
    sel = np.abs(phase) < width

    ax.plot(phase[sel], lc.flux.value[sel], ".", ms=2, alpha=0.35, color=DATA_COLOR)
    if sel.sum() > 20:
        cx, cy = _bin_phase(phase[sel], lc.flux.value[sel])
        ax.plot(cx, cy, "o", ms=3.5, color=BINNED_COLOR, label="binned")

    if fit is not None:
        from astrolab.science.exoplanets.fit import _q_to_u, transit_model

        post = fit.posterior
        median = {k: float(np.median(v.value)) for k, v in post.samples.items()}
        u1, u2 = _q_to_u(median["q1"], median["q2"])
        grid = np.linspace(-width, width, 600) / 24.0  # hours -> days
        model = transit_model(
            grid + median["t0"],
            t0=median["t0"],
            period=median["period"],
            rp=median["rp"],
            a_rs=median["a_rs"],
            b=median["b"],
            u1=u1,
            u2=u2,
            exposure_days=float(fit.metadata["exposure_days"]),
            supersample=int(fit.metadata["supersample"]),
        )
        ax.plot(grid * 24.0, model, "-", lw=1.4, color=MODEL_COLOR, label="fitted model")

    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("normalised flux")
    ax.set_title(
        f"P = {candidate.period.value:.5f} d, depth = {candidate.depth_ppm:.0f} ppm, "
        f"{candidate.n_transits} transits"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def posterior_figure(fit: TransitFit, params: tuple[str, str] = ("rp", "b")) -> Figure:
    """Joint posterior of two parameters, to make a degeneracy visible.

    The default pair is the radius-ratio / impact-parameter degeneracy: a larger planet near
    the limb mimics a smaller one crossing the centre. Seeing the banana is the point -- it is
    why marginal error bars alone would misrepresent the result.
    """
    fig, ax = _figure(figsize=(4.6, 4.2))
    x = fit.posterior.samples[params[0]].value
    y = fit.posterior.samples[params[1]].value
    ax.plot(x, y, ".", ms=1.0, alpha=0.25, color=DATA_COLOR)
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_title(f"joint posterior, correlation = {fit.posterior.correlation(*params):+.2f}")
    fig.tight_layout()
    return fig
