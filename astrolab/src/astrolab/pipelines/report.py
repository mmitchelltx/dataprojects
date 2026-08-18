"""HTML report generation, with a provenance appendix.

The report is the deliverable a human reads, so it is built around one principle: **the
caveats are as prominent as the numbers.** A report that puts a fitted depth in large type and
buries "these uncertainties were estimated, not measured" in an appendix is worse than no
report, because it manufactures confidence the analysis does not support.

So quality flags appear at the top, before any result; every measurement is shown with its
credible interval rather than a bare value; and the provenance appendix records what was run,
from what, with which seeds, and what could not be checked.

Self-contained HTML with figures inlined as base64 PNG: a report that depends on files next to
it stops being readable the moment someone emails it.
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy import units as u

from astrolab.core.logging import get_logger
from astrolab.core.plotting import (
    fold_figure,
    lightcurve_figure,
    periodogram_figure,
    posterior_figure,
)
from astrolab.core.quality import Severity

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from astrolab.core.uncertainty import Measurement
    from astrolab.pipelines.transit import TransitPipelineResult

__all__ = ["write_report"]

log = get_logger(__name__)

_CSS = """
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#5a5a5a; --line:#e2e2e2;
        --unreliable:#8a1c1c; --caution:#8a5a00; --info:#2a4d80; --ok:#1c6b34; }
* { box-sizing:border-box; }
body { margin:0 auto; max-width:60rem; padding:2rem 1.25rem 4rem;
       font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       color:var(--fg); background:var(--bg); }
h1 { font-size:1.6rem; margin:0 0 .25rem; }
h2 { font-size:1.15rem; margin:2.25rem 0 .6rem; padding-bottom:.3rem;
     border-bottom:1px solid var(--line); }
h3 { font-size:1rem; margin:1.4rem 0 .4rem; }
.sub { color:var(--muted); margin:0 0 1.5rem; }
table { border-collapse:collapse; width:100%; margin:.5rem 0 1rem; font-size:13px; }
th,td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line);
        vertical-align:top; }
th { font-weight:600; color:var(--muted); font-weight:500; }
td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
.flag { border-left:3px solid var(--line); padding:.5rem .75rem; margin:.4rem 0;
        background:#fafafa; }
.flag .lvl { font-weight:600; font-size:12px; letter-spacing:.03em; }
.flag.UNRELIABLE { border-left-color:var(--unreliable); }
.flag.UNRELIABLE .lvl { color:var(--unreliable); }
.flag.CAUTION { border-left-color:var(--caution); }
.flag.CAUTION .lvl { color:var(--caution); }
.flag.INFO { border-left-color:var(--info); }
.flag.INFO .lvl { color:var(--info); }
.banner { padding:.9rem 1rem; border-radius:4px; margin:1rem 0 1.5rem;
          border:1px solid var(--line); background:#fafafa; }
.disposition { display:inline-block; padding:.15rem .5rem; border-radius:3px;
               font-size:12px; font-weight:600; background:#f0f0f0; }
.KNOWN_OBJECT { background:#e8f0e8; color:var(--ok); }
.FALSE_POSITIVE { background:#f6e8e8; color:var(--unreliable); }
figure { margin:1rem 0; }
img { max-width:100%; height:auto; display:block; border:1px solid var(--line); }
figcaption { color:var(--muted); font-size:12px; margin-top:.35rem; }
pre { background:#fafafa; border:1px solid var(--line); padding:.75rem;
      overflow-x:auto; font-size:12px; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.status { font-size:12px; color:var(--muted); }
"""


def _fig_to_img(fig: Figure, alt: str, caption: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _measurement_row(label: str, m: Measurement, unit: str = "", scale: float = 1.0) -> str:
    minus, plus = m.uncertainty_asymmetric
    value = float(m.value.value) * scale
    lo = float(minus.value) * scale
    hi = float(plus.value) * scale
    asym = " (asymmetric)" if m.asymmetry > 0.1 else ""
    return (
        f"<tr><th>{html.escape(label)}</th>"
        f'<td class="num">{value:.6g} <span class="status">+{hi:.3g} / -{lo:.3g}</span>'
        f"{html.escape(unit)}{asym}</td></tr>"
    )


def _flags_html(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return '<p class="status">No quality reservations were raised.</p>'
    out = []
    for f in flags:
        out.append(
            f'<div class="flag {html.escape(f["severity"])}">'
            f'<div class="lvl">{html.escape(f["severity"])} &middot; '
            f"{html.escape(f['flag'])}</div>"
            f"<div>{html.escape(f['message'])}</div></div>"
        )
    return "\n".join(out)


def write_report(result: TransitPipelineResult, path: str | Path) -> Path:
    """Render the pipeline result to a self-contained HTML file."""
    cfg = result.config
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append(f"<h1>{html.escape(cfg.target.label)} &mdash; transit analysis</h1>")
    parts.append(
        f'<p class="sub">Run <code>{html.escape(cfg.run.name)}</code> &middot; '
        f"generated {datetime.now(UTC).isoformat(timespec='seconds')} &middot; "
        f"config hash <code>{cfg.content_hash()[:16]}</code></p>"
    )

    # -- Caveats first, deliberately ---------------------------------------------------
    severity = result.quality.severity
    verdict = (
        "No reservations were raised."
        if severity is None
        else f"Highest reservation: <strong>{severity.name}</strong>. "
        + (
            "Results below are not supported by the data and must not be quoted."
            if severity >= Severity.UNRELIABLE
            else "Results below are usable only with the caveats stated."
        )
    )
    parts.append(
        f'<div class="banner"><strong>Read this first.</strong> {verdict}<br>'
        "<span class='status'>This pipeline produces <em>candidates</em>, never confirmed "
        "planets. A disposition of PLANET_CANDIDATE means nothing tested here killed the "
        "signal &mdash; not that it is a planet.</span></div>"
    )
    parts.append("<h2>Quality reservations</h2>")
    parts.append(_flags_html(result.quality.to_list()))

    # -- Data ---------------------------------------------------------------------------
    parts.append("<h2>Data</h2>")
    lc_summary = result.detrended.summary()
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td class='num'>{html.escape(str(v))}</td></tr>"
        for k, v in lc_summary.items()
    )
    parts.append(f"<table>{rows}</table>")
    parts.append(
        _fig_to_img(
            lightcurve_figure(result.raw, result.detrended, result.detrend_result.trend),
            "light curve before and after detrending",
            "Raw light curve with the fitted trend (top) and the detrended series (bottom). "
            "If the trend followed the transits rather than passing over them, the detrending "
            "would be eating the signal.",
        )
    )

    # -- Search --------------------------------------------------------------------------
    parts.append("<h2>Search</h2>")
    parts.append(f"<p>{html.escape(result.search.significance_note())}</p>")
    if result.search.periods.size:
        parts.append(
            _fig_to_img(
                periodogram_figure(result.search),
                "search periodogram",
                "An isolated peak well above the forest is a signal; a peak barely above a "
                "comb of similar peaks is a trial-factor problem.",
            )
        )

    if not result.candidates:
        parts.append("<p><strong>No candidates above the detection threshold.</strong></p>")

    # -- Candidates ----------------------------------------------------------------------
    for i, (candidate, vetting) in enumerate(zip(result.candidates, result.vetting, strict=True)):
        parts.append(
            f"<h2>Candidate {i + 1} &mdash; P = {candidate.period.value:.5f} d "
            f'<span class="disposition {html.escape(vetting.disposition)}">'
            f"{html.escape(vetting.disposition)}</span></h2>"
        )

        fit = result.fits.get(i)
        parts.append(
            _fig_to_img(
                fold_figure(result.detrended, candidate, fit),
                f"phase-folded transit for candidate {i + 1}",
                "Phase-folded photometry with binned points. A flat-bottomed profile with "
                "finite ingress is transit-like; a V-shape suggests a grazing or stellar "
                "eclipse.",
            )
        )

        parts.append("<h3>Vetting</h3><table>")
        parts.append("<tr><th>test</th><th>result</th></tr>")
        for test in vetting.tests:
            parts.append(
                f"<tr><th>{html.escape(test.name)}</th>"
                f"<td><strong>{html.escape(test.status)}</strong> &mdash; "
                f"{html.escape(test.detail)}</td></tr>"
            )
        parts.append("</table>")
        if vetting.not_run:
            parts.append(
                '<p class="status"><strong>Not tested:</strong> '
                + html.escape(", ".join(t.name for t in vetting.not_run))
                + ". The corresponding false-positive scenarios are not excluded.</p>"
            )

        if fit is not None:
            parts.append("<h3>Fitted parameters</h3><table>")
            parts.append(_measurement_row("period", fit.period, " d"))
            parts.append(_measurement_row("Rp/R*", fit.radius_ratio))
            parts.append(_measurement_row("depth", fit.depth, " ppm", scale=1e6))
            parts.append(_measurement_row("impact parameter", fit.impact_parameter))
            parts.append(_measurement_row("duration (T14)", fit.duration.to(u.hour), " h"))
            parts.append(
                f"<tr><th>ln(Bayes factor) vs flat</th>"
                f'<td class="num">{fit.log_bayes_factor:.1f}</td></tr>'
            )
            parts.append("</table>")
            parts.append(
                '<p class="status">Intervals are equal-tailed 68.27% credible regions from '
                "posterior samples, not symmetric error bars. The Bayes factor compares a "
                "transit against a flat line only &mdash; it says nothing about eclipsing "
                "binaries or systematics, which is what vetting is for.</p>"
            )
            parts.append(
                _fig_to_img(
                    posterior_figure(fit),
                    "joint posterior",
                    "Radius ratio against impact parameter. The curved degeneracy is real: a "
                    "larger planet near the limb mimics a smaller one crossing the centre, so "
                    "marginal error bars alone would misrepresent the result.",
                )
            )

    # -- Provenance appendix --------------------------------------------------------------
    parts.append("<h2>Provenance appendix</h2>")
    manifest = result.manifest.to_dict()
    assessment = manifest["reproducibility"]
    parts.append(
        "<p><strong>Fully reproducible: "
        f"{'yes' if assessment['fully_reproducible'] else 'no'}</strong></p>"
    )
    if assessment["caveats"]:
        parts.append(
            "<ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in assessment["caveats"]) + "</ul>"
        )
    git = manifest["git"]
    env = manifest["environment"]
    parts.append(
        "<table>"
        f"<tr><th>git SHA</th><td class='num'>{html.escape(str(git.get('sha')))}</td></tr>"
        f"<tr><th>working tree dirty</th><td>{html.escape(str(git.get('dirty')))}</td></tr>"
        f"<tr><th>python</th><td class='num'>{html.escape(env['python_version'])}</td></tr>"
        f"<tr><th>lockfile hash</th><td class='num'>"
        f"{html.escape(str(env['lockfile_hash']))}</td></tr>"
        f"<tr><th>config hash</th><td class='num'>{html.escape(manifest['config_hash'])}</td></tr>"
        f"<tr><th>seeds</th><td class='num'>{html.escape(json.dumps(manifest['seeds']))}</td></tr>"
        "</table>"
    )
    parts.append("<h3>Software versions</h3>")
    parts.append(
        "<table>"
        + "".join(
            f"<tr><th>{html.escape(k)}</th><td class='num'>{html.escape(v)}</td></tr>"
            for k, v in env["packages"].items()
        )
        + "</table>"
    )
    parts.append("<h3>Light-curve lineage</h3>")
    parts.append(
        f"<pre><code>{html.escape(json.dumps(result.raw.provenance, indent=2, default=str))}"
        "</code></pre>"
    )
    parts.append("<h3>Resolved configuration</h3>")
    parts.append(
        f"<pre><code>{html.escape(json.dumps(manifest['config'], indent=2, default=str))}"
        "</code></pre>"
    )

    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(cfg.target.label)} - astrolab</title>"
        f"<style>{_CSS}</style></head><body>" + "\n".join(parts) + "</body></html>"
    )
    out.write_text(document)
    log.info("pipeline.report.written", path=str(out), bytes=len(document))
    return out
