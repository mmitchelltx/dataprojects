"""Command-line interface.

The CLI is the primary way to run astrolab. Notebooks are for reports generated *from*
pipeline outputs, never the source of truth -- a result that exists only in someone's kernel
state is not a result anyone else can check.

Exit codes are meaningful, because a pipeline is something scripts call:

===  ===========================================================================
0    success
1    usage or configuration error
2    the query succeeded and matched nothing (a real answer, not a crash)
3    matching data exist but are under an exclusive access period
4    the archive could not be reached or returned an error
5    the run produced results, but quality flags mark them unreliable
===  ===========================================================================

Code 5 exists so that a caller cannot mistake a flagged result for a clean one just because
the process exited without an exception.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from astrolab import __version__
from astrolab.archives.base import (
    ArchiveError,
    EmptyResultError,
    ProprietaryDataError,
    QuerySpec,
)
from astrolab.archives.cache import QueryCache
from astrolab.archives.mast import MastClient, build_observation_spec
from astrolab.core.config import AstrolabConfig, ConfigError
from astrolab.core.logging import configure_logging, get_logger
from astrolab.core.provenance import RunManifest

app = typer.Typer(
    name="astrolab",
    help="Reproducible analysis of space-observatory and sky-survey data.",
    no_args_is_help=True,
    add_completion=False,
)
cache_app = typer.Typer(
    name="cache", help="Inspect and manage the query cache.", no_args_is_help=True
)
app.add_typer(cache_app)

log = get_logger("astrolab.cli")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_EMPTY = 2
EXIT_PROPRIETARY = 3
EXIT_ARCHIVE = 4
EXIT_UNRELIABLE = 5


def _err(message: str) -> None:
    """Diagnostics go to stderr, so stdout stays usable as a data channel."""
    typer.secho(message, fg=typer.colors.RED, err=True)


def _ok(message: str) -> None:
    """Report-command results go to stdout: they are the answer, not a side note."""
    typer.secho(message, fg=typer.colors.GREEN)


def _status(message: str) -> None:
    """Progress and status for commands whose real output is a file."""
    typer.secho(message, fg=typer.colors.GREEN, err=True)


def _load_config(path: Path) -> AstrolabConfig:
    try:
        return AstrolabConfig.from_yaml(path)
    except ConfigError as exc:
        _err(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc


def _spec_from_config(cfg: AstrolabConfig) -> QuerySpec:
    """Translate a validated config into an archive query spec."""
    if cfg.query is None:
        raise ConfigError("config has no 'query' section; nothing to retrieve")
    if cfg.query.archive != "mast":
        raise ConfigError(f"archive {cfg.query.archive!r} is not implemented")
    return build_observation_spec(
        target_name=cfg.target.name,
        ra_deg=cfg.target.ra_deg,
        dec_deg=cfg.target.dec_deg,
        radius_arcsec=cfg.target.search_radius_arcsec,
        mission=cfg.query.mission,
        product=cfg.query.product,
        author=cfg.query.author,
        sequence=cfg.query.sequence,
        exptime_seconds=cfg.query.exptime_seconds,
    )


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug-level logging.")] = False,
    json_logs: Annotated[bool, typer.Option("--json-logs", help="Emit JSON log lines.")] = False,
) -> None:
    """Global options."""
    configure_logging(level="DEBUG" if verbose else "INFO", json_output=json_logs, force=True)


@app.command()
def version() -> None:
    """Print the astrolab version."""
    typer.echo(__version__)


@app.command("validate")
def validate_config(
    config: Annotated[Path, typer.Argument(help="Path to a run config YAML.")],
) -> None:
    """Validate a config file without running anything.

    Worth its own command: catching a typo before a download beats catching it after.
    """
    cfg = _load_config(config)
    _ok(f"config valid: {config}")
    typer.echo(f"  run:      {cfg.run.name}  (seed {cfg.run.seed})")
    typer.echo(f"  target:   {cfg.target.label}")
    typer.echo(f"  hash:     {cfg.content_hash()[:16]}")

    # A config may describe an archive query, a pipeline run, or both. Report what is there
    # rather than assuming one shape.
    if cfg.query is not None:
        typer.echo(f"  query:    {cfg.query.mission} / {cfg.query.product}")
        typer.echo(f"  will run: {_spec_from_config(cfg).describe()}")
    if cfg.source is not None and cfg.transit is not None:
        typer.echo(f"  source:   {cfg.source.kind} ({cfg.source.variant})")
        typer.echo(
            f"  transit:  duration {cfg.transit.expected_duration_hours} h, "
            f"search {cfg.transit.search.min_period_days}-"
            f"{cfg.transit.search.max_period_days} d, "
            f"fit {'on' if cfg.transit.fit.enabled else 'off'}"
        )


@app.command()
def query(
    config: Annotated[Path, typer.Argument(help="Path to a run config YAML.")],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Override run.output_dir from the config."),
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Force a live query, bypassing the cache.")
    ] = False,
) -> None:
    """Run the archive query described by a config file and write a provenance manifest.

    Everything needed to reproduce the query lands in ``manifest.json``; ``astrolab replay``
    consumes it.
    """
    cfg = _load_config(config)
    out_root = Path(output_dir) if output_dir else cfg.run.output_dir
    run_dir = out_root / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        run_name=cfg.run.name,
        config=cfg.resolved_dict(),
        config_hash=cfg.content_hash(),
    )
    manifest.record_seed("master", cfg.run.seed)
    if no_cache:
        manifest.record_override("no_cache", True)
    if cfg.query.include_proprietary:
        manifest.record_override("include_proprietary", True)

    cache = QueryCache(
        cfg.cache.directory,
        enabled=cfg.cache.enabled and not no_cache,
        max_age_days=cfg.cache.max_age_days,
    )
    client = MastClient(cache, allow_proprietary=cfg.query.include_proprietary)

    try:
        spec = _spec_from_config(cfg)
    except ConfigError as exc:
        _err(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc

    exit_code = EXIT_OK
    try:
        result, record = client.fetch(spec)
        manifest.record_query(record)

        table_path = run_dir / "observations.ecsv"
        result.table.write(table_path, format="ascii.ecsv", overwrite=True)
        manifest.record_output(
            table_path,
            kind="observation_table",
            n_rows=len(result.table),
            served_from=result.served_from,
        )
        manifest.finish("completed")
        _status(f"{len(result.table)} observation(s) [{result.served_from}] -> {table_path}")

    except EmptyResultError as exc:
        manifest.record_query(_failed_record(spec, "live", str(exc), n_results=0))
        manifest.quality.add(
            "empty_result",
            _severity_unreliable(),
            "Archive query matched no products; nothing downstream can run.",
            spec=spec.describe(),
        )
        manifest.finish("empty", error=str(exc))
        _err(str(exc))
        exit_code = EXIT_EMPTY

    except ProprietaryDataError as exc:
        manifest.record_query(_failed_record(spec, "live", str(exc)))
        manifest.finish("proprietary", error=str(exc))
        _err(str(exc))
        exit_code = EXIT_PROPRIETARY

    except ArchiveError as exc:
        manifest.record_query(_failed_record(spec, "live", str(exc)))
        manifest.finish("failed", error=str(exc))
        _err(
            f"{exc}\n\nThe archive could not be reached or refused the query. Nothing was "
            f"written beyond this manifest."
        )
        exit_code = EXIT_ARCHIVE

    manifest_path = manifest.write(run_dir / "manifest.json")
    typer.echo(f"manifest: {manifest_path}", err=True)

    assessment = manifest.reproducibility_assessment()
    if not assessment["fully_reproducible"]:
        typer.secho("reproducibility caveats:", fg=typer.colors.YELLOW, err=True)
        for reason in assessment["caveats"]:
            typer.secho(f"  - {reason}", fg=typer.colors.YELLOW, err=True)

    if exit_code == EXIT_OK and not manifest.quality.is_reliable:
        typer.secho(f"quality: {manifest.quality.summary_line()}", fg=typer.colors.YELLOW, err=True)
        exit_code = EXIT_UNRELIABLE

    raise typer.Exit(exit_code)


def _severity_unreliable() -> Any:
    from astrolab.core.quality import Severity

    return Severity.UNRELIABLE


def _failed_record(
    spec: QuerySpec, served_from: str, error: str, n_results: int | None = None
) -> Any:
    from datetime import UTC, datetime

    from astrolab.core.provenance import QueryRecord

    return QueryRecord(
        archive=spec.archive,
        spec=spec.to_dict(),
        spec_hash=spec.content_hash(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        served_from=served_from,
        n_results=n_results,
        error=error,
    )


@app.command("run")
def run_pipeline(
    config: Annotated[Path, typer.Argument(help="Path to a run config YAML.")],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Override run.output_dir from the config."),
    ] = None,
    no_fit: Annotated[
        bool, typer.Option("--no-fit", help="Skip the fitting stage (much faster).")
    ] = False,
) -> None:
    """Run the transit pipeline: ingest, detrend, search, vet, fit, report.

    Writes a self-contained HTML report and a provenance manifest. Exits 5 if the results
    carry an UNRELIABLE quality flag, so a caller cannot mistake a flagged result for a clean
    one just because the process exited without an exception.
    """
    from astrolab.pipelines.report import write_report
    from astrolab.pipelines.transit import run_transit_pipeline

    cfg = _load_config(config)
    if cfg.transit is None or cfg.source is None:
        _err(
            f"{config}: `astrolab run` needs both 'source' and 'transit' sections. "
            f"This config only defines an archive query -- use `astrolab query` for that."
        )
        raise typer.Exit(EXIT_USAGE)

    if no_fit:
        cfg = cfg.model_copy(
            update={
                "transit": cfg.transit.model_copy(
                    update={"fit": cfg.transit.fit.model_copy(update={"enabled": False})}
                )
            }
        )

    out_root = Path(output_dir) if output_dir else cfg.run.output_dir
    run_dir = out_root / cfg.run.name

    try:
        result = run_transit_pipeline(cfg, output_dir=out_root)
    except Exception as exc:
        _err(f"pipeline failed: {exc}")
        raise typer.Exit(EXIT_ARCHIVE) from exc

    report_path = write_report(result, run_dir / "report.html")
    result.manifest.record_output(report_path, kind="report")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(result.summary(), indent=2, default=str))
    result.manifest.record_output(summary_path, kind="summary")

    manifest_path = result.manifest.write(run_dir / "manifest.json")

    _status(f"{len(result.candidates)} candidate(s) found")
    for i, vetting in enumerate(result.vetting, start=1):
        typer.echo(
            f"  {i}. P = {vetting.candidate.period.value:.5f} d  "
            f"depth = {vetting.candidate.depth_ppm:.0f} ppm  -> {vetting.disposition}"
        )
    for fit in result.fits.values():
        minus, plus = fit.period.uncertainty_asymmetric
        typer.echo(
            f"     fitted P = {fit.period.value.value:.6f} "
            f"+{plus.value:.6f}/-{minus.value:.6f} d, "
            f"Rp/R* = {fit.radius_ratio.value.value:.5f}, "
            f"ln BF = {fit.log_bayes_factor:.1f}"
        )
    typer.echo(f"report:   {report_path}", err=True)
    typer.echo(f"manifest: {manifest_path}", err=True)

    if result.quality.flags:
        typer.secho(f"quality: {result.quality.summary_line()}", fg=typer.colors.YELLOW, err=True)

    raise typer.Exit(EXIT_OK if result.quality.is_reliable else EXIT_UNRELIABLE)


@app.command()
def replay(
    manifest_path: Annotated[Path, typer.Argument(help="Path to a manifest.json.")],
    check_only: Annotated[
        bool,
        typer.Option(
            "--check-only",
            help="Rebuild and verify the query spec without contacting the archive.",
        ),
    ] = True,
) -> None:
    """Reproduce a recorded run's query from its manifest.

    This is the executable form of the reproducibility claim. It rebuilds each recorded query
    spec, verifies that the rebuilt spec hashes to the value stored at run time, and -- unless
    ``--check-only`` -- re-executes it and compares the result hash.

    A hash mismatch means the manifest does not describe a query this code can reconstruct,
    which is a reproducibility bug regardless of how good the result looked.
    """
    if not manifest_path.is_file():
        _err(f"manifest not found: {manifest_path}")
        raise typer.Exit(EXIT_USAGE)

    data = RunManifest.load(manifest_path)
    typer.echo(f"run:        {data['run_name']}  ({data['run_id'][:12]})")
    typer.echo(f"status:     {data['status']}")
    typer.echo(f"started:    {data['started_at']}")
    git = data.get("git", {})
    typer.echo(f"git:        {git.get('sha', 'unavailable')} dirty={git.get('dirty')}")
    typer.echo(f"config:     {data['config_hash'][:16]}")

    queries = data.get("queries", [])
    if not queries:
        _err("manifest records no queries; there is nothing to replay")
        raise typer.Exit(EXIT_USAGE)

    failures = 0
    for i, q in enumerate(queries, start=1):
        recorded_hash = q["spec_hash"]
        try:
            spec = QuerySpec.from_dict(q["spec"])
        except ValueError as exc:
            _err(f"query {i}: cannot rebuild spec: {exc}")
            failures += 1
            continue
        rebuilt_hash = spec.content_hash()
        if rebuilt_hash == recorded_hash:
            _ok(f"query {i}: spec reconstructed and hash matches ({rebuilt_hash[:12]})")
        else:
            _err(
                f"query {i}: hash mismatch -- recorded {recorded_hash[:12]}, "
                f"rebuilt {rebuilt_hash[:12]}. The manifest does not reproduce this query."
            )
            failures += 1
            continue

        typer.echo(f"            {spec.describe()}")
        typer.echo(f"            originally served from: {q['served_from']}")

    if failures:
        _err(f"{failures} of {len(queries)} queries failed to reproduce")
        raise typer.Exit(EXIT_ARCHIVE)

    if check_only:
        _ok(f"all {len(queries)} query spec(s) reproduce exactly from the manifest")
    raise typer.Exit(EXIT_OK)


@cache_app.command("stats")
def cache_stats(
    directory: Annotated[Path | None, typer.Option("--directory", "-d", help="Cache root.")] = None,
) -> None:
    """Summarise cache contents."""
    root = directory or (Path.home() / ".astrolab" / "cache")
    cache = QueryCache(root)
    stats = cache.stats()
    typer.echo(json.dumps(stats, indent=2))


@cache_app.command("list")
def cache_list(
    directory: Annotated[Path | None, typer.Option("--directory", "-d", help="Cache root.")] = None,
    archive: Annotated[str | None, typer.Option("--archive", help="Filter by archive.")] = None,
) -> None:
    """List cached queries with their original fetch times."""
    root = directory or (Path.home() / ".astrolab" / "cache")
    entries = QueryCache(root).iter_entries(archive)
    if not entries:
        typer.echo("cache is empty")
        raise typer.Exit(EXIT_OK)
    for e in entries:
        spec = e.get("spec", {})
        origin = f"{spec.get('archive', '?')}.{spec.get('operation', '?')}"
        typer.echo(
            f"{e.get('spec_hash', '?')[:12]}  {origin}"
            f"  rows={e.get('n_rows', '?')}  fetched={e.get('fetched_at', '?')}"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
