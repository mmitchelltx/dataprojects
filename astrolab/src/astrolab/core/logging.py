"""Structured logging.

Every archive query, reference file, and seed is logged as a structured event rather than a
sentence. The reason is practical: when a result looks wrong six months later, the question is
"which CRDS context did that run use" or "was that query served from cache", and answering it
by grepping prose is miserable. Structured events are queryable.

The manifest (:mod:`astrolab.core.provenance`) is the durable record; logs are the live view of
the same facts. Where they overlap that is deliberate -- logs survive a crash that happens
before the manifest is written.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = False,
    force: bool = False,
) -> None:
    """Set up structlog once per process.

    Parameters
    ----------
    level
        Standard logging level name.
    json_output
        Emit JSON lines instead of the human-readable console renderer. Use this when logs are
        being collected rather than read directly.
    force
        Reconfigure even if already configured. Mainly for tests.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        # Route through stdlib logging rather than writing to a captured stream object.
        # A factory holding a direct reference to sys.stderr binds whatever stream existed at
        # configuration time; anything that later replaces the stream -- a test harness, a
        # subprocess wrapper -- leaves the logger writing to a closed file. Going through
        # logging resolves the handler's stream per write instead.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Reconfiguration must take effect. Caching bound loggers keeps stale configuration
        # alive across a configure(force=True), which is exactly what tests do.
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True


def get_logger(name: str) -> Any:
    """Return a bound structlog logger, configuring logging on first use."""
    configure_logging()
    return structlog.get_logger(name)
