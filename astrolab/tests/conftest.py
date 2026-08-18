"""Shared test fixtures and the network-test gate.

Tests marked ``network`` contact live archives and are skipped unless ``--run-network`` is
passed. They are not optional extras: they are the only tests that can confirm the archive
query vocabulary is right, and they must be run on an unrestricted network before any claim
that retrieval works. See ``docs/design.md`` on the archive-access limitation.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that contact live archives.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network (contacts a live archive)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def observation_table() -> Table:
    """A small table shaped like a MAST observations result.

    Column names and the ``dataRights`` vocabulary follow MAST's schema. This is a fixture for
    testing *our* cache and filtering logic, not a stand-in for archive data: no test uses it
    to make a scientific claim, and the retrieval path it would mask is covered only by the
    network-marked tests.
    """
    return Table(
        {
            "obsid": ["17000000001", "17000000002", "17000000003"],
            "obs_collection": ["TESS"] * 3,
            "target_name": ["WASP-18"] * 3,
            "sequence_number": [2, 3, 29],
            "t_exptime": [120.0, 120.0, 120.0] * u.s,
            "dataRights": ["PUBLIC", "PUBLIC", "EXCLUSIVE_ACCESS"],
        }
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260818)
