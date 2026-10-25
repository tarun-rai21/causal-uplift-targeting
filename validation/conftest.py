"""Session-scoped harness runs shared by the slow validation tests: each
(scenario, seed) is simulated, split and fitted ONCE, instead of once per
test file that needs it. Seed 42 is the default config's own seed."""

from __future__ import annotations

import pytest

from validation.harness import SeedRun, run_seed


@pytest.fixture(scope="session")
def balanced_run() -> SeedRun:
    return run_seed("balanced", 42)


@pytest.fixture(scope="session")
def imbalanced_run() -> SeedRun:
    return run_seed("imbalanced", 42)
