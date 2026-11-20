"""Shared fixtures. `mini_run` runs the whole pipeline once on small SYNTHETIC data in a temp lake."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from techforge import lake  # noqa: E402


@pytest.fixture(scope="session")
def mini_run(tmp_path_factory):
    import run_pipeline
    from techforge.step02_generate import small_settings

    lake_dir = tmp_path_factory.mktemp("lake")
    lake.set_root(lake_dir)
    run_pipeline.run("mini", list(range(1, 12)), settings=small_settings(), verbose=False)
    yield {"run_id": "mini", "lake": lake_dir}


def part(pid: str, category: str, **attrs):
    """Small helper to build a dim_part-like dict for unit tests."""
    base = {"part_id": pid, "category": category, "name": pid, "price": 100.0, "price_band": "mid",
            "socket": None, "memory_type": None, "form_factor": None, "accepted_form_factors": None,
            "tdp_w": None, "board_power_w": None, "wattage_w": None, "case_type": None}
    base.update(attrs)
    return base
