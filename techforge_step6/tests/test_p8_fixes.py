"""Regression tests for the P8 fixes that the P7 adversarial tests do not cover directly."""
import pandas as pd
import pytest

from scripts.evaluate_labels import check_complete
from techforge import lake
from techforge.step05_clean_events import enforce_key
from techforge.step07_cooccurrence import run_model_version


def test_model_bound_to_run(mini_run):
    """Issue 08: a run uses the model version of its own step 7, not another run's latest."""
    lake.set_root(mini_run["lake"])
    v = run_model_version(mini_run["run_id"])
    assert v is not None and mini_run["run_id"] in v
    assert run_model_version("no-such-run") is None


def test_label_sheet_must_be_complete():
    """Issue 06: a missing or duplicated label is an error, not silently dropped."""
    manifest = pd.DataFrame({"label_id": ["L001", "L002"]})
    check_complete(pd.DataFrame({"label_id": ["L001", "L002"]}), manifest, "ok")
    with pytest.raises(ValueError):
        check_complete(pd.DataFrame({"label_id": ["L001"]}), manifest, "missing")
    with pytest.raises(ValueError):
        check_complete(pd.DataFrame({"label_id": ["L001", "L001", "L002"]}), manifest, "duplicated")


def test_conflicting_sales_rows_are_quarantined():
    """Issue 07: two rows with the same key but different values are not both kept."""
    df = pd.DataFrame({"part_id": ["a", "a", "a"], "warehouse_id": ["KYIV"] * 3,
                       "date": ["2026-01-01", "2026-01-01", "2026-01-02"], "units_sold": [3, 5, 4]})
    kept, conflicts = enforce_key(df, ["part_id", "warehouse_id", "date"])
    assert len(kept) == 1 and len(conflicts) == 2


def test_every_injected_build_accounted_for(mini_run):
    """Issue 05: the detection table counts all injected builds."""
    lake.set_root(mini_run["lake"])
    det = lake.read_table("outputs", "eval_injected_detection", mini_run["run_id"])
    truth = lake.read_table("cleaned", "build_truth", mini_run["run_id"])
    n = det.loc[det["group"] == "injected incompatible", "builds"].sum()
    assert n == int(truth["injected"].astype(bool).sum())
