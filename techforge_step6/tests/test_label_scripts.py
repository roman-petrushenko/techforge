"""Label scripts (team decision 8). Tested ONLY with clearly marked DUMMY labels."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from techforge import lake

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


draw_mod = _load("draw_label_sample")
eval_mod = _load("evaluate_labels")


def test_kappa_hand_example():
    # po = 0.75, pe = 0.5*0.25 + 0.5*0.75 = 0.5 -> kappa = 0.5
    assert math.isclose(eval_mod.cohen_kappa(["y", "y", "n", "n"], ["y", "n", "n", "n"]), 0.5)
    assert eval_mod.cohen_kappa(["y", "n"], ["y", "n"]) == 1.0


def test_sample_sizes_split_seed_and_hidden_verdict(mini_run, tmp_path):
    lake.set_root(mini_run["lake"])
    dim = lake.read_table("curated", "dim_part", mini_run["run_id"])
    m1 = draw_mod.draw(dim, seed=11)
    m2 = draw_mod.draw(dim, seed=11)
    pd.testing.assert_frame_equal(m1, m2)                                    # reproducible
    pairs = m1[m1["rule_id"] != "R4"]
    assert len(pairs) == 150 and (pairs["subset"] == "test").sum() == 100 and (pairs["subset"] == "tuning").sum() == 50
    assert pairs[pairs["subset"] == "test"].groupby("rule_id").size().min() >= 25
    assert (m1["rule_id"] == "R4").sum() == 30
    draw_mod.write_sheets(m1, tmp_path)
    sheet = pd.read_csv(tmp_path / "labelling_sheet_labeller_A.csv")
    assert "stratum" not in sheet.columns and sheet["compatible"].isna().all()


def test_evaluation_runs_on_dummy_labels_and_marks_them(mini_run, tmp_path):
    lake.set_root(mini_run["lake"])
    dim = lake.read_table("curated", "dim_part", mini_run["run_id"])
    m = draw_mod.draw(dim, seed=5)
    rng = np.random.default_rng(0)
    def dummy_sheet(name):
        s = m.reindex(columns=draw_mod.SHEET_COLUMNS).copy()
        s["compatible"] = np.where(m["rule_id"] == "R4", "", rng.choice(["yes", "no"], len(m)))
        s["gpu_maker_recommended_psu_w"] = np.where(m["rule_id"] == "R4", rng.choice(["550", "750", "850"], len(m)), "")
        s["reason"] = "DUMMY"
        s["labeller"] = f"DUMMY_{name} - NOT A REAL LABEL"
        return s.astype(str)
    a, b = dummy_sheet("A"), dummy_sheet("B")
    resolved = m[["label_id"]].copy()
    resolved["final_compatible"] = a["compatible"].values
    resolved["final_gpu_maker_recommended_psu_w"] = a["gpu_maker_recommended_psu_w"].values
    res = eval_mod.evaluate(a, b, resolved.astype(str), m.astype(str), dim)
    assert res["dummy"] is True and res["kappa_status"] == "DUMMY - NOT EVIDENCE"
    assert set(res["accuracy"]["rule_id"]) == {"R1", "R2", "R3", "R4"}
    assert -1.0 <= res["kappa_overall"] <= 1.0
