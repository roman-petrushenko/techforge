"""Draw the label sample of Table 11 [R] and write empty labelling sheets.

  150 part pairs for R1-R3 (50 tuning, 100 test; at least 25 test pairs per rule -> 34/33/33)
  30 CPU-GPU-PSU triples for R4 (all held-out test)
Stratified per rule by the CURRENT engine verdict (PASS / FAIL / UNKNOWN) so that both
compatible and incompatible cases appear [A]; the verdict is kept in the manifest only and is
NOT shown on the labelling sheets. Fixed seed -> reproducible.

Usage:  python scripts/draw_label_sample.py --run-id demo [--seed 2026] [--out labels]
Writes: sample_manifest.csv, labelling_sheet_labeller_A.csv, labelling_sheet_labeller_B.csv,
        labels_resolved_TEMPLATE.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from techforge import lake, rules  # noqa: E402

PAIR_RULES = {"R1": ("cpu", "motherboard"), "R2": ("memory", "motherboard"), "R3": ("motherboard", "case")}
TEST_PER_RULE = {"R1": 34, "R2": 33, "R3": 33}      # 100 test pairs, >= 25 per rule [R]
TUNING_PER_RULE = {"R1": 17, "R2": 17, "R3": 16}    # 50 tuning pairs [R]
N_TRIPLES = 30                                      # [R]
STRATA_SHARE = {"PASS": 0.4, "FAIL": 0.4, "UNKNOWN": 0.2}  # [A]
SHEET_COLUMNS = ["label_id", "subset", "rule_id", "category_1", "part_id_1", "name_1", "category_2", "part_id_2",
                 "name_2", "category_3", "part_id_3", "name_3", "compatible", "gpu_maker_recommended_psu_w",
                 "reason", "source_document", "labeller", "labelled_on"]
CHECK = {"R1": rules.check_r1, "R2": rules.check_r2, "R3": rules.check_r3}


def _quota(n: int) -> dict[str, int]:
    q = {k: int(round(n * v)) for k, v in STRATA_SHARE.items()}
    q["PASS"] += n - sum(q.values())
    return q


def draw(dim: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = {c: g.to_dict("records") for c, g in dim.groupby("category")}
    rows, used = [], set()
    for rule_id, (c1, c2) in PAIR_RULES.items():
        need = _quota(TEST_PER_RULE[rule_id] + TUNING_PER_RULE[rule_id])
        got: dict[str, list] = {k: [] for k in need}
        tries = 0
        while any(len(got[k]) < need[k] for k in need) and tries < 400_000:
            tries += 1
            a = parts[c1][int(rng.integers(len(parts[c1])))]
            b = parts[c2][int(rng.integers(len(parts[c2])))]
            key = (rule_id, a["part_id"], b["part_id"])
            if key in used:
                continue
            args = (a, b) if rule_id != "R3" else (a, b)
            v = CHECK[rule_id](*args).verdict
            if len(got[v]) < need[v]:
                got[v].append((a, b))
                used.add(key)
        pairs = [(v, a, b) for v, lst in got.items() for a, b in lst]
        order = rng.permutation(len(pairs))
        n_test = TEST_PER_RULE[rule_id]
        for i, idx in enumerate(order):
            v, a, b = pairs[idx]
            rows.append({"rule_id": rule_id, "subset": "test" if i < n_test else "tuning", "stratum": v,
                         "category_1": c1, "part_id_1": a["part_id"], "name_1": a["name"],
                         "category_2": c2, "part_id_2": b["part_id"], "name_2": b["name"]})
    # R4 triples: GPUs from the reference chipsets so a maker's recommendation can be found
    need = _quota(N_TRIPLES)
    got = {k: [] for k in need}
    tries = 0
    while any(len(got[k]) < need[k] for k in need) and tries < 400_000:
        tries += 1
        c = parts["cpu"][int(rng.integers(len(parts["cpu"])))]
        g = parts["video-card"][int(rng.integers(len(parts["video-card"])))]
        p = parts["power-supply"][int(rng.integers(len(parts["power-supply"])))]
        v = rules.check_r4(c, g, p).verdict
        if len(got[v]) < need[v]:
            got[v].append((c, g, p))
    triples = [(v, t) for v, lst in got.items() for t in lst]
    for i in rng.permutation(len(triples)):   # P8 (P7 issue 06): shuffled, not grouped by verdict
        v, (c, g, p) = triples[i]
        if True:
            rows.append({"rule_id": "R4", "subset": "test", "stratum": v,
                         "category_1": "cpu", "part_id_1": c["part_id"], "name_1": c["name"],
                         "category_2": "video-card", "part_id_2": g["part_id"], "name_2": g["name"],
                         "category_3": "power-supply", "part_id_3": p["part_id"], "name_3": p["name"]})
    m = pd.DataFrame(rows)
    m.insert(0, "label_id", [f"L{i:03d}" for i in range(1, len(m) + 1)])
    m["seed"] = seed
    return m


def write_sheets(manifest: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "sample_manifest.csv", index=False)
    sheet = manifest.reindex(columns=SHEET_COLUMNS)   # no stratum / engine verdict on the sheet
    for who in ("A", "B"):
        s = sheet.copy()
        s["labeller"] = who
        s.to_csv(out / f"labelling_sheet_labeller_{who}.csv", index=False)
    # P8 fix (P7 issue 04): the template carries the categories that injection needs
    tmpl = manifest.reindex(columns=["label_id", "subset", "rule_id", "category_1", "part_id_1",
                                     "category_2", "part_id_2", "category_3", "part_id_3"]).copy()
    for col in ("final_compatible", "final_gpu_maker_recommended_psu_w", "resolver", "resolution_note"):
        tmpl[col] = ""
    tmpl.to_csv(out / "labels_resolved_TEMPLATE.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "labels"))
    ap.add_argument("--lake", default=None)
    a = ap.parse_args(argv)
    if a.lake:
        lake.set_root(a.lake)
    dim = lake.read_table("curated", "dim_part", a.run_id)
    m = draw(dim, a.seed)
    write_sheets(m, Path(a.out))
    print(m.groupby(["rule_id", "subset", "stratum"]).size().to_string())
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
