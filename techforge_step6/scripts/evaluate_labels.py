"""Evaluate the team's labels: Cohen's kappa and the Table 11 rule-accuracy measures [R].

Inputs
  --sheet-a / --sheet-b : the two labellers' sheets (compatible = yes / no / unsure;
                          R4 rows: gpu_maker_recommended_psu_w)
  --resolved            : third member's resolved file (final_compatible, final_gpu_maker_recommended_psu_w)
  --manifest            : sample_manifest.csv (subset, rule, part ids)
  --run-id              : run whose curated dim_part is used to recompute the engine verdicts now
Outputs (to --out): label_kappa.csv, label_disagreements.csv, label_rule_accuracy.csv,
                    label_evaluation.json; and the curated `labels` table of the run.

R1-R3 (test subset):   wrongly passed = engine PASS on a pair labelled incompatible (target 0)
                       wrongly failed = engine FAIL on a pair labelled compatible (target <= 5% of compatible)
                       UNKNOWN share  (target <= 15%)
R4 (30 triples):       reference = PSU wattage >= GPU maker's recommended PSU wattage
                       error = engine PASS although below the recommendation (target 0)
                       stricter = engine FAIL although meeting it (reported separately; R4 is deliberately stricter)
With zero errors in n cases the 95% upper bound of the true error rate is about 3/n (rule of three).

If any labeller name contains "DUMMY", every output is marked DUMMY - NOT EVIDENCE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from techforge import lake, rules  # noqa: E402

CHECK = {"R1": rules.check_r1, "R2": rules.check_r2, "R3": rules.check_r3}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement beyond chance for two raters over the same items."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    # P8 fix (P7 issue 06): if both raters use one single class, kappa is undefined (0/0)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def check_complete(sheet: pd.DataFrame, manifest: pd.DataFrame, name: str) -> None:
    """P8 fix (P7 issue 06): every sampled item must be labelled exactly once."""
    ids, expected = sheet["label_id"], set(manifest["label_id"])
    dup = ids[ids.duplicated()].tolist()
    missing = sorted(expected - set(ids))
    extra = sorted(set(ids) - expected)
    if dup or missing or extra:
        raise ValueError(f"{name}: duplicated {dup[:5]}, missing {missing[:5]} ({len(missing)}), "
                         f"unknown {extra[:5]} label ids")


def _r4_label(watts_rec, psu_w) -> str | None:
    try:
        return "yes" if float(psu_w) >= float(watts_rec) else "no"
    except (TypeError, ValueError):
        return None


def evaluate(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame, resolved: pd.DataFrame, manifest: pd.DataFrame,
             dim: pd.DataFrame) -> dict:
    dummy = any("DUMMY" in str(x).upper() for x in pd.concat([sheet_a["labeller"], sheet_b["labeller"]]))
    check_complete(sheet_a, manifest, "sheet A")
    check_complete(sheet_b, manifest, "sheet B")
    check_complete(resolved, manifest, "resolved labels")
    parts = {r["part_id"]: r for r in dim.to_dict("records")}
    # --- kappa on R1-R3 'compatible' and on R4 derived labels
    ab = sheet_a.merge(sheet_b, on="label_id", suffixes=("_a", "_b")).merge(manifest[["label_id", "part_id_3"]], on="label_id")
    def lab(row, who):
        if row["rule_id_a"] == "R4":
            psu = parts.get(row["part_id_3"], {}).get("wattage_w")
            return _r4_label(row[f"gpu_maker_recommended_psu_w_{who}"], psu) or "unsure"
        return str(row[f"compatible_{who}"]).strip().lower() or "unsure"
    ab["label_a"] = ab.apply(lambda r: lab(r, "a"), axis=1)
    ab["label_b"] = ab.apply(lambda r: lab(r, "b"), axis=1)
    kappa_rows = [{"scope": "all", "items": len(ab), "kappa": round(cohen_kappa(list(ab["label_a"]), list(ab["label_b"])), 4)}]
    for rid, g in ab.groupby("rule_id_a"):
        kappa_rows.append({"scope": rid, "items": len(g), "kappa": round(cohen_kappa(list(g["label_a"]), list(g["label_b"])), 4)})
    disagree = ab[ab["label_a"] != ab["label_b"]][["label_id", "rule_id_a", "label_a", "label_b", "reason_a", "reason_b"]]

    # --- accuracy on the test subset with resolved labels and current engine verdicts
    m = manifest.merge(resolved[["label_id", "final_compatible", "final_gpu_maker_recommended_psu_w"]], on="label_id")
    acc_rows = []
    for rid, g in m[m["subset"] == "test"].groupby("rule_id"):
        verdicts, truth = [], []
        for r in g.itertuples(index=False):
            if rid == "R4":
                c, gp, p = parts[r.part_id_1], parts[r.part_id_2], parts[r.part_id_3]
                verdicts.append(rules.check_r4(c, gp, p).verdict)
                truth.append(_r4_label(r.final_gpu_maker_recommended_psu_w, p.get("wattage_w")))
            else:
                verdicts.append(CHECK[rid](parts[r.part_id_1], parts[r.part_id_2]).verdict)
                truth.append(str(r.final_compatible).strip().lower())
        d = pd.DataFrame({"verdict": verdicts, "truth": truth})
        d = d[d["truth"].isin(["yes", "no"])]
        inc, comp = d[d["truth"] == "no"], d[d["truth"] == "yes"]
        wp = int((inc["verdict"] == rules.PASS).sum())
        wf = int((comp["verdict"] == rules.FAIL).sum())
        acc_rows.append({
            "rule_id": rid, "test_cases_labelled": len(d), "incompatible": len(inc), "compatible": len(comp),
            ("passed_below_recommendation" if rid == "R4" else "wrongly_passed"): wp,
            "wrongly_passed_share_of_incompatible": round(wp / len(inc), 4) if len(inc) else None,
            ("failed_although_meeting_it" if rid == "R4" else "wrongly_failed"): wf,
            "wrongly_failed_share_of_compatible": round(wf / len(comp), 4) if len(comp) else None,
            "unknown_share": round((d["verdict"] == rules.UNKNOWN).mean(), 4) if len(d) else None,
            "upper_bound_95_if_zero_errors": round(3 / len(inc), 3) if len(inc) and wp == 0 else None,
            "targets": "0; <= 5%; <= 15%",
            # P8 (P7 issue 06): the sample is stratified by engine verdict, so these are rates in a
            # challenge sample, not estimates for the whole catalogue
            "sample_note": "challenge sample stratified by engine verdict - not a representative estimate",
        })
    k = kappa_rows[0]["kappa"]
    return {"dummy": dummy, "kappa": pd.DataFrame(kappa_rows), "disagreements": disagree,
            "accuracy": pd.DataFrame(acc_rows), "kappa_overall": k,
            "kappa_status": ("DUMMY - NOT EVIDENCE" if dummy else ("met" if k >= 0.8 else "not met"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet-a", required=True)
    ap.add_argument("--sheet-b", required=True)
    ap.add_argument("--resolved", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lake", default=None)
    a = ap.parse_args(argv)
    if a.lake:
        lake.set_root(a.lake)
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    res = evaluate(read(a.sheet_a), read(a.sheet_b), read(a.resolved), read(a.manifest),
                   lake.read_table("curated", "dim_part", a.run_id))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "DUMMY_" if res["dummy"] else ""
    res["kappa"].assign(dummy=res["dummy"]).to_csv(out / f"{tag}label_kappa.csv", index=False)
    res["disagreements"].to_csv(out / f"{tag}label_disagreements.csv", index=False)
    res["accuracy"].assign(dummy=res["dummy"]).to_csv(out / f"{tag}label_rule_accuracy.csv", index=False)
    (out / "label_evaluation.json").write_text(json.dumps(
        {"dummy": res["dummy"], "kappa_overall": res["kappa_overall"], "kappa_status": res["kappa_status"]}, indent=2),
        encoding="utf-8")
    if not res["dummy"]:
        labels = read(a.resolved)
        lake.write_table(labels, "curated", "labels", a.run_id, source="team labels (resolved)")
    print(res["kappa"].to_string(index=False))
    print(res["accuracy"].to_string(index=False))
    print("DUMMY LABELS - NOT EVIDENCE" if res["dummy"] else "team labels evaluated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
