"""Step 4 - Add socket and memory type from the reference tables; log data gaps (Table 12 [R]).

Outputs (curated zone): dim_part, ref_* copies, dq_issue, attribute_coverage.
Done when: coverage and UNKNOWN share per attribute are reported.
"""
from __future__ import annotations

import pandas as pd

from techforge import config, lake, reference
from techforge.parts import RULE_ATTRIBUTES
from techforge.step03_clean_parts import load_cleaned

DIM_COLUMNS = ["part_id", "category", "name", "price", "price_band", "socket", "memory_type",
               "form_factor", "accepted_form_factors", "tdp_w", "board_power_w", "wattage_w",
               "microarchitecture", "chipset", "case_type", "source_rows", "fill_note", "version"]


def _price_band(prices: pd.Series) -> pd.Series:
    """low / mid / high by price tertile within the category; missing price -> missing band."""
    known = prices.dropna()
    if known.empty:
        return pd.Series(None, index=prices.index, dtype="object")
    q1, q2 = known.quantile([1 / 3, 2 / 3])
    band = pd.Series("mid", index=prices.index, dtype="object")
    band[prices <= q1] = "low"
    band[prices > q2] = "high"
    band[prices.isna()] = None
    return band


def curate_category(df: pd.DataFrame, category: str) -> tuple[pd.DataFrame, list[dict]]:
    out = pd.DataFrame({"part_id": df["part_id"], "category": category, "name": df["name"],
                        "price": df["price"], "source_rows": df["source_rows"]})
    out["price_band"] = _price_band(df["price"])
    for col in DIM_COLUMNS:
        if col not in out.columns:
            out[col] = None
    issues: list[dict] = []
    notes: list[str] = []

    def issue(pid: str, attr: str, kind: str, detail: str) -> None:
        issues.append({"part_id": pid, "category": category, "attribute": attr,
                       "issue_type": kind, "detail": detail})

    if category == "cpu":
        out["microarchitecture"] = df["microarchitecture"]
        out["tdp_w"] = df["tdp_w"]
        sockets = []
        for pid, name, ua in zip(df["part_id"], df["name"], df["microarchitecture"]):
            s, how = reference.cpu_socket(name, ua)
            sockets.append(s)
            notes.append(f"socket: {how}")
            if s is None:
                issue(pid, "socket", "unresolved", how)
        out["socket"] = sockets
    elif category == "motherboard":
        out["socket"] = df["socket"]
        out["form_factor"] = df["form_factor"]
        mem = []
        for pid, name, sock in zip(df["part_id"], df["name"], df["socket"]):
            m, how = reference.board_memory_type(name, sock)
            mem.append(m)
            notes.append(f"memory_type: {how}")
            if m is None:
                issue(pid, "memory_type", "unresolved", how)
        out["memory_type"] = mem
        known = reference.known_form_factors()
        for pid, ff in zip(df["part_id"], df["form_factor"]):
            if pd.notna(ff) and ff not in known:
                issue(pid, "form_factor", "not_covered", f"form factor '{ff}' not in case reference table")
    elif category == "memory":
        out["memory_type"] = df["ddr_gen"].map(lambda g: f"DDR{int(g)}" if pd.notna(g) else None)
        notes = ["memory_type: speed field"] * len(df)
    elif category == "video-card":
        out["chipset"] = df["chipset"]
        powers = []
        for pid, chip in zip(df["part_id"], df["chipset"]):
            p, how = reference.gpu_board_power(chip)
            powers.append(p)
            notes.append(f"board_power_w: {how}")
            if p is None:
                issue(pid, "board_power_w", "unresolved", how)
        out["board_power_w"] = powers
    elif category == "case":
        out["case_type"] = df["case_type"]
        acc = []
        for pid, ct in zip(df["part_id"], df["case_type"]):
            a, how = reference.case_form_factors(ct)
            acc.append(a)
            notes.append(f"accepted_form_factors: {how}")
            if a is None:
                issue(pid, "accepted_form_factors", "unresolved", how)
        out["accepted_form_factors"] = acc
    elif category == "power-supply":
        out["wattage_w"] = df["wattage_w"]
    out["fill_note"] = notes if notes else None

    # Missing rule attributes (including values removed by merge conflicts in step 3)
    for attr in RULE_ATTRIBUTES[category]:
        missing = out[attr].isna()
        already = {(i["part_id"], i["attribute"]) for i in issues}
        for pid in out.loc[missing, "part_id"]:
            if (pid, attr) not in already:
                issue(pid, attr, "missing", "value missing or conflicting in source rows")
    return out, issues


def run(run_id: str) -> pd.DataFrame:
    reference.clear_cache()
    version = f"{run_id}|ref:{lake.reference_version()}|code:{lake.code_version()}"
    for name in reference.REF_FILES:
        t = reference.load(name)
        lake.write_table(t, "curated", name, run_id, source=f"reference_tables/{reference.REF_FILES[name]}",
                         extra={"rows_verified": int((t["verified"] == "yes").sum())})
    dims, issues = [], []
    for category in config.CATEGORY_FILES:
        d, iss = curate_category(load_cleaned(run_id, category), category)
        dims.append(d)
        issues += iss
    # P8: drop all-empty columns per category before concatenating (pandas FutureWarning)
    dims = [d.dropna(axis=1, how="all").reindex(columns=[c for c in DIM_COLUMNS if c in d.columns and d[c].notna().any()])
            for d in dims]
    dim = pd.concat(dims, ignore_index=True).reindex(columns=DIM_COLUMNS)
    dim["version"] = version
    lake.write_table(dim, "curated", "dim_part", run_id, source="cleaned/parts_* + curated/ref_*")
    dq = pd.DataFrame(issues, columns=["part_id", "category", "attribute", "issue_type", "detail"])
    lake.write_table(dq, "curated", "dq_issue", run_id, source="step04")
    cov = attribute_coverage(dim)
    lake.write_table(cov, "curated", "attribute_coverage", run_id, source="step04")
    return cov


def attribute_coverage(dim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, attrs in RULE_ATTRIBUTES.items():
        d = dim[dim["category"] == category]
        for a in attrs:
            known = int(d[a].notna().sum())
            rows.append({"category": category, "attribute": a, "parts": len(d), "known": known,
                         "unknown": len(d) - known,
                         "unknown_share": round(1 - known / len(d), 4) if len(d) else None})
    return pd.DataFrame(rows)
