"""Step 5 - Validate, deduplicate and sort events, stock and sales (Table 12 [R]).

Malformed or duplicate events go to the quarantine area and are counted (Section 4.6 [R]).
Done when: rejected events are counted (cleaned/event_validation_log).
True demand is copied to its own table and is used only for evaluation [R].
"""
from __future__ import annotations

import json

import pandas as pd

from techforge import lake

REQUIRED = ("event_id", "session_id", "build_id", "event_ts", "ingest_ts", "action", "category",
            "part_id", "warehouse_id")
ACTIONS = {"add"}


FACT_KEYS = {   # P8 (P7 issue 07): declared key of each cleaned fact table, enforced below
    "sales_daily": ["part_id", "warehouse_id", "date"],
    "inventory_daily": ["part_id", "warehouse_id", "date"],
    "supplier_deliveries": ["part_id", "warehouse_id", "order_date"],
    "fa_sku": ["part_id"],
}


def enforce_key(df: pd.DataFrame, key: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact duplicates; rows that share the key but differ in values are conflicts and are
    quarantined (none of the conflicting rows is kept)."""
    df = df.drop_duplicates()
    dup = df.duplicated(key, keep=False)
    return df[~dup].reset_index(drop=True), df[dup].reset_index(drop=True)


def validate_events(lines: list[tuple[str, str]], known_parts: set[str],
                    part_category: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid events, quarantined records with reason). Input: (file name, line) pairs."""
    good, bad, seen = [], [], set()
    for fname, line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad.append({"file": fname, "event_id": None, "reason": "not_valid_json", "record": line[:200]})
            continue
        if not isinstance(rec, dict):   # P8 fix (P7 issue 07): e.g. a JSON array
            bad.append({"file": fname, "event_id": None, "reason": "not_an_object", "record": line[:200]})
            continue
        missing = [k for k in REQUIRED if rec.get(k) in (None, "")]
        reason = None
        if missing:
            reason = "missing_field:" + ",".join(missing)
        else:
            try:
                rec["event_ts"] = pd.Timestamp(rec["event_ts"])
                rec["ingest_ts"] = pd.Timestamp(rec["ingest_ts"])
                if pd.isna(rec["event_ts"]) or pd.isna(rec["ingest_ts"]):   # P8: NaT is not a time
                    reason = "bad_timestamp"
            except (ValueError, TypeError):
                reason = "bad_timestamp"
        if reason is None and rec["action"] not in ACTIONS:
            reason = "unknown_action"
        if reason is None and rec["part_id"] not in known_parts:
            reason = "unknown_part_id"
        if reason is None and part_category and part_category.get(rec["part_id"]) != rec["category"]:
            reason = "category_mismatch"   # P8: the event's category must match the part
        if reason is None and rec["event_id"] in seen:
            reason = "duplicate_event_id"
        if reason:
            bad.append({"file": fname, "event_id": rec.get("event_id"), "reason": reason, "record": line[:200]})
            continue
        seen.add(rec["event_id"])
        good.append(rec)
    ev = pd.DataFrame(good)
    if len(ev):
        ev = ev.sort_values(["event_ts", "event_id"]).reset_index(drop=True)
    q = pd.DataFrame(bad, columns=["file", "event_id", "reason", "record"])
    return ev, q


def run(run_id: str) -> pd.DataFrame:
    dim = lake.read_table("curated", "dim_part", run_id)
    known = set(dim["part_id"])
    folder = lake.table_dir("raw", "builder_events", run_id)
    lines = []
    for p in sorted(folder.glob("builder_events_*.jsonl")):
        lines += [(p.name, ln) for ln in p.read_text(encoding="utf-8").splitlines()]
    ev, q = validate_events(lines, known, dict(zip(dim["part_id"], dim["category"])))
    ev["late"] = (ev["ingest_ts"] - ev["event_ts"]) > pd.Timedelta(hours=1)
    lake.write_table(ev, "cleaned", "builder_event", run_id, source="raw/builder_events/*.jsonl (SYNTHETIC)")
    lake.write_table(q, "quarantine", "builder_event", run_id, source="step05 rejected events")
    log = q.assign(reason=q["reason"].str.split(":").str[0]).groupby("reason").size().rename("count").reset_index()
    log = pd.concat([log, pd.DataFrame([{"reason": "accepted", "count": len(ev)},
                                        {"reason": "late_accepted (processed in next batch)", "count": int(ev["late"].sum())}])],
                    ignore_index=True)
    lake.write_table(log, "cleaned", "event_validation_log", run_id, source="step05")

    # stock snapshots
    st = pd.read_csv(lake.table_dir("raw", "stock_snapshots", run_id) / "stock_snapshots.csv",
                     parse_dates=["snapshot_ts"])
    bad_stock = st["on_hand"].isna() | (st["on_hand"] < 0) | ~st["part_id"].isin(known)
    st = st[~bad_stock].drop_duplicates(["part_id", "warehouse_id", "snapshot_ts"]).sort_values(
        ["warehouse_id", "snapshot_ts", "part_id"])
    lake.write_table(st, "cleaned", "stock_snapshot", run_id, source="raw/stock_snapshots (SYNTHETIC)",
                     extra={"rejected_rows": int(bad_stock.sum())})

    # Focus A tables
    for name, dates in (("sales_daily", ["date"]), ("inventory_daily", ["date"]),
                        ("supplier_deliveries", ["order_date"]), ("fa_sku", [])):
        df = pd.read_csv(lake.table_dir("raw", name, run_id) / f"{name}.csv", parse_dates=dates or None)
        df, conflicts = enforce_key(df, FACT_KEYS[name])
        lake.write_table(df, "cleaned", name, run_id, source=f"raw/{name} (SYNTHETIC)",
                         extra={"key": FACT_KEYS[name], "key_conflict_rows": len(conflicts)})
        if len(conflicts):
            lake.write_table(conflicts, "quarantine", name, run_id, source="step05 key conflicts")
    td = pd.read_csv(lake.table_dir("raw", "true_demand", run_id) / "true_demand.csv", parse_dates=["date"])
    lake.write_table(td, "cleaned", "sales_true_demand", run_id,
                     source="raw/true_demand (SYNTHETIC) - stored separately, EVALUATION ONLY")
    asm = pd.read_csv(lake.table_dir("raw", "assortment", run_id) / "assortment.csv")
    lake.write_table(asm, "cleaned", "assortment", run_id, source="raw/assortment (SYNTHETIC)")
    truth = pd.read_csv(lake.table_dir("raw", "build_truth", run_id) / "build_truth.csv")
    lake.write_table(truth, "cleaned", "build_truth", run_id, source="raw/build_truth (SYNTHETIC, evaluation only)")
    return log
