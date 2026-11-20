"""Step 10 - Run the evaluation scripts (Table 12 [R]) and fill Table 11.

Accuracy evidence (rows 1-3) needs the team's labels; without them those rows are reported
as "pending labels" - this module never invents accuracy results.
Mechanics checks (rows 4-8) run on SYNTHETIC data and show only that the implementation
behaves as specified [R Section 4.5].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from techforge import forecasting as fc, lake, rules
from techforge.recommender import Recommender, StockView


def _row(test: str, measure: str, value, target: str, status: str, evidence: str, data: str, note: str = "") -> dict:
    return {"test": test, "measure": measure, "value": value, "target": target, "status": status,
            "evidence_type": evidence, "data": data, "note": note}


def injected_detection(run_id: str) -> pd.DataFrame:
    """P8 fix (P7 issue 05): every injected build is accounted for. Builds whose last replayed
    state does not have all six parts (or that were never replayed) are counted as such
    instead of being dropped by an inner join."""
    truth = lake.read_table("cleaned", "build_truth", run_id).set_index("build_id")
    bv = lake.read_table("outputs", "build_verdict", run_id).sort_values(["event_ts", "event_id"])
    last = bv.groupby("build_id").last()[["verdict", "rules_checked"]]
    t = truth.join(last, how="left")
    t["outcome"] = np.where(t["verdict"].isna(), "not replayed (training period only)",
                            np.where(t["rules_checked"] < 4, "incomplete in replay", t["verdict"]))
    t["group"] = np.where(t["injected"].astype(bool), "injected incompatible", "generator-compatible")
    out = t.groupby(["group", "outcome"]).size().rename("builds").reset_index()
    replayed = ~out["outcome"].str.startswith("not replayed")
    out["share_of_replayed"] = np.where(replayed, out["builds"] / out[replayed].groupby("group")["builds"]
                                        .transform("sum").reindex(out.index), np.nan)
    inj = t[t["injected"].astype(bool) & ~t["outcome"].str.startswith("not replayed")]
    by_v = inj.groupby(["violation", "outcome"]).size().rename("builds").reset_index()
    by_v["group"] = "injected: " + by_v["violation"]
    by_v["share_of_replayed"] = by_v["builds"] / by_v.groupby("violation")["builds"].transform("sum")
    out = pd.concat([out, by_v.drop(columns="violation")], ignore_index=True)
    return out.rename(columns={"outcome": "verdict"})


def _snapshot_independent(stock: pd.DataFrame):
    """Independent stock lookup for the audit (does not use StockView): latest snapshot at or
    before the event time per warehouse."""
    idx = {}
    for wh, g in stock.groupby("warehouse_id"):
        by_ts = {ts: dict(zip(h["part_id"], h["on_hand"])) for ts, h in g.groupby("snapshot_ts")}
        times = np.array(sorted(by_ts), dtype="datetime64[ns]")
        idx[wh] = (times, by_ts)

    def lookup(wh, at, pid):
        if wh not in idx:
            return None, None
        times, by_ts = idx[wh]
        i = int(np.searchsorted(times, np.datetime64(at, "ns"), side="right")) - 1
        if i < 0:
            return None, None
        ts = pd.Timestamp(times[i])
        return ts, int(by_ts[ts].get(pid, 0))
    return lookup


def suggestion_validity(run_id: str) -> dict:
    """Re-check every suggestion independently of the replay code path (P8: the stock snapshot
    and its age are now looked up independently as well)."""
    dim = lake.read_table("curated", "dim_part", run_id)
    reco = lake.read_table("outputs", "reco_next_part", run_id)
    log = lake.read_table("outputs", "replay_log", run_id).set_index("event_id")
    stock = lake.read_table("cleaned", "stock_snapshot", run_id)
    rec = Recommender(dim, [], None)
    lookup = _snapshot_independent(stock)
    fail = unknown = oos = stale = 0
    for r in reco.itertuples(index=False):
        build = json.loads(log.loc[r.event_id, "build_state"])
        if rec.is_unknown_part(r.part_id):
            unknown += 1
        trial = {c: rec.parts[p] for c, p in build.items() if c != r.target_category}
        trial[r.target_category] = rec.parts[r.part_id]
        verdicts = {x.verdict for x in rules.check_rules_involving(trial, r.target_category)}
        fail += rules.FAIL in verdicts
        unknown += rules.UNKNOWN in verdicts
        ts, on_hand = lookup(r.warehouse_id, pd.Timestamp(r.event_ts), r.part_id)
        stale += ts is None or (pd.Timestamp(r.event_ts) - ts) > pd.Timedelta(hours=1)
        oos += (on_hand is None) or on_hand <= 0
    return {"suggestions": len(reco), "fail": int(fail), "unknown": int(unknown),
            "out_of_stock": int(oos), "stale_stock": int(stale)}


def coverage_and_speed(run_id: str) -> dict:
    lg = lake.read_table("outputs", "replay_log", run_id)
    dim = lake.read_table("curated", "dim_part", run_id)
    rec = Recommender(dim, [], None)
    req = lg[lg["status"] != "build_complete"].copy()
    served = req[req["status"].isin(["ok", "ok_partial"])].copy()
    # builds whose parts so far all have their rule attributes (no UNKNOWN part blocks the candidates)
    served["unknown_part_in_build"] = served["build_state"].map(
        lambda s: any(rec.is_unknown_part(p) for p in json.loads(s).values()))
    no_unknown = served[~served["unknown_part_in_build"]]
    return {
        "requests": len(req),
        "share_5_all_requests": round((req["n_suggestions"] == 5).mean(), 4),
        "requests_fresh_stock": len(served),
        "share_5_fresh_stock": round((served["n_suggestions"] == 5).mean(), 4),
        "requests_fresh_no_unknown_build": len(no_unknown),
        "share_5_fresh_no_unknown_build": round((no_unknown["n_suggestions"] == 5).mean(), 4),
        "share_requests_with_5_eligible": round((served["eligible"] >= 5).mean(), 4),
        "share_5_where_5_eligible": round((served.loc[served["eligible"] >= 5, "n_suggestions"] == 5).mean(), 4),
        "stale_stock_requests": int((req["status"] == "stale_stock").sum()),
        "model_missing_requests": int((req["status"] == "model_missing").sum()),
        "events": len(lg), "late_events": int(lg["late"].sum()),
        "mean_ms": round(lg["elapsed_ms"].mean(), 3),
        "p95_ms": round(lg["elapsed_ms"].quantile(0.95), 3),
        "max_ms": round(lg["elapsed_ms"].max(), 3),
    }


def forecast_metrics(run_id: str) -> pd.DataFrame:
    f = lake.read_table("outputs", "forecast", run_id)
    td = lake.read_table("cleaned", "sales_true_demand", run_id)   # scoring only [R]
    skus = lake.read_table("cleaned", "fa_sku", run_id)
    # P8 fix (P7 issue 05): the method is chosen BEFORE scoring - the "selected policy" is
    # analogue_launch for a launch SKU in its first 4 weeks, otherwise ses_weekday.
    from techforge.step09_focus_a import ANALOGUE_WEEKS
    launch = skus.set_index("part_id")["launch_date"]
    f = f.copy()
    ld = pd.to_datetime(f["part_id"].map(launch))
    young = (f["role"] == "launch") & ((pd.to_datetime(f["target_date"]) - ld).dt.days < ANALOGUE_WEEKS * 7)
    pick = np.where(young, "analogue_launch", "ses_weekday")
    sel = f[f["method"] == pd.Series(pick, index=f.index)].assign(method="selected_policy")
    f = pd.concat([f, sel], ignore_index=True)
    m = f.merge(td[["part_id", "warehouse_id", "date", "true_demand"]],
                left_on=["part_id", "warehouse_id", "target_date"], right_on=["part_id", "warehouse_id", "date"])
    rows = []
    for (role, method), g in m.groupby(["role", "method"]):
        rows.append({"role": role, "method": method, "forecasts": len(g),
                     "wape": round(fc.wape(g["forecast"].to_numpy(), g["true_demand"].to_numpy()), 4),
                     "bias": round(fc.bias(g["forecast"].to_numpy(), g["true_demand"].to_numpy()), 4),
                     "data": "SYNTHETIC"})
    return pd.DataFrame(rows)


def sim_summary(run_id: str) -> pd.DataFrame:
    sim = lake.read_table("outputs", "fa_sim_daily", run_id)
    alerts = lake.read_table("outputs", "fa_alert", run_id)
    rows = []
    for keys, g in list(sim.groupby(["policy"])) + list(sim.groupby(["policy", "role"])):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({"policy": keys[0], "role": keys[1] if len(keys) > 1 else "all",
                     "periods": len(g), "periods_without_stockout": round(1 - g["stockout"].mean(), 4),
                     # P8 fix (P7 C3): fill rate is undefined when there was no demand
                     "fill_rate": round(g["sales"].sum() / g["demand"].sum(), 4) if g["demand"].sum() > 0 else np.nan,
                     "average_stock": round(g["on_hand_end"].mean(), 2)})
    out = pd.DataFrame(rows)
    out["alert_share_of_periods"] = None
    n_periods = len(sim[sim["policy"] == "order_up_to"])
    out.loc[(out["policy"] == "order_up_to") & (out["role"] == "all"), "alert_share_of_periods"] = round(
        len(alerts) / max(1, n_periods), 4)
    # D2 (P8): alerts for which no open order is due before the shelf runs empty need action
    out["alerts_needing_action_share"] = None
    if "open_order_due_in_time" in alerts and len(alerts):
        out.loc[(out["policy"] == "order_up_to") & (out["role"] == "all"), "alerts_needing_action_share"] = round(
            float((~alerts["open_order_due_in_time"].astype(bool)).sum()) / max(1, n_periods), 4)
    out["data"] = "SYNTHETIC"
    return out


def label_accuracy_rows(labels_dir: Path | None) -> list[dict]:
    """Rows 1-3 of Table 11. Real values only from the team's resolved label files."""
    pending = "pending: requires the team's labels (scripts/evaluate_labels.py)"
    if labels_dir is None or not (labels_dir / "label_evaluation.json").exists():
        return [_row("Label quality", "Cohen's kappa", None, ">= 0.8", pending, "accuracy evidence", "team labels"),
                _row("Rule accuracy R1-R3", "wrongly passed; wrongly failed; UNKNOWN share (per rule)", None,
                     "0; <= 5%; <= 15%", pending, "accuracy evidence", "team labels"),
                _row("Rule accuracy R4", "passed below maker recommendation; failed although meeting it; UNKNOWN", None,
                     "0; <= 5%; <= 15%", pending, "accuracy evidence", "team labels")]
    res = json.loads((labels_dir / "label_evaluation.json").read_text(encoding="utf-8"))
    tag = "DUMMY LABELS - NOT EVIDENCE" if res.get("dummy") else "team labels"
    return [_row("Label quality", "Cohen's kappa", res["kappa_overall"], ">= 0.8", res["kappa_status"],
                 "accuracy evidence", tag),
            _row("Rule accuracy R1-R3 / R4", "see label_rule_accuracy.csv", None, "0; <= 5%; <= 15%",
                 "see file", "accuracy evidence", tag)]


def run(run_id: str, labels_dir: Path | None = None) -> pd.DataFrame:
    rows = label_accuracy_rows(labels_dir)
    det = injected_detection(run_id)
    lake.write_table(det, "outputs", "eval_injected_detection", run_id, source="step10 (SYNTHETIC)")
    g = det[det["group"] == "injected incompatible"].set_index("verdict")["builds"]
    n_inj = int(g.sum())
    n_rep = int(g[[v for v in g.index if not str(v).startswith("not replayed")]].sum())
    n_inc = int(g.get("incomplete in replay", 0))
    n_complete = n_rep - n_inc
    inj = g / max(1, n_complete)
    caught = float(inj.get("FAIL", 0) + inj.get("UNKNOWN", 0))
    truth = lake.read_table("cleaned", "build_truth", run_id)
    src = truth.loc[truth["injected"].astype(bool), "inject_source"].iloc[0] if truth["injected"].any() else ""
    rows.append(_row("Build-level verdict and replay", "injected builds detected as FAIL / UNKNOWN / missed",
                     f"{inj.get('FAIL', 0):.4f} / {inj.get('UNKNOWN', 0):.4f} / {1 - caught:.4f}",
                     "100% FAIL or UNKNOWN; UNKNOWN reported", "met" if caught >= 0.9999 else "not met",
                     "mechanics check", "SYNTHETIC",
                     f"{n_inj} injected builds: {n_inj - n_rep} only in the training period, {n_inc} not complete "
                     f"at the end of the replay, {n_complete} complete and scored; injection source: {src}"
                     + ("" if src == "labels" else
                                                   " (Table 11 asks for held-out labelled pairs - pending labels)")))
    val = suggestion_validity(run_id)
    bad = val["fail"] + val["unknown"] + val["out_of_stock"] + val["stale_stock"]
    rows.append(_row("Suggestion validity", "suggestions that FAIL / are UNKNOWN / out of stock / on stock > 1 h old",
                     f"{val['fail']} / {val['unknown']} / {val['out_of_stock']} / {val['stale_stock']} of {val['suggestions']}", "0",
                     "met" if bad == 0 else "not met", "mechanics check", "SYNTHETIC events, real catalog"))
    cs = coverage_and_speed(run_id)
    rows.append(_row("Coverage", "requests with five suggestions (fresh stock and model)", cs["share_5_fresh_stock"],
                     ">= 90%", "met" if cs["share_5_fresh_stock"] >= 0.9 else "not met", "mechanics check", "SYNTHETIC",
                     f"all requests incl. stale stock: {cs['share_5_all_requests']}; builds without an UNKNOWN part: "
                     f"{cs['share_5_fresh_no_unknown_build']}; requests where the spec allows five (lift or same-band "
                     f"fallback): {cs['share_requests_with_5_eligible']}, of which served with five: "
                     f"{cs['share_5_where_5_eligible']}; stale-stock requests (no suggestion by design): "
                     f"{cs['stale_stock_requests']}"))
    rows.append(_row("Speed", "time per event, ms (mean / p95 / max)", f"{cs['mean_ms']} / {cs['p95_ms']} / {cs['max_ms']}",
                     "< 200 ms", "met" if cs["p95_ms"] < 200 else "not met", "mechanics check", "SYNTHETIC",
                     "measured on this machine; repeat on the team laptop"))
    fm = forecast_metrics(run_id)
    lake.write_table(fm, "outputs", "eval_forecast_metrics", run_id, source="step10 (SYNTHETIC)")
    for role, g in fm.groupby("role"):
        base = g[g["method"] == "seasonal_naive"]
        sel = g[g["method"] == "selected_policy"]
        if len(base) and len(sel):
            ok = sel["wape"].iloc[0] < base["wape"].iloc[0]
            rows.append(_row(f"Focus A forecast ({role})", "WAPE / bias: selected policy (chosen before scoring) vs seasonal naive",
                             f"{sel['wape'].iloc[0]} / {sel['bias'].iloc[0]} vs {base['wape'].iloc[0]} / {base['bias'].iloc[0]}",
                             "lower error than the baseline", "met" if ok else "not met", "mechanics check", "SYNTHETIC",
                             "selected policy = analogue curve for a launch SKU in its first 4 weeks, else SES with weekday factors"))
    ss = sim_summary(run_id)
    lake.write_table(ss, "outputs", "eval_sim_summary", run_id, source="step10 (SYNTHETIC)")
    o = ss[(ss["policy"] == "order_up_to") & (ss["role"] == "all")].iloc[0]
    m = ss[(ss["policy"] == "min_max") & (ss["role"] == "all")].iloc[0]
    rows.append(_row("Focus A ordering", "periods without stock-out; fill rate; average stock (order-up-to vs min/max)",
                     f"{o['periods_without_stockout']} ; {o['fill_rate']} ; {o['average_stock']} vs "
                     f"{m['periods_without_stockout']} ; {m['fill_rate']} ; {m['average_stock']}",
                     "approx. 95%; reported", ("met" if abs(o["periods_without_stockout"] - 0.95) <= 0.03 else
                      "above target (more stock than needed)" if o["periods_without_stockout"] > 0.95 else "not met"),
                     "mechanics check", "SYNTHETIC",
                     f"low-stock alerts (on-hand rule of the report) on {o['alert_share_of_periods']} of periods; "
                     f"without an open order due in time: {o['alerts_needing_action_share']}"))
    res = pd.DataFrame(rows)
    res["value"] = res["value"].astype(str)
    lake.write_table(res, "outputs", "evaluation_results", run_id, source="step10")
    return res
