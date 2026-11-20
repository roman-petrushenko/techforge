"""Step 8 - Replay events: rule check -> filters -> ranking (Table 12 [R]); simulated stream.

Events of the replay period are processed in hourly micro-batches by arrival time
(ingest_ts); inside a batch they are sorted by event time. Events that arrive late are
therefore processed in the next batch (Section 4.6 [R]). Each event sees only the stock
snapshot valid at its event time.

Outputs (outputs zone): compat_check, build_verdict, reco_next_part, replay_log.
Done when: replay log complete (one row per replayed event).
"""
from __future__ import annotations

import json
import time

import pandas as pd

from techforge import lake, rules
from techforge.recommender import Recommender, StockView, next_category
from techforge.step02_generate import load_settings
from techforge.step07_cooccurrence import load_model, run_model_version


def run(run_id: str, model_version: str | None = None) -> dict:
    dim = lake.read_table("curated", "dim_part", run_id)
    ev = lake.read_table("cleaned", "builder_event", run_id)
    stock = lake.read_table("cleaned", "stock_snapshot", run_id)
    assortment = lake.read_table("cleaned", "assortment", run_id)
    train_end = pd.Timestamp(load_settings(run_id)["train_end"])
    version = model_version or run_model_version(run_id)   # P8 fix (P7 issue 08)
    model = load_model(version) if version else None
    rec = Recommender(dim, list(assortment["part_id"]), model)
    sv = StockView(stock)
    parts = rec.parts

    # P8 fix (P7 issue 02): builds that started before the cutoff continue from their state at
    # the cutoff; a part replaces the current one only if its event time is later (late events
    # never restore an older part). Ties are broken by event_id.
    builds: dict[str, dict[str, str]] = {}
    added_at: dict[str, dict[str, tuple]] = {}
    before = ev[ev["event_ts"] < train_end].sort_values(["event_ts", "event_id"])
    for r in before.itertuples(index=False):
        builds.setdefault(r.build_id, {})[r.category] = r.part_id
        added_at.setdefault(r.build_id, {})[r.category] = (r.event_ts, r.event_id)

    replay = ev[ev["event_ts"] >= train_end].copy()
    replay["batch"] = replay["ingest_ts"].dt.floor("h")
    replay = replay.sort_values(["batch", "event_ts", "event_id"])
    checks, verdicts, recos, log = [], [], [], []
    for r in replay.itertuples(index=False):
        t0 = time.perf_counter()
        b = builds.setdefault(r.build_id, {})
        when = added_at.setdefault(r.build_id, {})
        superseded = r.category in when and (r.event_ts, r.event_id) < when[r.category]
        if not superseded:
            b[r.category] = r.part_id
            when[r.category] = (r.event_ts, r.event_id)
        results = rules.check_build({c: parts[p] for c, p in b.items()})
        verdict = rules.build_verdict(results)
        target = next_category(b)
        if superseded:
            sugg, info = [], {"status": "late_superseded", "snapshot_ts": None, "stock_age_min": None}
        elif target is not None:
            sugg, info = rec.suggest(b, target, r.warehouse_id, r.event_ts, sv, last_added=r.part_id)
        else:
            sugg, info = [], {"status": "build_complete", "snapshot_ts": None, "stock_age_min": None}
        elapsed_ms = (time.perf_counter() - t0) * 1000
        for x in results:
            checks.append({"build_id": r.build_id, "rule_id": x.rule_id, "event_ts": r.event_ts,
                           "event_id": r.event_id, "parts_checked": ";".join(x.part_ids),
                           "verdict": x.verdict, "reason": x.reason, "rule_version": x.rule_version})
        verdicts.append({"build_id": r.build_id, "event_ts": r.event_ts, "event_id": r.event_id,
                         "verdict": verdict, "message": rules.customer_message(verdict, results),
                         "rules_checked": len(results), "ruleset_version": rules.RULESET_VERSION})
        for s in sugg:
            recos.append({"build_id": r.build_id, "event_ts": r.event_ts, "rank": s.rank,
                          "event_id": r.event_id, "target_category": target, "part_id": s.part_id,
                          "score": s.score, "reason": s.reason, "on_hand": s.on_hand,
                          "warehouse_id": r.warehouse_id, "model_version": s.model_version})
        log.append({"event_id": r.event_id, "build_id": r.build_id, "event_ts": r.event_ts,
                    "batch_hour": r.batch, "late": bool(r.late), "warehouse_id": r.warehouse_id,
                    "target_category": target, "status": info["status"], "n_suggestions": len(sugg),
                    "snapshot_ts": info.get("snapshot_ts"), "stock_age_min": info.get("stock_age_min"),
                    "candidates": info.get("candidates"), "after_rules": info.get("after_rules"),
                    "after_stock": info.get("after_stock"), "eligible": info.get("eligible"),
                    "price_band": info.get("price_band"), "elapsed_ms": round(elapsed_ms, 3),
                    "build_state": json.dumps(b, sort_keys=True), "model_version": rec.model_version})
    src = "cleaned/builder_event replay period (SYNTHETIC) + curated/dim_part (real catalog)"
    lake.write_table(pd.DataFrame(checks), "outputs", "compat_check", run_id, source=src)
    lake.write_table(pd.DataFrame(verdicts), "outputs", "build_verdict", run_id, source=src)
    reco_cols = ["build_id", "event_ts", "rank", "event_id", "target_category", "part_id", "score",
                 "reason", "on_hand", "warehouse_id", "model_version"]
    lake.write_table(pd.DataFrame(recos, columns=reco_cols), "outputs", "reco_next_part", run_id, source=src)
    lg = pd.DataFrame(log)
    lake.write_table(lg, "outputs", "replay_log", run_id, source=src,
                     extra={"model_version": rec.model_version})
    return {"events": len(lg), "model_version": rec.model_version,
            "status_counts": lg["status"].value_counts().to_dict(),
            "mean_ms": round(lg["elapsed_ms"].mean(), 2), "p95_ms": round(lg["elapsed_ms"].quantile(0.95), 2)}
