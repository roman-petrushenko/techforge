"""Data access for the Streamlit app (kept out of the UI so it can be tested with pytest)."""
from __future__ import annotations

import pandas as pd

from techforge import config, lake, rules
from techforge.recommender import Recommender, StockView
from techforge.step07_cooccurrence import load_model, run_model_version


def latest_complete_run() -> str | None:
    """Most recent run that has replay outputs."""
    for run_id in reversed(lake.list_runs()):
        if lake.table_exists("outputs", "replay_log", run_id):
            return run_id
    return None


class BuilderBackend:
    """Everything the builder page needs: parts, rules, suggestions with stock."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dim = lake.read_table("curated", "dim_part", run_id)
        self.assortment = lake.read_table("cleaned", "assortment", run_id)
        self.stock_df = lake.read_table("cleaned", "stock_snapshot", run_id)
        version = run_model_version(run_id)  # P8: the model of this run, not another run's latest
        self.model = load_model(version) if version else None
        self.rec = Recommender(self.dim, list(self.assortment["part_id"]), self.model)
        self.stock = StockView(self.stock_df)

    def options(self, category: str, assortment_only: bool) -> pd.DataFrame:
        d = self.dim[self.dim["category"] == category]
        if assortment_only:
            d = d[d["part_id"].isin(self.assortment["part_id"])]
        d = d.assign(label=d["name"] + d["price"].map(lambda p: f"  ·  ${p:,.0f}" if pd.notna(p) else "  ·  no price")
                     + "  ·  " + d["part_id"])
        return d.sort_values("name")[["part_id", "label"]]

    def snapshot_times(self, warehouse: str) -> list[pd.Timestamp]:
        return self.stock.snapshot_times(warehouse)

    def check(self, build: dict[str, str]) -> tuple[str, str, list[rules.RuleResult]]:
        results = rules.check_build({c: self.rec.parts[p] for c, p in build.items()})
        verdict = rules.build_verdict(results)
        return verdict, rules.customer_message(verdict, results), results

    def suggest(self, build: dict[str, str], target: str, warehouse: str, at: pd.Timestamp):
        sugg, info = self.rec.suggest(build, target, warehouse, at, self.stock,
                                      last_added=list(build.values())[-1] if build else None)
        rows = [{"rank": s.rank, "part": self.rec.parts[s.part_id]["name"], "part_id": s.part_id,
                 "price": self.rec.parts[s.part_id]["price"], "reason": s.reason,
                 "lift": s.score, "in stock": s.on_hand, "model": s.model_version} for s in sugg]
        return pd.DataFrame(rows), info


def focus_a_tables(run_id: str) -> dict[str, pd.DataFrame]:
    return {name: lake.read_table(zone, name, run_id) for zone, name in (
        ("outputs", "forecast"), ("cleaned", "sales_true_demand"), ("outputs", "fa_sim_daily"),
        ("outputs", "fa_alert"), ("outputs", "eval_forecast_metrics"), ("outputs", "eval_sim_summary"),
        ("cleaned", "fa_sku"))}


def dq_tables(run_id: str) -> dict[str, pd.DataFrame]:
    return {
        "coverage": lake.read_table("curated", "attribute_coverage", run_id),
        "dq_reasons": lake.read_table("outputs", "dashboard_dq_reasons", run_id),
        "quarantine": lake.read_table("cleaned", "event_validation_log", run_id),
        "replay_status": lake.read_table("outputs", "dashboard_replay_status", run_id),
        "cleaning": lake.read_table("cleaned", "parts_cleaning_log", run_id),
        "dq_issue": lake.read_table("curated", "dq_issue", run_id),
        "evaluation": lake.read_table("outputs", "evaluation_results", run_id),
    }


CATEGORY_LABELS = {"cpu": "Processor (CPU)", "motherboard": "Motherboard", "memory": "Memory",
                   "video-card": "Graphics card", "case": "Case", "power-supply": "Power supply"}
BUILD_ORDER = config.BUILD_ORDER
