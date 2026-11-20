"""Compatible-Next-Part recommender (Section 4.3 [R]).

Order of operations (filter BEFORE ranking, so no score can override compatibility):
  1. no model                      -> no suggestion ("model_missing")
  2. stock snapshot valid at event time missing or older than 1 h -> no suggestion ("stale_stock")
  3. candidates = TechForge assortment of the target category
  4. remove UNKNOWN parts (any rule attribute missing) and candidates with a FAIL or
     UNKNOWN verdict against the parts already in the build
  5. remove candidates out of stock in the serving warehouse
  6. rank by lift (mean over build parts with co >= MIN_SUPPORT), trained on the training period only
  7. fill up to five with the most popular compatible part in the same price band ("fallback")
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import pandas as pd

from techforge import config, rules
from techforge.parts import RULE_ATTRIBUTES


@dataclass
class Suggestion:
    rank: int
    part_id: str
    score: float | None
    reason: str          # "lift" or "fallback"
    on_hand: int
    model_version: str


class StockView:
    """Hourly snapshots per warehouse; returns the snapshot valid at a given time."""

    def __init__(self, snapshots: pd.DataFrame):
        self._times: dict[str, list[pd.Timestamp]] = {}
        self._stock: dict[tuple[str, pd.Timestamp], dict[str, int]] = {}
        for (wh, ts), g in snapshots.groupby(["warehouse_id", "snapshot_ts"], sort=True):
            self._times.setdefault(wh, []).append(ts)
            self._stock[(wh, ts)] = dict(zip(g["part_id"], g["on_hand"].astype(int)))

    def valid_snapshot(self, warehouse: str, at: pd.Timestamp) -> tuple[pd.Timestamp | None, dict[str, int] | None]:
        times = self._times.get(warehouse, [])
        i = bisect.bisect_right(times, at) - 1
        if i < 0:
            return None, None
        return times[i], self._stock[(warehouse, times[i])]

    def snapshot_times(self, warehouse: str) -> list[pd.Timestamp]:
        return list(self._times.get(warehouse, []))


class Recommender:
    def __init__(self, dim_part: pd.DataFrame, assortment_ids: list[str], model: dict | None,
                 min_support: int = config.MIN_SUPPORT, top_n: int = config.TOP_N,
                 max_age_minutes: int = config.STOCK_MAX_AGE_MINUTES):
        self.parts: dict[str, dict[str, Any]] = {r["part_id"]: r for r in dim_part.to_dict("records")}
        self.min_support, self.top_n = min_support, top_n
        self.max_age = pd.Timedelta(minutes=max_age_minutes)
        self.by_category: dict[str, list[str]] = {}
        for pid in assortment_ids:
            p = self.parts.get(pid)
            if p is not None:
                self.by_category.setdefault(p["category"], []).append(pid)
        self.model_version: str | None = None
        self.lift: dict[str, dict[str, tuple[int, float]]] = {}
        self.popularity: dict[str, int] = {}
        if model is not None:
            self.model_version = model["version"]
            for r in model["pairs"].itertuples(index=False):
                self.lift.setdefault(r.part_id_i, {})[r.part_id_j] = (int(r.co), float(r.lift))
            self.popularity = dict(zip(model["popularity"]["part_id"], model["popularity"]["n_builds"]))

    # --------------------------------------------------------------- helpers
    def is_unknown_part(self, pid: str) -> bool:
        p = self.parts[pid]
        return any(rules.missing(p.get(a)) for a in RULE_ATTRIBUTES[p["category"]])

    def compatible_with_build(self, build: dict[str, str], candidate: str) -> bool:
        cat = self.parts[candidate]["category"]
        trial = {c: self.parts[pid] for c, pid in build.items() if c != cat}
        trial[cat] = self.parts[candidate]
        return all(r.verdict == rules.PASS for r in rules.check_rules_involving(trial, cat))

    def build_price_band(self, build: dict[str, str], last_added: str | None) -> str | None:
        """Price band of the most recently added part with a known band [A]."""
        order = ([last_added] if last_added else []) + list(build.values())
        for pid in order:
            band = self.parts.get(pid, {}).get("price_band")
            if band and not rules.missing(band):
                return band
        return None

    # --------------------------------------------------------------- main entry
    def suggest(self, build: dict[str, str], target_category: str, warehouse: str, at: pd.Timestamp,
                stock: StockView, last_added: str | None = None) -> tuple[list[Suggestion], dict[str, Any]]:
        info: dict[str, Any] = {"status": "ok", "snapshot_ts": None, "stock_age_min": None,
                                "candidates": 0, "after_rules": 0, "after_stock": 0}
        if self.model_version is None:
            info["status"] = "model_missing"
            return [], info
        snap_ts, snap = stock.valid_snapshot(warehouse, at)
        if snap_ts is not None:
            info["snapshot_ts"] = snap_ts
            info["stock_age_min"] = round((at - snap_ts).total_seconds() / 60, 1)
        if snap is None or at - snap_ts > self.max_age:
            info["status"] = "stale_stock"
            return [], info
        cands = self.by_category.get(target_category, [])
        info["candidates"] = len(cands)
        ok = [c for c in cands if not self.is_unknown_part(c) and self.compatible_with_build(build, c)]
        info["after_rules"] = len(ok)
        ok = [c for c in ok if snap.get(c, 0) > 0]
        info["after_stock"] = len(ok)

        scored = []
        for c in ok:
            lifts = [self.lift.get(a, {}).get(c) for a in build.values()]
            lifts = [lf for co_lf in lifts if co_lf and co_lf[0] >= self.min_support for lf in [co_lf[1]]]
            if lifts:
                scored.append((sum(lifts) / len(lifts), c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = [Suggestion(i + 1, c, round(s, 4), "lift", snap[c], self.model_version)
               for i, (s, c) in enumerate(scored[: self.top_n])]
        band = self.build_price_band(build, last_added)
        used = {s.part_id for s in out}
        rest = [c for c in ok if c not in used and band is not None and self.parts[c].get("price_band") == band]
        # candidates the specification allows: all lift-scored ones + same-band fallback ones
        info["eligible"] = len(scored) + len(rest)
        info["price_band"] = band
        if len(out) < self.top_n:
            rest.sort(key=lambda c: (-self.popularity.get(c, 0), c))
            for c in rest[: self.top_n - len(out)]:
                out.append(Suggestion(len(out) + 1, c, None, "fallback", snap[c], self.model_version))
        if len(out) < self.top_n:
            info["status"] = "ok_partial"
        return out, info


def next_category(build: dict[str, str]) -> str | None:
    for c in config.BUILD_ORDER:
        if c not in build:
            return c
    return None
