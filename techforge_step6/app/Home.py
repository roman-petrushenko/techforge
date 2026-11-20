"""TechForge Step 6 - PC builder demo (Focus B "Compatible-Next-Part").

Start:  streamlit run app/Home.py      (after:  python run_pipeline.py --run-id demo)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from techforge import app_data, config  # noqa: E402

st.set_page_config(page_title="TechForge builder demo", layout="wide")
st.title("TechForge PC builder - compatibility check and next-part suggestions")

run_id = app_data.latest_complete_run()
if run_id is None:
    st.error("No pipeline run found. Run `python run_pipeline.py --run-id demo` first.")
    st.stop()


@st.cache_resource
def backend(rid: str) -> app_data.BuilderBackend:
    return app_data.BuilderBackend(rid)


be = backend(run_id)
st.info(f"Run **{run_id}**. Part specifications are **real** (pc-part-dataset c52a04c). "
        "Stock, popularity and the co-occurrence model are **SYNTHETIC data**.")

with st.sidebar:
    st.header("Serving context (synthetic)")
    warehouse = st.selectbox("Serving warehouse", config.WAREHOUSES)
    times = be.snapshot_times(warehouse)
    at = st.select_slider("Event time (stock snapshot valid at this time)", options=times[1:],
                          value=times[len(times) // 2], format_func=lambda t: t.strftime("%Y-%m-%d %H:%M"))
    minutes = st.slider("Minutes after the full hour", 0, 59, 15)
    event_time = pd.Timestamp(at) + pd.Timedelta(minutes=minutes)
    assortment_only = st.checkbox("Only TechForge assortment (synthetic)", value=True,
                                  help="Untick to choose from the whole real catalog (rule checks only).")
    st.caption(f"Model version: {be.rec.model_version or 'none - suggestions disabled'}")

st.subheader("1. Choose parts")
build: dict[str, str] = {}
cols = st.columns(3)
for i, cat in enumerate(app_data.BUILD_ORDER):
    opts = be.options(cat, assortment_only)
    labels = ["-"] + list(opts["label"])
    choice = cols[i % 3].selectbox(app_data.CATEGORY_LABELS[cat], labels, key=f"sel_{cat}")
    if choice != "-":
        build[cat] = opts.loc[opts["label"] == choice, "part_id"].iloc[0]

st.subheader("2. Compatibility check (rules R1-R4)")
verdict, message, results = be.check(build)
{"PASS": st.success, "FAIL": st.warning, "UNKNOWN": st.info}.get(verdict, st.caption)(message)
if results:
    st.dataframe(pd.DataFrame([{"rule": r.rule_id, "verdict": r.verdict, "reason": r.reason,
                                "rule version": r.rule_version} for r in results]),
                 hide_index=True, use_container_width=True)
st.caption("PASS means only that the build passes these four checks. GPU length vs case space is not checked. "
           "Checkout is never blocked.")

# P8: when a rule fails, the part named below is offered for replacement (same filters and ranking)
REPLACE_FOR_RULE = {"R4": "power-supply", "R3": "case", "R2": "memory", "R1": "motherboard"}
failed = [r.rule_id for r in results if r.verdict == "FAIL"]
st.subheader("3. Suggestions for the next part" if not failed else "3. Suggestions to fix the build")
target = next((c for c in app_data.BUILD_ORDER if c not in build), None)
if failed:
    replace = REPLACE_FOR_RULE[failed[0]]
    st.caption(f"Rule {failed[0]} fails - compatible, in-stock alternatives for the "
               f"{app_data.CATEGORY_LABELS[replace].lower()}:")
    target = replace
    build = {c: p for c, p in build.items() if c != replace}
if target is None:
    st.write("Build complete - no further suggestion.")
else:
    if not failed:
        target = st.selectbox("Suggest parts for", [c for c in app_data.BUILD_ORDER if c not in build],
                              format_func=lambda c: app_data.CATEGORY_LABELS[c])
    table, info = be.suggest(build, target, warehouse, event_time)
    if info["status"] == "stale_stock":
        st.error(f"Stock data is stale (snapshot {info['snapshot_ts']}, {info['stock_age_min']} min old > 60 min): "
                 "no suggestion is shown. The compatibility check above still runs.")
    elif info["status"] == "model_missing":
        st.error("No co-occurrence model available: no suggestion is shown.")
    else:
        st.caption(f"{info['candidates']} candidates -> {info['after_rules']} pass all rules and are not UNKNOWN -> "
                   f"{info['after_stock']} in stock at {warehouse} (snapshot {info['snapshot_ts']}, "
                   f"{info['stock_age_min']} min old). SYNTHETIC stock and popularity.")
        if table.empty:
            st.warning("No compatible part in stock for this build.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
