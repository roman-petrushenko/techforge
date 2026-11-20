"""Staff dashboard: data quality, UNKNOWN attributes, quarantined events, stale data."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from techforge import app_data  # noqa: E402

st.set_page_config(page_title="Data quality", layout="wide")
st.title("Data quality and reference-table gaps (staff)")
run_id = app_data.latest_complete_run()
if run_id is None:
    st.error("No pipeline run found.")
    st.stop()
t = app_data.dq_tables(run_id)

st.subheader("UNKNOWN share per rule attribute - real catalog")
cov = t["coverage"].assign(unknown_pct=lambda d: (d["unknown_share"] * 100).round(1))
st.bar_chart(cov.set_index(cov["category"] + " · " + cov["attribute"])["unknown_pct"], horizontal=True)
st.dataframe(cov, hide_index=True)
st.caption("UNKNOWN parts are never suggested. Fill the reference tables (and verify rows) to reduce these shares.")

st.subheader("Why attributes are missing - real catalog")
st.dataframe(t["dq_reasons"], hide_index=True)
cat = st.selectbox("Show parts with gaps for category", sorted(t["dq_issue"]["category"].unique()))
st.dataframe(t["dq_issue"][t["dq_issue"]["category"] == cat].head(500), hide_index=True)

st.subheader("Part merging (step 3) - real catalog")
st.dataframe(t["cleaning"], hide_index=True)

st.subheader("Quarantined and late events - SYNTHETIC data")
st.dataframe(t["quarantine"], hide_index=True)
rs = t["replay_status"].set_index("status")["events"]
stale = int(rs.get("stale_stock", 0))
if stale:
    st.warning(f"STALE DATA: {stale} replayed events saw a stock snapshot older than one hour; "
               "no suggestion was shown for them.")
st.dataframe(t["replay_status"], hide_index=True)

st.subheader("Table 11 - evaluation results of this run")
st.dataframe(t["evaluation"], hide_index=True, use_container_width=True)
st.caption("Accuracy rows stay 'pending' until the team's labels are evaluated. Mechanics rows use SYNTHETIC data.")
