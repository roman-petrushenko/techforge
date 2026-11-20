"""Focus A slice: forecast -> order-up-to -> alerts (all SYNTHETIC)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from techforge import app_data, config  # noqa: E402

st.set_page_config(page_title="Focus A - restock", layout="wide")
st.title("Focus A slice: forecast -> restock -> alert")
st.warning("SYNTHETIC data. This page demonstrates the mechanics of the chain only; it makes no claim about "
           "forecast accuracy for real launches.")
run_id = app_data.latest_complete_run()
if run_id is None:
    st.error("No pipeline run found.")
    st.stop()
t = app_data.focus_a_tables(run_id)
sku = t["fa_sku"]
c1, c2 = st.columns(2)
pid = c1.selectbox("SKU", sku["part_id"], format_func=lambda p: f"{p} ({sku.set_index('part_id').loc[p, 'role']})")
wh = c2.selectbox("Warehouse", config.WAREHOUSES)

f = t["forecast"][(t["forecast"]["part_id"] == pid) & (t["forecast"]["warehouse_id"] == wh)]
truth = t["sales_true_demand"]
truth = truth[(truth["part_id"] == pid) & (truth["warehouse_id"] == wh)].set_index("date")["true_demand"]
truth = truth.loc[f["target_date"].min(): f["target_date"].max()]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(truth.index, truth.values, color="#0b0b0b", linewidth=1.2, label="true demand (scoring only)")
colors = {"seasonal_naive": "#2a78d6", "ses_weekday": "#eb6834", "analogue_launch": "#1baf7a"}
for m, g in f[f["method"].isin(colors)].groupby("method"):
    c = colors[m]
    g = g.groupby("target_date")["forecast"].mean()
    ax.plot(g.index, g.values, color=c, linewidth=2, label=m)
ax.set_title("Rolling backtest forecasts vs true demand (SYNTHETIC)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)   # P8: below the plot
fig.tight_layout()
st.pyplot(fig)

sim = t["fa_sim_daily"]
s = sim[(sim["part_id"] == pid) & (sim["warehouse_id"] == wh)]
a = t["fa_alert"][(t["fa_alert"]["part_id"] == pid) & (t["fa_alert"]["warehouse_id"] == wh)]
fig, ax = plt.subplots(figsize=(10, 3.5))
for (pol, g), c in zip(s.groupby("policy"), ["#eb6834", "#2a78d6"]):
    ax.plot(g["date"], g["on_hand_end"], color=c, linewidth=2, label=f"stock on hand - {pol}")
o = s[(s["policy"] == "order_up_to") & (s["order_qty"] > 0)]
ax.scatter(o["date"], o["order_qty"], color="#1baf7a", s=20, label="order (order-up-to)", zorder=3)
ax.scatter(a["date"], [0] * len(a), marker="^", color="#fab219", s=20, label="low-stock alert", zorder=3)
ax.set_title("Held-out simulation: stock, orders and alerts (SYNTHETIC)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)   # P8: below the plot
fig.tight_layout()
st.pyplot(fig)

st.subheader("Evaluation (SYNTHETIC - mechanics checks)")
st.dataframe(t["eval_forecast_metrics"], hide_index=True)
st.dataframe(t["eval_sim_summary"], hide_index=True)
