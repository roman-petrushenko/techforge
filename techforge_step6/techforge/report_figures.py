"""Report figures (PNG) and result tables (CSV) for Step 6, written to report_outputs/.

Static matplotlib charts for the report. Colours: validated categorical slots in fixed order
(blue, orange, aqua, yellow) and the status colours for PASS / UNKNOWN / FAIL, always with
a legend or direct labels, never colour alone. Every figure title says what data it shows.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from techforge import config, lake  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
METHOD_COLOR = {"seasonal_naive": SERIES[0], "ses_weekday": SERIES[1], "analogue_launch": SERIES[2],
                "selected_policy": INK2 if False else "#52514e"}  # fixed per method
STATUS = {"PASS": "#0ca30c", "UNKNOWN": "#fab219", "FAIL": "#d03b3b", "NOT_CHECKED": "#b4b2aa"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_unknown_share(run_id: str, out: Path) -> Path:
    cov = lake.read_table("curated", "attribute_coverage", run_id)
    cov["label"] = cov["category"] + " · " + cov["attribute"]
    cov = cov.sort_values("unknown_share")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(cov["label"], cov["unknown_share"] * 100, color=SERIES[0], height=0.6)
    for y, (v, n) in enumerate(zip(cov["unknown_share"] * 100, cov["unknown"])):
        ax.text(v + 1, y, f"{v:.1f}%  ({n})", va="center", color=INK2, fontsize=8.5)
    ax.set_xlabel("parts with the attribute UNKNOWN (%)")
    ax.set_xlim(0, max(100, cov["unknown_share"].max() * 115))
    ax.grid(axis="y", visible=False)
    ax.set_title("UNKNOWN share per rule attribute - real catalog")
    return _save(fig, out, "fig01_unknown_share_per_attribute.png")


def _stacked_status(ax, table: pd.DataFrame, order: list[str]) -> None:
    left = pd.Series(0.0, index=table.index)
    for v in order:
        if v in table.columns:
            ax.barh(table.index, table[v], left=left, color=STATUS[v], height=0.6, label=v,
                    edgecolor=SURFACE, linewidth=2)
            for y, (l0, w) in enumerate(zip(left, table[v])):
                if w >= 0.06:
                    ax.text(l0 + w / 2, y, f"{w:.0%}", ha="center", va="center", color=INK, fontsize=8.5)
            left = left + table[v]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(order), loc="upper center", bbox_to_anchor=(0.5, -0.15))


def fig_rule_verdicts(run_id: str, out: Path) -> Path:
    cc = lake.read_table("outputs", "compat_check", run_id)
    t = pd.crosstab(cc["rule_id"], cc["verdict"], normalize="index").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    _stacked_status(ax, t, ["PASS", "UNKNOWN", "FAIL"])
    ax.set_title("Rule verdicts during replay - SYNTHETIC builds, real catalog")
    return _save(fig, out, "fig02_rule_verdicts_replay.png")


def fig_injected_detection(run_id: str, out: Path) -> Path:
    det = lake.read_table("outputs", "eval_injected_detection", run_id)
    det = det[det["verdict"].isin(["FAIL", "UNKNOWN", "PASS"])]   # complete builds only (P8)
    det = det.assign(share=det["builds"] / det.groupby("group")["builds"].transform("sum"))
    t = det.pivot_table(index="group", columns="verdict", values="share", fill_value=0)
    t = t.loc[[g for g in t.index if g != "generator-compatible"] + ["generator-compatible"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    _stacked_status(ax, t, ["FAIL", "UNKNOWN", "PASS"])
    ax.set_title("Final build verdict: injected vs generator-compatible builds - SYNTHETIC")
    return _save(fig, out, "fig03_injected_build_detection.png")


def fig_quarantine(run_id: str, out: Path) -> Path:
    q = lake.read_table("cleaned", "event_validation_log", run_id)
    q = q[~q["reason"].str.startswith("accepted")].sort_values("count")
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.barh(q["reason"], q["count"], color=SERIES[0], height=0.6)
    for y, v in enumerate(q["count"]):
        ax.text(v, y, f" {v}", va="center", color=INK2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("events")
    ax.set_title("Quarantined and late events by reason - SYNTHETIC")
    return _save(fig, out, "fig04_quarantine_counts.png")


def fig_time_per_event(run_id: str, out: Path) -> Path:
    lg = lake.read_table("outputs", "replay_log", run_id)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.hist(lg["elapsed_ms"], bins=60, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    p95 = lg["elapsed_ms"].quantile(0.95)
    ax.axvline(p95, color=INK2, linewidth=1, linestyle="--")
    ax.text(p95, ax.get_ylim()[1] * 0.9, f"  p95 = {p95:.1f} ms (target < 200 ms)", color=INK2, fontsize=8.5)
    ax.set_xlabel("time per replayed event (ms, this machine)")
    ax.set_ylabel("events")
    ax.set_title("Rule check + suggestion time per event - SYNTHETIC replay")
    return _save(fig, out, "fig05_time_per_event.png")


def fig_forecast_launch(run_id: str, out: Path, warehouse: str = "KYIV") -> Path:
    f = lake.read_table("outputs", "forecast", run_id)
    td = lake.read_table("cleaned", "sales_true_demand", run_id)
    sku = f.loc[f["role"] == "launch", "part_id"].iloc[0]
    f = f[(f["part_id"] == sku) & (f["warehouse_id"] == warehouse)]
    truth = td[(td["part_id"] == sku) & (td["warehouse_id"] == warehouse)].set_index("date")["true_demand"]
    launch = pd.Timestamp(lake.read_table("cleaned", "fa_sku", run_id).set_index("part_id").loc[sku, "launch_date"])
    lo, hi = launch - pd.Timedelta(days=21), f["target_date"].max()
    f = f[f["target_date"] >= lo]
    truth = truth.loc[lo:hi]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(truth.index, truth.values, color=INK, linewidth=1.2, label="true demand (scoring only)")
    for m, g in f.groupby("method"):
        g = g.groupby("target_date")["forecast"].mean()   # mean over the overlapping origins
        ax.plot(g.index, g.values, color=METHOD_COLOR[m], linewidth=2, label=m)
    ax.axvline(launch, color=INK2, linewidth=1, linestyle=":")
    ax.text(launch, ax.get_ylim()[1] * 0.95, "  launch date (known from F10)", color=INK2, fontsize=8.5)
    ax.set_ylabel("units per day")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"Launch GPU, {warehouse}: backtest forecasts vs true demand - SYNTHETIC")
    return _save(fig, out, "fig06_forecast_vs_true_demand_launch.png")


def fig_stock_orders_alerts(run_id: str, out: Path, warehouse: str = "KYIV", role: str = "launch") -> Path:
    sim = lake.read_table("outputs", "fa_sim_daily", run_id)
    alerts = lake.read_table("outputs", "fa_alert", run_id)
    sku = sim.loc[sim["role"] == role, "part_id"].iloc[0]
    s = sim[(sim["part_id"] == sku) & (sim["warehouse_id"] == warehouse) & (sim["policy"] == "order_up_to")]
    a = alerts[(alerts["part_id"] == sku) & (alerts["warehouse_id"] == warehouse)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(s["date"], s["on_hand_end"], color=SERIES[0], linewidth=2, label="stock on hand (end of day)")
    ax.plot(s["date"], s["demand"], color=INK2, linewidth=1, label="true demand")
    o = s[s["order_qty"] > 0]
    ax.scatter(o["date"], o["order_qty"], color=SERIES[1], s=22, zorder=3, label="order placed (quantity)",
               edgecolor=SURFACE, linewidth=1)
    so = s[s["stockout"]]
    ax.scatter(so["date"], [0] * len(so), marker="x", color=STATUS["FAIL"], s=30, zorder=4, label="stock-out day")
    ymax = ax.get_ylim()[1]
    ax.scatter(a["date"], [ymax * 0.97] * len(a), marker="v", color=STATUS["UNKNOWN"], s=18, zorder=3,
               label="low-stock alert", edgecolor=INK2, linewidth=0.3)
    ax.set_ylabel("units")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    ax.set_title(f"Order-up-to policy, launch GPU, {warehouse}, held-out period - SYNTHETIC")
    return _save(fig, out, "fig07_stock_orders_alerts.png")


def fig_wape(run_id: str, out: Path) -> Path:
    fm = lake.read_table("outputs", "eval_forecast_metrics", run_id)
    roles = list(fm["role"].unique())
    methods = ["seasonal_naive", "ses_weekday", "analogue_launch", "selected_policy"]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    w = 0.2
    for i, m in enumerate(methods):
        c = METHOD_COLOR[m]
        g = fm[fm["method"] == m].set_index("role").reindex(roles)
        xs = [r + (i - 1.5) * w for r in range(len(roles))]
        ax.bar(xs, g["wape"], width=w - 0.03, color=c, label=m)
        for x, v in zip(xs, g["wape"]):
            if pd.notna(v):
                ax.text(x, v + 0.01, f"{v:.2f}", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(range(len(roles)), roles)
    ax.set_ylabel("WAPE (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Rolling backtest: WAPE vs true demand by SKU role - SYNTHETIC")
    return _save(fig, out, "fig08_backtest_wape.png")


def fig_policy(run_id: str, out: Path) -> Path:
    ss = lake.read_table("outputs", "eval_sim_summary", run_id)
    ss = ss[ss["role"] == "all"].set_index("policy")
    metrics = [("periods_without_stockout", "periods without stock-out", True),
               ("fill_rate", "fill rate", True), ("average_stock", "average stock (units)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for ax, (col, label, pct) in zip(axes, metrics):
        vals = ss.loc[["order_up_to", "min_max"], col]
        ax.bar(["order-up-to", "min/max"], vals, color=[SERIES[0], SERIES[1]], width=0.55)
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.1%}" if pct else f"{v:.1f}", ha="center", va="bottom", color=INK2, fontsize=8.5)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", visible=False)
        if pct:
            ax.set_ylim(0, 1.08)
    fig.suptitle("Held-out simulation: order-up-to vs fixed min/max - SYNTHETIC", fontweight="bold", color=INK)
    return _save(fig, out, "fig09_policy_comparison.png")


def write_all(run_id: str, out: Path = config.REPORT_OUTPUTS) -> list[Path]:
    """All report figures and result tables for one run."""
    paths = [f(run_id, out) for f in (fig_unknown_share, fig_rule_verdicts, fig_injected_detection, fig_quarantine,
                                      fig_time_per_event, fig_forecast_launch, fig_stock_orders_alerts,
                                      fig_wape, fig_policy)]
    tables = {
        "table11_evaluation_results.csv": ("outputs", "evaluation_results"),
        "attribute_coverage.csv": ("curated", "attribute_coverage"),
        "parts_cleaning_log.csv": ("cleaned", "parts_cleaning_log"),
        "event_validation_log.csv": ("cleaned", "event_validation_log"),
        "injected_detection.csv": ("outputs", "eval_injected_detection"),
        "forecast_metrics.csv": ("outputs", "eval_forecast_metrics"),
        "policy_simulation_summary.csv": ("outputs", "eval_sim_summary"),
        "builds_per_day.csv": ("curated", "builds_per_day"),
        "replay_status.csv": ("outputs", "dashboard_replay_status"),
    }
    real = {"attribute_coverage.csv", "parts_cleaning_log.csv"}   # real catalogue; all others SYNTHETIC
    out.mkdir(parents=True, exist_ok=True)
    for fname, (zone, ds) in tables.items():
        p = out / fname
        df = lake.read_table(zone, ds, run_id)
        # P8 fix (P7 issue 09): every exported table says what its data is
        label = "real catalog (pc-part-dataset c52a04c)" if fname in real else "SYNTHETIC"
        if fname == "table11_evaluation_results.csv":
            label = None   # has its own per-row "data" column
        if label and "data_label" not in df.columns:
            df.insert(0, "data_label", label)
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
