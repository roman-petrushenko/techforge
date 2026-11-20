"""Step 9 - Forecast -> order-up-to level -> orders and alerts (Table 12 [R]); daily batch.

Part 1 - rolling backtest: origins move forward one week; each forecast uses only recorded
         sales up to the origin with stock-out days excluded [R]. Output: forecast (no true demand).
Part 2 - simulation over the held-out period: the daily order-up-to policy [R] against a
         fixed min/max baseline [R Table 8]. Demand in the simulation is the generator's true
         demand (it plays the role of the customers); the policy sees only what it sold.
Outputs (outputs zone): forecast, fa_order, fa_alert, fa_sim_daily.
Done when: backtest finished.  All data SYNTHETIC.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from techforge import config, forecasting as fc, lake, policy
from techforge.step02_generate import load_settings

HORIZON = 7               # days forecast at each origin [A]
FIRST_ORIGIN_DAY = 120    # first backtest origin (days after start) [A]
ANALOGUE_WEEKS = 4        # launch SKU uses the analogue curve for its first 4 weeks [A]
MINMAX_COVER_DAYS = 14    # baseline max level = d_avg * (L + 14) [A]


def _series(sales: pd.DataFrame, pid: str, wh: str) -> tuple[pd.Series, pd.Series]:
    s = sales[(sales["part_id"] == pid) & (sales["warehouse_id"] == wh)].set_index("date").sort_index()
    return s["units_sold"].astype(float), s["stockout_flag"].astype(bool)


def backtest(sales: pd.DataFrame, skus: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    analogue = skus[skus["role"] == "analogue_launch"].iloc[0]
    for sku in skus.itertuples(index=False):
        for wh in config.WAREHOUSES:
            units, so = _series(sales, sku.part_id, wh)
            hist_all = fc.clean_history(units, so)
            start = hist_all.index.min() + pd.Timedelta(days=FIRST_ORIGIN_DAY)
            origins = pd.date_range(start, end_date - pd.Timedelta(days=HORIZON), freq="7D")
            a_hist = None
            if sku.role == "launch":
                a_units, a_so = _series(sales, analogue.part_id, wh)
                a_hist = fc.clean_history(a_units, a_so)
            for origin in origins:
                hist = hist_all.loc[:origin]
                preds = {"seasonal_naive": fc.seasonal_naive(hist, origin, HORIZON),
                         "ses_weekday": fc.SesWeekday(hist).forecast(origin, HORIZON)}
                if a_hist is not None:
                    # P8 fix (P7 issue 03): the analogue curve uses only analogue sales up to the origin
                    curve = fc.analogue_curve(a_hist.loc[:origin], pd.Timestamp(analogue.launch_date))
                    launch = pd.Timestamp(sku.launch_date)
                    preds["analogue_launch"] = fc.analogue_launch_forecast(curve, launch, hist, origin, HORIZON)
                targets = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, freq="D")
                for method, p in preds.items():
                    for k, (d, v) in enumerate(zip(targets, p), start=1):
                        rows.append((sku.part_id, wh, sku.role, origin, d, k, method, float(v)))
    return pd.DataFrame(rows, columns=["part_id", "warehouse_id", "role", "origin_date", "target_date",
                                       "horizon_days", "method", "forecast"])


def _received_before(deliv: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Deliveries that had ARRIVED before `before` (P8 fix, P7 issue 03): an order still on
    its way has no observed lead time yet."""
    arrival = deliv["order_date"] + pd.to_timedelta(deliv["lead_time_days"], unit="D")
    return deliv[arrival < before]


def _open_orders(deliv: pd.DataFrame, pid: str, wh: str, at: pd.Timestamp) -> dict:
    """Orders placed before `at` that arrive on or after `at`: {arrival_date: units}."""
    d = deliv[(deliv["part_id"] == pid) & (deliv["warehouse_id"] == wh) & (deliv["order_date"] < at)]
    out: dict = {}
    qty_col = next((c for c in ("qty", "quantity", "units", "order_qty") if c in d.columns), None)
    for r in d.itertuples(index=False):
        arrive = pd.Timestamp(r.order_date) + pd.Timedelta(days=int(r.lead_time_days))
        if arrive >= at and qty_col is not None:
            out[arrive] = out.get(arrive, 0) + int(getattr(r, qty_col))
    return out


def _lead_time_estimates(deliv: pd.DataFrame, pid: str, wh: str, before: pd.Timestamp) -> tuple[float, float]:
    deliv = _received_before(deliv, before)
    d = deliv[(deliv["part_id"] == pid) & (deliv["warehouse_id"] == wh)]
    if len(d) < 3:  # too few recorded deliveries: use all warehouses for this part [A]
        d = deliv[(deliv["part_id"] == pid)]
    if len(d) == 0:
        return 7.0, 2.0  # [A] default when no history
    return float(d["lead_time_days"].mean()), float(d["lead_time_days"].std(ddof=1)) if len(d) > 1 else 1.0


def simulate(sales: pd.DataFrame, truth: pd.DataFrame, inv: pd.DataFrame, deliv: pd.DataFrame,
             skus: pd.DataFrame, holdout_start: pd.Timestamp, holdout_days: int, seed: int) -> dict[str, pd.DataFrame]:
    daily, orders, alerts = [], [], []
    analogue = skus[skus["role"] == "analogue_launch"].iloc[0]
    days = pd.date_range(holdout_start, periods=holdout_days, freq="D")
    for sku_i, sku in enumerate(skus.itertuples(index=False)):
        for wh_i, wh in enumerate(config.WAREHOUSES):
            units, so = _series(sales, sku.part_id, wh)
            hist = fc.clean_history(units, so).loc[: holdout_start - pd.Timedelta(days=1)]
            dem = truth[(truth["part_id"] == sku.part_id) & (truth["warehouse_id"] == wh)].set_index("date")["true_demand"]
            start_inv = inv[(inv["part_id"] == sku.part_id) & (inv["warehouse_id"] == wh)
                            & (inv["date"] == holdout_start - pd.Timedelta(days=1))]
            on_hand0 = int(start_inv["on_hand"].iloc[0]) if len(start_inv) else 0
            L_hat, sL_hat = _lead_time_estimates(deliv, sku.part_id, wh, holdout_start)
            curve = None
            if sku.role == "launch":   # analogue sales known before the hold-out only (P8)
                a_units, a_so = _series(sales, analogue.part_id, wh)
                a_hist = fc.clean_history(a_units, a_so).loc[: holdout_start - pd.Timedelta(days=1)]
                curve = fc.analogue_curve(a_hist, pd.Timestamp(analogue.launch_date))
            open0 = _open_orders(deliv, sku.part_id, wh, holdout_start)   # P8: orders already on the way
            recent = hist.dropna().iloc[-56:]
            d_avg = float(recent.mean()) if len(recent) and recent.mean() > 0 else (
                float(curve[:28].mean()) if curve is not None else 0.0)
            mm_min, mm_max = d_avg * L_hat, d_avg * (L_hat + MINMAX_COVER_DAYS)
            for pol in ("order_up_to", "min_max"):
                rng = np.random.default_rng(seed + 1000 * sku_i + 10 * wh_i)  # same lead times for both policies
                model = fc.SesWeekday(hist)
                hist_live = hist.copy()
                on_hand, pipeline = on_hand0, dict(open0)
                an_errors: list[float] = []
                for d in days:
                    on_hand += pipeline.pop(d, 0)
                    on_order = sum(pipeline.values())
                    protect = max(1, int(round(L_hat)) + config.REVIEW_PERIOD_DAYS)
                    use_analogue = (curve is not None and
                                    (d - pd.Timestamp(sku.launch_date)).days < ANALOGUE_WEEKS * 7)
                    if use_analogue:
                        fcst = fc.analogue_launch_forecast(curve, pd.Timestamp(sku.launch_date), hist_live,
                                                           d - pd.Timedelta(days=1), protect)
                    else:
                        fcst = model.forecast(d - pd.Timedelta(days=1), protect)
                    d_bar = float(np.mean(fcst))
                    # P8 fix (P7 issue 03): on analogue days sigma comes from the analogue's own
                    # observed one-day errors once there are 7; before that the fallback [A]
                    sigma_d = (float(np.std(an_errors, ddof=1)) if len(an_errors) >= 7 else None) \
                        if use_analogue else model.sigma()
                    if sigma_d is None:
                        sigma_d = math.sqrt(max(d_bar, 1.0))  # Poisson-like fallback [A]
                    alert = policy.low_stock_alert(on_hand, d_bar, L_hat)
                    # D2 (P8): the report's on-hand alert is kept; we also record whether an open
                    # order is due before the shelf runs empty (then the alert needs no new action)
                    cover_days = on_hand / d_bar if d_bar > 0 else float("inf")
                    due = [(a - d).days for a in pipeline if a > d]
                    covered = bool(due) and min(due) <= cover_days
                    if pol == "order_up_to":
                        ss = policy.safety_stock(d_bar, sigma_d, L_hat, sL_hat)
                        S = policy.order_up_to_level(d_bar, L_hat, ss)
                        q = math.ceil(policy.order_quantity(S, on_hand, on_order))  # whole units [A]
                    else:
                        ss, S = float("nan"), mm_max
                        q = math.ceil(mm_max - (on_hand + on_order)) if on_hand + on_order <= mm_min else 0
                    if q > 0:
                        lt = max(1, int(round(rng.normal(sku.lead_time_mean_days, sku.lead_time_sd_days))))
                        if rng.random() < 0.05:
                            lt += 7
                        arrive = d + pd.Timedelta(days=lt)
                        pipeline[arrive] = pipeline.get(arrive, 0) + q
                    else:
                        rng.normal(); rng.random()  # keep random streams aligned between policies
                    demand = int(dem.get(d, 0))
                    sold = min(on_hand, demand)
                    on_hand_start = on_hand
                    on_hand -= sold
                    # P8 fix (P7 issue 03): censoring is decided from what the shop can see -
                    # the shelf is empty at the end of the day - not from the hidden demand
                    stockout_obs = on_hand == 0
                    recorded = None if stockout_obs else float(sold)
                    if use_analogue and recorded is not None:
                        an_errors.append(recorded - float(fcst[0]))
                    model.update(d, recorded)
                    hist_live.loc[d] = np.nan if recorded is None else recorded
                    daily.append((sku.part_id, wh, sku.role, pol, d, on_hand_start, demand, sold, on_hand,
                                  sold < demand, q, d_bar))
                    orders.append((sku.part_id, wh, pol, d, round(d_bar, 4), round(sigma_d, 4), round(L_hat, 4),
                                   round(sL_hat, 4), None if math.isnan(ss) else round(ss, 3), round(S, 3),
                                   on_hand_start + on_order, q))
                    if alert and pol == "order_up_to":
                        alerts.append((sku.part_id, wh, d, on_hand_start, round(d_bar, 3), round(L_hat, 2),
                                       "LOW_STOCK", round(on_hand_start / d_bar, 2), covered))
    return {
        "fa_sim_daily": pd.DataFrame(daily, columns=["part_id", "warehouse_id", "role", "policy", "date",
                                                     "on_hand_start", "demand", "sales", "on_hand_end",
                                                     "stockout", "order_qty", "d_bar"]),
        "fa_order": pd.DataFrame(orders, columns=["part_id", "warehouse_id", "policy", "date", "d_bar", "sigma_d",
                                                  "lead_time", "sigma_lead_time", "safety_stock",
                                                  "order_up_to_level", "inventory_position", "order_qty"]),
        "fa_alert": pd.DataFrame(alerts, columns=["part_id", "warehouse_id", "date", "on_hand", "d_bar",
                                                  "lead_time", "alert_type", "days_of_cover",
                                                  "open_order_due_in_time"]),
    }


def run(run_id: str) -> dict:
    st = load_settings(run_id)
    sales = lake.read_table("cleaned", "sales_daily", run_id)
    skus = lake.read_table("cleaned", "fa_sku", run_id)
    inv = lake.read_table("cleaned", "inventory_daily", run_id)
    deliv = lake.read_table("cleaned", "supplier_deliveries", run_id)
    end = sales["date"].max()
    holdout_start = end - pd.Timedelta(days=st["fa_holdout_days"] - 1)
    fcst = backtest(sales, skus, end)
    keys = {"forecast": ["part_id", "warehouse_id", "origin_date", "target_date", "method"],
            "fa_order": ["part_id", "warehouse_id", "policy", "date"],
            "fa_sim_daily": ["part_id", "warehouse_id", "policy", "date"],
            "fa_alert": ["part_id", "warehouse_id", "date"]}   # P8 (P7 issue 07): declared keys
    assert not fcst.duplicated(keys["forecast"]).any(), "forecast key not unique"
    lake.write_table(fcst, "outputs", "forecast", run_id, source="cleaned/sales_daily (SYNTHETIC recorded sales only)",
                     extra={"key": keys["forecast"]})
    # The world in the simulation needs the true demand (customers); the policy never reads it.
    truth = lake.read_table("cleaned", "sales_true_demand", run_id)
    sim = simulate(sales, truth, inv, deliv, skus, holdout_start, st["fa_holdout_days"], st["seed"])
    for name, df in sim.items():
        assert not df.duplicated(keys[name]).any(), f"{name} key not unique"
        lake.write_table(df, "outputs", name, run_id, source="step09 simulation (SYNTHETIC)",
                         extra={"holdout_start": str(holdout_start), "key": keys[name]})
    return {"forecast_rows": len(fcst), "holdout_start": str(holdout_start.date()),
            "alerts": len(sim["fa_alert"]), "orders": int((sim["fa_order"]["order_qty"] > 0).sum())}
