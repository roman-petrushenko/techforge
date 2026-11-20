"""Focus A formulas on a hand-computed example (Section 4.4 [R]) and forecast hygiene."""
import math

import numpy as np
import pandas as pd

from techforge import forecasting as fc, policy


def test_policy_hand_computed_example():
    # d_bar = 10/day, sigma_d = 3, L = 4 days, sigma_L = 1 day, R = 1, z = 1.65
    # SS = 1.65 * sqrt(5*9 + 100*1) = 1.65 * sqrt(145) = 19.8687...
    ss = policy.safety_stock(d_bar=10, sigma_d=3, lead_time=4, sigma_lead_time=1)
    assert math.isclose(ss, 1.65 * math.sqrt(145), rel_tol=1e-12)
    assert round(ss, 3) == 19.869
    S = policy.order_up_to_level(10, 4, ss)                 # 10*5 + 19.869 = 69.869
    assert round(S, 3) == 69.869
    assert round(policy.order_quantity(S, on_hand=25, on_order=15), 3) == 29.869
    assert policy.order_quantity(S, on_hand=60, on_order=20) == 0.0
    assert policy.low_stock_alert(on_hand=35, d_bar=10, lead_time=4) is True     # 3.5 days < 4
    assert policy.low_stock_alert(on_hand=40, d_bar=10, lead_time=4) is False    # 4 days, not < 4
    assert policy.low_stock_alert(on_hand=0, d_bar=0, lead_time=4) is False


def test_wape_and_bias_hand_example():
    f, a = np.array([10, 0, 5]), np.array([8, 2, 5])
    assert math.isclose(fc.wape(f, a), 4 / 15)
    assert math.isclose(fc.bias(f, a), 0.0)


def test_stockout_days_excluded_and_true_demand_never_used():
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    sales = pd.Series(np.r_[np.full(50, 5.0), np.zeros(10)], index=days)
    stockout = pd.Series(np.r_[np.zeros(50, bool), np.ones(10, bool)], index=days)
    hist = fc.clean_history(sales, stockout)
    assert hist.iloc[-10:].isna().all()                     # stock-out days removed
    m = fc.SesWeekday(hist)
    assert math.isclose(m.forecast(days[-1], 1)[0], 5.0, rel_tol=0.05)   # zeros of stock-out days ignored
    # The forecast functions take recorded sales only; changing a 'true demand' series cannot change them.
    sn1 = fc.seasonal_naive(hist, days[-1], 7)
    sn2 = fc.seasonal_naive(hist.copy(), days[-1], 7)
    assert (sn1 == sn2).all()


def test_seasonal_naive_skips_stockout_weeks():
    days = pd.date_range("2026-03-02", periods=21, freq="D")   # Monday start
    s = pd.Series(np.arange(21, dtype=float), index=days)
    s.iloc[14] = np.nan                                        # last Monday was a stock-out
    f = fc.seasonal_naive(s, days[-1], 1)                      # target = next Monday
    assert f[0] == 7.0                                         # takes the Monday two weeks earlier
