"""Forecast methods of the Focus A slice (Section 4.4 [R]).

All methods see only RECORDED sales up to the forecast origin; stock-out days are
excluded (set to NaN) before fitting. True demand is never an input here; it is used
only by `score_forecasts` in the evaluation.

Methods
  seasonal_naive  - baseline [R]: value of the same weekday one week earlier (earlier weeks
                    if that day was a stock-out day)
  ses_weekday     - simple exponential smoothing on weekday-adjusted sales, times the
                    weekday factor [P: weekday factors are a small addition to plain SES]
  analogue_launch - for a launch SKU in its first weeks: the recorded curve of an earlier
                    analogue launch, scaled by the ratio observed so far [optional, R Table 8]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHAS = np.linspace(0.05, 0.95, 19)


def clean_history(sales: pd.Series, stockout: pd.Series) -> pd.Series:
    """Recorded sales with stock-out days removed (NaN)."""
    return sales.where(~stockout.astype(bool))


def seasonal_naive(history: pd.Series, origin: pd.Timestamp, horizon: int) -> np.ndarray:
    """history: date-indexed recorded sales (NaN = excluded) up to and including origin."""
    h = history.loc[:origin]
    out = []
    for k in range(1, horizon + 1):
        target = origin + pd.Timedelta(days=k)
        val = np.nan
        back = target - pd.Timedelta(days=7)
        while back >= h.index.min():
            if back <= origin and back in h.index and not np.isnan(h.loc[back]):
                val = float(h.loc[back])
                break
            back -= pd.Timedelta(days=7)
        out.append(0.0 if np.isnan(val) else val)
    return np.array(out)


def weekday_factors(history: pd.Series, min_obs: int = 3) -> np.ndarray:
    h = history.dropna()
    if len(h) < 14 or h.mean() <= 0:
        return np.ones(7)
    by_wd = h.groupby(h.index.weekday).agg(["mean", "size"])
    f = np.ones(7)
    for wd, row in by_wd.iterrows():
        if row["size"] >= min_obs:
            f[wd] = row["mean"] / h.mean()
    f = np.where(f <= 0, 1.0, f)
    return f * 7 / f.sum()


def fit_ses(y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit SES on y (no NaN) by minimising one-step squared error over a grid of alphas.
    Returns (alpha, final level, one-step errors for the best alpha)."""
    if len(y) == 0:
        return 0.3, 0.0, np.array([])
    level = np.full(len(ALPHAS), y[0], dtype=float)
    sse = np.zeros(len(ALPHAS))
    errs = np.zeros((len(ALPHAS), len(y)))
    for t in range(1, len(y)):
        e = y[t] - level
        errs[:, t] = e
        sse += e ** 2
        level = level + ALPHAS * e
    best = int(np.argmin(sse)) if len(y) > 1 else 6
    return float(ALPHAS[best]), float(level[best]), errs[best, 1:]


class SesWeekday:
    """SES on weekday-adjusted recorded sales; can be updated day by day (used in the simulation)."""

    def __init__(self, history: pd.Series):
        self.factors = weekday_factors(history)
        h = history.dropna()
        adj = (h / self.factors[h.index.weekday]).to_numpy(dtype=float)
        self.alpha, self.level, e_adj = fit_ses(adj)
        # one-step errors in original units (for sigma_d)
        self.errors = list(e_adj * self.factors[h.index.weekday][1:]) if len(h) > 1 else []

    def forecast(self, origin: pd.Timestamp, horizon: int) -> np.ndarray:
        days = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        return np.maximum(0.0, self.level * self.factors[days.weekday])

    def update(self, day: pd.Timestamp, recorded: float | None) -> None:
        """Add one day of recorded sales; None = stock-out day (excluded)."""
        if recorded is None or np.isnan(recorded):
            return
        pred = self.level * self.factors[day.weekday()]
        self.errors.append(recorded - pred)
        self.level += self.alpha * (recorded / self.factors[day.weekday()] - self.level)

    def sigma(self, window: int = 56) -> float | None:
        e = np.array(self.errors[-window:])
        return float(e.std(ddof=1)) if len(e) >= 14 else None


def analogue_curve(analogue_history: pd.Series, analogue_launch: pd.Timestamp, max_age: int = 120) -> np.ndarray:
    """Recorded sales of the analogue SKU by days since its launch (stock-out days interpolated)."""
    ages = (analogue_history.index - analogue_launch).days
    s = pd.Series(analogue_history.to_numpy(), index=ages)
    s = s[(s.index >= 0) & (s.index < max_age)].reindex(range(max_age))
    return s.interpolate(limit_direction="both").fillna(0).to_numpy()


def analogue_launch_forecast(curve: np.ndarray, launch: pd.Timestamp, history: pd.Series,
                             origin: pd.Timestamp, horizon: int) -> np.ndarray:
    """Analogue curve at the target ages, scaled by observed/analogue sales so far (scale 1 before launch).
    The launch date is known in advance from the launch calendar (F10 [R])."""
    h = history.loc[launch:origin].dropna()
    ages_seen = (h.index - launch).days
    scale = 1.0
    if len(h) >= 3:
        base = curve[np.clip(ages_seen, 0, len(curve) - 1)].sum()
        if base > 0:
            scale = float(h.sum() / base)
    ages = (pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D") - launch).days.to_numpy()
    vals = np.where(ages < 0, 0.0, curve[np.clip(ages, 0, len(curve) - 1)] * scale)
    return np.maximum(vals, 0.0)


def wape(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Sum of absolute errors as a share of the true demand [R Table 11]."""
    denom = np.abs(actual).sum()
    return float(np.abs(forecast - actual).sum() / denom) if denom > 0 else float("nan")


def bias(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Sum of (forecast - actual) as a share of the true demand; > 0 = over-forecast."""
    denom = np.abs(actual).sum()
    return float((forecast - actual).sum() / denom) if denom > 0 else float("nan")
