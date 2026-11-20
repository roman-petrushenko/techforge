"""Daily order-up-to policy of the Focus A slice (Section 4.4 [R]).

  SS = z * sqrt((L + R) * sigma_d^2 + d_bar^2 * sigma_L^2)
  S  = d_bar * (L + R) + SS
  order = max(0, S - inventory position),  inventory position = on hand + on order
  low-stock alert when on_hand / d_bar < L
with R = 1 day and z = 1.65 [R].
"""
from __future__ import annotations

import math

from techforge import config


def safety_stock(d_bar: float, sigma_d: float, lead_time: float, sigma_lead_time: float,
                 z: float = config.Z_SERVICE, review: float = config.REVIEW_PERIOD_DAYS) -> float:
    return z * math.sqrt((lead_time + review) * sigma_d ** 2 + d_bar ** 2 * sigma_lead_time ** 2)


def order_up_to_level(d_bar: float, lead_time: float, ss: float,
                      review: float = config.REVIEW_PERIOD_DAYS) -> float:
    return d_bar * (lead_time + review) + ss


def order_quantity(S: float, on_hand: float, on_order: float) -> float:
    return max(0.0, S - (on_hand + on_order))


def low_stock_alert(on_hand: float, d_bar: float, lead_time: float) -> bool:
    """True if stock on hand covers fewer days than the lead time. No alert when d_bar <= 0."""
    if d_bar <= 0:
        return False
    return on_hand / d_bar < lead_time
