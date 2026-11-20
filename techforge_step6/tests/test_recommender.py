"""Recommender mechanics (Section 4.3 [R]) on a tiny hand-made catalog."""
import pandas as pd

from conftest import part
from techforge.recommender import Recommender, StockView

T0 = pd.Timestamp("2026-10-01 10:00")
CPU = part("CPU-a", "cpu", socket="AM5", tdp_w=105.0)
BOARDS = [
    part("MB-good1", "motherboard", socket="AM5", memory_type="DDR5", form_factor="ATX", price_band="mid"),
    part("MB-good2", "motherboard", socket="AM5", memory_type="DDR5", form_factor="ATX", price_band="mid"),
    part("MB-fail", "motherboard", socket="AM4", memory_type="DDR4", form_factor="ATX", price_band="mid"),
    part("MB-unknown", "motherboard", socket="AM5", memory_type=None, form_factor="ATX", price_band="mid"),
    part("MB-oos", "motherboard", socket="AM5", memory_type="DDR5", form_factor="ATX", price_band="mid"),
    part("MB-high", "motherboard", socket="AM5", memory_type="DDR5", form_factor="ATX", price_band="high"),
]
DIM = pd.DataFrame([CPU] + BOARDS)


def model(lifts: dict[str, float], co: int = 10):
    pairs = pd.DataFrame([{"part_id_i": "CPU-a", "part_id_j": j, "category_j": "motherboard", "co": co, "lift": lf}
                          for j, lf in lifts.items()])
    pop = pd.DataFrame([{"part_id": p["part_id"], "category": p["category"], "n_builds": n}
                        for p, n in zip([CPU] + BOARDS, [50, 5, 4, 30, 30, 30, 99])])
    return {"version": "test-model", "pairs": pairs, "popularity": pop}


def stock(at=T0 - pd.Timedelta(minutes=10), on_hand=None):
    on_hand = on_hand or {"MB-good1": 3, "MB-good2": 2, "MB-fail": 9, "MB-unknown": 9, "MB-oos": 0, "MB-high": 4}
    return StockView(pd.DataFrame([{"part_id": k, "warehouse_id": "KYIV", "snapshot_ts": at, "on_hand": v}
                                   for k, v in on_hand.items()]))


def rec(lifts=None, with_model=True):
    lifts = lifts if lifts is not None else {"MB-fail": 9.0, "MB-unknown": 8.0, "MB-oos": 7.0, "MB-good1": 2.0, "MB-good2": 1.5}
    return Recommender(DIM, [b["part_id"] for b in BOARDS], model(lifts) if with_model else None, min_support=3)


def test_filter_before_ranking_unknown_and_out_of_stock_never_suggested():
    out, info = rec().suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0, stock(), last_added="CPU-a")
    ids = [s.part_id for s in out]
    assert "MB-fail" not in ids        # highest lift but FAIL
    assert "MB-unknown" not in ids     # UNKNOWN part
    assert "MB-oos" not in ids         # out of stock
    assert ids[:2] == ["MB-good1", "MB-good2"] and all(s.reason == "lift" for s in out[:2])
    assert info["after_rules"] == 4 and info["after_stock"] == 3


def test_fallback_same_price_band_only():
    out, _ = rec(lifts={}).suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0, stock(), last_added="CPU-a")
    assert [s.part_id for s in out] == ["MB-good1", "MB-good2"]       # by popularity, band 'mid' only
    assert all(s.reason == "fallback" for s in out)
    assert "MB-high" not in [s.part_id for s in out]


def test_min_support_respected():
    m = model({"MB-good2": 5.0}, co=2)          # co below MIN_SUPPORT -> lift not used
    r = Recommender(DIM, [b["part_id"] for b in BOARDS], m, min_support=3)
    out, _ = r.suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0, stock(), last_added="CPU-a")
    assert all(s.reason == "fallback" for s in out)


def test_stale_stock_gives_no_suggestion():
    old = stock(at=T0 - pd.Timedelta(minutes=61))
    out, info = rec().suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0, old)
    assert out == [] and info["status"] == "stale_stock"
    fresh = stock(at=T0 - pd.Timedelta(minutes=60))     # exactly one hour old is still valid
    assert rec().suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0, fresh)[1]["status"] != "stale_stock"


def test_snapshot_from_the_future_is_not_used():
    future = stock(at=T0 + pd.Timedelta(minutes=5))
    out, info = rec().suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0, future)
    assert out == [] and info["status"] == "stale_stock"


def test_missing_model_gives_no_suggestion():
    out, info = rec(with_model=False).suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0, stock())
    assert out == [] and info["status"] == "model_missing"
