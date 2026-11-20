"""End-to-end mechanics checks of Table 11 [R] on a small SYNTHETIC run (mini_run fixture).

These show that the implementation behaves as specified. They are NOT accuracy evidence.
"""
import json

import pandas as pd

from techforge import lake
from techforge.step10_evaluate import coverage_and_speed, suggestion_validity


def test_all_steps_wrote_outputs_and_catalog(mini_run):
    lake.set_root(mini_run["lake"])
    cat = lake.read_catalog(mini_run["run_id"])
    for key in ("raw/synthetic_settings", "cleaned/builder_event", "quarantine/builder_event", "curated/dim_part",
                "curated/attribute_coverage", "curated/build_item", "outputs/model_cooccurrence",
                "outputs/compat_check", "outputs/reco_next_part", "outputs/replay_log", "outputs/forecast",
                "outputs/fa_order", "outputs/fa_alert", "outputs/evaluation_results"):
        assert key in cat["entries"], key
    assert cat["entries"]["raw/synthetic_settings"]["settings"]["seed"] == 7


def test_suggestions_never_fail_unknown_or_out_of_stock(mini_run):
    lake.set_root(mini_run["lake"])
    v = suggestion_validity(mini_run["run_id"])
    assert v["suggestions"] > 0
    assert v["fail"] == 0 and v["unknown"] == 0 and v["out_of_stock"] == 0


def test_replay_fills_five_whenever_the_specification_allows(mini_run):
    """Mechanics: every served request returns min(5, eligible candidates) - never fewer."""
    lake.set_root(mini_run["lake"])
    lg = lake.read_table("outputs", "replay_log", mini_run["run_id"])
    served = lg[lg["status"].isin(["ok", "ok_partial"])]
    assert len(served) > 0
    expected = served["eligible"].clip(upper=5)
    assert (served["n_suggestions"] == expected).all()
    feasible = served[served["eligible"] >= 5]
    assert (feasible["n_suggestions"] == 5).mean() >= 0.90     # Table 11 target where the data allows five


def test_coverage_is_reported_and_time_per_event_below_200ms(mini_run):
    lake.set_root(mini_run["lake"])
    cs = coverage_and_speed(mini_run["run_id"])
    # The Table 11 coverage value itself is data-dependent and is reported (met / not met) by step 10.
    assert 0 <= cs["share_5_fresh_stock"] <= 1
    assert cs["p95_ms"] < 200, cs
    res = lake.read_table("outputs", "evaluation_results", mini_run["run_id"])
    assert "Coverage" in set(res["test"]) and "Speed" in set(res["test"])


def test_stale_snapshot_events_get_no_suggestion(mini_run):
    lake.set_root(mini_run["lake"])
    lg = lake.read_table("outputs", "replay_log", mini_run["run_id"])
    stale = lg[lg["status"] == "stale_stock"]
    assert len(stale) > 0                                   # the generator's feed outage is visible
    assert (stale["n_suggestions"] == 0).all()
    fresh = lg[lg["status"].isin(["ok", "ok_partial"])]
    assert (fresh["stock_age_min"] <= 60).all()


def test_injected_builds_are_flagged(mini_run):
    lake.set_root(mini_run["lake"])
    det = lake.read_table("outputs", "eval_injected_detection", mini_run["run_id"])
    inj = det[det["group"] == "injected incompatible"]
    # P8: every injected build is now accounted for; builds that never reached six parts are
    # listed as such, and among complete ones none is missed (PASS)
    complete = inj[~inj["verdict"].isin(["incomplete in replay", "not replayed (training period only)"])]
    assert len(complete) > 0 and set(complete["verdict"]) <= {"FAIL", "UNKNOWN"}


def test_cooccurrence_uses_training_period_only(mini_run):
    lake.set_root(mini_run["lake"])
    items = lake.read_table("curated", "build_item", mini_run["run_id"])
    model = lake.read_table("outputs", "model_cooccurrence", mini_run["run_id"])
    replay_ids = set(items.loc[items["period"] == "replay", "part_id"])
    train_builds = items.loc[items["period"] == "train", "build_id"].nunique()
    assert len(replay_ids) > 0 and len(model) > 0
    entry = lake.read_catalog(mini_run["run_id"])["entries"]["outputs/model_cooccurrence"]
    assert entry["n_builds"] == train_builds                 # only training builds were counted


def test_quarantine_counted(mini_run):
    lake.set_root(mini_run["lake"])
    log = lake.read_table("cleaned", "event_validation_log", mini_run["run_id"]).set_index("reason")["count"]
    assert log.get("duplicate_event_id", 0) > 0 and log.get("accepted", 0) > 0


def test_forecast_tables_hold_no_true_demand(mini_run):
    lake.set_root(mini_run["lake"])
    f = lake.read_table("outputs", "forecast", mini_run["run_id"])
    assert "true_demand" not in f.columns
    o = lake.read_table("outputs", "fa_order", mini_run["run_id"])
    pol = o[o["policy"] == "order_up_to"].dropna(subset=["safety_stock"])
    # S = d_bar * (L + 1) + SS holds row by row
    # (stored values are rounded, hence the tolerance)
    assert ((pol["d_bar"] * (pol["lead_time"] + 1) + pol["safety_stock"] - pol["order_up_to_level"]).abs() < 0.05).all()
