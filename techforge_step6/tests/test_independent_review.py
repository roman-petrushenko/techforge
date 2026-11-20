"""Independent Step 4 contract checks. Run from the extracted project root."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_labels import cohen_kappa
from techforge import forecasting, parts, policy, reference, rules
from techforge import step08_replay as replay
from techforge import step09_focus_a as focus
from techforge.step03_clean_parts import merge_duplicates
from techforge.step05_clean_events import validate_events
from techforge.step06_builds import assemble
from techforge.step07_cooccurrence import count_cooccurrence, validate_model
from test_recommender import T0, rec, stock


def test_unresolved_generation():
    assert reference.cpu_socket("Unidentified model", "Zen 4")[0] is None


def test_incomplete_model_name():
    assert reference.cpu_socket("AMD Ryzen 7 engineering sample", "Zen 4")[0] is None


def test_contradictory_board():
    kind, _ = reference.board_memory_type("Example DDR4", "AM5")
    assert rules.check_r2({"memory_type": "DDR4"}, {"memory_type": kind}).verdict == "UNKNOWN"


def test_unknown_sentinel():
    assert rules.check_r2({"memory_type": "UNKNOWN"}, {"memory_type": "UNKNOWN"}).verdict == "UNKNOWN"


def test_exact_power():
    assert rules.check_r4({"tdp_w": 65}, {"board_power_w": 135}, {"wattage_w": 375}).verdict == "PASS"


@pytest.mark.parametrize("age,allowed", [(3600, True), (3601, False)])
def test_snapshot_boundary(age, allowed):
    suggestions, _ = rec().suggest({"cpu": "CPU-a"}, "motherboard", "KYIV", T0,
                                    stock(at=T0 - pd.Timedelta(seconds=age)))
    assert bool(suggestions) == allowed


def event(pid, category, when, arrival):
    return dict(build_id="b", session_id="s", event_id=pid, category=category, part_id=pid,
                event_ts=when, ingest_ts=arrival, warehouse_id="KYIV", action="add", late=arrival-when > pd.Timedelta(hours=1))


def test_straddling_build():
    rows = [event("a", "cpu", T0, T0), event("b", "memory", T0 + pd.Timedelta(minutes=2), T0 + pd.Timedelta(minutes=2))]
    items = assemble(pd.DataFrame(rows), T0 + pd.Timedelta(minutes=1))
    pairs, pop, _ = count_cooccurrence(items[items.period == "train"])
    assert "b" not in set(pop.part_id) and pairs.empty


def test_late_replacement(monkeypatch):
    hours = lambda n: T0 + pd.Timedelta(hours=n)
    rows = [event("old", "cpu", hours(0), hours(2)), event("new", "cpu", hours(1), hours(1)),
            event("ram", "memory", hours(3), hours(3))]
    tables = {"builder_event": pd.DataFrame(rows), "dim_part": pd.DataFrame(rows).drop_duplicates("part_id"),
              "assortment": pd.DataFrame(rows), "stock_snapshot": pd.DataFrame(columns=["warehouse_id", "snapshot_ts"])}
    outputs = {}
    monkeypatch.setattr(replay.lake, "read_table", lambda zone, name, run: tables[name])
    monkeypatch.setattr(replay.lake, "write_table", lambda df, zone, name, *a, **k: outputs.update({name: df}))
    monkeypatch.setattr(replay, "load_model", lambda _: None)
    monkeypatch.setattr(replay, "load_settings", lambda _: {"train_end": T0})
    replay.run("review")
    assert json.loads(outputs["replay_log"].iloc[-1].build_state)["cpu"] == "new"


def test_empty_training():
    pairs, pop, n = count_cooccurrence(pd.DataFrame(columns=["build_id", "part_id", "category"]))
    assert "no training builds" in validate_model(pairs, pop, n)


def test_no_demand():
    assert math.isnan(forecasting.wape(np.array([1]), np.array([0])))
    assert not policy.low_stock_alert(0, 0, 7)


def test_constant_zero_lead():
    assert policy.safety_stock(5, 0, 0, 0) == 0
    assert policy.order_quantity(policy.order_up_to_level(5, 0, 0), 1, 2) == 2


def test_conflicting_parts():
    raw = pd.DataFrame({"name": ["A", " a "], "price": [30, 31], "wattage": [350, 850],
                        "type": ["ATX"]*2, "efficiency": ["Gold"]*2})
    # P8 amendment: this case tests the original report key; with the extended key (default since
    # D1) the two wattages are two different parts - see test_extended_key_separates_variants
    merged, conflicts = merge_duplicates(parts.add_part_id(parts.parse_category(raw, "power-supply"),
                                                            "power-supply", "report"), "power-supply")
    assert len(merged) == 1 and pd.isna(merged.wattage_w.iloc[0])
    assert conflicts.attribute.tolist() == ["wattage_w"]


def test_degenerate_kappa():
    assert math.isnan(cohen_kappa(["yes"]*5, ["yes"]*5))


def test_non_object_json():
    good, bad = validate_events([("batch", "[]")], set())
    assert good.empty and len(bad) == 1


def test_nat_timestamp():
    row = event("a", "cpu", T0, T0)
    row["event_ts"] = "NaT"
    good, bad = validate_events([("batch", json.dumps(row, default=str))], {"a"})
    assert good.empty and len(bad) == 1


def test_unreceived_lead_time():
    df = pd.DataFrame({"part_id": ["a"]*4, "warehouse_id": ["KYIV"]*4,
                       "order_date": [T0-pd.Timedelta(days=n) for n in [30, 20, 10, 1]],
                       "lead_time_days": [2, 2, 2, 90]})
    assert focus._lead_time_estimates(df, "a", "KYIV", T0) == (2, 0)


def test_future_analogue_sales(monkeypatch):
    monkeypatch.setattr(focus.config, "WAREHOUSES", ["KYIV"])
    monkeypatch.setattr(focus, "FIRST_ORIGIN_DAY", 9)
    days = pd.date_range("2025-01-01", periods=17)
    skus = pd.DataFrame([("old", "analogue_launch", days[5]), ("new", "launch", days[8])],
                        columns=["part_id", "role", "launch_date"])
    sales = pd.concat([pd.DataFrame({"part_id": pid, "warehouse_id": "KYIV", "date": days,
                                     "units_sold": 3, "stockout_flag": False}) for pid in skus.part_id])
    def predictions():
        frame = focus.backtest(sales, skus, days[-1])
        return frame.loc[frame.method == "analogue_launch", "forecast"].tolist()
    before = predictions()
    sales.loc[(sales.part_id == "old") & (sales.date > days[9]), "units_sold"] = 300
    assert predictions() == before


def test_label_template_injection(mini_run, monkeypatch, tmp_path):
    from pathlib import Path
    from techforge import lake, step02_generate as generator

    lake.set_root(mini_run["lake"])
    template = Path(generator.__file__).parents[1] / "labels" / "labels_resolved_TEMPLATE.csv"
    labels = pd.read_csv(template).iloc[:1].copy()
    labels["final_compatible"] = "no"  # invented fixture, never accuracy evidence
    labels_file = tmp_path / "resolved.csv"
    labels.to_csv(labels_file, index=False)
    settings = generator.small_settings()
    settings.inject_from, settings.labels_file = "labels", str(labels_file)
    settings.incompatible_share = 1
    settings.train_days, settings.replay_days = 1, 1
    raw = generator._load_raw_parts("mini", settings.part_key)
    rng = np.random.default_rng(7)
    assortment = generator.build_assortment(raw, settings, rng)
    _, truth = generator.generate_builds(assortment, raw, settings, rng, pd.read_csv(labels_file))
    # P8 amendment: labelled pairs are reserved for replay days (ChatGPT's own fix for issue 04),
    # so only the injected builds are checked here
    assert set(truth.loc[truth.injected.astype(bool), "inject_source"]) == {"labels"}


def test_extended_key_separates_variants():
    """P8 (decision D1): with the extended key a 350 W and an 850 W unit of the same name are two
    parts, each with a known wattage."""
    raw = pd.DataFrame({"name": ["A", " a "], "price": [30, 31], "wattage": [350, 850],
                        "type": ["ATX"]*2, "efficiency": ["Gold"]*2})
    merged, conflicts = merge_duplicates(parts.add_part_id(parts.parse_category(raw, "power-supply"),
                                                            "power-supply"), "power-supply")
    assert len(merged) == 2 and sorted(merged.wattage_w) == [350, 850] and conflicts.empty
