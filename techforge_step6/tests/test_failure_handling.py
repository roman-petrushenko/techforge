"""Failure handling of Section 4.6 [R]: quarantine, re-run replaces, last good model, run ID."""
import json

import pandas as pd

from techforge import lake, step07_cooccurrence as s7
from techforge.step05_clean_events import validate_events


def _line(**kw):
    rec = {"event_id": "E1", "session_id": "S1", "build_id": "B1", "event_ts": "2026-10-01T10:00:00",
           "ingest_ts": "2026-10-01T10:00:05", "action": "add", "category": "cpu", "part_id": "CPU-x",
           "warehouse_id": "KYIV"}
    rec.update(kw)
    return ("f.jsonl", json.dumps(rec))


def test_quarantine_reasons_and_counts():
    lines = [_line(), _line(),                                  # duplicate
             _line(event_id="E2", part_id=None),                 # missing field
             _line(event_id="E3", event_ts="not-a-time"),        # bad timestamp
             _line(event_id="E4", part_id="GPU-unknown"),        # unknown part
             _line(event_id="E5", action="delete"),              # unknown action
             ("f.jsonl", '{"event_id": "E6", "sess')]            # truncated JSON
    ok, q = validate_events(lines, {"CPU-x"})
    assert list(ok["event_id"]) == ["E1"]
    reasons = q["reason"].str.split(":").str[0].value_counts().to_dict()
    assert reasons == {"duplicate_event_id": 1, "missing_field": 1, "bad_timestamp": 1, "unknown_part_id": 1,
                       "unknown_action": 1, "not_valid_json": 1}


def test_rerun_replaces_outputs_and_catalog(tmp_path):
    lake.set_root(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    lake.write_table(df, "outputs", "x", "r1", source="t")
    lake.write_table(df.head(2), "outputs", "x", "r1", source="t")
    assert len(lake.read_table("outputs", "x", "r1")) == 2
    files = list((tmp_path / "outputs" / "x" / "run_id=r1").glob("data.*"))
    assert len(files) == 1
    cat = lake.read_catalog("r1")
    assert cat["entries"]["outputs/x"]["row_count"] == 2 and len(cat["entries"]) == 1
    for key in ("dataset", "zone", "source", "row_count", "run_id", "created_at", "code_version", "reference_version"):
        assert key in cat["entries"]["outputs/x"]


def test_failed_model_job_keeps_last_good_version(mini_run):
    lake.set_root(mini_run["lake"])
    good = s7.latest_good_version()
    assert good is not None
    status = s7.run(mini_run["run_id"], simulate_failure=True)
    assert status["status"] == "failed" and status["kept_version"] == good
    assert s7.latest_good_version() == good
    assert s7.load_model() is not None
