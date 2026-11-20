"""Smoke tests for the app back-end and the report figures (the Streamlit UI itself is tested manually)."""
from techforge import app_data, lake, report_figures


def test_builder_backend(mini_run):
    lake.set_root(mini_run["lake"])
    be = app_data.BuilderBackend(mini_run["run_id"])
    cpu = be.options("cpu", assortment_only=True)["part_id"].iloc[0]
    verdict, message, results = be.check({"cpu": cpu})
    assert verdict == "NOT_CHECKED" and results == []
    at = be.snapshot_times("KYIV")[30]
    table, info = be.suggest({"cpu": cpu}, "motherboard", "KYIV", at)
    assert info["status"] in {"ok", "ok_partial"}
    assert len(table) <= 5


def test_report_figures_written(mini_run, tmp_path):
    lake.set_root(mini_run["lake"])
    paths = report_figures.write_all(mini_run["run_id"], tmp_path)
    assert sum(p.suffix == ".png" for p in paths) == 9
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
