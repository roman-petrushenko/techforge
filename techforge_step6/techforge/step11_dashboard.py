"""Step 11 - Query the outputs for the dashboard (Table 12 [R]).

Writes small summary tables that the Streamlit app and the notebook read. Every table
carries a `data_label` column ("SYNTHETIC" or "real catalog").
Done when (report): screenshots in the report - that part is manual.
"""
from __future__ import annotations

import pandas as pd

from techforge import lake


def run(run_id: str) -> dict[str, int]:
    out = {}
    cc = lake.read_table("outputs", "compat_check", run_id)
    t = cc.groupby(["rule_id", "verdict"]).size().rename("checks").reset_index()
    out["dashboard_rule_verdicts"] = t.assign(data_label="SYNTHETIC events, real catalog")
    cov = lake.read_table("curated", "attribute_coverage", run_id)
    out["dashboard_unknown_share"] = cov.assign(data_label="real catalog")
    dq = lake.read_table("curated", "dq_issue", run_id)
    out["dashboard_dq_reasons"] = (dq.groupby(["category", "attribute", "issue_type"]).size()
                                   .rename("parts").reset_index().assign(data_label="real catalog"))
    out["dashboard_quarantine"] = lake.read_table("cleaned", "event_validation_log", run_id).assign(data_label="SYNTHETIC")
    lg = lake.read_table("outputs", "replay_log", run_id)
    out["dashboard_replay_status"] = (lg.groupby("status").size().rename("events").reset_index()
                                      .assign(data_label="SYNTHETIC"))
    reco = lake.read_table("outputs", "reco_next_part", run_id)
    out["dashboard_reco_reason"] = (reco.groupby(["target_category", "reason"]).size().rename("suggestions")
                                    .reset_index().assign(data_label="SYNTHETIC"))
    for name, df in out.items():
        lake.write_table(df, "outputs", name, run_id, source="step11 dashboard query")
    return {k: len(v) for k, v in out.items()}
