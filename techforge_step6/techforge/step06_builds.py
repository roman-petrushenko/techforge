"""Step 6 - Assemble builds from events (Table 12 [R]).

Output (curated): build_item (one row per build and category, last part added wins),
builds_per_day.  Done when: builds per day are reported.
"""
from __future__ import annotations

import pandas as pd

from techforge import lake
from techforge.step02_generate import load_settings


def assemble(events: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """One row per build, category and period; the last part added (by event time) wins.

    P8 fix (P7 issue 02): events are split at train_end BEFORE assembly, so a part added
    after the cutoff never enters the training rows of a build that started before it.
    Ties on event time are broken by event_id (deterministic).
    """
    ev = events.sort_values(["build_id", "event_ts", "event_id"]).copy()
    ev["period"] = (ev["event_ts"] < train_end).map({True: "train", False: "replay"})
    items = ev.groupby(["build_id", "period", "category"], as_index=False).last()[
        ["build_id", "session_id", "period", "category", "part_id", "event_ts", "warehouse_id"]]
    start = ev.groupby("build_id")["event_ts"].min().rename("build_start_ts")
    items = items.merge(start, on="build_id")
    items["build_date"] = items["build_start_ts"].dt.normalize()
    return items.rename(columns={"event_ts": "added_ts"})


def run(run_id: str) -> pd.DataFrame:
    ev = lake.read_table("cleaned", "builder_event", run_id)
    train_end = pd.Timestamp(load_settings(run_id)["train_end"])
    items = assemble(ev, train_end)
    lake.write_table(items, "curated", "build_item", run_id, source="cleaned/builder_event (SYNTHETIC)",
                     extra={"train_end": str(train_end)})
    per_day = (items.drop_duplicates(["build_id", "period"]).groupby(["build_date", "period"]).size()
               .rename("builds").reset_index())
    lake.write_table(per_day, "curated", "builds_per_day", run_id, source="step06")
    return per_day
