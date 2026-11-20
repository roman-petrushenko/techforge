"""Run the TechForge Step 6 prototype pipeline (Table 12 steps 1-11).

Examples
  python run_pipeline.py --run-id demo                 # all steps
  python run_pipeline.py --run-id demo --step 8        # one step (earlier outputs of the run must exist)
  python run_pipeline.py --run-id demo --step 7-10     # a range
  python run_pipeline.py --run-id demo --engine spark  # step 7 with PySpark (optional)
  python run_pipeline.py --run-id quick --small        # small synthetic data (about 1 minute)

Re-running a run ID replaces that run's outputs (no duplicates).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from techforge import (lake, step01_load_raw, step02_generate, step03_clean_parts, step04_curate_parts,
                       step05_clean_events, step06_builds, step07_cooccurrence, step08_replay, step09_focus_a,
                       step10_evaluate, step11_dashboard)
from techforge.step02_generate import GeneratorSettings, small_settings

STEP_NAMES = {
    1: "Load pc-part-dataset files unchanged (raw)",
    2: "Generate SYNTHETIC events, stock, sales, lead times (raw)",
    3: "Parse, normalise, deduplicate parts (cleaned)",
    4: "Add socket/memory type from reference tables; log gaps (curated)",
    5: "Validate, deduplicate, sort events, stock, sales (cleaned + quarantine)",
    6: "Assemble builds from events (curated)",
    7: "Count co-occurrences on the training period (outputs, model)",
    8: "Replay events: rule check -> filters -> ranking (outputs)",
    9: "Forecast -> order-up-to -> orders and alerts (outputs)",
    10: "Run the evaluation scripts (outputs)",
    11: "Query the outputs for the dashboard (outputs)",
}


def parse_steps(text: str) -> list[int]:
    if text == "all":
        return list(STEP_NAMES)
    steps: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            steps += list(range(int(a), int(b) + 1))
        else:
            steps.append(int(part))
    return steps


def run(run_id: str, steps: list[int], engine: str = "pandas", part_key: str = "extended",
        settings: GeneratorSettings | None = None, labels_dir: Path | None = None,
        simulate_model_failure: bool = False, verbose: bool = True) -> dict[int, object]:
    settings = settings or GeneratorSettings()
    settings.part_key = part_key
    results: dict[int, object] = {}
    actions = {
        1: lambda: step01_load_raw.run(run_id),
        2: lambda: step02_generate.run(run_id, settings),
        3: lambda: step03_clean_parts.run(run_id, part_key),
        4: lambda: step04_curate_parts.run(run_id),
        5: lambda: step05_clean_events.run(run_id),
        6: lambda: step06_builds.run(run_id),
        7: lambda: step07_cooccurrence.run(run_id, engine, simulate_model_failure),
        8: lambda: step08_replay.run(run_id),
        9: lambda: step09_focus_a.run(run_id),
        10: lambda: step10_evaluate.run(run_id, labels_dir),
        11: lambda: step11_dashboard.run(run_id),
    }
    for s in steps:
        t0 = time.perf_counter()
        results[s] = actions[s]()
        if verbose:
            print(f"[run {run_id}] step {s:>2} {STEP_NAMES[s]:<70} {time.perf_counter() - t0:6.1f} s")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default=datetime.now().strftime("run%Y%m%d_%H%M%S"))
    p.add_argument("--step", default="all", help="all | 8 | 3,4 | 7-10")
    p.add_argument("--engine", choices=["pandas", "spark"], default="pandas", help="back-end for step 7")
    p.add_argument("--part-key", choices=["report", "extended"], default="extended",
                   help="extended = name + variant fields (default, Table 13 amended in P8, decision D1); "
                        "report = category + name (original key, for comparison)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--small", action="store_true", help="small synthetic data for a quick run")
    p.add_argument("--inject-from", choices=["generator", "labels"], default="generator")
    p.add_argument("--labels-file", default=None, help="resolved labels CSV (for --inject-from labels)")
    p.add_argument("--labels-dir", default=None, help="folder with label_evaluation.json for step 10")
    p.add_argument("--simulate-model-failure", action="store_true", help="test: step 7 fails, last good model kept")
    p.add_argument("--lake", default=None, help="lake folder (default ./lake)")
    a = p.parse_args(argv)
    if a.lake:
        lake.set_root(a.lake)
    settings = small_settings() if a.small else GeneratorSettings()
    if a.seed is not None:
        settings.seed = a.seed
    settings.inject_from, settings.labels_file = a.inject_from, a.labels_file
    run(a.run_id, parse_steps(a.step), a.engine, a.part_key, settings,
        Path(a.labels_dir) if a.labels_dir else None, a.simulate_model_failure)
    print(f"catalog: {lake.root() / 'catalog' / f'run_{a.run_id}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
