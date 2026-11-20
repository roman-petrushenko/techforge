"""Step 7 - Count co-occurrences on the training period (Table 12 [R]); daily batch.

lift(i, j) = co(i, j) * N / (n_i * n_j)   - how much more often i and j are in the same
build than expected by chance [R].  Only builds that started before train_end are used.

One function, two back-ends (team decision 5):
  engine="pandas" (default) and engine="spark" (optional, same logic as a Glue Spark job).

Model storage (outputs zone):  models/cooccurrence/<model_version>/{pairs,popularity}.csv
LATEST_GOOD.json points to the last version that passed validation; a failed job keeps
the previous pointer (Section 4.6 [R]).  Done when: model version saved.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from techforge import lake

MODEL_DIR_NAME = "models/cooccurrence"


def count_cooccurrence(build_items: pd.DataFrame, engine: str = "pandas") -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (pairs with co/lift in both directions, popularity per part, number of builds)."""
    if engine == "spark":
        return _count_spark(build_items)
    if engine != "pandas":
        raise ValueError("engine must be 'pandas' or 'spark'")
    bi = build_items[["build_id", "part_id", "category"]].drop_duplicates(["build_id", "part_id"])
    n_builds = bi["build_id"].nunique()
    pop = bi.groupby(["part_id", "category"]).size().rename("n_builds").reset_index()
    j = bi.merge(bi, on="build_id", suffixes=("_i", "_j"))
    j = j[j["category_i"] != j["category_j"]]
    pairs = j.groupby(["part_id_i", "part_id_j", "category_j"]).size().rename("co").reset_index()
    pairs = _add_lift(pairs, pop, n_builds)
    return pairs, pop, n_builds


def _add_lift(pairs: pd.DataFrame, pop: pd.DataFrame, n_builds: int) -> pd.DataFrame:
    n = pop.set_index("part_id")["n_builds"]
    pairs = pairs.copy()
    pairs["n_i"] = pairs["part_id_i"].map(n)
    pairs["n_j"] = pairs["part_id_j"].map(n)
    pairs["lift"] = pairs["co"] * n_builds / (pairs["n_i"] * pairs["n_j"])
    return pairs


def _count_spark(build_items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Same logic with PySpark (optional; requires pyspark and Java)."""
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.master("local[*]").appName("techforge-cooccurrence").getOrCreate()
    try:
        bi = spark.createDataFrame(build_items[["build_id", "part_id", "category"]]).dropDuplicates(["build_id", "part_id"])
        n_builds = bi.select("build_id").distinct().count()
        pop = bi.groupBy("part_id", "category").agg(F.count("*").alias("n_builds"))
        a = bi.select(F.col("build_id"), F.col("part_id").alias("part_id_i"), F.col("category").alias("category_i"))
        b = bi.select(F.col("build_id"), F.col("part_id").alias("part_id_j"), F.col("category").alias("category_j"))
        pairs = (a.join(b, "build_id").where(F.col("category_i") != F.col("category_j"))
                 .groupBy("part_id_i", "part_id_j", "category_j").agg(F.count("*").alias("co")))
        pop_pd = pop.toPandas()
        pairs_pd = pairs.toPandas()
    finally:
        spark.stop()
    return _add_lift(pairs_pd, pop_pd, n_builds), pop_pd, n_builds


def model_root() -> Path:
    return lake.root() / "outputs" / MODEL_DIR_NAME


def validate_model(pairs: pd.DataFrame, pop: pd.DataFrame, n_builds: int) -> list[str]:
    problems = []
    if n_builds == 0 or pop.empty:
        problems.append("no training builds")
    if pairs.empty:
        problems.append("no pairs")
    elif not pairs["lift"].between(0, float("inf"), inclusive="neither").all():
        problems.append("lift not finite and positive")
    return problems


def run(run_id: str, engine: str = "pandas", simulate_failure: bool = False) -> dict:
    items = lake.read_table("curated", "build_item", run_id)
    train = items[items["period"] == "train"]
    from techforge.step02_generate import load_settings
    train_end = pd.Timestamp(load_settings(run_id)["train_end"])
    version = f"cooc-{run_id}-{pd.Timestamp(train_end):%Y%m%d}-{engine}"
    status = {"model_version": version, "engine": engine, "status": "failed", "problems": []}
    try:
        if simulate_failure:
            raise RuntimeError("simulated failure of the daily job (test of 'keep last good model')")
        pairs, pop, n_builds = count_cooccurrence(train, engine)
        problems = validate_model(pairs, pop, n_builds)
        if problems:
            raise ValueError("; ".join(problems))
    except Exception as exc:  # failed job keeps the last good version [R]
        status["problems"] = [str(exc)]
        status["kept_version"] = latest_good_version()
        _write_status(run_id, status)
        return status
    # P8 fix (P7 issue 08): write to a staging folder, check it, then switch; the old version
    # stays intact until the new one is complete.
    folder = model_root() / version
    staging = model_root() / f".staging-{version}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    pairs.to_csv(staging / "pairs.csv", index=False)
    pop.to_csv(staging / "popularity.csv", index=False)
    meta = {"model_version": version, "n_builds": int(n_builds), "train_end": str(train_end),
            "engine": engine, "run_id": run_id, "synthetic": True, "pairs": len(pairs)}
    (staging / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if len(pd.read_csv(staging / "pairs.csv")) != len(pairs):
        raise IOError("model files incomplete after writing")
    if folder.exists():
        backup = model_root() / f".old-{version}"
        if backup.exists():
            shutil.rmtree(backup)
        folder.rename(backup)
        staging.rename(folder)
        shutil.rmtree(backup)
    else:
        staging.rename(folder)
    (model_root() / "LATEST_GOOD.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lake.write_table(pairs.assign(model_version=version), "outputs", "model_cooccurrence", run_id,
                     source="curated/build_item period=train (SYNTHETIC)", extra=meta)
    status.update({"status": "ok", "n_builds": int(n_builds), "pairs": len(pairs)})
    _write_status(run_id, status)
    return status


def _write_status(run_id: str, status: dict) -> None:
    lake.write_table(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                    for k, v in status.items()}]),
                     "outputs", "model_job_status", run_id, source="step07")


def latest_good_version() -> str | None:
    p = model_root() / "LATEST_GOOD.json"
    return json.loads(p.read_text(encoding="utf-8"))["model_version"] if p.exists() else None


def run_model_version(run_id: str) -> str | None:
    """Model version bound to this run (P8 fix, P7 issue 08): the version step 7 built for
    the run, or the version it kept after a failed job; never another run's newer model."""
    try:
        st = lake.read_table("outputs", "model_job_status", run_id).iloc[-1]
    except (FileNotFoundError, KeyError):
        return None
    if st["status"] == "ok":
        return st["model_version"]
    kept = st.get("kept_version")
    return None if pd.isna(kept) else kept


def load_model(version: str | None = None) -> dict | None:
    """Load the given or the last good model; None if there is none (-> no suggestions)."""
    version = version or latest_good_version()
    if not version or not (model_root() / version / "pairs.csv").exists():
        return None
    folder = model_root() / version
    return {"version": version,
            "pairs": pd.read_csv(folder / "pairs.csv"),
            "popularity": pd.read_csv(folder / "popularity.csv"),
            "meta": json.loads((folder / "meta.json").read_text(encoding="utf-8"))}
