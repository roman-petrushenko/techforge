"""Local data lake: zone folders, table read/write and the JSON run catalog.

Layout:  lake/<zone>/<dataset>/run_id=<run_id>/data.parquet
Catalog: lake/catalog/run_<run_id>.json  (one entry per zone/dataset; data lineage for Step 7)

Re-running a step for the same run_id deletes and rewrites its folders, so outputs
are replaced instead of duplicated (Section 4.6 [R]).

Parquet needs pyarrow (in requirements.txt). If pyarrow is missing the module falls
back to CSV and records the format in the catalog, so the pipeline still runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from techforge import config

_root: Path = Path(os.environ.get("TECHFORGE_LAKE", str(config.LAKE_ROOT)))


def set_root(path: str | Path) -> None:
    """Point the lake at another folder (used by tests)."""
    global _root
    _root = Path(path)


def root() -> Path:
    return _root


def storage_format() -> str:
    """'parquet' when pyarrow is installed, otherwise 'csv' (fallback)."""
    forced = os.environ.get("TECHFORGE_FORMAT")
    if forced in ("parquet", "csv"):
        return forced
    try:
        import pyarrow  # noqa: F401
        return "parquet"
    except ImportError:
        return "csv"


def table_dir(zone: str, dataset: str, run_id: str) -> Path:
    if zone not in config.ZONES:
        raise ValueError(f"unknown zone {zone}")
    return _root / zone / dataset / f"run_id={run_id}"


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def code_version() -> str:
    """Hash of the techforge package source; stored with every table."""
    return _hash_files(list((config.ROOT / "techforge").glob("*.py")))


def reference_version() -> str:
    """Hash of the reference CSV files; changes whenever a row is edited or verified."""
    return _hash_files(list(config.REFERENCE_DIR.glob("ref_*.csv")))


# ----------------------------------------------------------------- catalog
def _catalog_path(run_id: str) -> Path:
    return _root / "catalog" / f"run_{run_id}.json"


def read_catalog(run_id: str) -> dict[str, Any]:
    p = _catalog_path(run_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"run_id": run_id, "created_at": _now(), "entries": {}}


def register(run_id: str, zone: str, dataset: str, source: str, row_count: int,
             path: Path, extra: dict[str, Any] | None = None) -> None:
    """Add or replace one catalog entry (zone/dataset) for the run."""
    cat = read_catalog(run_id)
    cat["entries"][f"{zone}/{dataset}"] = {
        "dataset": dataset,
        "zone": zone,
        "source": source,
        "row_count": int(row_count),
        "run_id": run_id,
        "created_at": _now(),
        "path": str(path.relative_to(_root)).replace("\\", "/"),
        "code_version": code_version(),
        "reference_version": reference_version(),
        **(extra or {}),
    }
    p = _catalog_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cat, indent=2, default=str), encoding="utf-8")


def list_runs() -> list[str]:
    folder = _root / "catalog"
    if not folder.exists():
        return []
    files = sorted(folder.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    return [p.stem[len("run_"):] for p in files]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------- tables
def write_table(df: pd.DataFrame, zone: str, dataset: str, run_id: str, source: str,
                extra: dict[str, Any] | None = None) -> Path:
    """Write a table for a run, replacing an earlier version of the same run."""
    folder = table_dir(zone, dataset, run_id)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    fmt = storage_format()
    if fmt == "parquet":
        path = folder / "data.parquet"
        df.to_parquet(path, index=False)
    else:
        path = folder / "data.csv"
        df.to_csv(path, index=False)
    schema = {c: str(t) for c, t in df.dtypes.items()}
    (folder / "_schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    register(run_id, zone, dataset, source, len(df), path,
             {"format": fmt, "columns": list(df.columns), **(extra or {})})
    return path


def read_table(zone: str, dataset: str, run_id: str) -> pd.DataFrame:
    folder = table_dir(zone, dataset, run_id)
    pq, csv = folder / "data.parquet", folder / "data.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if not csv.exists():
        raise FileNotFoundError(f"{zone}/{dataset} for run {run_id} not found - run the earlier step first")
    df = pd.read_csv(csv, low_memory=False)
    schema_file = folder / "_schema.json"
    if schema_file.exists():
        df = _restore_types(df, json.loads(schema_file.read_text(encoding="utf-8")))
    return df


def table_exists(zone: str, dataset: str, run_id: str) -> bool:
    folder = table_dir(zone, dataset, run_id)
    return (folder / "data.parquet").exists() or (folder / "data.csv").exists()


def _restore_types(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Restore datetimes, booleans and integers after a CSV round trip."""
    for col, typ in schema.items():
        if col not in df.columns:
            continue
        if typ.startswith("datetime64"):
            df[col] = pd.to_datetime(df[col])
        elif typ == "bool":
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
        elif typ.startswith("int") and df[col].notna().all():
            df[col] = df[col].astype("int64")
        elif typ in ("object", "str", "string") and col.endswith("_id"):
            df[col] = df[col].astype("object").where(df[col].notna(), None)
    return df


def copy_raw_file(src: Path, dataset: str, run_id: str, source: str, row_count: int,
                  extra: dict[str, Any] | None = None, clear: bool = False) -> Path:
    """Copy a source file unchanged into the raw zone."""
    folder = table_dir("raw", dataset, run_id)
    if clear and folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / src.name
    shutil.copy2(src, dst)
    register(run_id, "raw", f"{dataset}/{src.name}", source, row_count, dst, extra)
    return dst
