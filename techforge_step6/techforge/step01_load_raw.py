"""Step 1 - Load the pc-part-dataset files unchanged into the raw zone (Table 12 [R]).

Done when: files and commit ID are stored (lake/raw/pc_part_dataset/run_id=.../).
If the local copy in data/ is missing, the files are downloaded from GitHub at the
pinned commit.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from techforge import config, lake

RAW_DATASET = "pc_part_dataset"


def ensure_local_copy(target: Path = config.DATASET_DIR) -> Path:
    """Download the six category files at the pinned commit if they are not present."""
    target.mkdir(parents=True, exist_ok=True)
    for fname in config.CATEGORY_FILES.values():
        dst = target / fname
        if not dst.exists():
            url = (f"https://raw.githubusercontent.com/docyx/pc-part-dataset/"
                   f"{config.DATASET_COMMIT}/data/csv/{fname}")
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, dst)
    return target


def _count_rows(path: Path) -> int:
    import pandas as pd
    return len(pd.read_csv(path, low_memory=False))


def run(run_id: str, source_dir: Path = config.DATASET_DIR) -> dict[str, int]:
    source_dir = ensure_local_copy(source_dir)
    counts: dict[str, int] = {}
    first = True
    for category, fname in config.CATEGORY_FILES.items():
        src = source_dir / fname
        n = _count_rows(src)
        lake.copy_raw_file(src, RAW_DATASET, run_id,
                           source=f"{config.DATASET_REPO} @ {config.DATASET_COMMIT} data/csv/{fname}",
                           row_count=n, extra={"commit": config.DATASET_COMMIT, "category": category},
                           clear=first)
        first = False
        counts[category] = n
    commit_file = lake.table_dir("raw", RAW_DATASET, run_id) / "COMMIT.txt"
    commit_file.write_text(f"{config.DATASET_REPO}\ncommit {config.DATASET_COMMIT}\n", encoding="utf-8")
    lake.register(run_id, "raw", f"{RAW_DATASET}/COMMIT.txt", "pipeline", 1, commit_file,
                  {"commit": config.DATASET_COMMIT})
    return counts


def raw_path(run_id: str, category: str) -> Path:
    return lake.table_dir("raw", RAW_DATASET, run_id) / config.CATEGORY_FILES[category]
