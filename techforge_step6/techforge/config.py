"""Central settings for the TechForge Step 6 prototype.

Values marked [R] come from the Steps 1-5 report, [A] are team assumptions
that must be confirmed, [P] are proposals that are not in the report.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Local data lake with the zones of Step 3 [R]. "quarantine" holds rejected events (Section 4.6 [R]).
LAKE_ROOT = ROOT / "lake"
ZONES = ("raw", "cleaned", "curated", "outputs", "quarantine")

# Real public dataset [R]: pc-part-dataset, commit c52a04c (Table 14).
DATASET_REPO = "https://github.com/docyx/pc-part-dataset"
DATASET_COMMIT = "c52a04ca9465c83997ed335f7767b09a2005dd26"
DATASET_DIR = ROOT / "data" / "pc-part-dataset-c52a04c"
CATEGORY_FILES = {
    "cpu": "cpu.csv",
    "motherboard": "motherboard.csv",
    "memory": "memory.csv",
    "video-card": "video-card.csv",
    "case": "case.csv",
    "power-supply": "power-supply.csv",
}
CATEGORY_PREFIX = {
    "cpu": "CPU", "motherboard": "MB", "memory": "MEM",
    "video-card": "GPU", "case": "CASE", "power-supply": "PSU",
}

# Order in which the PC builder asks for parts; the "next part" is the first missing one [A].
BUILD_ORDER = ["cpu", "motherboard", "memory", "video-card", "case", "power-supply"]

REFERENCE_DIR = ROOT / "reference_tables"
REPORT_OUTPUTS = ROOT / "report_outputs"

WAREHOUSES = ("KYIV", "LVIV")  # two warehouses [R]; names [A]

# Serving rules [R] Section 4.3 / 4.6
STOCK_MAX_AGE_MINUTES = 60      # snapshot older than one hour -> no suggestion
TOP_N = 5                       # five suggestions
MIN_SUPPORT = 3                 # minimum co-occurrence count before lift is trusted [A]

# Focus A policy [R] Section 4.4
Z_SERVICE = 1.65
REVIEW_PERIOD_DAYS = 1
