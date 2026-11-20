"""Step 3 - Parse, normalise and deduplicate parts (Table 12 [R]).

Output (cleaned zone): parts_<category> tables, parts_cleaning_log, merge_conflicts.
Done when: row counts and duplicates are logged.

Merging rule [R]: rows with the same category and normalised name become one part.
If the merged rows disagree on a rule attribute (e.g. two GPUs called "Sapphire PULSE"
with different chipsets), the attribute is set to missing, so the rules return
UNKNOWN instead of guessing ("when in doubt, show nothing"). Conflicts are logged.
"""
from __future__ import annotations

import pandas as pd

from techforge import config, lake, parts
from techforge.step01_load_raw import raw_path

MERGE_ATTRIBUTES: dict[str, list[str]] = {
    "cpu": ["microarchitecture", "tdp_w"],
    "motherboard": ["socket", "form_factor"],
    "memory": ["ddr_gen", "speed_mhz", "modules"],
    "video-card": ["chipset", "vram_gb", "length_mm"],
    "case": ["case_type"],
    "power-supply": ["wattage_w", "psu_type", "efficiency"],
}


def merge_duplicates(parsed: pd.DataFrame, category: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge rows sharing a part_id; conflicting attributes become missing."""
    attrs = MERGE_ATTRIBUTES[category]
    g = parsed.groupby("part_id", sort=False)
    merged = g.agg(category=("category", "first"), name=("name", "first"),
                   price=("price", "min"),  # cheapest listed offer [A]
                   source_rows=("name", "size")).reset_index()
    conflict_rows = []
    for a in attrs:
        n_distinct = g[a].nunique(dropna=True)
        first_val = g[a].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else None)
        merged[a] = merged["part_id"].map(first_val.where(n_distinct <= 1))
        bad = n_distinct[n_distinct > 1]
        for pid, k in bad.items():
            values = sorted(map(str, parsed.loc[parsed["part_id"] == pid, a].dropna().unique()))
            conflict_rows.append({"part_id": pid, "category": category, "attribute": a,
                                  "distinct_values": int(k), "values": "; ".join(values)[:300]})
    return merged, pd.DataFrame(conflict_rows,
                                columns=["part_id", "category", "attribute", "distinct_values", "values"])


def run(run_id: str, part_key: str = "extended") -> pd.DataFrame:
    log_rows, conflicts, crosswalk = [], [], []
    for category in config.CATEGORY_FILES:
        raw = pd.read_csv(raw_path(run_id, category), low_memory=False)
        parsed = parts.add_part_id(parts.parse_category(raw, category), category, part_key)
        merged, conf = merge_duplicates(parsed, category)
        merged["part_key_mode"] = part_key
        lake.write_table(merged, "cleaned", f"parts_{category.replace('-', '_')}", run_id,
                         source=f"raw/pc_part_dataset/{config.CATEGORY_FILES[category]}",
                         extra={"part_key_mode": part_key})
        conflicts.append(conf)
        # D1 (P8): crosswalk between the report key (category + name) and the extended key
        other = "report" if part_key == "extended" else "extended"
        alt = parts.add_part_id(parts.parse_category(raw, category), category, other)
        crosswalk.append(pd.DataFrame({"category": category, f"part_id_{part_key}": parsed["part_id"],
                                       f"part_id_{other}": alt["part_id"]}).drop_duplicates())
        log_rows.append({
            "category": category, "rows_in": len(raw), "parts_out": len(merged),
            "duplicate_rows_merged": len(raw) - len(merged),
            "parts_with_conflicts": conf["part_id"].nunique(),
            "rows_in_conflicting_parts": int(merged.loc[merged["part_id"].isin(conf["part_id"]),
                                                        "source_rows"].sum()),
            "parts_without_price": int(merged["price"].isna().sum()),
            "part_key_mode": part_key,
        })
    log = pd.DataFrame(log_rows)
    lake.write_table(log, "cleaned", "parts_cleaning_log", run_id, source="step03")
    lake.write_table(pd.concat(conflicts, ignore_index=True), "cleaned", "merge_conflicts", run_id,
                     source="step03")
    lake.write_table(pd.concat(crosswalk, ignore_index=True), "cleaned", "part_key_crosswalk", run_id,
                     source="step03 (report key <-> extended key, Table 13 amendment D1)")
    return log


def load_cleaned(run_id: str, category: str) -> pd.DataFrame:
    return lake.read_table("cleaned", f"parts_{category.replace('-', '_')}", run_id)
