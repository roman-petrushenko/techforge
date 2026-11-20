"""Parsing of the real pc-part-dataset fields and the part identifier.

Field names are the real ones of commit c52a04c (checked in the CSV headers):
  cpu:          name, price, core_count, core_clock, boost_clock, microarchitecture, tdp, graphics
  motherboard:  name, price, socket, form_factor, max_memory, memory_slots, color
  memory:       name, price, speed ("5,6000" = DDR5-6000), modules ("2,16" = 2 x 16 GB), ...
  video-card:   name, price, chipset, memory, core_clock, boost_clock, color, length
  case:         name, price, type, color, psu, side_panel, external_volume, internal_35_bays
  power-supply: name, price, type, efficiency, wattage, modular, color

part_id rule [R] (Section 4.6): built from category and normalised name; rows with the
same category and name are merged.  The optional "extended" key [P] also uses the
fields that distinguish variants sold under one name (memory speed/modules, GPU
chipset/VRAM, PSU wattage/type/efficiency); it needs team approval before use.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from techforge import config

PART_KEY_MODES = ("report", "extended")

# Attributes that the rules need, per category (dim_part column names).
RULE_ATTRIBUTES: dict[str, list[str]] = {
    "cpu": ["socket", "tdp_w"],
    "motherboard": ["socket", "memory_type", "form_factor"],
    "memory": ["memory_type"],
    "video-card": ["board_power_w"],
    "case": ["accepted_form_factors"],
    "power-supply": ["wattage_w"],
}


def normalise_name(name: pd.Series) -> pd.Series:
    """Lower case, trimmed, single spaces."""
    return name.astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)


def _first_number(s: pd.Series) -> pd.Series:
    """'5,6000' -> 5 ; anything without a comma pair -> NaN."""
    parts = s.astype("string").str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.Series(float("nan"), index=s.index)
    first = pd.to_numeric(parts[0], errors="coerce")
    return first.where(parts[1].notna())


def _second_number(s: pd.Series) -> pd.Series:
    parts = s.astype("string").str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.Series(float("nan"), index=s.index)
    return pd.to_numeric(parts[1], errors="coerce")


def parse_category(raw: pd.DataFrame, category: str) -> pd.DataFrame:
    """Map the raw columns of one category to the prototype's column names (no merging)."""
    out = pd.DataFrame({
        "category": category,
        "name": raw["name"].astype(str).str.strip(),
        "price": pd.to_numeric(raw["price"], errors="coerce"),
    })
    if category == "cpu":
        out["microarchitecture"] = raw["microarchitecture"]
        out["tdp_w"] = pd.to_numeric(raw["tdp"], errors="coerce")
    elif category == "motherboard":
        out["socket"] = raw["socket"]
        out["form_factor"] = raw["form_factor"]
    elif category == "memory":
        gen = _first_number(raw["speed"])
        # Valid DDR generations only; values such as "400" (no generation) stay missing.
        out["ddr_gen"] = gen.where(gen.isin([2, 3, 4, 5]))
        out["speed_mhz"] = _second_number(raw["speed"])
        out["modules"] = raw["modules"].astype("string").str.replace(",", " x ", regex=False) + " GB"
    elif category == "video-card":
        out["chipset"] = raw["chipset"]
        out["vram_gb"] = pd.to_numeric(raw["memory"], errors="coerce")
        out["length_mm"] = pd.to_numeric(raw["length"], errors="coerce")  # not used: no case clearance data [R]
    elif category == "case":
        out["case_type"] = raw["type"]
    elif category == "power-supply":
        out["wattage_w"] = pd.to_numeric(raw["wattage"], errors="coerce")
        out["psu_type"] = raw["type"]
        out["efficiency"] = raw["efficiency"]
    return out


def key_text(parsed: pd.DataFrame, category: str, mode: str = "extended") -> pd.Series:
    """Text that identifies a part. extended (default since P8, decision D1) = normalised name +
    variant fields (chipset, wattage, memory generation ...); report = normalised name only (original Table 13)."""
    if mode not in PART_KEY_MODES:
        raise ValueError(f"part key mode must be one of {PART_KEY_MODES}")
    key = normalise_name(parsed["name"])
    if mode == "extended":
        extra_cols = {
            "memory": ["ddr_gen", "speed_mhz", "modules"],
            "video-card": ["chipset", "vram_gb"],
            "power-supply": ["wattage_w", "psu_type", "efficiency"],
        }.get(category, [])
        for c in extra_cols:
            key = key + "|" + parsed[c].astype("string").fillna("?")
    return key


def make_part_id(category: str, key: pd.Series) -> pd.Series:
    """Stable id such as 'GPU-1a2b3c4d5e' from category and key text."""
    prefix = config.CATEGORY_PREFIX[category]
    return key.map(lambda k: f"{prefix}-" + hashlib.sha1(f"{category}|{k}".encode()).hexdigest()[:10])


def add_part_id(parsed: pd.DataFrame, category: str, mode: str = "extended") -> pd.DataFrame:
    out = parsed.copy()
    out["part_key"] = key_text(parsed, category, mode)
    out["part_id"] = make_part_id(category, out["part_key"])
    return out


# Explicit memory markers in motherboard names, e.g. "B760M-A D4-CSM", "Z790 ... DDR4".
_MEM_MARKERS = {
    "DDR5": re.compile(r"(?:\bDDR5\b|(?<![A-Z0-9])D5(?![0-9A-Z]))", re.IGNORECASE),
    "DDR4": re.compile(r"(?:\bDDR4\b|(?<![A-Z0-9])D4(?![0-9A-Z]))", re.IGNORECASE),
    "DDR3": re.compile(r"(?:\bDDR3L?\b|(?<![A-Z0-9])D3(?![0-9A-Z]))", re.IGNORECASE),
}


def memory_marker_in_name(name: str) -> str | None:
    """Return DDR3/DDR4/DDR5 if exactly one generation is named in the board name."""
    found = [gen for gen, rx in _MEM_MARKERS.items() if rx.search(name or "")]
    return found[0] if len(found) == 1 else None
