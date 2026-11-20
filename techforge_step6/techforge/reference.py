"""Loading and applying the reference tables (Table 10 [R]: filled attributes for R1-R4).

Every reference CSV has the columns source, verified (yes/no) and version.
Ambiguous rows have an empty mapped value and ambiguous=yes; they never produce a value.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from techforge import config, parts

REF_FILES = {
    "ref_uarch_socket": "ref_uarch_socket.csv",
    "ref_cpu_model_socket": "ref_cpu_model_socket.csv",
    "ref_socket_memory": "ref_socket_memory.csv",
    "ref_case_formfactor": "ref_case_formfactor.csv",
    "ref_gpu_power": "ref_gpu_power.csv",
}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(config.REFERENCE_DIR / REF_FILES[name], dtype=str, keep_default_na=False)


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    return {n: load(n) for n in REF_FILES}


def clear_cache() -> None:
    _tables.cache_clear()


def cpu_socket(name: str, microarchitecture: str | None) -> tuple[str | None, str]:
    """Return (socket, how) for a processor. how explains the source of the value or why it is missing."""
    if not microarchitecture or pd.isna(microarchitecture):
        return None, "microarchitecture missing"
    t = _tables()
    row = t["ref_uarch_socket"].loc[t["ref_uarch_socket"]["microarchitecture"] == microarchitecture]
    if row.empty:
        return None, f"generation '{microarchitecture}' not in reference table"
    if row.iloc[0]["ambiguous"] != "yes" and row.iloc[0]["socket"]:
        return row.iloc[0]["socket"], "generation"
    patterns = t["ref_cpu_model_socket"]
    patterns = patterns.loc[patterns["microarchitecture"] == microarchitecture].copy()
    patterns["priority"] = patterns["priority"].astype(int)
    for _, p in patterns.sort_values("priority").iterrows():
        if re.search(p["name_regex"], name, flags=re.IGNORECASE):
            return p["socket"], f"model pattern #{p['priority']}"
    return None, f"generation '{microarchitecture}' has several sockets and the model is not resolved"


def board_sockets(socket: str | None) -> set[str]:
    """'AM3+/AM3' -> {'AM3+','AM3'}; '2 x LGA2011' -> {'LGA2011'}; 'Integrated ...' -> {'INTEGRATED'}."""
    if not socket or pd.isna(socket):
        return set()
    s = str(socket)
    if s.startswith("Integrated"):
        return {"INTEGRATED"}
    s = re.sub(r"^\d+\s*x\s*", "", s)
    return {x.strip() for x in s.split("/") if x.strip()}


def board_memory_type(name: str, socket: str | None) -> tuple[str | None, str]:
    """Memory type of a motherboard from its socket and an explicit marker in its name."""
    marker = parts.memory_marker_in_name(name)
    socket_ok = bool(socket) and not pd.isna(socket)
    t = _tables()["ref_socket_memory"]
    row = t.loc[t["socket"] == socket] if socket_ok else t.iloc[0:0]
    default = None
    if not row.empty and row.iloc[0]["ambiguous"] != "yes" and row.iloc[0]["default_memory_type"]:
        default = row.iloc[0]["default_memory_type"]
    if marker:
        # P8 fix (P7 issue 01): a name marker that contradicts a single-type socket is a conflict, not a value
        if default and marker != default:
            return None, f"board name says {marker} but socket '{socket}' supports only {default} (conflict, check datasheet)"
        return marker, "board name"
    if not socket_ok:
        return None, "socket missing"
    if row.empty:
        return None, f"socket '{socket}' not in reference table"
    if default is None:
        return None, f"socket '{socket}' supports several memory types and the board name does not say which"
    return default, "socket"


def case_form_factors(case_type: str | None) -> tuple[str | None, str]:
    if not case_type or pd.isna(case_type):
        return None, "case type missing"
    t = _tables()["ref_case_formfactor"]
    row = t.loc[t["case_type"] == case_type]
    if row.empty:
        return None, f"case type '{case_type}' not in reference table"
    if row.iloc[0]["ambiguous"] == "yes" or not row.iloc[0]["accepted_form_factors"]:
        return None, f"case type '{case_type}' accepts different boards depending on the model"
    return row.iloc[0]["accepted_form_factors"], "case type"


def known_form_factors() -> set[str]:
    """All board sizes that appear in the case reference table; others are not covered (UNKNOWN)."""
    t = _tables()["ref_case_formfactor"]
    out: set[str] = set()
    for v in t["accepted_form_factors"]:
        out |= {x for x in v.split(";") if x}
    return out


def gpu_board_power(chipset: str | None) -> tuple[float | None, str]:
    if not chipset or pd.isna(chipset):
        return None, "chipset missing"
    t = _tables()["ref_gpu_power"]
    row = t.loc[t["chipset"] == chipset]
    if row.empty:
        return None, f"chipset '{chipset}' not in reference table"
    if row.iloc[0]["ambiguous"] == "yes" or not row.iloc[0]["board_power_w"]:
        return None, f"chipset '{chipset}' has variants with different power"
    return float(row.iloc[0]["board_power_w"]), "chipset"
