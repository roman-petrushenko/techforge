"""Rule engine R1-R4 exactly as Table 10 [R].

Each rule returns PASS, FAIL or UNKNOWN (a needed value is missing), a reason a
customer or staff member can read, and the rule version.
Build verdict: FAIL if any rule fails, else UNKNOWN if any is UNKNOWN, else PASS [R].
PASS means only "passes these four checks" [R].

Parts are plain dicts with the dim_part columns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from techforge import reference

RULESET_VERSION = "rules-v1.0"
RULE_VERSIONS = {"R1": "R1-v1.0", "R2": "R2-v1.0", "R3": "R3-v1.0", "R4": "R4-v1.0"}
R4_OVERHEAD_W = 100.0   # [R] Table 10 constant (an assumption with a safety margin, Section 4.3)
R4_FACTOR = 1.25        # [R] Table 10 constant

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"

# Which categories each rule needs
RULE_PARTS = {
    "R1": ("cpu", "motherboard"),
    "R2": ("memory", "motherboard"),
    "R3": ("motherboard", "case"),
    "R4": ("cpu", "video-card", "power-supply"),
}

Part = dict[str, Any]


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    verdict: str
    reason: str
    part_ids: tuple[str, ...]
    rule_version: str


_MISSING_MARKERS = {"", "unknown", "n/a", "na", "nan", "none", "null", "-"}


def missing(v: Any) -> bool:
    """None, NaN, empty text or a text marker such as 'UNKNOWN' (P8 fix, P7 issue 01)."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in _MISSING_MARKERS


def _ids(*parts: Part) -> tuple[str, ...]:
    return tuple(p.get("part_id", "?") for p in parts)


def check_r1(cpu: Part, mb: Part) -> RuleResult:
    """R1 Socket: processor socket = motherboard socket."""
    ids, v = _ids(cpu, mb), RULE_VERSIONS["R1"]
    if missing(cpu.get("socket")):
        return RuleResult("R1", UNKNOWN, "Processor socket unknown (generation not resolved in the reference table)", ids, v)
    if missing(mb.get("socket")):
        return RuleResult("R1", UNKNOWN, "Motherboard socket missing in the source data", ids, v)
    accepted = reference.board_sockets(mb["socket"])
    if "INTEGRATED" in accepted:
        return RuleResult("R1", FAIL, f"Motherboard has an integrated processor ({mb['socket']}); no separate processor fits", ids, v)
    if cpu["socket"] in accepted:
        return RuleResult("R1", PASS, f"Both use socket {cpu['socket']}", ids, v)
    return RuleResult("R1", FAIL, f"Processor socket {cpu['socket']} does not match motherboard socket {mb['socket']}", ids, v)


def check_r2(mem: Part, mb: Part) -> RuleResult:
    """R2 Memory: memory generation = type supported by the motherboard."""
    ids, v = _ids(mem, mb), RULE_VERSIONS["R2"]
    if missing(mem.get("memory_type")):
        return RuleResult("R2", UNKNOWN, "Memory generation unknown in the source data", ids, v)
    if missing(mb.get("memory_type")):
        return RuleResult("R2", UNKNOWN, "Motherboard memory type unclear (socket allows several types and the name does not say which)", ids, v)
    if mem["memory_type"] == mb["memory_type"]:
        return RuleResult("R2", PASS, f"Both use {mem['memory_type']}", ids, v)
    return RuleResult("R2", FAIL, f"Memory is {mem['memory_type']} but the motherboard supports {mb['memory_type']}", ids, v)


def check_r3(mb: Part, case: Part) -> RuleResult:
    """R3 Form factor: motherboard size fits the case."""
    ids, v = _ids(mb, case), RULE_VERSIONS["R3"]
    ff = mb.get("form_factor")
    if missing(ff):
        return RuleResult("R3", UNKNOWN, "Motherboard form factor missing in the source data", ids, v)
    if ff not in reference.known_form_factors():
        return RuleResult("R3", UNKNOWN, f"Board size '{ff}' is not covered by the case reference table", ids, v)
    acc = case.get("accepted_form_factors")
    if missing(acc):
        return RuleResult("R3", UNKNOWN, f"Board sizes accepted by case type '{case.get('case_type')}' are not known", ids, v)
    accepted = [x for x in str(acc).split(";") if x]
    if ff in accepted:
        return RuleResult("R3", PASS, f"{ff} board fits a {case.get('case_type')} case", ids, v)
    return RuleResult("R3", FAIL, f"{ff} board does not fit a {case.get('case_type')} case (accepts {', '.join(accepted)})", ids, v)


def r4_required_watts(tdp_w: float, board_power_w: float) -> float:
    return (tdp_w + board_power_w + R4_OVERHEAD_W) * R4_FACTOR


def check_r4(cpu: Part, gpu: Part, psu: Part) -> RuleResult:
    """R4 Power: PSU wattage >= (processor TDP + graphics card power + 100 W) x 1.25."""
    ids, v = _ids(cpu, gpu, psu), RULE_VERSIONS["R4"]
    gaps = [label for label, val in (("processor TDP", cpu.get("tdp_w")),
                                     ("graphics card power", gpu.get("board_power_w")),
                                     ("power supply wattage", psu.get("wattage_w"))) if missing(val)]
    if gaps:
        return RuleResult("R4", UNKNOWN, "Cannot check power: " + ", ".join(gaps) + " unknown", ids, v)
    need = r4_required_watts(float(cpu["tdp_w"]), float(gpu["board_power_w"]))
    have = float(psu["wattage_w"])
    if have >= need:
        return RuleResult("R4", PASS, f"Power supply {have:.0f} W >= required {need:.0f} W", ids, v)
    return RuleResult("R4", FAIL, f"Power supply {have:.0f} W < required {need:.0f} W "
                                  f"(({cpu['tdp_w']:.0f} + {gpu['board_power_w']:.0f} + 100) x 1.25)", ids, v)


_CHECKS = {"R1": check_r1, "R2": check_r2, "R3": check_r3, "R4": check_r4}


def check_build(parts_by_category: dict[str, Part]) -> list[RuleResult]:
    """Run every rule whose parts are all present; rules with a missing part are not checked yet [A]."""
    results = []
    for rule_id, cats in RULE_PARTS.items():
        if all(c in parts_by_category for c in cats):
            results.append(_CHECKS[rule_id](*(parts_by_category[c] for c in cats)))
    return results


def check_rules_involving(parts_by_category: dict[str, Part], category: str) -> list[RuleResult]:
    """Rules that involve `category` and whose other parts are present (used for candidates)."""
    out = []
    for rule_id, cats in RULE_PARTS.items():
        if category in cats and all(c in parts_by_category for c in cats):
            out.append(_CHECKS[rule_id](*(parts_by_category[c] for c in cats)))
    return out


def build_verdict(results: list[RuleResult]) -> str:
    """FAIL if any FAIL, else UNKNOWN if any UNKNOWN, else PASS; NOT_CHECKED if no rule applied."""
    verdicts = {r.verdict for r in results}
    if not verdicts:
        return "NOT_CHECKED"
    if FAIL in verdicts:
        return FAIL
    if UNKNOWN in verdicts:
        return UNKNOWN
    return PASS


def customer_message(verdict: str, results: list[RuleResult]) -> str:
    """Text shown in the builder [R] Section 4.3; checkout is never blocked."""
    if verdict == PASS:
        n = len(results)
        return f"Passes {n} of 4 compatibility checks" if n < 4 else "Passes 4 compatibility checks"
    if verdict == FAIL:
        return "Warning: " + "; ".join(r.reason for r in results if r.verdict == FAIL)
    if verdict == UNKNOWN:
        return "Could not be verified: " + "; ".join(r.reason for r in results if r.verdict == UNKNOWN)
    return "Select more parts to run the compatibility checks"
