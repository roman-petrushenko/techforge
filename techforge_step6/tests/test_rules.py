"""Rule engine edge cases (Table 10 [R]) - mechanics checks."""
import math

from conftest import part
from techforge import rules

CPU = part("C1", "cpu", socket="AM5", tdp_w=120.0)
MB = part("M1", "motherboard", socket="AM5", memory_type="DDR5", form_factor="ATX")
MEM = part("R1", "memory", memory_type="DDR5")
GPU = part("G1", "video-card", board_power_w=300.0)
CASE = part("K1", "case", case_type="ATX Mid Tower", accepted_form_factors="ATX;Micro ATX;Mini ITX")
PSU = part("P1", "power-supply", wattage_w=750.0)


def test_r1_pass_fail_unknown():
    assert rules.check_r1(CPU, MB).verdict == rules.PASS
    assert rules.check_r1(part("C2", "cpu", socket="AM4"), MB).verdict == rules.FAIL
    r = rules.check_r1(part("C3", "cpu", socket=None), MB)
    assert r.verdict == rules.UNKNOWN and "unknown" in r.reason.lower()
    assert rules.check_r1(CPU, part("M2", "motherboard", socket=float("nan"))).verdict == rules.UNKNOWN


def test_r1_multi_socket_and_integrated_boards():
    assert rules.check_r1(part("C", "cpu", socket="AM3"), part("M", "motherboard", socket="AM3+/AM3")).verdict == rules.PASS
    assert rules.check_r1(part("C", "cpu", socket="LGA2011"), part("M", "motherboard", socket="2 x LGA2011")).verdict == rules.PASS
    assert rules.check_r1(CPU, part("M", "motherboard", socket="Integrated Atom D525")).verdict == rules.FAIL


def test_r2():
    assert rules.check_r2(MEM, MB).verdict == rules.PASS
    assert rules.check_r2(part("R2", "memory", memory_type="DDR4"), MB).verdict == rules.FAIL
    assert rules.check_r2(MEM, part("M3", "motherboard", memory_type=None)).verdict == rules.UNKNOWN
    assert rules.check_r2(part("R3", "memory", memory_type=None), MB).verdict == rules.UNKNOWN


def test_r3():
    assert rules.check_r3(MB, CASE).verdict == rules.PASS
    small = part("K2", "case", case_type="Mini ITX Tower", accepted_form_factors="Mini ITX;Thin Mini ITX")
    assert rules.check_r3(MB, small).verdict == rules.FAIL
    assert rules.check_r3(MB, part("K3", "case", case_type="HTPC", accepted_form_factors=None)).verdict == rules.UNKNOWN
    odd = part("M4", "motherboard", form_factor="Mini DTX")   # not covered by the reference table
    assert rules.check_r3(odd, CASE).verdict == rules.UNKNOWN


def test_r4_formula_and_boundary():
    need = rules.r4_required_watts(120, 300)
    assert math.isclose(need, (120 + 300 + 100) * 1.25)          # 650 W
    assert rules.check_r4(CPU, GPU, part("P", "power-supply", wattage_w=650.0)).verdict == rules.PASS   # equal passes
    assert rules.check_r4(CPU, GPU, part("P", "power-supply", wattage_w=649.0)).verdict == rules.FAIL
    r = rules.check_r4(CPU, part("G2", "video-card", board_power_w=None), PSU)
    assert r.verdict == rules.UNKNOWN and "graphics card power" in r.reason


def test_build_verdict_precedence_and_applicability():
    assert rules.build_verdict([]) == "NOT_CHECKED"
    full = rules.check_build({"cpu": CPU, "motherboard": MB, "memory": MEM, "video-card": GPU, "case": CASE,
                              "power-supply": PSU})
    assert [r.rule_id for r in full] == ["R1", "R2", "R3", "R4"]
    assert rules.build_verdict(full) == rules.PASS
    assert rules.customer_message(rules.PASS, full) == "Passes 4 compatibility checks"
    with_unknown = rules.check_build({"cpu": CPU, "motherboard": MB, "memory": part("R", "memory")})
    assert rules.build_verdict(with_unknown) == rules.UNKNOWN
    with_fail_and_unknown = rules.check_build({"cpu": part("C", "cpu", socket="AM4"), "motherboard": MB,
                                               "memory": part("R", "memory")})
    assert rules.build_verdict(with_fail_and_unknown) == rules.FAIL
    only_cpu = rules.check_build({"cpu": CPU})
    assert only_cpu == []


def test_every_result_has_reason_and_version():
    for r in rules.check_build({"cpu": CPU, "motherboard": MB, "memory": MEM, "video-card": GPU,
                                "case": CASE, "power-supply": PSU}):
        assert r.reason and r.rule_version.startswith(r.rule_id)
