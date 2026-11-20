"""Step 2 - Generate SYNTHETIC builder events, stock, sales and lead times (Table 12 [R]).

Everything written here is synthetic and labelled so (column `synthetic` / catalog flag).
Done when: generator settings and seed are saved (catalog entry `raw/synthetic_settings`).

Independence: the generator reads the raw pc-part-dataset fields itself and uses its own
small platform table (GEN_PLATFORMS, GEN_CASE_FIT). It never calls the rule engine or the
reference tables, so the 10% injected incompatible builds are created independently of
the rule code (team decision 9). Table 11 asks for injection from held-out labelled
incompatible pairs; that mode (inject_from="labels") is available once the team's labels exist.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from techforge import config, lake, parts
from techforge.step01_load_raw import raw_path

# ---------------------------------------------------------------- generator knowledge
# Own, deliberately simple platform table of the generator (not the rule reference tables).
GEN_PLATFORMS = {
    "AM4": {"uarch": ["Zen 2", "Zen 3"], "ddr": 4, "weight": 0.25},
    "AM5": {"uarch": ["Zen 4", "Zen 5"], "ddr": 5, "weight": 0.30},
    "LGA1700": {"uarch": ["Alder Lake", "Raptor Lake", "Raptor Lake Refresh"], "ddr": None, "weight": 0.30},
    "LGA1851": {"uarch": ["Arrow Lake"], "ddr": 5, "weight": 0.10},
    "LGA1200": {"uarch": ["Comet Lake", "Rocket Lake"], "ddr": 4, "weight": 0.05},
}
GEN_CASE_FIT = {  # board size -> case types the generator considers suitable
    "ATX": ["ATX Mid Tower", "ATX Full Tower"],
    "Micro ATX": ["ATX Mid Tower", "MicroATX Mini Tower", "MicroATX Mid Tower"],
    "Mini ITX": ["ATX Mid Tower", "Mini ITX Tower", "MicroATX Mini Tower"],
}
GEN_SMALL_CASES = ["Mini ITX Tower", "Mini ITX Desktop"]  # used to inject form-factor violations
TIERS = ("low", "mid", "high")
WEEKDAY_FACTOR = np.array([0.9, 0.95, 0.95, 1.0, 1.1, 1.35, 0.75])  # Mon..Sun [A]
HOUR_PROFILE = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 7, 8, 9, 10, 10, 8, 5, 2], float)


@dataclass
class GeneratorSettings:
    seed: int = 42
    part_key: str = "extended"   # D1 (P8): Table 13 amended; "report" kept for comparison
    # Focus B builder events
    start_date: str = "2026-09-01"
    train_days: int = 21
    replay_days: int = 7
    builds_per_day: int = 250
    incompatible_share: float = 0.10       # [R] Table 11
    duplicate_share: float = 0.01          # [A]
    malformed_share: float = 0.005         # [A]
    late_share: float = 0.02               # [A] events that arrive > 1 h after they happened
    assortment_per_category: int = 150     # [A] TechForge assortment drawn from priced parts
    out_of_stock_share: float = 0.12       # [A] share of part-warehouse-days with zero stock
    stock_outage_day: int = 3              # replay day with a stock feed outage (stale-data test)
    stock_outage_hours: tuple[int, int] = (10, 13)  # snapshots 10:00-12:00 missing
    inject_from: str = "generator"         # "generator" or "labels"
    labels_file: str | None = None
    # Focus A slice
    fa_start_date: str = "2025-09-01"
    fa_days: int = 420
    fa_regular_skus: int = 8
    fa_analogue_launch_day: int = 150
    fa_launch_day: int = 343
    fa_holdout_days: int = 84
    notes: list[str] = field(default_factory=lambda: ["all generated data is SYNTHETIC"])

    @property
    def train_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_date) + pd.Timedelta(days=self.train_days)


def small_settings(seed: int = 7) -> GeneratorSettings:
    """Smaller data for tests and quick runs."""
    return GeneratorSettings(seed=seed, train_days=10, replay_days=3, builds_per_day=60,
                             assortment_per_category=60, stock_outage_day=1,
                             fa_days=200, fa_regular_skus=3, fa_analogue_launch_day=60,
                             fa_launch_day=150, fa_holdout_days=42)


# ---------------------------------------------------------------- catalog of the generator
def _load_raw_parts(run_id: str, part_key: str) -> dict[str, pd.DataFrame]:
    out = {}
    for cat in config.CATEGORY_FILES:
        raw = pd.read_csv(raw_path(run_id, cat), low_memory=False)
        p = parts.add_part_id(parts.parse_category(raw, cat), cat, part_key)
        out[cat] = p.drop_duplicates("part_id").reset_index(drop=True)
    return out


def _tier(prices: pd.Series) -> pd.Series:
    q1, q2 = prices.quantile([1 / 3, 2 / 3])
    return pd.Series(np.where(prices <= q1, "low", np.where(prices > q2, "high", "mid")), index=prices.index)


def build_assortment(raw: dict[str, pd.DataFrame], s: GeneratorSettings, rng: np.random.Generator) -> pd.DataFrame:
    """TechForge assortment [A]: priced, current-platform parts sampled from the real catalog."""
    n = s.assortment_per_category
    frames = []
    cpu = raw["cpu"]
    for plat, info in GEN_PLATFORMS.items():
        c = cpu[cpu["price"].notna() & cpu["microarchitecture"].isin(info["uarch"])
                & cpu["name"].str.contains(r"Ryzen [3579] |Core i[3579]|Core Ultra", regex=True)
                & ~cpu["name"].str.contains("Threadripper")].assign(platform=plat)
        frames.append(c)
        mb = raw["motherboard"]
        m = mb[mb["price"].notna() & (mb["socket"] == plat) & mb["form_factor"].isin(list(GEN_CASE_FIT))]
        m = m.sample(min(len(m), max(10, n // 4)), random_state=int(rng.integers(1e9))).assign(platform=plat)
        frames.append(m)
    mem = raw["memory"]
    for g in (4, 5):
        mm = mem[mem["price"].notna() & (mem["ddr_gen"] == g)]
        frames.append(mm.sample(min(len(mm), n // 2), random_state=int(rng.integers(1e9))))
    gpu = raw["video-card"]
    gg = gpu[gpu["price"].notna() & gpu["chipset"].str.contains(r"RTX (?:30|40|50)\d0|RX [679]\d00|Arc B", regex=True)]
    frames.append(gg.sample(min(len(gg), n), random_state=int(rng.integers(1e9))))
    case = raw["case"]
    all_fit = sorted({t for v in GEN_CASE_FIT.values() for t in v} | set(GEN_SMALL_CASES))
    cc = case[case["price"].notna() & case["case_type"].isin(all_fit)]
    frames.append(cc.sample(min(len(cc), n), random_state=int(rng.integers(1e9))))
    psu = raw["power-supply"]
    pp = psu[psu["price"].notna() & (psu["psu_type"] == "ATX") & psu["wattage_w"].between(450, 1600)]
    frames.append(pp.sample(min(len(pp), n), random_state=int(rng.integers(1e9))))
    a = pd.concat(frames, ignore_index=True).drop_duplicates("part_id")
    a["tier"] = a.groupby("category")["price"].transform(_tier)
    # Zipf-like popularity over a random order of parts within each category [A]
    a["popularity_weight"] = 0.0
    for cat, idx in a.groupby("category").groups.items():
        ranks = rng.permutation(len(idx)) + 1
        a.loc[idx, "popularity_weight"] = 1.0 / ranks ** 0.8
    if "platform" not in a.columns:
        a["platform"] = None
    return a.reset_index(drop=True)


def _choose(pool: pd.DataFrame, tier: str, rng: np.random.Generator) -> pd.Series:
    w = pool["popularity_weight"].to_numpy() * np.where(pool["tier"].to_numpy() == tier, 3.0, 0.5)
    return pool.iloc[int(rng.choice(len(pool), p=w / w.sum()))]


def _board_ddr(board: pd.Series) -> int:
    plat = GEN_PLATFORMS[board["platform"]]
    if plat["ddr"]:
        return plat["ddr"]
    name = str(board["name"]).upper()
    return 4 if ("DDR4" in name or " D4" in name) else 5


# ---------------------------------------------------------------- builds and events
def generate_builds(a: pd.DataFrame, raw: dict[str, pd.DataFrame], s: GeneratorSettings,
                    rng: np.random.Generator, labelled_pairs: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_cat = {c: g for c, g in a.groupby("category")}
    plats = list(GEN_PLATFORMS)
    plat_w = np.array([GEN_PLATFORMS[p]["weight"] for p in plats])
    low_psu = raw["power-supply"][raw["power-supply"]["wattage_w"].between(250, 400)]
    small_cases = raw["case"][raw["case"]["case_type"].isin(GEN_SMALL_CASES)]
    rows, truth = [], []
    start = pd.Timestamp(s.start_date)
    total_days = s.train_days + s.replay_days
    b = 0
    for d in range(total_days):
        day = start + pd.Timedelta(days=d)
        n_builds = rng.poisson(s.builds_per_day * WEEKDAY_FACTOR[day.weekday()])
        for _ in range(n_builds):
            b += 1
            tier = TIERS[int(rng.choice(3, p=[0.35, 0.45, 0.20]))]
            plat = plats[int(rng.choice(len(plats), p=plat_w / plat_w.sum()))]
            cpu = _choose(by_cat["cpu"][by_cat["cpu"]["platform"] == plat], tier, rng)
            mb = _choose(by_cat["motherboard"][by_cat["motherboard"]["platform"] == plat], tier, rng)
            ddr = _board_ddr(mb)
            mem = _choose(by_cat["memory"][by_cat["memory"]["ddr_gen"] == ddr], tier, rng)
            gpu = _choose(by_cat["video-card"], tier, rng)
            cases = by_cat["case"][by_cat["case"]["case_type"].isin(GEN_CASE_FIT[mb["form_factor"]])]
            case = _choose(cases, tier, rng)
            floor = {"low": 550, "mid": 750, "high": 850}[gpu["tier"]]
            if pd.Series([gpu["chipset"]]).str.contains(r"4090|5090|7900 XTX|5080").iloc[0]:
                floor = 1000
            psus = by_cat["power-supply"][by_cat["power-supply"]["wattage_w"] >= floor]
            psu = _choose(psus if len(psus) else by_cat["power-supply"], tier, rng)
            build = {"cpu": cpu, "motherboard": mb, "memory": mem, "video-card": gpu, "case": case,
                     "power-supply": psu}
            injected, violation, source = False, "", ""
            # P8 fix (P7 issue 04): labelled (held-out) pairs are injected only on replay days
            use_labels = s.inject_from == "labels"
            if rng.random() < s.incompatible_share and not (use_labels and d < s.train_days):
                injected = True
                if use_labels:
                    pair = labelled_pairs.iloc[int(rng.integers(len(labelled_pairs)))]
                    for cat_col, pid_col in (("category_1", "part_id_1"), ("category_2", "part_id_2")):
                        cat = pair[cat_col]
                        build[cat] = raw[cat].loc[raw[cat]["part_id"] == pair[pid_col]].iloc[0]
                    violation, source = f"labelled pair {pair['label_id']} ({pair['rule_id']})", "labels"
                else:
                    source = "generator"
                    options = ["socket", "memory", "power"] + (["form_factor"] if mb["form_factor"] != "Mini ITX" else [])
                    violation = options[int(rng.integers(len(options)))]
                    if violation == "socket":
                        other = [p for p in plats if p != plat]
                        op = other[int(rng.integers(len(other)))]
                        build["motherboard"] = _choose(by_cat["motherboard"][by_cat["motherboard"]["platform"] == op], tier, rng)
                        # keep memory/case consistent with the new board so only the socket is wrong
                        build["memory"] = _choose(by_cat["memory"][by_cat["memory"]["ddr_gen"] == _board_ddr(build["motherboard"])], tier, rng)
                        fit = GEN_CASE_FIT[build["motherboard"]["form_factor"]]
                        build["case"] = _choose(by_cat["case"][by_cat["case"]["case_type"].isin(fit)], tier, rng)
                    elif violation == "memory":
                        build["memory"] = _choose(by_cat["memory"][by_cat["memory"]["ddr_gen"] == (9 - ddr)], tier, rng)
                    elif violation == "form_factor":
                        build["case"] = small_cases.iloc[int(rng.integers(len(small_cases)))]
                    else:
                        # small PSU together with a high-price GPU (the generator does not know GPU power;
                        # a cheap GPU with a 400 W PSU can be genuinely compatible)
                        build["video-card"] = _choose(by_cat["video-card"][by_cat["video-card"]["tier"] == "high"], "high", rng)
                        build["power-supply"] = low_psu.iloc[int(rng.integers(len(low_psu)))]
            ts = day + pd.Timedelta(hours=int(rng.choice(24, p=HOUR_PROFILE / HOUR_PROFILE.sum())),
                                    minutes=int(rng.integers(60)), seconds=int(rng.integers(60)))
            wh = config.WAREHOUSES[int(rng.choice(2, p=[0.6, 0.4]))]
            for cat in config.BUILD_ORDER:
                rows.append({"build_id": f"B{b:07d}", "session_id": f"S{b:07d}", "event_ts": ts,
                             "category": cat, "part_id": build[cat]["part_id"], "warehouse_id": wh})
                ts = ts + pd.Timedelta(seconds=int(rng.integers(20, 240)))
            truth.append({"build_id": f"B{b:07d}", "injected": injected, "violation": violation,
                          "inject_source": source, "synthetic": True})
    return pd.DataFrame(rows), pd.DataFrame(truth)


def write_event_files(ev: pd.DataFrame, s: GeneratorSettings, rng: np.random.Generator,
                      run_id: str, known_ids: list[str]) -> dict[str, int]:
    """Write events as JSON lines per ingest day, with late, duplicate and malformed records."""
    ev = ev.copy()
    ev["event_id"] = [f"E{i:08d}" for i in range(1, len(ev) + 1)]
    delay = np.where(rng.random(len(ev)) < s.late_share,
                     rng.integers(65 * 60, 180 * 60, len(ev)), rng.integers(1, 30, len(ev)))
    ev["ingest_ts"] = ev["event_ts"] + pd.to_timedelta(delay, unit="s")
    records = []
    for r in ev.itertuples(index=False):
        records.append({"event_id": r.event_id, "session_id": r.session_id, "build_id": r.build_id,
                        "event_ts": r.event_ts.isoformat(), "ingest_ts": r.ingest_ts.isoformat(),
                        "action": "add", "category": r.category, "part_id": r.part_id,
                        "warehouse_id": r.warehouse_id, "synthetic": True})
    counts = {"events": len(records), "late": int((delay > 3600).sum()), "duplicates": 0, "malformed": 0}
    lines: list[tuple[str, str]] = []
    for rec in records:
        u = rng.random()
        if u < s.malformed_share:
            kind = int(rng.integers(4))
            bad = dict(rec)
            if kind == 0:
                bad["part_id"] = None
            elif kind == 1:
                bad["event_ts"] = "not-a-timestamp"
            elif kind == 2:
                bad["part_id"] = "GPU-0000000000"   # unknown part
            lines.append((rec["ingest_ts"][:10], json.dumps(bad)[: (25 if kind == 3 else None)]))
            counts["malformed"] += 1
            continue
        lines.append((rec["ingest_ts"][:10], json.dumps(rec)))
        if u > 1 - s.duplicate_share:
            lines.append((rec["ingest_ts"][:10], json.dumps(rec)))
            counts["duplicates"] += 1
    folder = lake.table_dir("raw", "builder_events", run_id)
    if folder.exists():
        import shutil
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    by_day: dict[str, list[str]] = {}
    for day, line in lines:
        by_day.setdefault(day, []).append(line)
    for day, ls in sorted(by_day.items()):
        p = folder / f"builder_events_{day}.jsonl"
        p.write_text("\n".join(ls) + "\n", encoding="utf-8")
        lake.register(run_id, "raw", f"builder_events/{p.name}", "SYNTHETIC generator (simulates flow F2)",
                      len(ls), p, {"synthetic": True})
    return counts


def generate_stock(a: pd.DataFrame, s: GeneratorSettings, rng: np.random.Generator) -> pd.DataFrame:
    """Hourly stock snapshots per warehouse for the replay period (+1 day before) [A]."""
    first_day = s.train_end - pd.Timedelta(days=1)
    days = pd.date_range(first_day, periods=s.replay_days + 1, freq="D")
    outage_day = s.train_end + pd.Timedelta(days=s.stock_outage_day)
    rows = []
    lam = 3 + 30 * a["popularity_weight"].to_numpy()
    for wh_i, wh in enumerate(config.WAREHOUSES):
        for day in days:
            level = rng.poisson(lam * (1.0 if wh_i == 0 else 0.6))
            level[rng.random(len(a)) < s.out_of_stock_share] = 0
            on_order = np.where(rng.random(len(a)) < 0.3, rng.integers(5, 30, len(a)), 0)
            for h in range(24):
                if day == outage_day and s.stock_outage_hours[0] <= h < s.stock_outage_hours[1]:
                    continue  # simulated feed outage -> later events see a snapshot older than 1 h
                ts = day + pd.Timedelta(hours=h)
                sold = rng.binomial(1, 0.05, len(a)) if h else 0
                level = np.maximum(0, level - sold)
                rows.append(pd.DataFrame({"part_id": a["part_id"].to_numpy(), "warehouse_id": wh,
                                          "snapshot_ts": ts, "on_hand": level, "on_order": on_order}))
    out = pd.concat(rows, ignore_index=True)
    out["synthetic"] = True
    return out


# ---------------------------------------------------------------- Focus A
def generate_focus_a(a: pd.DataFrame, s: GeneratorSettings, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Daily true demand, recorded (censored) sales, inventory and supplier deliveries [A]."""
    gpus = a[a["category"] == "video-card"].sample(frac=1, random_state=int(rng.integers(1e9)))
    psus = a[a["category"] == "power-supply"].sample(frac=1, random_state=int(rng.integers(1e9)))
    n_gpu = max(1, s.fa_regular_skus - 2)
    skus = [(pid, "regular") for pid in gpus["part_id"].iloc[:n_gpu]]
    skus += [(pid, "regular") for pid in psus["part_id"].iloc[:s.fa_regular_skus - n_gpu]]
    skus += [(gpus["part_id"].iloc[n_gpu], "analogue_launch"), (gpus["part_id"].iloc[n_gpu + 1], "launch")]
    dates = pd.date_range(s.fa_start_date, periods=s.fa_days, freq="D")
    wd = dates.weekday.to_numpy()
    t = np.arange(s.fa_days)
    sku_rows, sales, truth, inv, deliveries = [], [], [], [], []
    for pid, role in skus:
        L_mean, L_sd = int(rng.integers(4, 10)), float(rng.uniform(0.5, 2.5))
        launch_day = {"launch": s.fa_launch_day, "analogue_launch": s.fa_analogue_launch_day}.get(role)
        sku_rows.append({"part_id": pid, "role": role,
                         "launch_date": dates[launch_day].date().isoformat() if launch_day is not None else None,
                         "lead_time_mean_days": L_mean, "lead_time_sd_days": round(L_sd, 2), "synthetic": True})
        for wh, whf in zip(config.WAREHOUSES, (1.0, 0.6)):
            if role == "regular":
                mu = rng.uniform(2, 12) * whf
                lam = mu * WEEKDAY_FACTOR[wd] * (1 + 0.0005 * t)
                plan_mu = np.full(s.fa_days, mu)
            else:
                peak, tau, plateau = (30, 10, 6) if role == "launch" else (22, 12, 5)
                age = t - launch_day
                lam = np.where(age >= 0, (plateau + peak * np.exp(-np.maximum(age, 0) / tau)) * whf, 0.0)
                lam = lam * WEEKDAY_FACTOR[wd]
                plan_mu = np.where(age >= -L_mean - 3, plateau * whf, 0.0)  # planner underestimates the spike
            demand = rng.poisson(lam)
            on_hand = int(plan_mu[0] * (L_mean + 10)) if role == "regular" else 0
            pipeline: dict[int, int] = {}
            for i in range(s.fa_days):
                on_hand += pipeline.pop(i, 0)
                sold = min(on_hand, int(demand[i]))
                on_hand -= sold
                stockout = on_hand == 0 and plan_mu[i] > 0   # observable flag: shelf empty at day end [A]
                sales.append((pid, wh, dates[i], sold, bool(stockout)))
                truth.append((pid, wh, dates[i], int(demand[i])))
                on_order = sum(pipeline.values())
                ip = on_hand + on_order
                s_min, s_max = plan_mu[i] * (L_mean + 3), plan_mu[i] * (L_mean + 14)
                if plan_mu[i] > 0 and ip <= s_min:
                    q = int(np.ceil(s_max - ip))
                    lt = max(1, int(round(rng.normal(L_mean, L_sd))))
                    if rng.random() < 0.05:
                        lt += 7  # occasional supplier delay
                    if q > 0:
                        pipeline[i + lt] = pipeline.get(i + lt, 0) + q
                        deliveries.append((pid, wh, dates[i], lt, q))
                inv.append((pid, wh, dates[i], on_hand, sum(pipeline.values())))
    cols = ["part_id", "warehouse_id", "date"]
    return {
        "fa_sku": pd.DataFrame(sku_rows),
        "sales_daily": pd.DataFrame(sales, columns=cols + ["units_sold", "stockout_flag"]).assign(synthetic=True),
        "true_demand": pd.DataFrame(truth, columns=cols + ["true_demand"]).assign(synthetic=True),
        "inventory_daily": pd.DataFrame(inv, columns=cols + ["on_hand", "on_order"]).assign(synthetic=True),
        "supplier_deliveries": pd.DataFrame(deliveries, columns=["part_id", "warehouse_id", "order_date",
                                                                 "lead_time_days", "quantity"]).assign(synthetic=True),
    }


# ---------------------------------------------------------------- step entry point
def _write_raw_csv(df: pd.DataFrame, dataset: str, run_id: str, source: str) -> Path:
    folder = lake.table_dir("raw", dataset, run_id)
    if folder.exists():
        import shutil
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    p = folder / f"{dataset}.csv"
    df.to_csv(p, index=False)
    lake.register(run_id, "raw", f"{dataset}/{p.name}", source, len(df), p, {"synthetic": True})
    return p


def run(run_id: str, s: GeneratorSettings | None = None) -> dict[str, int]:
    s = s or GeneratorSettings()
    rng = np.random.default_rng(s.seed)
    raw = _load_raw_parts(run_id, s.part_key)
    a = build_assortment(raw, s, rng)
    labelled = None
    if s.inject_from == "labels":
        if not s.labels_file or not Path(s.labels_file).exists():
            raise FileNotFoundError("inject_from='labels' needs labels_file with the resolved team labels")
        lab = pd.read_csv(s.labels_file)
        if "category_1" not in lab.columns:   # older template: join the manifest by label_id (P8)
            man = pd.read_csv(Path(s.labels_file).parent / "sample_manifest.csv")
            lab = lab.merge(man[["label_id", "category_1", "category_2"]], on="label_id", how="left")
        labelled = lab[(lab["subset"] == "test") & (lab["final_compatible"] == "no") & (lab["rule_id"] != "R4")]
        if labelled.empty:   # P8 fix (P7 issue 04): never fall back silently to generator injection
            raise ValueError("inject_from='labels' but the labels file has no held-out pair labelled incompatible")
    builds, truth = generate_builds(a, raw, s, rng, labelled)
    counts = write_event_files(builds, s, rng, run_id, list(a["part_id"]))
    stock = generate_stock(a, s, rng)
    fa = generate_focus_a(a, s, rng)
    _write_raw_csv(a[["part_id", "category", "name", "price", "tier", "platform", "popularity_weight"]]
                   .assign(synthetic=True), "assortment", run_id, "SYNTHETIC generator: TechForge assortment")
    _write_raw_csv(truth, "build_truth", run_id, "SYNTHETIC generator: injected incompatible builds (evaluation only)")
    _write_raw_csv(stock, "stock_snapshots", run_id, "SYNTHETIC generator (simulates flow F4, hourly)")
    for name, df in fa.items():
        src = "SYNTHETIC generator: true demand - EVALUATION ONLY" if name == "true_demand" else "SYNTHETIC generator (Focus A)"
        _write_raw_csv(df, name, run_id, src)
    settings = asdict(s)
    settings["train_end"] = str(s.train_end)
    settings_path = lake.table_dir("raw", "synthetic_settings", run_id)
    settings_path.mkdir(parents=True, exist_ok=True)
    p = settings_path / "generator_settings.json"
    p.write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    lake.register(run_id, "raw", "synthetic_settings", "SYNTHETIC generator settings and seed", 1, p,
                  {"synthetic": True, "settings": settings})
    counts.update({"builds": len(truth), "injected": int(truth["injected"].sum()),
                   "assortment_parts": len(a), "stock_rows": len(stock)})
    return counts


def load_settings(run_id: str) -> dict:
    p = lake.table_dir("raw", "synthetic_settings", run_id) / "generator_settings.json"
    return json.loads(p.read_text(encoding="utf-8"))
