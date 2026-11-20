# TechForge – Step 6 DevDemo (local prototype)

This project implements the prototype that Step 4 of the TechForge report specifies (sections 4.3–4.7, Tables 10–14). It is a local demonstration that the algorithm works. Nothing is deployed to the cloud.

* **Focus B (core) – "Compatible-Next-Part":** rules R1–R4 check real product specifications, and the recommender suggests up to five compatible, in-stock next parts.
* **Focus A (slice):** forecast → daily order-up-to policy → low-stock alert, run on synthetic sales.
* **Not implemented:** problems 3 (prices) and 6 (deliveries), a check of GPU length against case clearance (there is no clearance data), real-time stock (see Step 5), and any cloud service.

## What is real and what is synthetic

| Data | Type | Where |
|---|---|---|
| pc-part-dataset, commit `c52a04c` (MIT licence, © docyx) | **Real**, public | `data/pc-part-dataset-c52a04c/` (6 CSV files copied unchanged) |
| Reference tables for R1–R4 | **Draft by the AI agent – every row has `verified = no`** | `reference_tables/ref_*.csv` |
| Builder events, stock, sales, true demand, lead times, assortment | **SYNTHETIC** (step 2, seed in the catalog) | `lake/raw/...` |
| Labels (150 pairs + 30 triples) | **Not created** – the team labels them | `labels/` holds the empty sheets |

Every result built on synthetic data is labelled "SYNTHETIC" in its table, figure, catalog entry and app page. The rule-accuracy rows of Table 11 stay *pending* until the team's labels are evaluated.

## Setup on Windows 11 (Python 3.11 or 3.12)

```powershell
cd techforge_step6
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# optional, for --engine spark: pip install -r requirements-spark.txt  (needs Java 17 and HADOOP_HOME/winutils)
```

## How to run

```powershell
python run_pipeline.py --run-id demo            # steps 1-11, about 1 min
python run_pipeline.py --run-id demo --step 8   # one step again (replaces that step's outputs)
python run_pipeline.py --run-id quick --small   # small data, about 15 s
python -m pytest -q                             # 58 tests: mechanics, P7 adversarial, P8 regression
streamlit run app/Home.py --server.port 8600 --server.address 127.0.0.1   # live demo (port 8501 may be blocked on Windows)
jupyter notebook step6_demo.ipynb               # A-F end to end; writes report_outputs/
```

Other options:

* `--engine spark` runs step 7 with PySpark (same logic as the Glue job in Step 5).
* `--part-key report` runs the original Table 13 key (category + name) for comparison; the default is `extended` (decision D1, see "Changes after the independent review").
* `--simulate-model-failure` tests "keep the last good model".
* `--inject-from labels --labels-file labels/labels_resolved.csv` injects incompatible builds from the held-out labelled pairs, as Table 11 asks, once the labels exist.

## Pipeline (Table 12) and data lake

| Step | Module | Output (zone) |
|---|---|---|
| 1 | `step01_load_raw.py` | `raw/pc_part_dataset` + `COMMIT.txt` |
| 2 | `step02_generate.py` | `raw/builder_events/*.jsonl`, `raw/stock_snapshots`, `raw/sales_daily`, `raw/true_demand`, `raw/synthetic_settings` |
| 3 | `step03_clean_parts.py` | `cleaned/parts_*`, `parts_cleaning_log`, `merge_conflicts` |
| 4 | `step04_curate_parts.py` (+ `reference.py`) | `curated/dim_part`, `ref_*`, `dq_issue`, `attribute_coverage` |
| 5 | `step05_clean_events.py` | `cleaned/builder_event`, `stock_snapshot`, `sales_daily`, `sales_true_demand`, `quarantine/builder_event`, `event_validation_log` |
| 6 | `step06_builds.py` | `curated/build_item`, `builds_per_day` |
| 7 | `step07_cooccurrence.py` | `outputs/models/cooccurrence/<version>/`, `LATEST_GOOD.json`, `outputs/model_cooccurrence` |
| 8 | `step08_replay.py` (+ `rules.py`, `recommender.py`) | `outputs/compat_check`, `build_verdict`, `reco_next_part`, `replay_log` |
| 9 | `step09_focus_a.py` (+ `forecasting.py`, `policy.py`) | `outputs/forecast`, `fa_order`, `fa_alert`, `fa_sim_daily` |
| 10 | `step10_evaluate.py` | `outputs/evaluation_results` (Table 11), `eval_*` |
| 11 | `step11_dashboard.py` | `outputs/dashboard_*` (app); screenshots for the report are taken by hand |

Tables are stored at `lake/<zone>/<dataset>/run_id=<id>/data.parquet`, with a CSV fallback if pyarrow is missing. Each run writes a catalog to `lake/catalog/run_<id>.json`: dataset, zone, source, row count, run ID, creation time, and code and reference-table hashes. This catalog is the data-lineage evidence for Step 7.

## Labels (team decision 8)

```powershell
python scripts/draw_label_sample.py --run-id demo          # 150 pairs (50 tuning / 100 test) + 30 triples, seed 2026
# two members fill labels/labelling_sheet_labeller_A.csv and _B.csv from datasheets;
# a third member fills labels_resolved_TEMPLATE.csv -> save as labels_resolved.csv
python scripts/evaluate_labels.py --sheet-a labels/labelling_sheet_labeller_A.csv --sheet-b labels/labelling_sheet_labeller_B.csv `
   --resolved labels/labels_resolved.csv --manifest labels/sample_manifest.csv --run-id demo --out labels/results
python run_pipeline.py --run-id demo --step 10 --labels-dir labels/results
```

The sheets do not show the engine's verdict. The scripts were tested only with labels marked DUMMY, and any output computed from dummy labels is marked "DUMMY – NOT EVIDENCE".

## Changes after the independent review (P7 → P8)

ChatGPT reviewed the code independently (P7) and wrote 19 adversarial tests; 11 failed. All nine reported issues were checked against the code and confirmed. P8 fixed them; the 19 tests are kept in `tests/test_independent_review.py` (two amended, see the comments) and 4 regression tests in `tests/test_p8_fixes.py`. Now: **58 passed**.

| P7 issue | Fix |
|---|---|
| 01 rules | Ryzen model patterns need a model number; a board-name memory marker that contradicts a single-type socket → UNKNOWN (conflict logged); text markers such as "UNKNOWN" count as missing |
| 02 time | events are split at the training cutoff **before** builds are assembled; replay continues builds from their state at the cutoff; a late event never replaces a newer part |
| 03 Focus A | analogue curve uses only sales up to the forecast origin; lead times only from deliveries that had arrived; orders already on the way at the hold-out start are restored; censoring from "shelf empty", not from hidden demand; analogue days use the analogue's own observed errors |
| 04 labels | resolved-labels template carries the categories; an empty labelled set is an error; labelled pairs are injected only on replay days |
| 05 evaluation | every injected build is accounted for (training-only, incomplete, scored); stock age audited independently; forecast method chosen **before** scoring ("selected policy") |
| 06 κ, sample | κ undefined (NaN) when both raters use one class; incomplete or duplicated label sheets are rejected; triples shuffled; sample rates marked as challenge-sample rates |
| 07 validation, keys | non-object JSON and NaT quarantined; event category must match the part; fact tables have declared keys, conflicting rows quarantined |
| 08 model | each run uses its own model version; models are written to a staging folder and switched only when complete |
| 09 labels on outputs | every exported CSV states real or SYNTHETIC; reference rows: **193**, all `verified = no` |

Team decisions D1–D4 (after P7): **D1** extended part key adopted (Table 13 amended; `cleaned/part_key_crosswalk` maps old ↔ new ids). **D2** the report's on-hand alert is kept; each alert also records whether an open order is due before the shelf runs empty. **D3** the ≥ 90 % coverage target is kept and the miss is reported with its causes. **D4** the analogue SKU not beating its baseline is reported as a result.

UI (team test on Windows): suggestions to replace the failing part after a FAIL; Focus A legends below the plots; pandas FutureWarning removed; `pyparsing` pinned.

## Deviations from the report and open points

1. **Part key (Table 13 amended, D1).** "Category + normalised name" merged different products that share a name (511 GPU names covered several chipsets, 415 PSU names several wattages). The default key now adds the variant fields (chipset and VRAM; wattage, type and efficiency; memory generation, speed and modules). With the extended key, 99 % of the assortment's graphics cards and 100 % of its power supplies have a known power value.
2. **Injection source.** Table 11 asks for incompatible builds created from held-out labelled pairs. Until the labels exist, the generator injects violations using its own platform table, which is independent of the rule code.
3. **Coverage diagnostic.** The ≥ 90 % target is reported (met or not met) together with its causes. pytest checks the mechanics property instead: "five suggestions whenever the specification allows five".
4. **Lowest-level choices [A]** that the report leaves open:
   * a rule is checked only when all of its parts are present;
   * the next category follows the fixed builder order;
   * the build's price band comes from its most recently added priced part;
   * `MIN_SUPPORT = 3`;
   * the observable stock-out flag is "shelf empty at the end of the day";
   * d̄ is the mean forecast over L + R;
   * σd defaults to √d̄ when there are fewer than 14 errors;
   * orders are rounded up to whole units.
