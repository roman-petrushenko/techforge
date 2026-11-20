# Run log after P8 (coordinator chat, 2026-11-19)

Environment: Linux, Python 3.13.x, pandas 3.0.5, CSV fallback (PyPI not reachable, so the pinned
versions, Parquet, Streamlit and Jupyter could not be installed here). The team re-runs on the
Windows laptop (Python 3.12 venv, pinned versions) - that run is the evidence for the report.

Commands and results
- `python -W error::FutureWarning run_pipeline.py --run-id demo` - all 11 steps, about 60 s, no warning
- `python -m pytest -q` - **58 passed** (34 original, 19 P7 adversarial, 1 extra P7-style, 4 P8 regression)
- `python run_pipeline.py --run-id reportkey --part-key report` - comparison run with the original key
- `python scripts/draw_label_sample.py --run-id demo` - new label sample (part ids changed with the key)
- `python -c "from techforge import report_figures; report_figures.write_all('demo')"` - figures and CSVs

P7 adversarial tests: 11 of 19 failed before P8, all 19 pass after P8 (two amended, see test comments).
Accuracy (kappa, rule accuracy) is still pending: it needs the team's labels.
