# Run log – AI agent sandbox (2026-11-18)

**Environment:** Linux, Python 3.13.18, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2, pytest 9.1.1.

PyPI was blocked in the sandbox (proxy 403), so the pinned requirements (pandas 2.2.3, pyarrow, streamlit, jupyter, pyspark) could **not** be installed. As a result:

* Tables were stored as CSV, the fallback in `lake.py`. The Parquet path was **not run**.
* The Streamlit app was only compiled (`py_compile`); it was **not started**. Its data back-end is covered by pytest.
* The notebook's code cells were executed as one script, without Jupyter. The `.ipynb` file is delivered without outputs.
* The `--engine spark` back-end was **not run**.

The team must repeat everything below on the Windows laptop with `requirements.txt`.

## Commands and observed results

```
python3 -m py_compile app/Home.py app/pages/*.py techforge/*.py scripts/*.py run_pipeline.py   -> OK
notebook code (run_pipeline.run("demo", steps 1-11, seed 42))                                  -> total 66.7 s
  step 2 29.2 s | step 3 8.1 s | step 4 4.7 s | step 8 14.5 s | step 9 4.8 s | others < 3 s
python3 -m pytest -q                                                                           -> 34 passed in 22.4 s
python3 run_pipeline.py --run-id demo_ext --part-key extended --lake /tmp/extlake              -> completed
python3 scripts/draw_label_sample.py --run-id demo                                             -> 150 pairs + 30 triples
```

Table 11 (demo run) is in `report_outputs/table11_evaluation_results.csv`. Accuracy rows are pending labels; all other rows are mechanics checks on SYNTHETIC data.
