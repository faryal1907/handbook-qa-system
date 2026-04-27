# Experiments

These scripts are standalone analyses run from the **project root** after the main system is set up.

> **Prerequisite:** Activate your virtual environment and ensure `chunks.json` exists (run `python src/preprocess.py` if not).

---

## Scripts

### `dataset_scaling.py`
Tests how retrieval performance changes as the dataset size scales up (e.g. 25%, 50%, 75%, 100% of chunks).

```bash
python src/experiments/dataset_scaling.py
```

---

### `parameter_sensitivity.py`
Sweeps over key LSH parameters (`threshold`, `num_perm`) to show how they affect result quality.

```bash
python src/experiments/parameter_sensitivity.py
```

---

### `scalability_test.py`
Benchmarks query latency for each retriever (TF-IDF, LSH, SimHash) as corpus size grows.

```bash
python src/experiments/scalability_test.py
```

---

### `simhash_analysis.py`
Analyses SimHash fingerprint distance distributions across the chunk corpus.

```bash
python src/experiments/simhash_analysis.py
```

---

## Notes
- All scripts read `chunks.json` via `config.py` — no hardcoded paths.
- Results are printed to stdout. Redirect to a file if needed:
  ```bash
  python src/experiments/scalability_test.py > outputs/scalability_results.txt
  ```
