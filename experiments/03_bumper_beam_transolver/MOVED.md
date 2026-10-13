# Code, papers and small results moved to PINNeAPPle-Benchmark (2026-10-02)

`src/`, `paper/` (incl. both PDFs), `figures/`, `model/` (the licensed Altair deck) and the small part of
`results/` (results.json, the .npz examples the papers' figures read) now live in
`PINNeAPPle-Benchmark/experiments/03_bumper_beam_transolver/`. That is the canonical copy.

What stays **only** here, because it is too large to commit to any git repo:

- `data/` (1.4 GB — the packaged `training_data.hdf5`; regenerate with `src/package_hdf5.py` once `runs/`
  exists, from the Benchmark copy)
- `runs/` (9.4 GB — raw OpenRadioss output for all 100 DoE points; regenerate with `src/run_doe.py`,
  ~2 days of wall-clock time including Docker/OpenRadioss overhead)
- `solver/` (516 MB — the OpenRadioss install itself)
- `results_smoke/` (smoke-test artifacts, not the reported results)
