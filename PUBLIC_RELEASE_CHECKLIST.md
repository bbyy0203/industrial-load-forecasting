# Public release checklist

Prepared on 2026-10-15. No remote repository has been configured and nothing
has been pushed.

- Repository name: `industrial-load-forecasting`
- License: none; public portfolio display only

## Verification completed

- Git repository initialized locally.
- Industrial source files pass Python syntax compilation.
- Six synthetic industrial tests pass when called directly:
  - causal sources precede the prediction origin;
  - mutating future actuals cannot alter a forecast;
  - missing recent residual falls back to Last-week;
  - missing Last-week input raises a hard error;
  - zero residual reconstructs Last-week exactly;
  - positive residual is added with the correct sign.
- The current environment does not have `pytest` installed, so the same test
  functions were executed directly with `D:\Anaconda\envs\load15\python.exe`.
- No common private-key, GitHub-token, Google-key, or OpenAI-key pattern was
  found in the candidate files.
- No candidate model artifact exceeds GitHub's 100 MB per-file limit.

## Explicitly excluded

- `input/`: all raw and prepared industrial station data.
- `my_outputs/`: all local predictions, per-origin metrics and run artifacts.
- `my_load15_config.json`, `station4_load15_config.json`: local station configs.
- `01_load15_*.ipynb`, `02_load15_*.ipynb`: executed industrial notebooks.
- `industrial_load_*.csv`, `industrial_load_*.parquet`.
- `.env`, caches, virtual environments and local validation logs.

## Included industrial material

- General predictor, CLI and dataset/model source code.
- Reproducible diagnostic and selection scripts.
- Synthetic causality and residual-sign tests.
- `station_config.example.json` with placeholder paths and dates.
- Frozen data-free `recent_residual_v1` and `recent_residual_global_v1` JSON.
- Pre-registration protocol and anonymized aggregate blind-test conclusion.
- Chinese project README and change log.

## Legacy repository material excluded

The pre-existing German day-ahead forecasting source, dashboard, tests,
workflows, public SMARD data and deployed model artifacts remain available in
the local workspace but are excluded from this industrial portfolio release.

## License status

No license is intentionally included. The repository may be publicly viewed,
but it must not be described as open source and no reuse permission is granted.
