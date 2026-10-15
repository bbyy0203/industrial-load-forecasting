# Public release scope

Planned GitHub repository name: `industrial-load-forecasting`.

This repository publishes source code and aggregate, anonymized model evidence.
It does **not** publish industrial minute/15-minute measurements, per-origin
predictions, local experiment outputs, station-specific configs, executed
industrial notebooks, credentials, or environment files.

The unrelated legacy German day-ahead forecasting application, dashboard,
SMARD data, model artifacts and tests are also excluded from this release.

No open-source license is granted. Public visibility is for portfolio review
and does not imply permission to copy, modify, distribute or commercially use
the project.

## Published industrial components

- Production predictor: `load15_recent_residual_predictor.py`
- CLI: `predict_recent_residual.py`
- Frozen config: `model_checkpoints/recent_residual_global_v1/model.json`
- Causality/unit tests: `tests/test_recent_residual_predictor.py` and
  `tests/test_residual_reconstruction.py`
- Data/model experiment source code for reproducibility
- Pre-registration protocol and aggregate blind-test result
- Example configuration: `station_config.example.json`

## Data contract

Users provide their own timezone-aware 15-minute load series in kW. Forecasts
require 96 complete Last-week values. The correction additionally needs the 16
most recent values and their week-lagged counterparts; otherwise inference
falls back to the pure Last-week baseline.

## Minimal verification

```powershell
D:\Anaconda\envs\load15\python.exe -m pytest tests/test_recent_residual_predictor.py tests/test_residual_reconstruction.py -q
```

No industrial data are needed for these synthetic tests.
