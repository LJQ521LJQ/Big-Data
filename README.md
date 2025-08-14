# NYC Taxi ML: Temporal & Weather-Aware Modelling

This repository provides a complete, reproducible pipeline to:
- Clean & engineer features from NYC TLC Yellow Taxi **January 2024** trip data (Parquet).
- Join hourly **weather** features (precipitation, temperature).
- Forecast **hourly demand** with Prophet.
- Predict **fare** (XGBoost) and **duration** (Random Forest).
- Produce evaluation metrics and figures.

## Quick Start

1) **Install dependencies** (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

> If Prophet fails to build on your platform, install a compatible binary (e.g., `pip install prophet`).
> Some environments may require `cmdstanpy` to be installed and set up automatically.

2) **Prepare data** (place files in `data/` or update `config.yaml` paths):
- `yellow_tripdata_2024-01.parquet` — NYC TLC Yellow Taxi January 2024.
  - Download from NYC TLC Trip Record Data portal.
- `weather_hourly_nyc.csv` — hourly weather with at least columns:
  - `timestamp` (ISO format, e.g., `2024-01-01 00:00:00` in local NYC time or UTC),
  - `precip_mm` (numeric),
  - `temp_c` (numeric).

3) **Configure** paths & hyperparameters in `config.yaml`.

4) **Run the full pipeline**:
```bash
python -m src.main --config config.yaml
```

Outputs go to `outputs/`:
- `metrics.json`
- `forecast_prophet.csv`
- `models/*.joblib`
- Figures in `outputs/figures/`

## Notes
- Visualisations use **matplotlib** only (no seaborn) and don't specify colors.
- For speed during testing, set `sample_n_rows` in `config.yaml` to a smaller number (e.g., 1_000_000).

## Citation
This repo is a practical implementation corresponding to an academic analysis/report that uses:
- Prophet for hourly demand forecasting.
- XGBoost for fare prediction.
- Random Forest for duration prediction.
