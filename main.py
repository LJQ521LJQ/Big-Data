import argparse
import json
import yaml
import os
import joblib
import pandas as pd
from .utils import ensure_dir, resolve_path
from . import data_prep, weather as wmod, features as feat, modeling, evaluation, visualization as viz

def run(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    out_dir = resolve_path(cfg['output_dir'])
    figs_dir = os.path.join(out_dir, 'figures')
    models_dir = os.path.join(out_dir, 'models')
    ensure_dir(out_dir); ensure_dir(figs_dir); ensure_dir(models_dir)

    # ---------- Load & clean trips ----------
    trips_path = resolve_path(cfg['data']['trips_parquet_path'])
    df = data_prep.load_trips_parquet(trips_path, sample_n_rows=cfg.get('sample_n_rows'))
    df = data_prep.clean_trips(df,
                               max_fare=cfg['filters']['max_fare'],
                               min_distance=cfg['filters']['min_distance'],
                               min_passengers=cfg['filters']['min_passengers'])

    # ---------- Hourly aggregation for Prophet ----------
    hourly = data_prep.aggregate_hourly_counts(df)
    # Plot line
    viz.plot_hourly_line(hourly, figs_dir)

    # ---------- Weather join ----------
    weather_path = resolve_path(cfg['data']['weather_csv_path'])
    wdf = wmod.load_weather_csv(weather_path)
    df = wmod.join_weather(df, wdf)

    # ---------- Zone clustering ----------
    df = feat.add_zone_clustering(df, n_clusters=cfg['clustering']['n_clusters'], random_state=cfg['clustering']['random_state'])

    # ---------- Prophet ----------
    prophet_model, forecast, prophet_metrics = modeling.train_prophet(hourly, cfg)
    forecast.to_csv(os.path.join(out_dir, 'forecast_prophet.csv'), index=False)
    joblib.dump(prophet_model, os.path.join(models_dir, 'prophet_model.joblib'))

    # ---------- Regressors ----------
    fare_pipe, fare_metrics, dur_pipe, dur_metrics = modeling.train_regressors(df, cfg)
    joblib.dump(fare_pipe, os.path.join(models_dir, 'xgboost_fare.joblib'))
    joblib.dump(dur_pipe, os.path.join(models_dir, 'rf_duration.joblib'))

    # ---------- Save metrics ----------
    all_metrics = {
        'prophet': prophet_metrics,
        'fare_xgboost': fare_metrics,
        'duration_random_forest': dur_metrics
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)

    print('Pipeline completed. Metrics:\n', json.dumps(all_metrics, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    run(args.config)
