from datetime import datetime
import pandas as pd
import numpy as np

def load_trips_parquet(path: str, sample_n_rows=None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if sample_n_rows is not None and len(df) > sample_n_rows:
        df = df.sample(sample_n_rows, random_state=42)
    # Standardise timestamp columns (TLC schema columns commonly named tpep_pickup_datetime / tpep_dropoff_datetime)
    pickup_col = 'tpep_pickup_datetime' if 'tpep_pickup_datetime' in df.columns else 'pickup_datetime'
    dropoff_col = 'tpep_dropoff_datetime' if 'tpep_dropoff_datetime' in df.columns else 'dropoff_datetime'
    df[pickup_col] = pd.to_datetime(df[pickup_col])
    df[dropoff_col] = pd.to_datetime(df[dropoff_col])
    df['pickup_dt'] = df[pickup_col]
    df['dropoff_dt'] = df[dropoff_col]
    # Duration minutes
    df['trip_duration_min'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60.0
    return df

def clean_trips(df: pd.DataFrame, max_fare: float, min_distance: float, min_passengers: int) -> pd.DataFrame:
    df = df.copy()
    # basic columns
    if 'fare_amount' not in df.columns and 'fare' in df.columns:
        df['fare_amount'] = df['fare']
    if 'trip_distance' not in df.columns and 'distance' in df.columns:
        df['trip_distance'] = df['distance']

    mask = (
        (df['trip_distance'] > min_distance) &
        (df['fare_amount'] > 0) &
        (df['fare_amount'] <= max_fare) &
        (df.get('passenger_count', 1) >= min_passengers) &
        (df['trip_duration_min'] > 0)
    )
    df = df.loc[mask].copy()
    # Temporal features
    df['hour'] = df['pickup_dt'].dt.hour
    df['weekday'] = df['pickup_dt'].dt.weekday  # Monday=0
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
    df['day_type'] = np.where(df['is_weekend'] == 1, 'weekend', 'weekday')
    # Floor to the hour for joins
    df['pickup_hour'] = df['pickup_dt'].dt.floor('H')
    return df

def aggregate_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.groupby('pickup_hour').size().reset_index(name='trip_count')
    hourly = hourly.sort_values('pickup_hour')
    hourly = hourly.rename(columns={'pickup_hour': 'ds'})
    return hourly
