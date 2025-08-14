import pandas as pd

def load_weather_csv(path: str) -> pd.DataFrame:
    # Expected columns: timestamp, precip_mm, temp_c
    w = pd.read_csv(path)
    # Flexible timestamp naming
    ts_col = 'timestamp' if 'timestamp' in w.columns else 'datetime'
    w[ts_col] = pd.to_datetime(w[ts_col])
    w = w.rename(columns={ts_col: 'timestamp'})
    # Floor to hour
    w['timestamp'] = w['timestamp'].dt.floor('H')
    # Forward fill for any gaps after resampling to hourly
    w = w.set_index('timestamp').resample('H').ffill().reset_index()
    # Ensure required columns
    if 'precip_mm' not in w.columns:
        w['precip_mm'] = 0.0
    if 'temp_c' not in w.columns:
        w['temp_c'] = 0.0
    return w

def join_weather(df_trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    # Join on floored pickup_hour
    merged = df_trips.merge(weather, left_on='pickup_hour', right_on='timestamp', how='left')
    merged = merged.drop(columns=['timestamp'])
    # Fill missing with 0 for precipitation and with forward/back fill for temperature
    merged['precip_mm'] = merged['precip_mm'].fillna(0.0)
    merged['temp_c'] = merged['temp_c'].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    return merged
