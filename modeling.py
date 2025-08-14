import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .evaluation import regression_metrics, mape

def train_prophet(hourly_df: pd.DataFrame, cfg: dict):
    # hourly_df columns: ds (datetime), trip_count (int)
    model = Prophet(
        changepoint_prior_scale=cfg['prophet']['changepoint_prior_scale'],
        seasonality_prior_scale=cfg['prophet']['seasonality_prior_scale'],
        daily_seasonality=cfg['prophet']['daily_seasonality'],
        weekly_seasonality=cfg['prophet']['weekly_seasonality']
    )
    model.fit(hourly_df.rename(columns={'trip_count':'y'}))
    # Construct future only up to test_end
    test_end = pd.to_datetime(cfg['prophet']['test_end'])
    future = pd.DataFrame({'ds': pd.date_range(start=hourly_df['ds'].min(), end=test_end, freq='H')})
    forecast = model.predict(future)
    # Evaluate on test segment only
    train_end = pd.to_datetime(cfg['prophet']['train_end'])
    test_mask = (hourly_df['ds'] > train_end) & (hourly_df['ds'] <= test_end)
    y_true = hourly_df.loc[test_mask, 'trip_count'].values
    y_pred = forecast.set_index('ds').loc[hourly_df.loc[test_mask, 'ds'], 'yhat'].values
    metrics = {'MAPE': mape(y_true, y_pred),
               'RMSE': float(np.sqrt(np.mean((y_true - y_pred)**2)))}
    return model, forecast, metrics

def _build_regression_pipeline(model, categorical_features, numeric_features):
    preproc = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(with_mean=False), numeric_features)
        ]
    )
    pipe = Pipeline(steps=[('prep', preproc), ('mdl', model)])
    return pipe

def train_regressors(df: pd.DataFrame, cfg: dict):
    # Feature set for fare & duration
    # Choose widely available columns to avoid schema surprises
    features = ['trip_distance', 'hour', 'is_weekend', 'precip_mm', 'temp_c', 'zone_cluster']
    df_model = df.dropna(subset=features + ['fare_amount', 'trip_duration_min']).copy()

    X = df_model[features]
    y_fare = df_model['fare_amount']
    y_dur = df_model['trip_duration_min']

    cat_feats = ['hour', 'is_weekend', 'zone_cluster']
    num_feats = ['trip_distance', 'precip_mm', 'temp_c']

    # Fare: XGBoost
    xgb = XGBRegressor(
        n_estimators=cfg['regression']['xgboost']['n_estimators'],
        max_depth=cfg['regression']['xgboost']['max_depth'],
        learning_rate=cfg['regression']['xgboost']['learning_rate'],
        subsample=cfg['regression']['xgboost']['subsample'],
        colsample_bytree=cfg['regression']['xgboost']['colsample_bytree'],
        random_state=cfg['regression']['random_state']
    )
    pipe_fare = _build_regression_pipeline(xgb, cat_feats, num_feats)

    # Duration: RandomForest
    rf = RandomForestRegressor(
        n_estimators=cfg['regression']['random_forest']['n_estimators'],
        max_depth=cfg['regression']['random_forest']['max_depth'],
        min_samples_split=cfg['regression']['random_forest']['min_samples_split'],
        random_state=cfg['regression']['random_state'],
        n_jobs=-1
    )
    pipe_dur = _build_regression_pipeline(rf, cat_feats, num_feats)

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X, y_fare, test_size=cfg['regression']['test_size'], random_state=cfg['regression']['random_state'], stratify=X['hour']
    )
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X, y_dur, test_size=cfg['regression']['test_size'], random_state=cfg['regression']['random_state'], stratify=X['hour']
    )

    pipe_fare.fit(X_train_f, y_train_f)
    fare_pred = pipe_fare.predict(X_test_f)
    fare_metrics = regression_metrics(y_test_f, fare_pred)

    pipe_dur.fit(X_train_d, y_train_d)
    dur_pred = pipe_dur.predict(X_test_d)
    dur_metrics = regression_metrics(y_test_d, dur_pred)

    return pipe_fare, fare_metrics, pipe_dur, dur_metrics
