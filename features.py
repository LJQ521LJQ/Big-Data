import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def add_zone_clustering(df: pd.DataFrame, n_clusters=5, random_state=42) -> pd.DataFrame:
    # Cluster by pickup zone frequency if PULocationID exists
    if 'PULocationID' not in df.columns:
        df['PULocationID'] = -1  # fallback single cluster
    zone_counts = df.groupby('PULocationID').size().reset_index(name='pickup_count')
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    zone_counts['zone_cluster'] = km.fit_predict(zone_counts[['pickup_count']])
    df = df.merge(zone_counts[['PULocationID', 'zone_cluster']], on='PULocationID', how='left')
    df['zone_cluster'] = df['zone_cluster'].fillna(-1).astype(int)
    return df
