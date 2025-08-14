import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utils import ensure_dir

def plot_hourly_line(hourly_df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    fig, ax = plt.subplots()
    ax.plot(hourly_df['ds'], hourly_df['trip_count'])
    ax.set_title('Hourly Trip Count')
    ax.set_xlabel('Time')
    ax.set_ylabel('Trips')
    fig.autofmt_xdate()
    fig.savefig(os.path.join(out_dir, 'hourly_trip_count.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

def plot_residual_hist(residuals: np.ndarray, out_dir: str, name: str):
    ensure_dir(out_dir)
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=50)
    ax.set_title(f'Residual Distribution: {name}')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    fig.savefig(os.path.join(out_dir, f'{name}_residuals.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

def plot_feature_importance(model, feature_names, out_dir: str, name: str):
    ensure_dir(out_dir)
    # Works for tree-based models that expose feature_importances_ after preprocessing? We plot only when plain model is used.
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots()
        ax.bar(range(len(importances)), importances[idx])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right')
        ax.set_title(f'Feature Importance: {name}')
        ax.set_ylabel('Importance')
        fig.savefig(os.path.join(out_dir, f'{name}_feature_importance.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)
