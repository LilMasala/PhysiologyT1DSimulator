"""Feature frame builder."""
from __future__ import annotations

import pandas as pd


def build_feature_frames(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly feature frames from merged raw hourly records."""
    df = raw.sort_values(["user_id", "hour_utc"]).copy()
    g = df.groupby("user_id", group_keys=False)
    df["bg_delta_avg_7h"] = df["avg_bg"] - g["avg_bg"].shift(24).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["bg_z_avg_7h"] = (df["avg_bg"] - g["avg_bg"].rolling(24*7, min_periods=24).mean().reset_index(level=0, drop=True)) / g["avg_bg"].rolling(24*7, min_periods=24).std().reset_index(level=0, drop=True)
    df["hr_delta_7h"] = df["heart_rate"] - g["heart_rate"].shift(24).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["hr_z_7h"] = (df["heart_rate"] - g["heart_rate"].rolling(24*7, min_periods=24).mean().reset_index(level=0, drop=True)) / g["heart_rate"].rolling(24*7, min_periods=24).std().reset_index(level=0, drop=True)
    df["kcal_active_last3h"] = g["active_energy"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["kcal_active_last6h"] = g["active_energy"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    df["ex_min_last3h"] = g["exercise_minutes"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    return df
