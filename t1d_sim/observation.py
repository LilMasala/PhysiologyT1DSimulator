"""Observation/noise model."""
from __future__ import annotations

import numpy as np


def observe_cgm(true_bg: np.ndarray, day_index: int, patient_seed: int) -> np.ndarray:
    """Apply Dexcom-like noise and missingness."""
    rng = np.random.default_rng(patient_seed * 10000 + day_index)
    obs = true_bg.copy()
    lag = int(rng.integers(1, 3))
    obs[lag:] = obs[:-lag]
    obs = obs * (1 + rng.normal(0, 0.02, size=obs.size)) + rng.normal(0, 5, size=obs.size)
    drift = np.linspace(0, rng.uniform(-3, 3), obs.size)
    obs += drift
    obs[rng.random(obs.size) < 0.002] = np.nan
    if rng.random() < 0.005:
        s = int(rng.integers(0, 270))
        obs[s:s+int(rng.integers(6, 18))] = np.nan
    if day_index % 10 == 0:
        obs[:24] = np.nan
    return obs


def synthesize_hr(base_rhr: float, stress: float, exercise_minutes: float, exercise_intensity: float, day_index: int, patient_seed: int) -> np.ndarray:
    """Produce hourly heart-rate estimates."""
    rng = np.random.default_rng(patient_seed * 10000 + day_index)
    hr = np.full(24, base_rhr)
    for h in range(24):
        if h < 6:
            hr[h] *= 0.85
        if 7 <= h <= 22:
            hr[h] *= (1 + 0.15 * stress)
    if exercise_minutes > 0:
        h = int(rng.integers(6, 20))
        boost = 1.5 if exercise_intensity < 0.6 else 1.75
        hr[h:min(24, h+2)] *= boost
    hr = hr + rng.normal(0, 3, size=24)
    hr[rng.random(24) < 0.005] = np.nan
    return hr
