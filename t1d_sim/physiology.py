"""Physiology model wrapper."""
from __future__ import annotations

import numpy as np

from t1d_sim.behavior import ContextState


def apply_context_effectors(base_params: dict, ctx: ContextState) -> dict:
    """Apply context-driven parameter modulation."""
    params = base_params.copy()
    if ctx.hours_since_exercise < 48:
        decay = np.exp(-ctx.hours_since_exercise / 16.0)
        max_boost = 0.40 * ctx.exercise_intensity
        boost = 1.0 + max_boost * decay
        params["k1"] *= boost
        params["k2"] *= boost
    if ctx.cycle_phase == "luteal":
        r = 1.0 - (0.03 + 0.15 * ctx.cycle_sensitivity)
        params["k1"] *= r
        params["k2"] *= r
    elif ctx.cycle_phase == "menstrual":
        params["k1"] *= 1.0 - (0.02 + 0.06 * ctx.cycle_sensitivity)
    elif ctx.cycle_phase == "follicular":
        params["k1"] *= 1.0 + 0.02 * ctx.cycle_sensitivity
        params["k2"] *= 1.0 + 0.02 * ctx.cycle_sensitivity
    elif ctx.cycle_phase == "ovulation":
        params["k1"] *= 1.0 + 0.05 * ctx.cycle_sensitivity
        params["k2"] *= 1.0 + 0.05 * ctx.cycle_sensitivity

    sleep_debt_h = max(0.0, (7.5 * 60 - ctx.sleep_min_last_night) / 60.0)
    sleep_resistance = 1.0 + min(0.30, 0.04 * sleep_debt_h + 0.008 * sleep_debt_h ** 2) * (0.5 + 0.5 * ctx.stress_reactivity)
    params["EGP0"] *= sleep_resistance
    params["k1"] /= (1.0 + 0.4 * (sleep_resistance - 1.0))
    if ctx.sleep_efficiency < 0.80:
        params["EGP0"] *= 1.0 + 0.08 * (0.80 - ctx.sleep_efficiency) * 10

    if ctx.stress > 0.2:
        effect = 1.0 + 0.22 * (ctx.stress - 0.2) * ctx.stress_reactivity
        params["EGP0"] *= effect
        params["k1"] *= (1.0 / (1.0 + 0.5 * (effect - 1.0)))
    if ctx.stress_baseline > 0.2:
        params["EGP0"] *= 1.0 + 0.10 * (ctx.stress_baseline - 0.2) * ctx.stress_reactivity
    if ctx.is_ill:
        params["EGP0"] *= 1.40
        params["k1"] *= 0.75
    return params


def simulate_day_cgm(base_params: dict, modified_params: dict, meals: list[tuple], seed: int) -> np.ndarray:
    """Generate 288 five-minute BG points in mg/dL."""
    rng = np.random.default_rng(seed)
    bg = np.zeros(288, dtype=float)
    bg[0] = 120.0 + rng.normal(0, 8)
    meal_effect = np.zeros(288)
    for t, carbs in meals:
        idx = int((t.hour * 60 + t.minute) / 5)
        for k in range(36):
            j = idx + k
            if j < 288:
                meal_effect[j] += carbs * np.exp(-k / 8.0) * 0.8
    sens = modified_params["k1"] / max(1e-6, base_params["k1"])
    egp = modified_params["EGP0"] / max(1e-6, base_params["EGP0"])
    for i in range(1, 288):
        drift = (egp - 1.0) * 1.8 - (sens - 1.0) * 1.2
        bg[i] = max(45.0, bg[i-1] + 0.06 * meal_effect[i] + drift + rng.normal(0, 2.8))
    return bg
