"""Physiology model wrapper."""
from __future__ import annotations

from enum import Enum
import numpy as np

from t1d_sim.behavior import ContextState


class MealProfile(str, Enum):
    """Meal absorption profile types."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


_MEAL_PROFILE_PARAMS: dict[MealProfile, tuple[int, float, float]] = {
    MealProfile.FAST: (48, 0.22, 1.20),
    MealProfile.MEDIUM: (72, 0.12, 1.00),
    MealProfile.SLOW: (96, 0.075, 0.85),
}


def _meal_absorption_kernel(profile: MealProfile) -> np.ndarray:
    """Dual-exponential meal absorption kernel in 5-min bins."""
    length, decay, gain = _MEAL_PROFILE_PARAMS[profile]
    t = np.arange(length, dtype=float)
    # Difference of exponentials: fast gastric emptying vs slower elimination tail.
    gastric = np.exp(-t * decay * 0.55)
    elimination = np.exp(-t * decay * 1.35)
    kernel = np.maximum(0.0, (gastric - elimination) * gain)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return np.zeros(length, dtype=float)
    return kernel / kernel_sum


_MEAL_KERNELS: dict[MealProfile, np.ndarray] = {
    profile: _meal_absorption_kernel(profile) for profile in MealProfile
}


def _parse_meal_event(meal: tuple, rng: np.random.Generator) -> tuple:
    """Normalize meal event tuple as (time, carbs, profile)."""
    if len(meal) == 2:
        t, carbs = meal
        profile = MealProfile.MEDIUM
    else:
        t, carbs, raw_profile = meal[:3]
        if raw_profile is None:
            profile = MealProfile.MEDIUM
        elif isinstance(raw_profile, MealProfile):
            profile = raw_profile
        elif str(raw_profile) == "grazer":
            profile = rng.choice(
                [MealProfile.FAST, MealProfile.MEDIUM, MealProfile.SLOW],
                p=[0.30, 0.45, 0.25],
            )
        else:
            profile = MealProfile(str(raw_profile).lower())
    return t, float(carbs), profile


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

    # Research: Spiegel et al. (1999) Lancet — 14% SI drop per 4h restriction → λ=0.035/hr
    # Van Cauter lab: 20-30% EGP rise from overnight deprivation
    # Reference baseline: 8.5h (510 min) — used in sleep deprivation studies
    SLEEP_REF_MIN = 510.0
    sleep_deficit_h = max(0.0, (SLEEP_REF_MIN - ctx.sleep_min_last_night) / 60.0)
    lam = 0.035  # 14% SI drop per 4h = 0.035/hr
    si_reduction = min(0.30, lam * sleep_deficit_h) * (0.7 + 0.3 * ctx.stress_reactivity)
    params["k1"] *= (1.0 - si_reduction)
    params["k2"] *= (1.0 - si_reduction * 0.6)
    egp_rise = min(0.28, 0.025 * sleep_deficit_h * (0.7 + 0.3 * ctx.stress_reactivity))
    params["EGP0"] *= (1.0 + egp_rise)
    # Sleep fragmentation penalty — T1D population mean efficiency is 0.82, not 0.80
    # Research: fragmentation produces ~25% SI reduction equivalent to similar total deprivation
    if ctx.sleep_efficiency < 0.82:
        frag_penalty = 0.25 * max(0.0, (0.82 - ctx.sleep_efficiency) / 0.82)
        params["k1"] *= (1.0 - frag_penalty * 0.5)
        params["EGP0"] *= (1.0 + frag_penalty * 0.3)

    # Research tiers (from pharmacological stress studies):
    #   mild psych stress (0.1-0.3):  5-10% TDI increase
    #   acute TSST/cold (0.3-0.6):    13-30% TDI increase
    #   pharmacological-like (0.6-0.9): 40-70% TDI increase
    if ctx.stress > 0.1:
        if ctx.stress < 0.3:
            stress_egp_factor = 0.12 * (ctx.stress - 0.1)
        elif ctx.stress < 0.6:
            stress_egp_factor = 0.024 + 0.35 * (ctx.stress - 0.3)
        else:
            stress_egp_factor = 0.129 + 0.65 * (ctx.stress - 0.6)
        effect = 1.0 + stress_egp_factor * ctx.stress_reactivity
        params["EGP0"] *= effect
        params["k1"] *= (1.0 / (1.0 + 0.45 * (effect - 1.0)))
    # Chronic background stress — 10-20% persistent TDI increase
    if ctx.stress_baseline > 0.15:
        chronic = min(0.20, 0.15 * (ctx.stress_baseline - 0.15) * ctx.stress_reactivity)
        params["EGP0"] *= (1.0 + chronic)
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
    for meal in meals:
        t, carbs, profile = _parse_meal_event(meal, rng)
        idx = int((t.hour * 60 + t.minute) / 5)
        kernel = _MEAL_KERNELS[profile]
        end = min(288, idx + kernel.size)
        window = end - idx
        if window > 0:
            meal_effect[idx:end] += carbs * kernel[:window] * 9.0
    sens = modified_params["k1"] / max(1e-6, base_params["k1"])
    egp = modified_params["EGP0"] / max(1e-6, base_params["EGP0"])
    for i in range(1, 288):
        drift = (egp - 1.0) * 1.8 - (sens - 1.0) * 1.2
        bg[i] = max(45.0, bg[i-1] + 0.06 * meal_effect[i] + drift + rng.normal(0, 2.8))
    return bg
