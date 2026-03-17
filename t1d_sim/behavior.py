"""Behavioral generator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np

from t1d_sim.population import PatientConfig
from t1d_sim.constants import SITE_LOCATIONS


@dataclass
class ContextState:
    cycle_phase: str | None
    cycle_day: int
    cycle_sensitivity: float
    sleep_min_last_night: float
    sleep_efficiency: float
    hours_since_exercise: float
    exercise_intensity: float
    stress: float
    stress_reactivity: float
    stress_baseline: float
    mood_valence: float
    mood_arousal: float
    is_ill: bool
    day_of_week: int
    is_weekend: bool


def _cycle_phase(is_female: bool, day_idx: int, cycle_len: int = 28) -> tuple[str | None, int]:
    if not is_female:
        return None, -1
    d = day_idx % cycle_len
    f = d / cycle_len
    if f < 0.18:
        return "menstrual", d
    if f < 0.45:
        return "follicular", d
    if f < 0.55:
        return "ovulation", d
    return "luteal", d


def generate_day_behavior(cfg: PatientConfig, start_day: datetime, day_index: int, prev_mood: tuple[float, float]) -> dict:
    """Generate one day of behavior and context."""
    rng = np.random.default_rng(cfg.seed * 10000 + day_index)
    phase, cycle_day = _cycle_phase(cfg.is_female, day_index)
    is_weekend = start_day.weekday() >= 5

    sleep_min = np.clip(rng.normal(cfg.sleep_total_min_mean, 50 * (1 - cfg.sleep_regularity)), 240, 600)
    stress = max(cfg.stress_baseline, float(np.clip(rng.normal(cfg.stress_baseline + 0.12 * (1 - cfg.mood_stability), 0.15), 0, 1)))
    exercise_today = rng.random() < cfg.activity_propensity
    ex_minutes = float(rng.uniform(20, 75)) if exercise_today else 0.0
    ex_intensity = float(np.clip(rng.normal(cfg.exercise_intensity_mean, 0.1), 0.1, 1.0)) if exercise_today else 0.0
    prev_v, prev_a = prev_mood
    v = float(np.clip(0.7 * prev_v + rng.normal(0, 0.4 * (1 - cfg.mood_stability)) - 0.25 * max(0, stress - 0.4), -1, 1))
    a = float(np.clip(0.7 * prev_a + rng.normal(0, 0.4 * (1 - cfg.mood_stability)) + 0.25 * stress, -1, 1))
    if phase == "luteal":
        v -= cfg.luteal_mood_drop
        v = max(-1.0, v)

    n_meals = int(np.clip(round(rng.normal(cfg.n_meals_per_day_mean, 0.8)), 1, 6))
    windows = [(8, (30, 60), cfg.skips_breakfast_p), (13, (40, 80), cfg.skips_lunch_p), (19, (50, 90), 0.05)]
    meals: list[tuple[datetime, float]] = []
    for h, (low, high), skip_p in windows:
        if rng.random() < skip_p + 0.15 * stress:
            continue
        minute_std = 15 + 75 * (1 - cfg.meal_regularity)
        hh = h + cfg.meal_schedule_offset_h + (0.7 if (is_weekend and h == 8) else 0)
        t = start_day + timedelta(hours=float(rng.normal(hh, minute_std / 60)), minutes=float(rng.normal(0, minute_std)))
        carb = float(rng.uniform(low, high) * cfg.meal_size_multiplier * (1.0 + (cfg.luteal_meal_size_boost if phase == "luteal" else 0.0)))
        meals.append((t, carb))
    while len(meals) < n_meals:
        h = float(rng.uniform(10, 22))
        t = start_day + timedelta(hours=h)
        meals.append((t, float(rng.uniform(10, 30) * cfg.meal_size_multiplier)))

    site_days = max(0, int(np.floor((day_index % 12) / 3)))
    site_loc = SITE_LOCATIONS[(day_index // 3) % len(SITE_LOCATIONS)]
    ctx = ContextState(
        cycle_phase=phase,
        cycle_day=cycle_day,
        cycle_sensitivity=cfg.cycle_sensitivity,
        sleep_min_last_night=float(sleep_min),
        sleep_efficiency=cfg.sleep_efficiency,
        hours_since_exercise=0.0 if exercise_today else 24.0,
        exercise_intensity=ex_intensity,
        stress=stress,
        stress_reactivity=cfg.stress_reactivity,
        stress_baseline=cfg.stress_baseline,
        mood_valence=v,
        mood_arousal=a,
        is_ill=bool(rng.random() < 0.01),
        day_of_week=start_day.weekday(),
        is_weekend=is_weekend,
    )
    return {
        "context": ctx,
        "meals": sorted(meals, key=lambda x: x[0]),
        "exercise_minutes": ex_minutes,
        "exercise_intensity": ex_intensity,
        "sleep_minutes": float(sleep_min),
        "site_days_since_change": site_days,
        "site_location": site_loc,
        "mood_event": {
            "timestamp": (start_day + timedelta(hours=14, minutes=int(rng.uniform(0, 59)))).astimezone(timezone.utc),
            "valence": v,
            "arousal": a,
        },
    }
