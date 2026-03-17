"""Constants for t1d synthetic simulation."""

from __future__ import annotations

THERAPY_PROFILE_ID = "profile_001"
THERAPY_PROFILE_NAME = "Default"
SNAPSHOT_TIMESTAMP = "2025-03-01T00:00:00Z"

CGM_LOW = 70.0
CGM_HIGH = 180.0

SITE_LOCATIONS = [
    "abdomen_left",
    "abdomen_right",
    "thigh_left",
    "thigh_right",
    "arm_left",
    "arm_right",
]

PERSONAS = {
    "solid_sleeper": {
        "weight": 0.2,
        "sleep_regularity": (0.85, 0.08),
        "sleep_total_min_mean": (450, 30),
        "sleep_efficiency": (0.88, 0.06),
        "activity_propensity": (0.55, 0.2),
        "meal_regularity": (0.65, 0.15),
        "stress_reactivity": (0.4, 0.15),
        "mood_stability": (0.7, 0.12),
        "logging_quality": (0.6, 0.15),
        "fitness_level": (0.55, 0.2),
        "base_rhr": (60, 6),
    },
    "casual_mover": {
        "weight": 0.2,
        "activity_propensity": (0.45, 0.15),
        "exercise_intensity_mean": (0.4, 0.12),
        "sleep_regularity": (0.6, 0.15),
        "meal_regularity": (0.55, 0.15),
        "stress_reactivity": (0.45, 0.15),
        "mood_stability": (0.6, 0.12),
        "logging_quality": (0.55, 0.15),
        "fitness_level": (0.45, 0.15),
        "base_rhr": (64, 6),
    },
    "sedentary": {
        "weight": 0.2,
        "activity_propensity": (0.12, 0.08),
        "sleep_regularity": (0.5, 0.2),
        "meal_regularity": (0.4, 0.2),
        "stress_reactivity": (0.55, 0.15),
        "mood_stability": (0.5, 0.15),
        "logging_quality": (0.45, 0.15),
        "fitness_level": (0.2, 0.1),
        "base_rhr": (70, 6),
        "isf_base_multiplier": (0.85, 0.08),
    },
    "athlete": {
        "weight": 0.15,
        "activity_propensity": (0.92, 0.05),
        "exercise_intensity_mean": (0.75, 0.10),
        "sleep_regularity": (0.8, 0.10),
        "sleep_total_min_mean": (480, 30),
        "meal_regularity": (0.75, 0.10),
        "meal_size_multiplier": (1.35, 0.10),
        "stress_reactivity": (0.3, 0.15),
        "mood_stability": (0.75, 0.10),
        "logging_quality": (0.65, 0.15),
        "fitness_level": (0.9, 0.07),
        "base_rhr": (50, 5),
        "isf_base_multiplier": (1.2, 0.1),
    },
    "high_stress": {
        "weight": 0.15,
        "stress_reactivity": (0.85, 0.08),
        "stress_baseline": (0.55, 0.15),
        "mood_stability": (0.25, 0.10),
        "sleep_regularity": (0.35, 0.15),
        "sleep_total_min_mean": (360, 60),
        "meal_regularity": (0.35, 0.15),
        "activity_propensity": (0.3, 0.15),
        "logging_quality": (0.35, 0.15),
        "fitness_level": (0.3, 0.15),
        "base_rhr": (70, 6),
        "isf_base_multiplier": (0.88, 0.08),
    },
    "cycle_sensitive": {
        "weight": 0.05,
        "cycle_sensitivity": (0.9, 0.07),
        "luteal_meal_size_boost": (0.18, 0.05),
        "luteal_mood_drop": (0.35, 0.10),
    },
    "cycle_insensitive": {
        "weight": 0.05,
        "cycle_sensitivity": (0.1, 0.06),
    },
}
