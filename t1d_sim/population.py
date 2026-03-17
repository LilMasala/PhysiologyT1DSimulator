"""Population sampling."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from t1d_sim.constants import PERSONAS


@dataclass
class PatientConfig:
    patient_id: str
    persona: str
    is_female: bool
    activity_propensity: float
    sleep_regularity: float
    sleep_total_min_mean: float
    sleep_efficiency: float
    sleep_schedule_offset_h: float
    stress_reactivity: float
    stress_baseline: float
    cycle_sensitivity: float
    mood_stability: float
    meal_regularity: float
    meal_schedule_offset_h: float
    meal_size_multiplier: float
    n_meals_per_day_mean: float
    skips_breakfast_p: float
    skips_lunch_p: float
    luteal_meal_size_boost: float
    luteal_mood_drop: float
    exercise_intensity_mean: float
    logging_quality_raw: float
    fitness_level: float
    base_rhr: float
    isf_multiplier: float
    cr_multiplier: float
    basal_multiplier: float
    base_patient_name: str
    seed: int
    n_days: int
    split: str

    @property
    def logging_quality(self) -> str:
        if self.logging_quality_raw > 0.7:
            return "good"
        if self.logging_quality_raw >= 0.35:
            return "mediocre"
        return "poor"

    def to_record(self) -> dict:
        d = asdict(self)
        d["logging_quality"] = self.logging_quality
        return d


def sample_population(n_patients: int, seed: int = 42, male_fraction: float = 0.45) -> list[PatientConfig]:
    """Sample synthetic patient configurations."""
    rng = np.random.default_rng(seed)
    personas = list(PERSONAS.keys())
    patients: list[PatientConfig] = []

    def trait(persona: dict, key: str, lo: float = 0.0, hi: float = 1.0) -> float:
        mean, std = persona.get(key, (0.5, 0.2))
        return float(np.clip(rng.normal(mean, std), lo, hi))

    for i in range(n_patients):
        is_female = rng.random() > male_fraction
        eligible = [(p, PERSONAS[p]["weight"]) for p in personas if is_female or p not in ("cycle_sensitive", "cycle_insensitive")]
        p_names, p_weights = zip(*eligible)
        weights = np.array(p_weights, dtype=float)
        weights /= weights.sum()
        persona_name = str(rng.choice(p_names, p=weights))
        persona = PERSONAS[persona_name]
        patients.append(PatientConfig(
            patient_id=f"sim_{'f' if is_female else 'm'}_{i:03d}",
            persona=persona_name,
            is_female=is_female,
            activity_propensity=trait(persona, "activity_propensity"),
            sleep_regularity=trait(persona, "sleep_regularity"),
            sleep_total_min_mean=float(np.clip(rng.normal(*persona.get("sleep_total_min_mean", (420, 45))), 240, 570)),
            sleep_efficiency=float(np.clip(rng.normal(*persona.get("sleep_efficiency", (0.84, 0.07))), 0.55, 0.97)),
            sleep_schedule_offset_h=float(rng.normal(*persona.get("sleep_schedule_offset_h", (0.0, 0.3)))),
            stress_reactivity=trait(persona, "stress_reactivity"),
            stress_baseline=trait(persona, "stress_baseline") if "stress_baseline" in persona else float(np.clip(rng.normal(0.15, 0.10), 0, 0.6)),
            cycle_sensitivity=trait(persona, "cycle_sensitivity") if is_female else 0.0,
            mood_stability=trait(persona, "mood_stability"),
            meal_regularity=trait(persona, "meal_regularity"),
            meal_schedule_offset_h=float(rng.normal(*persona.get("meal_schedule_offset_h", (0.0, 0.2)))),
            meal_size_multiplier=float(np.clip(rng.normal(*persona.get("meal_size_multiplier", (1.0, 0.12))), 0.4, 2.0)),
            n_meals_per_day_mean=float(np.clip(rng.normal(*persona.get("n_meals_per_day", (3.0, 0.5))), 1.0, 7.0)),
            skips_breakfast_p=float(persona.get("skips_breakfast_p", 0.15)),
            skips_lunch_p=float(persona.get("skips_lunch_p", 0.08)),
            luteal_meal_size_boost=float(persona.get("luteal_meal_size_boost", (0.08, 0.03))[0]),
            luteal_mood_drop=float(persona.get("luteal_mood_drop", (0.15, 0.05))[0]),
            exercise_intensity_mean=float(np.clip(rng.normal(*persona.get("exercise_intensity_mean", (0.5, 0.15))), 0.1, 1.0)),
            logging_quality_raw=trait(persona, "logging_quality"),
            fitness_level=trait(persona, "fitness_level"),
            base_rhr=float(np.clip(rng.normal(*persona.get("base_rhr", (63, 6))), 45, 85)),
            isf_multiplier=float(np.clip(rng.normal(*persona.get("isf_base_multiplier", (1.0, 0.12))), 0.70, 1.35)),
            cr_multiplier=float(np.clip(rng.normal(*persona.get("cr_base_multiplier", (1.0, 0.12))), 0.70, 1.35)),
            basal_multiplier=float(np.clip(rng.normal(1.0, 0.10), 0.75, 1.25)),
            base_patient_name=f"adult#0{rng.integers(1,10):02d}",
            seed=int(rng.integers(0, 1_000_000)),
            n_days=180,
            split="train",
        ))
    return patients
