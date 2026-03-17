"""SQLite backend writer."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from t1d_sim.writers.base_writer import BaseWriter


SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (user_id TEXT PRIMARY KEY, split TEXT, logging_quality TEXT, is_female INTEGER, activity_propensity REAL, sleep_regularity REAL, stress_reactivity REAL, cycle_sensitivity REAL, mood_stability REAL, meal_regularity REAL, base_rhr REAL, fitness_level REAL, base_patient_name TEXT, isf_multiplier REAL, cr_multiplier REAL, basal_multiplier REAL);
CREATE TABLE IF NOT EXISTS bg_hourly (user_id TEXT, hour_utc TEXT, start_bg REAL, end_bg REAL, avg_bg REAL, percent_low REAL, percent_high REAL, uroc REAL, expected_end_bg REAL, therapy_profile_id TEXT, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS hr_hourly (user_id TEXT, hour_utc TEXT, heart_rate REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS energy_hourly (user_id TEXT, hour_utc TEXT, basal_energy REAL, active_energy REAL, total_energy REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS exercise_hourly (user_id TEXT, hour_utc TEXT, move_minutes REAL, exercise_minutes REAL, total_minutes REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS sleep_daily (user_id TEXT, date_utc TEXT, awake REAL, asleep_core REAL, asleep_deep REAL, asleep_rem REAL, asleep_unspecified REAL, PRIMARY KEY (user_id, date_utc));
CREATE TABLE IF NOT EXISTS menstrual_daily (user_id TEXT, date_utc TEXT, days_since_period_start INTEGER, PRIMARY KEY (user_id, date_utc));
CREATE TABLE IF NOT EXISTS site_changes_daily (user_id TEXT, date_utc TEXT, days_since_change INTEGER, location TEXT, PRIMARY KEY (user_id, date_utc));
CREATE TABLE IF NOT EXISTS therapy_settings_hourly (user_id TEXT, hour_utc TEXT, profile_id TEXT, profile_name TEXT, carb_ratio REAL, basal_rate REAL, insulin_sensitivity REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS mood_hourly (user_id TEXT, hour_utc TEXT, valence REAL, arousal REAL, quad_pos_pos INTEGER, quad_pos_neg INTEGER, quad_neg_pos INTEGER, quad_neg_neg INTEGER, hours_since_mood REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS mood_events (user_id TEXT, event_id TEXT, timestamp_utc TEXT, valence REAL, arousal REAL, PRIMARY KEY (user_id, event_id));
CREATE TABLE IF NOT EXISTS feature_frames (user_id TEXT, hour_utc TEXT, bg_avg REAL, bg_tir REAL, bg_percent_low REAL, bg_percent_high REAL, bg_uroc REAL, bg_delta_avg_7h REAL, bg_z_avg_7h REAL, hr_mean REAL, hr_delta_7h REAL, hr_z_7h REAL, rhr_daily REAL, kcal_active REAL, kcal_active_last3h REAL, kcal_active_last6h REAL, kcal_active_delta_7h REAL, kcal_active_z_7h REAL, sleep_prev_total_min REAL, sleep_debt_7d_min REAL, minutes_since_wake INTEGER, ex_move_min REAL, ex_exercise_min REAL, ex_min_last3h REAL, ex_hours_since REAL, days_since_period_start INTEGER, cycle_follicular INTEGER, cycle_ovulation INTEGER, cycle_luteal INTEGER, days_since_site_change INTEGER, site_loc_current TEXT, site_loc_same_as_last INTEGER, mood_valence REAL, mood_arousal REAL, mood_quad_pos_pos INTEGER, mood_quad_pos_neg INTEGER, mood_quad_neg_pos INTEGER, mood_quad_neg_neg INTEGER, mood_hours_since REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS ground_truth_daily (user_id TEXT, date_utc TEXT, true_isf REAL, true_cr REAL, true_basal REAL, true_meal_times TEXT, true_meal_carbs TEXT, true_exercise_min INTEGER, true_exercise_intensity REAL, true_sleep_min INTEGER, true_cycle_phase TEXT, true_mood_valence REAL, true_mood_arousal REAL, true_stress REAL, PRIMARY KEY (user_id, date_utc));
"""


class SQLiteWriter(BaseWriter):
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def write_patient(self, payload: dict) -> None:
        c = self.conn.cursor()
        p = payload["patient"]
        c.execute("INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            p["patient_id"], p["split"], p["logging_quality"], int(p["is_female"]), p["activity_propensity"], p["sleep_regularity"], p["stress_reactivity"], p["cycle_sensitivity"], p["mood_stability"], p["meal_regularity"], p["base_rhr"], p["fitness_level"], p["base_patient_name"], p["isf_multiplier"], p["cr_multiplier"], p["basal_multiplier"],
        ))
        c.executemany("INSERT OR REPLACE INTO bg_hourly VALUES (?,?,?,?,?,?,?,?,?,?)", payload["bg_hourly"])
        c.executemany("INSERT OR REPLACE INTO hr_hourly VALUES (?,?,?)", payload["hr_hourly"])
        c.executemany("INSERT OR REPLACE INTO energy_hourly VALUES (?,?,?,?,?)", payload["energy_hourly"])
        c.executemany("INSERT OR REPLACE INTO exercise_hourly VALUES (?,?,?,?,?)", payload["exercise_hourly"])
        c.executemany("INSERT OR REPLACE INTO sleep_daily VALUES (?,?,?,?,?,?,?)", payload["sleep_daily"])
        c.executemany("INSERT OR REPLACE INTO menstrual_daily VALUES (?,?,?)", payload["menstrual_daily"])
        c.executemany("INSERT OR REPLACE INTO site_changes_daily VALUES (?,?,?,?)", payload["site_daily"])
        c.executemany("INSERT OR REPLACE INTO therapy_settings_hourly VALUES (?,?,?,?,?,?,?)", payload["therapy"])
        c.executemany("INSERT OR REPLACE INTO mood_hourly VALUES (?,?,?,?,?,?,?,?,?)", payload["mood_hourly"])
        c.executemany("INSERT OR REPLACE INTO mood_events VALUES (?,?,?,?,?)", payload["mood_events"])
        c.executemany("INSERT OR REPLACE INTO ground_truth_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", payload["ground_truth"])
        self.conn.commit()

    def write_features(self, rows: list[tuple]) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO feature_frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def raw_for_features(self):
        return self.conn.execute("SELECT b.user_id,b.hour_utc,b.avg_bg,b.percent_low,b.percent_high,b.uroc,h.heart_rate,e.active_energy,x.exercise_minutes,m.valence,m.arousal,s.days_since_change,s.location FROM bg_hourly b LEFT JOIN hr_hourly h ON b.user_id=h.user_id AND b.hour_utc=h.hour_utc LEFT JOIN energy_hourly e ON b.user_id=e.user_id AND b.hour_utc=e.hour_utc LEFT JOIN exercise_hourly x ON b.user_id=x.user_id AND b.hour_utc=x.hour_utc LEFT JOIN mood_hourly m ON b.user_id=m.user_id AND b.hour_utc=m.hour_utc LEFT JOIN site_changes_daily s ON b.user_id=s.user_id AND substr(b.hour_utc,1,10)=s.date_utc").fetchall()

    def finalize(self) -> None:
        self.conn.commit()
        self.conn.close()
