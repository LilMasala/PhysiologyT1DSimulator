"""Per-patient simulation orchestrator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import uuid
import numpy as np

from t1d_sim.behavior import generate_day_behavior
from t1d_sim.constants import THERAPY_PROFILE_ID, THERAPY_PROFILE_NAME
from t1d_sim.observation import observe_cgm, synthesize_hr
from t1d_sim.physiology import apply_context_effectors, simulate_day_cgm
from t1d_sim.population import PatientConfig


def _iso_hour(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def simulate_patient(cfg: PatientConfig, days: int, start_utc: datetime) -> dict:
    base = {"k1": 1.0 * cfg.isf_multiplier, "k2": 1.0 * cfg.isf_multiplier, "EGP0": 1.0}
    bg_rows = []
    hr_rows = []
    energy_rows = []
    exercise_rows = []
    sleep_rows = []
    menstrual_rows = []
    site_rows = []
    therapy_rows = []
    mood_hourly_rows = []
    mood_events = []
    gt_rows = []
    prev_mood = (0.0, 0.0)
    for d in range(days):
        day_start = start_utc + timedelta(days=d)
        beh = generate_day_behavior(cfg, day_start, d, prev_mood)
        ctx = beh["context"]
        prev_mood = (ctx.mood_valence, ctx.mood_arousal)
        mod = apply_context_effectors(base, ctx)
        true_bg = simulate_day_cgm(base, mod, beh["meals"], cfg.seed * 10000 + d)
        cgm = observe_cgm(true_bg, d, cfg.seed)
        hr = synthesize_hr(cfg.base_rhr, ctx.stress, beh["exercise_minutes"], beh["exercise_intensity"], d, cfg.seed)
        for h in range(24):
            s = h * 12
            e = s + 12
            ts = _iso_hour(day_start + timedelta(hours=h))
            vals = cgm[s:e]
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                start_bg = end_bg = avg_bg = 120.0
                pct_low = pct_high = 0.0
            else:
                start_bg = float(np.nan_to_num(vals[0], nan=float(finite[0])))
                end_bg = float(np.nan_to_num(vals[-1], nan=float(finite[-1])))
                avg_bg = float(np.nanmean(vals))
                pct_low = float(np.mean(finite < 70))
                pct_high = float(np.mean(finite > 180))
            uroc = float((end_bg - start_bg) / 60.0)
            bg_rows.append((cfg.patient_id, ts, start_bg, end_bg, avg_bg, pct_low, pct_high, uroc, avg_bg + uroc * 60.0, THERAPY_PROFILE_ID))
            hr_rows.append((cfg.patient_id, ts, float(hr[h]) if np.isfinite(hr[h]) else None))
            active = beh["exercise_minutes"] * beh["exercise_intensity"] * (6 / 24 if abs(h - 17) < 1 else 0)
            basal = 65.0 + 5 * cfg.basal_multiplier
            energy_rows.append((cfg.patient_id, ts, basal, active, basal + active))
            ex = beh["exercise_minutes"] if abs(h - 17) < 1 else 0.0
            exercise_rows.append((cfg.patient_id, ts, ex, ex, ex))
            therapy_rows.append((cfg.patient_id, ts, THERAPY_PROFILE_ID, THERAPY_PROFILE_NAME, 12.0 * cfg.cr_multiplier, 0.85 * cfg.basal_multiplier, 45.0 / cfg.isf_multiplier))
            mood_hourly_rows.append((cfg.patient_id, ts, ctx.mood_valence, ctx.mood_arousal, int(ctx.mood_valence >= 0 and ctx.mood_arousal >= 0), int(ctx.mood_valence >= 0 and ctx.mood_arousal < 0), int(ctx.mood_valence < 0 and ctx.mood_arousal >= 0), int(ctx.mood_valence < 0 and ctx.mood_arousal < 0), float(h)))
        dstr = day_start.strftime("%Y-%m-%d")
        sleep_rows.append((cfg.patient_id, dstr, (1-cfg.sleep_efficiency)*beh["sleep_minutes"], 0.5*beh["sleep_minutes"], 0.2*beh["sleep_minutes"], 0.23*beh["sleep_minutes"], 0.0))
        if cfg.is_female:
            menstrual_rows.append((cfg.patient_id, dstr, ctx.cycle_day))
        site_rows.append((cfg.patient_id, dstr, beh["site_days_since_change"], beh["site_location"]))
        mood_events.append((cfg.patient_id, str(uuid.uuid4()), beh["mood_event"]["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"), beh["mood_event"]["valence"], beh["mood_event"]["arousal"]))
        gt_rows.append((cfg.patient_id, dstr, 45.0 / (mod["k1"] / base["k1"]), 12.0 * cfg.cr_multiplier, 0.85 * cfg.basal_multiplier, json.dumps([m[0].strftime("%H:%M") for m in beh["meals"]]), json.dumps([round(m[1], 1) for m in beh["meals"]]), int(beh["exercise_minutes"]), beh["exercise_intensity"], int(beh["sleep_minutes"]), str(ctx.cycle_phase), ctx.mood_valence, ctx.mood_arousal, ctx.stress))
    return {
        "patient": cfg.to_record(), "bg_hourly": bg_rows, "hr_hourly": hr_rows,
        "energy_hourly": energy_rows, "exercise_hourly": exercise_rows, "sleep_daily": sleep_rows,
        "menstrual_daily": menstrual_rows, "site_daily": site_rows, "therapy": therapy_rows,
        "mood_hourly": mood_hourly_rows, "mood_events": mood_events, "ground_truth": gt_rows,
    }
