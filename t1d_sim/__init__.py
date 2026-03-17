"""t1d synthetic simulator."""
from __future__ import annotations

from datetime import datetime, timezone
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

from t1d_sim.features import build_feature_frames
from t1d_sim.patient import simulate_patient
from t1d_sim.population import sample_population, PatientConfig
from t1d_sim.writers import SQLiteWriter, FirebaseWriter


def _simulate_one(args: tuple[PatientConfig, int, datetime]) -> dict:
    return simulate_patient(*args)


def simulate_population(outdb: str | None = None, firebase_project: str | None = None, outdir: str = "./t1d_sim_output/", n_patients: int = 100, days: int = 180, seed: int = 42, split: str = "80/10/10", jobs: int = 1, male_fraction: float = 0.45, aid_fraction: float = 0.35) -> None:
    """Run full population simulation."""
    patients = sample_population(n_patients, seed=seed, male_fraction=male_fraction, aid_fraction=aid_fraction)
    train, val, test = [int(x) for x in split.split("/")]
    cuts = [n_patients * train // 100, n_patients * (train + val) // 100]
    for i, p in enumerate(patients):
        p.n_days = days
        p.split = "train" if i < cuts[0] else "val" if i < cuts[1] else "test"
    writer = SQLiteWriter(outdb) if outdb else FirebaseWriter(firebase_project or "")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    tasks = [(p, days, start) for p in patients]
    results = []
    if jobs > 1:
        with Pool(jobs) as pool:
            for payload in tqdm(pool.imap_unordered(_simulate_one, tasks), total=len(tasks), desc="patients"):
                writer.write_patient(payload)
                results.append(payload)
    else:
        for t in tqdm(tasks, total=len(tasks), desc="patients"):
            payload = _simulate_one(t)
            writer.write_patient(payload)
            results.append(payload)
    if outdb:
        raw = writer.raw_for_features()
        rdf = pd.DataFrame(raw, columns=["user_id","hour_utc","avg_bg","percent_low","percent_high","uroc","heart_rate","active_energy","exercise_minutes","valence","arousal","days_since_change","location"])
        ff = build_feature_frames(rdf)
        rows = []
        for _, r in ff.iterrows():
            rows.append((r.user_id,r.hour_utc,r.avg_bg,1-r.percent_low-r.percent_high,r.percent_low,r.percent_high,r.uroc,r.bg_delta_avg_7h,r.bg_z_avg_7h,r.heart_rate,r.hr_delta_7h,r.hr_z_7h,None,r.active_energy,r.kcal_active_last3h,r.kcal_active_last6h,None,None,None,None,None,None,r.exercise_minutes,r.ex_min_last3h,None,None,0,0,0,r.days_since_change,r.location,1,r.valence,r.arousal,int(r.valence>=0 and r.arousal>=0),int(r.valence>=0 and r.arousal<0),int(r.valence<0 and r.arousal>=0),int(r.valence<0 and r.arousal<0),1.0))
        writer.write_features(rows)
    writer.finalize()
    all_bg = [x[4] for payload in results for x in payload["bg_hourly"]]
    mean_meals = sum(len(p["ground_truth"]) for p in results) and sum(len(eval(r[5])) for p in results for r in p["ground_truth"]) / max(1, sum(len(p["ground_truth"]) for p in results))
    print(f"patients={n_patients} days={days} mean_meals/day={mean_meals:.2f} mean_cgm={sum(all_bg)/len(all_bg):.2f}")
