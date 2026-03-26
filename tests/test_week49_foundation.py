from datetime import datetime, timezone

from t1d_sim.feature_frame import FeatureFrameHourly
from t1d_sim.population import sample_population
from t1d_sim.simulate import SimulationCarryState, simulate_day
from t1d_sim.therapy import make_default_schedule


def test_feature_frame_signal_payload_normalizes_percentages():
    frame = FeatureFrameHourly(
        hour_start_utc=datetime(2026, 1, 1, 23, tzinfo=timezone.utc),
        bg_tir=72.5,
        bg_percent_low=4.0,
        bg_percent_high=23.5,
        mood_valence=-0.2,
        mood_arousal=0.8,
        days_since_period_start=2,
    )

    payload = frame.to_signal_dict()

    assert payload["tir_7d"] == 0.725
    assert payload["pct_low_7d"] == 0.04
    assert payload["pct_high_7d"] == 0.235
    assert payload["stress_acute"] == 1.0
    assert payload["cycle_phase_menstrual"] == 1


def test_simulate_day_produces_app_shaped_records():
    cfg = sample_population(1, seed=7)[0]
    schedule = cfg.therapy_schedule or cfg.baseline_therapy_schedule or make_default_schedule(cfg)

    result = simulate_day(
        cfg=cfg,
        schedule=schedule,
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        rng_seed=7,
        day_index=0,
        carry_state=SimulationCarryState(),
    )

    assert len(result.feature_frames) == 24
    assert result.decision_frame.hour_start_utc.hour == 23
    assert len(result.bg_hourly) == 24
    assert len(result.therapy_hourly) == 24
    assert result.site_daily["dateUtc"] == "2026-01-01"
    assert result.hr_daily_average["dateUtc"] == "2026-01-01"

    signals = result.decision_frame.to_signal_dict()
    assert "bg_avg" in signals
    assert "tir_7d" in signals
