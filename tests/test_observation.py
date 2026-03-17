import numpy as np
from t1d_sim.observation import observe_cgm


def test_observation_noise_nan():
    bg = np.full(288, 120.0)
    obs = observe_cgm(bg, 1, 42)
    assert obs.shape == (288,)
    assert np.isnan(obs).mean() < 0.2
