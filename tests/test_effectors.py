from t1d_sim.behavior import ContextState
from t1d_sim.physiology import apply_context_effectors


def test_effectors_directionality():
    base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}
    luteal = ContextState("luteal", 10, 1.0, 450, 0.9, 72, 0.0, 0.1, 0.5, 0.1, 0, 0, False, 0, False)
    ex = ContextState(None, -1, 0.0, 450, 0.9, 1, 1.0, 0.1, 0.5, 0.1, 0, 0, False, 0, False)
    sleep_debt = ContextState(None, -1, 0.0, 300, 0.7, 72, 0.0, 0.1, 0.5, 0.1, 0, 0, False, 0, False)
    assert apply_context_effectors(base, luteal)["k1"] < 1.0
    assert apply_context_effectors(base, ex)["k1"] > 1.0
    assert apply_context_effectors(base, sleep_debt)["EGP0"] > 1.0
