import numpy as np

from src.encoder.eval_exits import cost, pick, policies, sweep


def fake_outs(n, p_act_early, correct_early, seed=0):
    """Exit 1 has the given ACT prob and is right or wrong for all samples. Exit 3 is always right and confident."""
    ys = np.zeros((n, 4), dtype=int)
    outs = []
    for _ in range(n):
        def o(p, right):
            lab = 0 if right else 1
            return {"action": np.eye(3)[lab] * 9, "obj": np.eye(3)[0] * 9, "location": np.eye(3)[0] * 9,
                    "slot": np.zeros(15), "decision": np.log(np.array([p, (1 - p) / 2, (1 - p) / 2]))}
        outs.append([o(p_act_early, correct_early), o(0.5, True), o(0.99, True)])
    return outs, ys


def test_full_policy_never_stops_early():
    outs, ys = fake_outs(20, 0.99, False)     # exit 1 is confident and wrong
    r = sweep(outs, ys, 5, "valid")
    full = [x for x in r if x["policy"] == "full_reject" and x["act_thresh"] == 0.9][0]
    early = [x for x in r if x["policy"] == "early_reject" and x["act_thresh"] == 0.9][0]
    assert full["mean_exit"] == 3.0 and full["false_action_rate"] == 0.0
    assert early["mean_exit"] == 1.0 and early["false_action_rate"] == 1.0   # it trusted the wrong exit


def test_noreject_never_rejects():
    outs, ys = fake_outs(20, 0.2, True)
    for x in sweep(outs, ys, 5, "valid"):
        if x["policy"] == "early_noreject":
            assert x["reject_rate"] == 0.0


def test_pick_uses_asymmetric_cost():
    rows = [{"policy": "p", "act_thresh": 0.5, "false_action_rate": 0.10, "reject_rate": 0.0},
            {"policy": "p", "act_thresh": 0.9, "false_action_rate": 0.02, "reject_rate": 0.10}]
    assert pick(rows, 1)["p"] == 0.5      # lam 1: 0.10 vs 0.12, acting is cheaper
    assert pick(rows, 10)["p"] == 0.9     # lam 10: 1.0 vs 0.30, rejecting is cheaper
    assert cost(rows[0], 10) == 1.0


def test_policy_names():
    assert set(policies(0.9)) == {"early_reject", "early_noreject", "full_reject"}
