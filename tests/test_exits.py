import numpy as np
import torch

from src.encoder.exits import ACT, CONTINUE, NO_OPTION, REJECT, ExitHead, ExitModel
from src.encoder.policy import decide, evaluate_policy, run_policy
from src.encoder.student import WhisperEncoder

N_ACTION, N_OBJECT, N_LOC, SLOT = 3, 4, 2, 15


def tiny_model():
    torch.manual_seed(0)
    enc = WhisperEncoder(32, 4, 2, 64, 80, 40)
    return ExitModel(enc, [2, 3, 4], 16, N_ACTION, N_OBJECT, N_LOC, SLOT)


def test_each_head_output_shape_at_each_exit():
    outs = tiny_model()(torch.randn(5, 80, 40))
    assert len(outs) == 3
    for o in outs:
        assert o["action"].shape == (5, N_ACTION)
        assert o["obj"].shape == (5, N_OBJECT)
        assert o["location"].shape == (5, N_LOC)
        assert o["slot"].shape == (5, SLOT)
        assert o["decision"].shape == (5, 3)


def test_last_exit_has_no_continue():
    outs = tiny_model()(torch.randn(5, 80, 40))
    assert torch.softmax(outs[-1]["decision"], 1)[:, CONTINUE].max() < 1e-6
    assert torch.softmax(outs[0]["decision"], 1)[:, CONTINUE].min() > 1e-6   # earlier exits can continue
    head = ExitHead(8, 8, 2, 2, 2, 3, can_continue=False)
    assert (head(torch.randn(2, 5, 8))["decision"][:, CONTINUE] == NO_OPTION).all()


# policy tests use hand made logits so we know exactly what should happen
def logits(p_act, p_cont, p_rej, action=1):
    dec = np.log(np.array([p_act, p_cont, p_rej]) + 1e-12)
    a = np.zeros(3)
    a[action] = 5.0
    return {"decision": dec, "action": a, "obj": np.array([0.0, 1.0]), "location": np.array([1.0, 0.0]),
            "slot": np.zeros(15)}


ACT_T, REJ_T = [0.9, 0.85, 0.5], [0.9, 0.9, 0.5]


def test_policy_stops_at_first_act():
    calls = []

    def fn(k):
        calls.append(k)
        return logits(0.95, 0.04, 0.01)

    r = run_policy(fn, ACT_T, REJ_T)
    assert r.decision == ACT and r.exit_idx == 0
    assert calls == [0]   # later exits were never asked for


def test_policy_continues_then_acts_at_second_exit():
    outs = [logits(0.3, 0.6, 0.1), logits(0.9, 0.05, 0.05), logits(0.9, 0.0, 0.1)]
    r = run_policy(lambda k: outs[k], ACT_T, REJ_T)
    assert r.decision == ACT and r.exit_idx == 1 and r.action == 1


def test_policy_rejects_when_reject_is_confident():
    outs = [logits(0.02, 0.0, 0.98), logits(0.9, 0.05, 0.05), logits(0.9, 0.0, 0.1)]
    r = run_policy(lambda k: outs[k], ACT_T, REJ_T)
    assert r.decision == REJECT and r.exit_idx == 0


def test_policy_never_continues_at_last_exit():
    outs = [logits(0.2, 0.7, 0.1)] * 2 + [logits(0.3, 0.0, 0.7)]
    r = run_policy(lambda k: outs[k], ACT_T, REJ_T)
    assert r.decision == REJECT and r.exit_idx == 2
    # even with all probability on "continue" it has to end
    outs = [logits(0.0, 1.0, 0.0)] * 3
    assert run_policy(lambda k: outs[k], ACT_T, REJ_T).decision in (ACT, REJECT)
    assert decide(np.array([0.1, 0.8, 0.1]), 0.5, 0.5, last=True) != CONTINUE


def test_no_reject_mode_always_acts():
    outs = [logits(0.0, 1.0, 0.0)] * 3
    r = run_policy(lambda k: outs[k], ACT_T, REJ_T, allow_reject=False)
    assert r.decision == ACT and r.exit_idx == 2


def test_evaluate_policy_counts_false_actions():
    good = [logits(0.95, 0.0, 0.05, action=1)] * 3
    bad = [logits(0.95, 0.0, 0.05, action=2)] * 3     # confident but wrong
    rej = [logits(0.0, 0.0, 1.0)] * 3
    ys = np.array([[1, 1, 0, -100], [1, 1, 0, -100], [1, 1, 0, -100]])
    res = evaluate_policy([good, bad, rej], ys, ACT_T, REJ_T)
    assert abs(res["false_action_rate"] - 1 / 3) < 1e-9
    assert abs(res["reject_rate"] - 1 / 3) < 1e-9
    assert res["exit_counts"][0] == 3
