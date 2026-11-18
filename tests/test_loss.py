import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.encoder.data import IGNORE, FakeSLUData
from src.encoder.exits import ExitModel
from src.encoder.loss import cost_matrix, decision_loss, prob_correct, total_loss
from src.encoder.student import WhisperEncoder

LOSS_CFG = {"reject_cost": 1.0, "continue_cost": 0.2, "exit_weights": [0.3, 0.3, 0.4],
            "early_distill_weight": 0.5, "decision_weight": 1.0}


def hand_out(decision_idx: int, intent_right: bool):
    """One sample, decision is a near one-hot on decision_idx, intent heads are near one-hot right or wrong."""
    dec = torch.full((1, 3), -30.0)
    dec[0, decision_idx] = 30.0
    right, wrong = 30.0, -30.0

    def head(n, true):
        z = torch.full((1, n), wrong)
        z[0, true if intent_right else (true + 1) % n] = right
        return z

    return {"action": head(3, 0), "obj": head(4, 1), "location": head(2, 0),
            "slot": torch.zeros(1, 15), "decision": dec}


Y = torch.tensor([[0, 1, 0, IGNORE]])


@pytest.mark.parametrize("lam", [1.0, 2.0, 5.0, 10.0])
def test_wrong_act_costs_lambda_times_a_reject(lam):
    wrong_act = decision_loss(hand_out(0, intent_right=False), Y, lam, LOSS_CFG).item()
    reject = decision_loss(hand_out(2, intent_right=False), Y, lam, LOSS_CFG).item()
    assert reject == pytest.approx(1.0, abs=1e-4)
    assert wrong_act == pytest.approx(lam * reject, abs=1e-4)


def test_right_act_costs_nothing_and_lambda_one_is_symmetric():
    assert decision_loss(hand_out(0, True), Y, 5.0, LOSS_CFG).item() == pytest.approx(0.0, abs=1e-4)
    # lam=1: wrong act and reject cost the same
    a = decision_loss(hand_out(0, False), Y, 1.0, LOSS_CFG).item()
    r = decision_loss(hand_out(2, False), Y, 1.0, LOSS_CFG).item()
    assert a == pytest.approx(r, abs=1e-4)


def test_cost_matrix_values():
    c = cost_matrix(torch.tensor([1.0, 0.0, 0.5]), lam=4.0, continue_cost=0.2)
    assert c[:, 0].tolist() == [0.0, 4.0, 2.0]    # act
    assert c[:, 1].tolist() == pytest.approx([0.2] * 3)   # continue
    assert c[:, 2].tolist() == [1.0, 1.0, 1.0]    # reject


def test_prob_correct_uses_slot_only_when_value_present():
    out = hand_out(0, True)
    assert prob_correct(out, Y).item() == pytest.approx(1.0, abs=1e-4)
    y_val = torch.tensor([[0, 1, 0, 3]])
    out["slot"] = torch.full((1, 15), -30.0)
    out["slot"][0, 3] = 30.0
    assert prob_correct(out, y_val).item() == pytest.approx(1.0, abs=1e-4)
    out["slot"][0, 3], out["slot"][0, 4] = -30.0, 30.0
    assert prob_correct(out, y_val).item() < 1e-4


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return ExitModel(WhisperEncoder(32, 4, 2, 64, 80, 40), [2, 3, 4], 16, 3, 4, 2, 15)


def test_gradients_reach_every_head():
    model = tiny_model()
    x, y = next(iter(DataLoader(FakeSLUData(16, 80, 40, 3, 4, 2), batch_size=16)))
    y[:4, 3] = 2    # a few samples with a value so the slot head gets a gradient
    loss, _ = total_loss(model(x), y, 5.0, LOSS_CFG)
    loss.backward()
    for i, head in enumerate(model.heads):
        for name, p in head.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad in exit {i} {name}"
    assert model.enc.conv1.weight.grad.abs().sum() > 0


def add_label_noise(data, seed, frac=0.3):
    """Replace the action label of ~30% of samples with a random one. The mel still shows the old label."""
    g = np.random.default_rng(seed)
    flip = g.random(len(data)) < frac
    data.y[:, 0] = np.where(flip, g.integers(0, 3, len(data)), data.y[:, 0])


def val_false_action_rate(model, lam_steps, lam, seed=0, device="cpu"):
    """Train briefly with lam on noisy fake data, return false action rate on a val set using a fixed policy."""
    from src.encoder.policy import evaluate_policy
    torch.manual_seed(seed)
    model = model.to(device)
    train = FakeSLUData(512, 80, 40, 3, 4, 2, seed=1)
    add_label_noise(train, seed=2)    # mistakes have to be unavoidable or the test says nothing
    loader = DataLoader(train, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    it = iter(loader)
    for _ in range(lam_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        loss, _ = total_loss(model(x.to(device)), y.to(device), lam, LOSS_CFG)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    val = FakeSLUData(200, 80, 40, 3, 4, 2, seed=7)
    add_label_noise(val, seed=8)
    with torch.no_grad():
        outs = model(torch.from_numpy(val.x).to(device))
    per_sample = [[{k: v[i].cpu().numpy() for k, v in o.items()} for o in outs] for i in range(len(val))]
    return evaluate_policy(per_sample, val.y, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])["false_action_rate"]


@pytest.mark.gpu
def test_higher_lambda_lowers_false_action_rate():
    """The hypothesis. If this fails, record it in docs/milestones.md, do not tune it away."""
    rates = {lam: val_false_action_rate(tiny_model(), 300, lam, device="cuda") for lam in [1.0, 10.0]}
    print(rates)
    assert rates[10.0] <= rates[1.0]


def test_higher_lambda_lowers_false_action_rate_cpu_tiny():
    # same hypothesis on cpu with a tiny model, kept as a plain test so the laptop checks it too
    rates = {lam: val_false_action_rate(tiny_model(), 150, lam) for lam in [1.0, 10.0]}
    print(rates)
    assert rates[1.0] > 0.05   # lambda=1 really makes mistakes here, so the check is not vacuous
    assert rates[10.0] < rates[1.0]
