"""Asymmetric action loss for the exits (written by us).

Idea: a wrong device action is much worse than asking the user to repeat. Per sample, at one exit:
    cost(ACT)      = 0 if the intent is right, lam if it is wrong
    cost(REJECT)   = 1
    cost(CONTINUE) = continue_cost (small, the price of running another segment)
The decision head gives probs d over these three, and we minimize the expected cost sum_a d_a * cost_a.
"Right" is the probability the intent heads give to the true labels (soft, so gradients reach the intent
heads too). lam = 1 means a wrong act and a reject cost the same, so it is the symmetric case.
"""
import torch
import torch.nn.functional as F

from src.encoder.data import IGNORE
from src.encoder.distill import kd_loss


def prob_correct(out: dict, y: torch.Tensor) -> torch.Tensor:
    """P(all labels right) = product of the true class probs. Slot only counts when the sample has a value."""
    p = 1.0
    for key, col in [("action", 0), ("obj", 1), ("location", 2)]:
        p = p * F.softmax(out[key], dim=-1).gather(1, y[:, col:col + 1]).squeeze(1)
    has_val = y[:, 3] != IGNORE
    p_slot = F.softmax(out["slot"], dim=-1).gather(1, y[:, 3].clamp(min=0).unsqueeze(1)).squeeze(1)
    return p * torch.where(has_val, p_slot, torch.ones_like(p_slot))


def cost_matrix(p_right: torch.Tensor, lam: float, continue_cost: float, reject_cost: float = 1.0) -> torch.Tensor:
    """(B, 3) cost of [ACT, CONTINUE, REJECT]. Acting wrong costs lam times a reject when reject_cost is 1."""
    act = lam * reject_cost * (1 - p_right)
    cont = torch.full_like(p_right, continue_cost)
    rej = torch.full_like(p_right, reject_cost)
    return torch.stack([act, cont, rej], dim=1)  # order matches ACT, CONTINUE, REJECT in exits.py


def decision_loss(out: dict, y: torch.Tensor, lam: float, loss_cfg: dict) -> torch.Tensor:
    d = F.softmax(out["decision"], dim=-1)
    costs = cost_matrix(prob_correct(out, y), lam, loss_cfg["continue_cost"], loss_cfg["reject_cost"])
    return (d * costs).sum(dim=1).mean()


def supervised_loss(out: dict, y: torch.Tensor) -> torch.Tensor:
    """Plain CE on the labels so the intent heads learn even before the decision head is useful."""
    loss = sum(F.cross_entropy(out[k], y[:, i]) for i, k in enumerate(["action", "obj", "location"]))
    if (y[:, 3] != IGNORE).any():
        loss = loss + F.cross_entropy(out["slot"], y[:, 3], ignore_index=IGNORE)
    return loss


def exit_loss(out: dict, y: torch.Tensor, lam: float, loss_cfg: dict) -> torch.Tensor:
    return supervised_loss(out, y) + loss_cfg["decision_weight"] * decision_loss(out, y, lam, loss_cfg)


def total_loss(outs: list, y: torch.Tensor, lam: float, loss_cfg: dict) -> tuple:
    """Weighted sum over exits, plus the last exit teaching the earlier ones (its probs are the target)."""
    w = loss_cfg["exit_weights"]
    per_exit = [exit_loss(o, y, lam, loss_cfg) for o in outs]
    total = sum(wk * l for wk, l in zip(w, per_exit))
    final = outs[-1]
    teach = 0.0
    for o in outs[:-1]:
        for k in ["action", "obj", "location"]:
            teach = teach + kd_loss(o[k], final[k].detach(), 1.0)
    total = total + loss_cfg["early_distill_weight"] * teach
    return total, {"exit_losses": [l.item() for l in per_exit]}
