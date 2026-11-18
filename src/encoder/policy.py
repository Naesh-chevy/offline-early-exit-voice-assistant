"""Inference policy: walk the exits in order, stop at the first ACT or REJECT.

exit_fn(k) runs the model up to exit k (one more segment) and gives numpy logits for that exit.
It is only called for exits we actually reach, so the later layers never run after an early stop.
"""
from dataclasses import dataclass

import numpy as np

from src.encoder.exits import ACT, CONTINUE, REJECT


@dataclass
class Result:
    decision: int            # ACT or REJECT, never CONTINUE
    exit_idx: int            # which exit decided (0 based)
    action: int
    obj: int
    location: int
    slot: int
    p_act: float


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def decide(probs: np.ndarray, act_thresh: float, reject_thresh: float, last: bool, allow_reject: bool = True) -> int:
    """probs = [p_act, p_continue, p_reject] at one exit."""
    if probs[ACT] >= act_thresh:
        return ACT
    if last:   # nothing left to run, so it is act or reject
        return REJECT if allow_reject else ACT
    if allow_reject and probs[REJECT] >= reject_thresh:
        return REJECT
    return CONTINUE


def run_policy(exit_fn, act_thresh: list, reject_thresh: list, allow_reject: bool = True) -> Result:
    n = len(act_thresh)
    for k in range(n):
        out = exit_fn(k)
        probs = softmax(out["decision"])
        d = decide(probs, act_thresh[k], reject_thresh[k], k == n - 1, allow_reject)
        if d != CONTINUE:
            return Result(d, k, int(out["action"].argmax()), int(out["obj"].argmax()),
                          int(out["location"].argmax()), int(out["slot"].argmax()), float(probs[ACT]))
    raise AssertionError("unreachable, the last exit never continues")


def evaluate_policy(outs_per_sample: list, ys: np.ndarray, act_thresh, reject_thresh, allow_reject=True) -> dict:
    """outs_per_sample[i][k] = numpy logits dict of sample i at exit k. ys is (N, 4) labels.

    false action rate = acted AND wrong, over all samples (this is the number the asymmetric loss should lower).
    """
    n = len(ys)
    acted = wrong = rejected = 0
    exit_counts = np.zeros(len(act_thresh), dtype=int)
    for outs, y in zip(outs_per_sample, ys):
        r = run_policy(lambda k: outs[k], act_thresh, reject_thresh, allow_reject)
        exit_counts[r.exit_idx] += 1
        if r.decision == REJECT:
            rejected += 1
            continue
        acted += 1
        if (r.action, r.obj, r.location) != tuple(y[:3]):
            wrong += 1
    return {"false_action_rate": wrong / n, "reject_rate": rejected / n, "act_rate": acted / n,
            "acc_when_acted": (acted - wrong) / max(acted, 1), "mean_exit": float((exit_counts * np.arange(1, len(act_thresh) + 1)).sum() / n),
            "exit_counts": exit_counts.tolist()}
