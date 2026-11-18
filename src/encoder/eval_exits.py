"""Threshold sweep, reject vs no reject, early exit vs full encoder, and the int8 check on real audio.

usage: python -m src.encoder.eval_exits --lams 1 2 5 10 --int8 5 10
Thresholds are picked on valid, then the same thresholds are reported on test (never tuned on test).
"""
import argparse
import csv
import random

import numpy as np
import torch
import yaml

from src.audio.features import clip_features, load_wav
from src.audio.labels import load_label_maps
from src.encoder.exits import from_config as exits_from_config
from src.encoder.policy import evaluate_policy
from src.encoder.student import from_config as enc_from_config
from src.encoder.train_exits import collect_outputs
from src.measure.common import new_run_dir, write_csv
from src.runtime.segmented import SegmentedEncoder

GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
EARLY_REJECT = 0.9      # reject threshold at the early exits, fixed. the last exit just acts or rejects


def load_split(cfg: dict, split: str, limit: int | None = None) -> tuple:
    """All mels of a split in memory (N, n_mels, frames) plus labels (N, 4)."""
    prep, a = cfg["prepared_dir"], cfg["audio"]
    maps = load_label_maps(f"{prep}/fsc_labels.json")
    rows = list(csv.DictReader(open(f"{prep}/fsc_{split}.csv", newline="")))
    if limit:
        random.Random(cfg["seed"]).shuffle(rows)
        rows = rows[:limit]
    mels = np.stack([clip_features(load_wav(r["path"], a["sample_rate"]), a["window_s"], a["n_mels"], a["n_fft"],
                                   a["hop"], a["sample_rate"]) for r in rows])
    ys = np.array([[maps["action"][r["action"]], maps["object"][r["object"]], maps["location"][r["location"]], -100]
                   for r in rows])
    return mels, ys


def torch_outputs(model, mels: np.ndarray, ys: np.ndarray, bs: int = 64) -> list:
    batches = [(torch.from_numpy(mels[i:i + bs]), torch.from_numpy(ys[i:i + bs])) for i in range(0, len(mels), bs)]
    outs, _ = collect_outputs(model, batches, "cpu")
    return outs


def tflite_outputs(paths: list, mels: np.ndarray) -> list:
    """Every exit for every clip, from the int8 (or fp32) tflite segments. Offline eval, so all three run."""
    enc = SegmentedEncoder(paths)
    outs = []
    for m in mels:
        enc.start(m)
        outs.append([enc.exit_fn(k) for k in range(len(paths))])
    return outs


def policies(t: float) -> dict:
    """The three policies we compare at act threshold t."""
    return {
        "early_reject": dict(act=[t, t, t], rej=[EARLY_REJECT, EARLY_REJECT, 1.0], allow=True),
        "early_noreject": dict(act=[t, t, t], rej=[2.0, 2.0, 2.0], allow=False),
        "full_reject": dict(act=[2.0, 2.0, t], rej=[2.0, 2.0, 2.0], allow=True),    # never stops early
    }


def cost(r: dict, lam: float) -> float:
    """Same idea as the training loss: wrong action costs lam, a reject costs 1."""
    return lam * r["false_action_rate"] + r["reject_rate"]


def sweep(outs: list, ys: np.ndarray, lam: float, split: str) -> list:
    rows = []
    for t in GRID:
        for name, p in policies(t).items():
            r = evaluate_policy(outs, ys, p["act"], p["rej"], allow_reject=p["allow"])
            r.update({"lambda": lam, "policy": name, "act_thresh": t, "split": split, "exit_counts": str(r["exit_counts"])})
            rows.append(r)
    return rows


def pick(val_rows: list, lam: float) -> dict:
    """Best act threshold per policy on valid by the asymmetric cost."""
    best = {}
    for r in val_rows:
        c = cost(r, lam)
        if r["policy"] not in best or c < best[r["policy"]][0]:
            best[r["policy"]] = (c, r["act_thresh"])
    return {k: v[1] for k, v in best.items()}


def agreement(a: list, b: list) -> dict:
    """How much two sets of per exit outputs agree (a = torch fp32, b = tflite). Per exit."""
    res = {}
    for k in range(len(a[0])):
        same_intent = np.mean([(x[k]["action"].argmax(), x[k]["obj"].argmax(), x[k]["location"].argmax()) ==
                               (y[k]["action"].argmax(), y[k]["obj"].argmax(), y[k]["location"].argmax())
                               for x, y in zip(a, b)])
        same_dec = np.mean([x[k]["decision"].argmax() == y[k]["decision"].argmax() for x, y in zip(a, b)])
        diff = np.mean([np.abs(x[k]["action"] - y[k]["action"]).max() for x, y in zip(a, b)])
        res[f"exit{k + 1}_intent_agree"] = float(same_intent)
        res[f"exit{k + 1}_decision_agree"] = float(same_dec)
        res[f"exit{k + 1}_mean_max_logit_diff"] = float(diff)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_encoder.yaml")
    ap.add_argument("--lams", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--int8", type=int, nargs="*", default=[], help="lambdas to also run through the int8 segments")
    ap.add_argument("--tag", default="distill")
    ap.add_argument("--limit", type=int, default=None, help="use a random subset of each split (quick check)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    data = {s: load_split(cfg, s, args.limit) for s in ["valid", "test"]}
    print({s: len(v[0]) for s, v in data.items()})
    out_dir = new_run_dir(f"exits_eval_{args.tag}")
    sweep_rows, chosen_rows, parity_rows = [], [], []
    for lam in args.lams:
        model = exits_from_config(cfg, enc_from_config(cfg, "student"))
        model.load_state_dict(torch.load(f"checkpoints/exits_{args.tag}_lam{lam}.pt", map_location="cpu"))
        outs = {s: torch_outputs(model, *data[s]) for s in data}
        rows = {s: sweep(outs[s], data[s][1], lam, s) for s in data}
        sweep_rows += rows["valid"] + rows["test"]
        chosen = pick(rows["valid"], lam)
        for r in rows["test"]:
            if chosen[r["policy"]] == r["act_thresh"]:
                chosen_rows.append({**r, "precision": "fp32 torch", "chosen_on": "valid"})
                print("lam", lam, r["policy"], "t", r["act_thresh"], {k: round(v, 4) if isinstance(v, float) else v
                                                                      for k, v in r.items() if k in
                                                                      ["false_action_rate", "reject_rate", "acc_when_acted", "mean_exit", "exit_counts"]})
        if lam in args.int8:
            for prec in ["int8", "fp32"]:
                paths = [f"export/lam{lam}/seg{i}_{prec}.tflite" for i in (1, 2, 3)]
                tf_outs = tflite_outputs(paths, data["test"][0])
                parity_rows.append({"lambda": lam, "precision": prec, "n_clips": len(tf_outs),
                                    **agreement(outs["test"], tf_outs)})
                print(parity_rows[-1])
                for name, t in chosen.items():
                    p = policies(t)[name]
                    r = evaluate_policy(tf_outs, data["test"][1], p["act"], p["rej"], allow_reject=p["allow"])
                    r.update({"lambda": lam, "policy": name, "act_thresh": t, "split": "test",
                              "exit_counts": str(r["exit_counts"]), "precision": f"{prec} tflite", "chosen_on": "valid"})
                    chosen_rows.append(r)
                    print(prec, name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()
                                       if k in ["false_action_rate", "reject_rate", "acc_when_acted", "mean_exit"]})
    write_csv(out_dir / "sweep.csv", sweep_rows)
    write_csv(out_dir / "chosen_test.csv", chosen_rows)
    if parity_rows:
        write_csv(out_dir / "int8_parity.csv", parity_rows)
    print("wrote", out_dir)
