"""Train the exit heads (and fine tune the encoder) with the asymmetric loss, one run per lambda.

usage: python -m src.encoder.train_exits --encoder checkpoints/student/encoder.pt --tag distill
Saves checkpoints/exits_<tag>_lam<lambda>.pt and results/<run>/exits_val.csv. Same seed for every lambda.
"""
import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.encoder.ckpt import load_resume, save_resume
from src.encoder.data import FSCDataset
from src.encoder.exits import from_config as exits_from_config
from src.encoder.loss import total_loss
from src.encoder.policy import evaluate_policy
from src.encoder.student import from_config as enc_from_config
from src.measure.common import new_run_dir, write_csv


@torch.no_grad()
def collect_outputs(model, loader, device) -> tuple:
    """Run the full model once, return per sample per exit numpy logits and the labels."""
    model.eval().to(device)
    per_sample, ys = [], []
    for x, y in loader:
        outs = model(x.to(device))
        for i in range(len(y)):
            per_sample.append([{k: v[i].cpu().numpy() for k, v in o.items()} for o in outs])
        ys.append(y.numpy())
    return per_sample, np.concatenate(ys)


def train_one(cfg, lam, enc_ckpt, train_loader, device, epochs, seed, tag, resume=False):
    """Train one lambda. With resume: reuse a finished model, or continue from the last saved epoch."""
    torch.manual_seed(seed)
    enc = enc_from_config(cfg, "student")
    if enc_ckpt:
        enc.load_state_dict(torch.load(enc_ckpt, map_location="cpu"))
    model = exits_from_config(cfg, enc).to(device)
    final = f"checkpoints/exits_{tag}_lam{lam}.pt"
    rpath = f"checkpoints/resume_exits_{tag}_lam{lam}.pt"
    t = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # a lambda that already finished all epochs just loads its state and the loop below does nothing
    start = load_resume(rpath, opt, model=model) if resume and os.path.exists(rpath) else 0
    for ep in range(start, epochs):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            loss, _ = total_loss(model(x.to(device)), y.to(device), lam, cfg["loss"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"lam {lam} epoch {ep} loss {tot / len(train_loader):.4f}")
        save_resume(rpath, ep + 1, opt, model=model)
    torch.save(model.state_dict(), final)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_encoder.yaml")
    ap.add_argument("--encoder", default=None, help="distilled (or no-distill) encoder state dict")
    ap.add_argument("--tag", default="distill")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip finished lambdas, continue unfinished ones")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep = cfg["prepared_dir"]
    train = FSCDataset(f"{prep}/fsc_train.csv", f"{prep}/fsc_labels.json", cfg["audio"])
    valid = FSCDataset(f"{prep}/fsc_valid.csv", f"{prep}/fsc_labels.json", cfg["audio"])
    tl = DataLoader(train, cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    vl = DataLoader(valid, cfg["train"]["batch_size"])
    os.makedirs("checkpoints", exist_ok=True)
    rows = []
    for lam in cfg["loss"]["lambda_sweep"]:
        model = train_one(cfg, lam, args.encoder, tl, dev, args.epochs or cfg["train"]["epochs"], cfg["seed"],
                          args.tag, args.resume)
        outs, ys = collect_outputs(model, vl, dev)
        p = cfg["policy"]
        for allow in [True, False]:
            r = evaluate_policy(outs, ys, p["act_thresh"], p["reject_thresh"], allow_reject=allow)
            r.update({"tag": args.tag, "lambda": lam, "allow_reject": allow, "split": "valid"})
            r["exit_counts"] = str(r["exit_counts"])
            print(r)
            rows.append(r)
    path = new_run_dir(f"exits_{args.tag}") / "exits_val.csv"
    write_csv(path, rows)
    print("wrote", path)
