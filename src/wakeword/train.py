"""Train the wake word model: fp32 -> fold bn -> QAT -> int8 tflite -> eval both.

usage: python -m src.wakeword.train --config configs/train_wakeword.yaml
Needs data/prepared/sc_*.csv (data/prepare.py sc). Run on the GPU box.
"""
import argparse
import csv
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.audio.features import clip_features, load_wav
from src.measure.common import new_run_dir, write_csv
from src.runtime.tflite_runner import Segment
from src.wakeword.export import export_wakeword_int8
from src.wakeword.model import DSCNN, enable_qat, fold_bn, n_params


def load_noise(noise_dir: str, sr: int):
    """Split each noise file 80/20 so the false accept test never hears training noise."""
    train, held = [], []
    for p in sorted(glob.glob(os.path.join(noise_dir, "*.wav"))):
        w = load_wav(p, sr)
        cut = int(len(w) * 0.8)
        train.append(w[:cut])
        held.append(w[cut:])
    return train, held


class WakeDataset(Dataset):
    def __init__(self, manifest: str, cfg: dict, noises: list, train: bool, seed: int = 0):
        rows = list(csv.DictReader(open(manifest, newline="")))
        rng = random.Random(seed)
        wake = [r for r in rows if r["is_wake"] == "1"]
        other = [r for r in rows if r["is_wake"] == "0"]
        if train:  # keep it balanced-ish, 'other' is way bigger than 'wake'
            other = rng.sample(other, min(len(other), len(wake) * cfg["train"]["other_per_wake"]))
        self.rows = wake + other
        if train:
            # the live mic is mostly room noise and silence, so the model has to see those as "other" too.
            # first run had none of these and fired on ~11% of pure noise windows
            n_empty = int(len(self.rows) * cfg["train"].get("noise_neg_frac", 0.0))
            self.rows += [{"path": "", "is_wake": "0"} for _ in range(n_empty)]
        self.cfg, self.noises, self.train = cfg, noises, train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, a = self.rows[i], self.cfg["audio"]
        n = int(a["sample_rate"] * a["window_s"])
        if r["path"] == "":     # noise only (or near silence) negative
            w = np.random.randn(n).astype(np.float32) * 1e-4
            if self.noises and self.rng.random() < 0.8:
                nz = self.rng.choice(self.noises)
                s = self.rng.randrange(0, max(1, len(nz) - n))
                w = w + np.pad(nz[s:s + n], (0, max(0, n - len(nz[s:s + n])))) * self.rng.uniform(0.05, 1.0)
            return torch.from_numpy(clip_features(w, a["window_s"], a["n_mels"], a["n_fft"], a["hop"], a["sample_rate"]))[None], 0
        w = load_wav(r["path"], a["sample_rate"])[:n]
        w = np.pad(w, (0, n - len(w)))
        if self.train and self.noises and self.rng.random() < self.cfg["train"]["noise_prob"]:
            nz = self.rng.choice(self.noises)
            s = self.rng.randrange(0, max(1, len(nz) - n))
            w = w + nz[s:s + n] * self.rng.uniform(0, self.cfg["train"]["noise_gain_max"])
        mel = clip_features(w, a["window_s"], a["n_mels"], a["n_fft"], a["hop"], a["sample_rate"])
        return torch.from_numpy(mel)[None], int(r["is_wake"])


def fit(model, loader, epochs, lr, device, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    model.to(device)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"{tag} epoch {ep} loss {tot / len(loader):.4f}")


def wake_probs_torch(model, mels: np.ndarray) -> np.ndarray:
    model.eval().cpu()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(mels)[:, None]), 1)[:, 1].numpy()


def wake_probs_tflite(seg: Segment, mels: np.ndarray) -> np.ndarray:
    out = []
    for m in mels:
        logits = seg.run(m.reshape(1, *m.shape, 1))[0][0]
        e = np.exp(logits - logits.max())
        out.append(e[1] / e.sum())
    return np.array(out)


def metrics(probs, labels, thr) -> dict:
    pred = probs > thr
    labels = np.asarray(labels).astype(bool)
    return {"accuracy": float((pred == labels).mean()),
            "false_reject_rate": float((~pred[labels]).mean()),   # wake word said but missed
            "false_accept_clips": float(pred[~labels].mean())}


def false_accepts_per_hour(prob_fn, held_noise: list, cfg: dict) -> float:
    """Slide a 1 s window (hop 0.25 s) over held out noise, count windows over the threshold."""
    a = cfg["audio"]
    n, hop = int(a["sample_rate"] * a["window_s"]), int(a["sample_rate"] * 0.25)
    mels, secs = [], 0.0
    for w in held_noise:
        for s in range(0, len(w) - n, hop):
            mels.append(clip_features(w[s:s + n], a["window_s"], a["n_mels"], a["n_fft"], a["hop"], a["sample_rate"]))
        secs += len(w) / a["sample_rate"]
    hits = int((prob_fn(np.stack(mels)) > cfg["eval"]["threshold"]).sum())
    # windows overlap so one noise burst counts several times, this is an upper bound
    return hits / (secs / 3600)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_wakeword.yaml")
    ap.add_argument("--run_name", default="wakeword")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sr = cfg["audio"]["sample_rate"]
    noise_tr, noise_held = load_noise(cfg["noise_dir"], sr)
    mk = lambda s, tr: WakeDataset(f"data/prepared/sc_{s}.csv", cfg, noise_tr, tr, cfg["seed"])
    tl = DataLoader(mk("train", True), cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    test = mk("test", False)
    m = cfg["model"]
    model = DSCNN(m["widths"], 2, tuple(m["first_kernel"]), tuple(m["first_stride"]))
    assert n_params(model) < m["max_params"], n_params(model)
    os.makedirs("checkpoints", exist_ok=True)

    fit(model, tl, cfg["train"]["epochs"], cfg["train"]["lr"], dev, "fp32")
    torch.save(model.state_dict(), cfg["out"]["ckpt"])
    fold_bn(model.cpu())
    enable_qat(model)
    fit(model, tl, cfg["train"]["qat_epochs"], cfg["train"]["lr"] / 10, dev, "qat")
    torch.save(model.state_dict(), cfg["out"]["qat_ckpt"])

    # calibration = some train clips, test = held out
    calib = [tl.dataset[i][0][0].numpy() for i in random.sample(range(len(tl.dataset)), 200)]
    model.cpu().eval()
    size = export_wakeword_int8(model, calib, cfg["out"]["tflite"], cfg["audio"]["n_mels"],
                                int(cfg["audio"]["window_s"] * sr / cfg["audio"]["hop"]))
    seg = Segment(cfg["out"]["tflite"])

    mels = np.stack([test[i][0][0].numpy() for i in range(len(test))])
    labels = [r["is_wake"] == "1" for r in test.rows]
    thr = cfg["eval"]["threshold"]
    rows = []
    enable_qat(model, False)   # fp32 reference = folded weights without fake quant
    for name, fn in [("fp32", lambda x: wake_probs_torch(model, x)), ("int8", lambda x: wake_probs_tflite(seg, x))]:
        r = {"variant": name, "model_kb": size / 1024 if name == "int8" else n_params(model) * 4 / 1024,
             **metrics(fn(mels), labels, thr),
             "false_accepts_per_hour": false_accepts_per_hour(fn, noise_held, cfg)}
        print(r)
        rows.append(r)
    out = new_run_dir(args.run_name) / "wakeword_eval.csv"
    write_csv(out, rows)
    print("wrote", out)
