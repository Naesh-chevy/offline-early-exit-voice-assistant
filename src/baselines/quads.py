"""Prior method: QUADS style quantized distillation (Biswas et al., Interspeech 2025, arXiv 2505.14723).

We reproduce the core mechanism only, in our own code. No code was copied from their repo (it has no licence),
we read it to see what the training loop does. What is the same as QUADS:
  - one student model with a single intent classifier, no exits, no reject
  - training alternates between a distillation phase and a k-means quantization phase, and ends with
    quantization only (the "final quantization phase")
  - distillation phase: L1 between teacher and student encoder output + cross entropy on the labels
  - quantization phase: k-means codebook per weight matrix (2^bits centroids). after every optimizer step the
    centroids are moved to the mean of their cluster and the weights are snapped to the centroids
What is different (our setup, on purpose): whisper tiny student, our FSC finetuned whisper base teacher, 4 s input,
three heads (action, object, location) like the naive baseline. The paper does not give alpha, so alpha is a config value.

Weights stay fp32 in memory (as in QUADS), so the saving is storage only. Compute is the same as the FP32 student.
Train: python -m src.baselines.quads --bits 8 4
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def kmeans_1d(w: torch.Tensor, k: int, iters: int = 20, seed: int = 0) -> tuple:
    """Lloyd's k-means on a flat tensor. 1D so assignment is just a bucketize against midpoints of sorted centroids."""
    w = w.flatten()
    g = torch.Generator().manual_seed(seed)
    # random start (QUADS also starts random), picked from the weights themselves so no centroid is far from the data
    cent = w[torch.randperm(len(w), generator=g)[:k].to(w.device)].sort().values
    for _ in range(iters):
        idx = torch.bucketize(w, (cent[1:] + cent[:-1]) / 2)
        cent = recenter(w, idx, cent).sort().values      # sorted again in case an empty cluster kept an old value
    idx = torch.bucketize(w, (cent[1:] + cent[:-1]) / 2)
    return cent, idx


def recenter(w: torch.Tensor, idx: torch.Tensor, cent: torch.Tensor) -> torch.Tensor:
    """Each centroid = mean of the weights assigned to it. An empty cluster keeps its old value."""
    sums = torch.zeros_like(cent).index_add_(0, idx, w)
    counts = torch.zeros_like(cent).index_add_(0, idx, torch.ones_like(w))
    return torch.where(counts > 0, sums / counts.clamp(min=1), cent)


class KMeansQuantizer:
    """Codebook per weight matrix (anything with dim > 1 that trains). apply() is called after every optimizer step."""

    def __init__(self, model: nn.Module, bits: int):
        self.bits, self.books = bits, {}
        for name, p in model.named_parameters():
            if p.dim() > 1 and p.requires_grad:
                k = min(2 ** bits, p.numel())
                cent, idx = kmeans_1d(p.detach().flatten().cpu(), k)
                self.books[name] = [cent.to(p.device), idx.to(p.device)]
        self.apply(model, update_centroids=False)

    @torch.no_grad()
    def apply(self, model: nn.Module, update_centroids: bool = True) -> None:
        for name, p in model.named_parameters():
            if name in self.books:
                cent, idx = self.books[name]
                if update_centroids:       # the sgd step moved the weights, so move the centroids with them
                    cent = recenter(p.flatten(), idx, cent)
                    self.books[name][0] = cent
                p.copy_(cent[idx].view_as(p))

    def size_mb(self, model: nn.Module) -> float:
        """Storage if the codebooks were saved as indices (bits per weight) plus fp32 centroids and fp32 for the rest."""
        total = 0.0
        for name, p in model.named_parameters():
            if name in self.books:
                total += p.numel() * self.bits / 8 + len(self.books[name][0]) * 4
            else:
                total += p.numel() * 4
        return total / 1e6


def distill_phase(student, teacher, proj, loader, opt, l1_weight, device) -> float:
    """One epoch: l1_weight * L1(pooled encoder output) + CE on the three heads."""
    student.train()
    teacher.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_feat = teacher.enc(x).mean(1)
        s_out, hidden = student(x, return_hidden=True)
        s_feat = student.enc.layer_norm(hidden[-1]).mean(1)      # same pooled vector the heads use
        ce = sum(F.cross_entropy(o, y[:, i]) for i, o in enumerate(s_out))
        loss = ce + l1_weight * F.l1_loss(proj(s_feat), t_feat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def quant_phase(student, quantizer, loader, lr, device) -> float:
    """One epoch of CE only, snapping the weights to the codebooks after every step."""
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(loader))
    student.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = student(x)
        loss = sum(F.cross_entropy(o, y[:, i]) for i, o in enumerate(out))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        quantizer.apply(student, update_centroids=True)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def run_mct(student, teacher, proj, loader, valid_loader, bits, q_cfg, device, evaluate) -> tuple:
    """Multi-stage combined training. schedule is a string like 'dqdqdqq': d = distill epoch, q = quantize epoch."""
    quantizer = KMeansQuantizer(student, bits)
    opt = torch.optim.Adam([{"params": student.enc.parameters(), "lr": q_cfg["enc_lr"]},
                            {"params": [p for n, p in student.named_parameters() if not n.startswith("enc.")]
                             + list(proj.parameters()), "lr": q_cfg["head_lr"]}])
    log = []
    for i, ph in enumerate(q_cfg["schedule"]):
        if ph == "d":
            loss = distill_phase(student, teacher, proj, loader, opt, q_cfg["l1_weight"], device)
            # distill moved the weights off the codebook. move each centroid to its cluster mean and snap again
            # (snapping to the old centroids would throw the whole epoch away, an earlier version of this did that)
            quantizer.apply(student, update_centroids=True)
        else:
            loss = quant_phase(student, quantizer, loader, q_cfg["quant_lr"], device)
        m = evaluate(student, valid_loader, device)
        print(f"{bits}-bit phase {i} ({'distill' if ph == 'd' else 'quant'}) loss {loss:.3f} val {m}")
        log.append({"bits": bits, "phase": i, "kind": ph, "loss": loss, "val_intent": m["intent"]})
    return quantizer, log


if __name__ == "__main__":
    import os

    import yaml
    from torch.utils.data import DataLoader

    from src.baselines.naive import NaiveModel, evaluate, export_naive_fp32
    from src.encoder.data import FSCDataset
    from src.encoder.student import from_config, load_pretrained
    from src.measure.common import new_run_dir, write_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_encoder.yaml")
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--schedule", default=None, help="override quads.schedule, e.g. dq for a quick check")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    q_cfg = dict(cfg["quads"])
    if args.schedule:
        q_cfg["schedule"] = args.schedule
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep, h, tr = cfg["prepared_dir"], cfg["heads"], cfg["train"]
    mk = lambda s: FSCDataset(f"{prep}/fsc_{s}.csv", f"{prep}/fsc_labels.json", cfg["audio"])
    tl = DataLoader(mk("train"), tr["batch_size"], shuffle=True, num_workers=2)
    vl, tel = DataLoader(mk("valid"), tr["batch_size"]), DataLoader(mk("test"), tr["batch_size"])

    def make(which):
        enc = load_pretrained(from_config(cfg, which), cfg[which]["init_from"])
        return NaiveModel(enc, h["n_action"], h["n_object"], h["n_location"])

    teacher = make("teacher").to(dev)
    teacher.load_state_dict(torch.load(cfg["teacher"]["ckpt"], map_location=dev))
    teacher.eval()
    out_dir = new_run_dir("quads")
    os.makedirs("export", exist_ok=True)
    rows, log_rows = [], []
    for bits in args.bits:
        torch.manual_seed(cfg["seed"])
        student = make("student").to(dev)
        proj = nn.Linear(cfg["student"]["d_model"], cfg["teacher"]["d_model"]).to(dev)
        quantizer, log = run_mct(student, teacher, proj, tl, vl, bits, q_cfg, dev, evaluate)
        log_rows += log
        v, t = evaluate(student, vl, dev), evaluate(student, tel, dev)
        fp32_mb = sum(p.numel() for p in student.parameters()) * 4 / 1e6
        rows.append({"bits": bits, "val_intent": v["intent"], "test_intent": t["intent"], "test_action": t["action"],
                     "test_object": t["object"], "test_location": t["location"],
                     "model_mb_codebook": quantizer.size_mb(student), "model_mb_fp32": fp32_mb,
                     "schedule": q_cfg["schedule"], "l1_weight": q_cfg["l1_weight"]})
        print(rows[-1])
        torch.save(student.state_dict(), f"checkpoints/quads_{bits}bit.pt")
        # decoded weights are plain fp32 numbers, so this is the same graph as the naive baseline on the device
        export_naive_fp32(student.cpu().eval(), f"export/quads_{bits}bit_fp32.tflite")
    write_csv(out_dir / "quads_summary.csv", rows)
    write_csv(out_dir / "quads_phases.csv", log_rows)
    print("wrote", out_dir)
