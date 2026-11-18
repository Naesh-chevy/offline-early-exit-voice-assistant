"""Distill whisper base (teacher, finetuned on fsc) into the whisper tiny encoder (student).

loss = alpha_task * CE + alpha_kd * KL(logits at temperature T) + alpha_feat * MSE(projected hidden states)
Stages:  python -m src.encoder.distill --stage teacher
         python -m src.encoder.distill --stage student            (distilled)
         python -m src.encoder.distill --stage student --no_distill  (ablation, same seed and budget)
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, temp: float) -> torch.Tensor:
    """KL(teacher || student) on softened probs. temp^2 keeps the grad scale the same for any temp."""
    log_s = F.log_softmax(s_logits / temp, dim=-1)
    p_t = F.softmax(t_logits / temp, dim=-1)
    return F.kl_div(log_s, p_t, reduction="batchmean") * temp ** 2


class FeatureMatcher(nn.Module):
    """Linear projections student d -> teacher d, one per matched layer pair."""

    def __init__(self, d_student: int, d_teacher: int, n_pairs: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d_student, d_teacher) for _ in range(n_pairs)])

    def forward(self, s_hidden: list, t_hidden: list) -> torch.Tensor:
        """Both lists are the matched layers only, each (B, T, d)."""
        losses = [F.mse_loss(p(s), t) for p, s, t in zip(self.proj, s_hidden, t_hidden)]
        return sum(losses) / len(losses)


def distill_loss(s_logits, t_logits, s_hidden, t_hidden, matcher, y, kd_cfg, student_layers, teacher_layers):
    """s_logits / t_logits are lists of (B, n_class) for action, object, location.

    student_layers / teacher_layers are 1-indexed layer numbers to match, e.g. [2,3,4] and [2,4,6].
    """
    task = sum(F.cross_entropy(s, y[:, i]) for i, s in enumerate(s_logits))
    kd = sum(kd_loss(s, t, kd_cfg["temperature"]) for s, t in zip(s_logits, t_logits))
    feat = matcher([s_hidden[l - 1] for l in student_layers], [t_hidden[l - 1] for l in teacher_layers])
    total = kd_cfg["alpha_task"] * task + kd_cfg["alpha_kd"] * kd + kd_cfg["alpha_feat"] * feat
    return total, {"task": task.item(), "kd": kd.item(), "feat": feat.item()}


def distill_steps(student, teacher, matcher, loader, steps, opt, kd_cfg, student_layers, teacher_layers,
                  device="cpu", use_distill=True) -> list:
    """Train student for `steps` batches. use_distill=False is the no-distillation ablation (task loss only)."""
    student.train().to(device)
    teacher.eval().to(device)
    matcher.to(device)
    losses, it = [], iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        s_logits, s_hidden = student(x, return_hidden=True)
        if use_distill:
            with torch.no_grad():
                t_logits, t_hidden = teacher(x, return_hidden=True)
            loss, _ = distill_loss(s_logits, t_logits, s_hidden, t_hidden, matcher, y, kd_cfg,
                                   student_layers, teacher_layers)
        else:
            loss = sum(F.cross_entropy(s, y[:, i]) for i, s in enumerate(s_logits))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    import os

    import yaml
    from torch.utils.data import DataLoader

    from src.baselines.naive import NaiveModel, evaluate, make_opt, train_steps
    from src.encoder.ckpt import load_resume, save_resume
    from src.encoder.data import FSCDataset
    from src.encoder.student import from_config, load_pretrained

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_encoder.yaml")
    ap.add_argument("--stage", choices=["teacher", "student"], required=True)
    ap.add_argument("--no_distill", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/resume_*.pt if it exists")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep, h, tr = cfg["prepared_dir"], cfg["heads"], cfg["train"]
    train = FSCDataset(f"{prep}/fsc_train.csv", f"{prep}/fsc_labels.json", cfg["audio"])
    valid = FSCDataset(f"{prep}/fsc_valid.csv", f"{prep}/fsc_labels.json", cfg["audio"])
    tl = DataLoader(train, tr["batch_size"], shuffle=True, num_workers=2)
    vl = DataLoader(valid, tr["batch_size"])
    epochs = args.epochs or tr["epochs"]
    os.makedirs("checkpoints", exist_ok=True)

    def make(which):
        enc = load_pretrained(from_config(cfg, which), cfg[which]["init_from"])
        return NaiveModel(enc, h["n_action"], h["n_object"], h["n_location"])

    if args.stage == "teacher":
        teacher = make("teacher").to(dev)
        opt = make_opt(teacher, tr["lr"])
        rpath = "checkpoints/resume_teacher.pt"
        start = load_resume(rpath, opt, teacher=teacher) if args.resume and os.path.exists(rpath) else 0
        # the last epoch can be worse than an earlier one, so keep the best by val intent accuracy
        bpath = cfg["teacher"]["ckpt"] + ".best"
        best = float(open(bpath).read()) if args.resume and os.path.exists(bpath) else -1.0
        for ep in range(start, epochs):
            train_steps(teacher, tl, len(tl), opt, dev)
            m = evaluate(teacher, vl, dev)
            print("teacher epoch", ep, m)
            if m["intent"] > best:
                best = m["intent"]
                torch.save(teacher.state_dict(), cfg["teacher"]["ckpt"])
                open(bpath, "w").write(str(best))
            save_resume(rpath, ep + 1, opt, teacher=teacher)
        print("best teacher val intent:", best)
    else:
        teacher = make("teacher").to(dev)
        teacher.load_state_dict(torch.load(cfg["teacher"]["ckpt"]))
        student = make("student").to(dev)
        d = cfg["distill"]
        s_layers, t_layers = cfg["heads"]["exits"], d["teacher_layer_map"]
        matcher = FeatureMatcher(cfg["student"]["d_model"], cfg["teacher"]["d_model"], len(s_layers)).to(dev)
        opt = torch.optim.AdamW(list(student.parameters()) + list(matcher.parameters()), lr=tr["lr"],
                                weight_decay=tr["weight_decay"])
        rpath = f"checkpoints/resume_{'nodistill' if args.no_distill else 'student'}.pt"
        start = load_resume(rpath, opt, student=student, matcher=matcher) if args.resume and os.path.exists(rpath) else 0
        out = cfg["out"]["nodistill_dir"] if args.no_distill else cfg["out"]["student_dir"]
        os.makedirs(out, exist_ok=True)
        bpath = f"{out}/encoder.best"
        best = float(open(bpath).read()) if args.resume and os.path.exists(bpath) else -1.0
        for ep in range(start, epochs):
            ls = distill_steps(student, teacher, matcher, tl, len(tl), opt, d, s_layers, t_layers, dev,
                               use_distill=not args.no_distill)
            m = evaluate(student, vl, dev)
            print(f"student epoch {ep} loss {sum(ls) / len(ls):.3f}", m)
            if m["intent"] > best:   # keep the best epoch, not the last
                best = m["intent"]
                torch.save(student.enc.state_dict(), f"{out}/encoder.pt")
                open(bpath, "w").write(str(best))
            save_resume(rpath, ep + 1, opt, student=student, matcher=matcher)
        print("best student val intent:", best)
