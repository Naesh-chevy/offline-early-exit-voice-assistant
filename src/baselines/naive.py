"""Naive baseline: FP32 whisper tiny encoder (all 4 layers), mean pool, linear classifiers.

No exits, no distillation, no quantization. This is the "no tricks" reference.
Train: python -m src.baselines.naive --config configs/train_encoder.yaml
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder.student import WhisperEncoder, from_config, load_pretrained


class NaiveModel(nn.Module):
    def __init__(self, enc: WhisperEncoder, n_action: int, n_object: int, n_location: int):
        super().__init__()
        self.enc = enc
        d = enc.layer_norm.normalized_shape[0]
        self.action = nn.Linear(d, n_action)
        self.obj = nn.Linear(d, n_object)
        self.location = nn.Linear(d, n_location)

    def forward(self, mel: torch.Tensor, return_hidden: bool = False):
        out, hidden = self.enc(mel, return_hidden=True)
        feats = out.mean(dim=1)  # mean pool over time
        logits = (self.action(feats), self.obj(feats), self.location(feats))
        return (logits, hidden) if return_hidden else logits


def export_naive_fp32(model: NaiveModel, out_path: str) -> None:
    """FP32 tflite of the whole thing. Input is (1, frames, n_mels), channels last."""
    import tensorflow as tf

    from src.export import tf_encoder as T

    p = T.np_params(model)
    enc_p = {k[4:]: v for k, v in p.items() if k.startswith("enc.")}
    n_heads = model.enc.layers[0].n_heads
    n_layers, n_frames = model.enc.n_layers, model.enc.n_pos * 2
    n_mels = model.enc.conv1.in_channels

    def fn(mel):
        x = T.frontend(enc_p, mel)
        x = T.run_layers(enc_p, x, 1, n_layers, n_heads)
        x = T.layer_norm(x, enc_p["layer_norm.weight"], enc_p["layer_norm.bias"])
        f = tf.reduce_mean(x, axis=1)
        return (T.linear(f, p["action.weight"], p["action.bias"]),
                T.linear(f, p["obj.weight"], p["obj.bias"]),
                T.linear(f, p["location.weight"], p["location.bias"]))

    data = T.to_tflite(fn, [tf.TensorSpec([1, n_frames, n_mels], tf.float32)])
    with open(out_path, "wb") as f:
        f.write(data)


def naive_loss(outs, y):
    return sum(F.cross_entropy(o, y[:, i]) for i, o in enumerate(outs))


def make_opt(model, lr: float):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)


def train_steps(model, loader, steps: int, opt, device="cpu") -> list:
    """Plain training loop, returns the loss per step."""
    losses, it = [], iter(loader)
    model.train().to(device)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        loss = naive_loss(model(x.to(device)), y.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate(model, loader, device="cpu") -> dict:
    """Accuracy per slot and full intent (all three right)."""
    model.eval().to(device)
    right = torch.zeros(3)
    full = n = 0
    for x, y in loader:
        outs = model(x.to(device))
        ok = torch.stack([o.argmax(1).cpu() == y[:, i] for i, o in enumerate(outs)], dim=1)
        right += ok.sum(0)
        full += ok.all(1).sum().item()
        n += len(y)
    return {"action": right[0].item() / n, "object": right[1].item() / n,
            "location": right[2].item() / n, "intent": full / n}


if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader

    from src.encoder.data import FSCDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_encoder.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/naive_fp32.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep = cfg["prepared_dir"]
    train = FSCDataset(f"{prep}/fsc_train.csv", f"{prep}/fsc_labels.json", cfg["audio"])
    valid = FSCDataset(f"{prep}/fsc_valid.csv", f"{prep}/fsc_labels.json", cfg["audio"])
    enc = load_pretrained(from_config(cfg, "student"), cfg["student"]["init_from"])
    h = cfg["heads"]
    model = NaiveModel(enc, h["n_action"], h["n_object"], h["n_location"])
    tl = DataLoader(train, cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    vl = DataLoader(valid, cfg["train"]["batch_size"])
    opt = make_opt(model, cfg["train"]["lr"])
    best = -1.0
    for ep in range(args.epochs or cfg["train"]["epochs"]):
        losses = train_steps(model, tl, len(tl), opt, dev)
        m = evaluate(model, vl, dev)
        print(f"epoch {ep} loss {sum(losses) / len(losses):.3f} val {m}")
        if m["intent"] > best:      # same rule as teacher/student: keep the best val epoch, not the last
            best = m["intent"]
            torch.save(model.state_dict(), args.out)
    print("best naive val intent:", best)
    model.load_state_dict(torch.load(args.out, map_location="cpu"))
    export_naive_fp32(model.cpu().eval(), "export/naive_fp32.tflite")
