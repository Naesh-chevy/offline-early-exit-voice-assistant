"""Segmented TFLite export (written by us, no off the shelf tool).

The encoder is cut into separate tflite files so layers after an exit never run:
    seg1 = frontend + layers 1..2 + exit 1 heads
    seg2 = layer 3 + exit 2 heads
    seg3 = layer 4 + exit 3 heads
Each file takes the hidden state from the previous one (python passes it along) and returns
[hidden for the next segment (not on the last one), action, obj, location, slot, decision] logits.
The CONTINUE mask of the last exit is NOT baked in (a -1e4 logit would wreck the int8 range), the runtime applies it.

usage: python -m src.export.segments --config configs/train_encoder.yaml --ckpt checkpoints/exits_distill_lam5.pt
"""
import argparse
import os

import numpy as np
import tensorflow as tf
import torch
import yaml

from src.export import tf_encoder as T
from src.measure.common import write_csv

HEAD_KEYS = ["action", "obj", "location", "slot", "decision"]


def head_tf(p: dict, i: int, h):
    """TF version of ExitHead: mean pool, layernorm, shared layer, five linear heads."""
    pre = f"heads.{i}."
    m = tf.reduce_mean(h, axis=1)                                   # (1, d)
    m = T.layer_norm(m, p[pre + "norm.weight"], p[pre + "norm.bias"])
    z = T.gelu(T.linear(m, p[pre + "shared.weight"], p[pre + "shared.bias"]))
    return [T.linear(z, p[pre + k + ".weight"], p[pre + k + ".bias"]) for k in HEAD_KEYS]


def build_segment_fn(p: dict, seg_idx: int, layers: list, n_heads: int, is_last: bool):
    first, last = layers[0], layers[-1]
    enc_p = {k[4:]: v for k, v in p.items() if k.startswith("enc.")}

    def fn(x):
        h = T.frontend(enc_p, x) if seg_idx == 0 else x
        h = T.run_layers(enc_p, h, first, last, n_heads)
        heads = head_tf(p, seg_idx, h)
        return tuple(heads) if is_last else (h, *heads)

    return fn


@torch.no_grad()
def segment_inputs(model, mels: np.ndarray, seg_cfg: list) -> list:
    """Float inputs each segment would see, for calibration. mels (N, n_mels, frames). One list per segment."""
    model.eval()
    mel_t = torch.from_numpy(mels)
    x = model.enc.frontend(mel_t)
    ins = [[m.transpose(0, 1).numpy()[None] for m in mel_t]]   # seg1 takes (1, frames, n_mels)
    done = 0
    for s in seg_cfg[:-1]:
        x = model.enc.run_layers(x, done + 1, s["layers"][-1])
        done = s["layers"][-1]
        ins.append([h.numpy()[None] for h in x])
    return ins


def export_segments(model, seg_cfg: list, calib_mels: np.ndarray, out_dir: str, int8: bool = True) -> list:
    """Writes seg*.tflite into out_dir, returns [(path, size_bytes)]."""
    os.makedirs(out_dir, exist_ok=True)
    p = T.np_params(model)
    n_heads = model.enc.layers[0].n_heads
    n_mels, n_frames, d = model.enc.conv1.in_channels, model.enc.n_pos * 2, model.enc.layer_norm.normalized_shape[0]
    ins = segment_inputs(model, calib_mels, seg_cfg) if int8 else None
    files = []
    for i, s in enumerate(seg_cfg):
        is_last = i == len(seg_cfg) - 1
        fn = build_segment_fn(p, i, s["layers"], n_heads, is_last)
        spec = tf.TensorSpec([1, n_frames, n_mels] if i == 0 else [1, model.enc.n_pos, d], tf.float32)
        calib = None
        if int8:
            def calib(data=ins[i]):
                for x in data:
                    yield [x.astype(np.float32)]
        data = T.to_tflite(fn, [spec], calib)
        path = os.path.join(out_dir, f"{s['name']}{'_int8' if int8 else '_fp32'}.tflite")
        with open(path, "wb") as f:
            f.write(data)
        files.append((path, len(data)))
        print(f"{path}: {len(data) / 1024:.0f} KB")
    print(f"total {sum(sz for _, sz in files) / 1024:.0f} KB")
    write_csv(os.path.join(out_dir, f"segments_size_{'int8' if int8 else 'fp32'}.csv"),
              [{"file": os.path.basename(f), "size_kb": sz / 1024} for f, sz in files])
    return files


if __name__ == "__main__":
    from src.audio.features import clip_features, load_wav
    from src.encoder.exits import from_config as exits_from_config
    from src.encoder.student import from_config as enc_from_config
    import csv
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_encoder.yaml")
    ap.add_argument("--export_config", default="configs/export.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="export")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ecfg = yaml.safe_load(open(args.export_config))
    model = exits_from_config(cfg, enc_from_config(cfg, "student"))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    # calibration clips from fsc train, fixed seed
    rows = list(csv.DictReader(open(f"{cfg['prepared_dir']}/fsc_train.csv")))
    random.Random(ecfg["calib"]["seed"]).shuffle(rows)
    a = cfg["audio"]
    mels = np.stack([clip_features(load_wav(r["path"], a["sample_rate"]), a["window_s"], a["n_mels"], a["n_fft"],
                                   a["hop"], a["sample_rate"]) for r in rows[: ecfg["calib"]["n_samples"]]])
    export_segments(model, ecfg["segments"], mels, args.out_dir, int8=True)
    export_segments(model, ecfg["segments"], mels, args.out_dir, int8=False)
