import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from src.baselines.naive import NaiveModel
from src.encoder.data import FakeSLUData
from src.encoder.distill import FeatureMatcher, distill_steps, kd_loss
from src.encoder.student import WhisperEncoder, from_config


def test_shortened_input_gives_right_sequence_length():
    cfg = yaml.safe_load(open("configs/train_encoder.yaml"))
    enc = from_config(cfg, "student").eval()
    frames = int(cfg["audio"]["window_s"] * cfg["audio"]["sample_rate"] / cfg["audio"]["hop"])
    assert frames == 400
    with torch.no_grad():
        out, hidden = enc(torch.randn(1, 80, frames), return_hidden=True)
    assert enc.embed_positions.weight.shape == (200, 384)   # cropped from 1500
    assert out.shape == (1, 200, 384)
    assert len(hidden) == 4


def test_kd_zero_when_logits_match_positive_otherwise():
    t = torch.randn(8, 6)
    assert kd_loss(t.clone(), t, 2.0).item() < 1e-6
    assert kd_loss(torch.randn(8, 6), t, 2.0).item() > 1e-3


def test_kd_temperature_scale_is_finite_and_nonneg():
    for temp in [1.0, 2.0, 4.0]:
        assert kd_loss(torch.randn(4, 5), torch.randn(4, 5), temp).item() >= 0


def test_feature_projection_shapes_line_up():
    m = FeatureMatcher(32, 64, 3)
    s = [torch.randn(2, 20, 32) for _ in range(3)]
    t = [torch.randn(2, 20, 64) for _ in range(3)]
    loss = m(s, t)
    assert loss.shape == () and loss.item() > 0
    loss.backward()
    assert all(p.grad is not None for p in m.proj.parameters())


def tiny(d, layers, heads, seed):
    torch.manual_seed(seed)
    return NaiveModel(WhisperEncoder(d, layers, heads, d * 2, 80, 40), 3, 4, 2)


KD = {"temperature": 2.0, "alpha_task": 1.0, "alpha_kd": 1.0, "alpha_feat": 0.5}


def run_distill(steps, device="cpu"):
    teacher, student = tiny(64, 6, 4, 1), tiny(32, 4, 2, 2)
    data = FakeSLUData(256, 80, 40, 3, 4, 2)
    loader = DataLoader(data, batch_size=32, shuffle=True)
    # teacher gets a short head start so its logits mean something
    from src.baselines.naive import make_opt, train_steps
    train_steps(teacher, loader, 40, make_opt(teacher, 1e-3), device)
    matcher = FeatureMatcher(32, 64, 3)
    opt = torch.optim.AdamW(list(student.parameters()) + list(matcher.parameters()), lr=1e-3)
    return distill_steps(student, teacher, matcher, loader, steps, opt, KD, [2, 3, 4], [2, 4, 6], device)


def test_distill_loss_decreases_cpu():
    losses = run_distill(60)
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


@pytest.mark.gpu
def test_distill_loss_decreases_gpu():
    losses = run_distill(100, "cuda")
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
