import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.baselines.naive import NaiveModel, export_naive_fp32, make_opt, train_steps
from src.encoder.data import FakeSLUData
from src.encoder.student import WhisperEncoder
from src.runtime.tflite_runner import Segment

# tiny encoder so tests run in seconds: 40 mel frames -> 20 positions
N_MELS, N_FRAMES = 80, 40


def tiny_model(seed=0):
    torch.manual_seed(seed)
    enc = WhisperEncoder(d_model=32, n_layers=4, n_heads=2, ffn=64, n_mels=N_MELS, n_frames=N_FRAMES)
    return NaiveModel(enc, 3, 4, 2).eval()


def test_forward_shapes():
    outs = tiny_model()(torch.randn(5, N_MELS, N_FRAMES))
    assert [tuple(o.shape) for o in outs] == [(5, 3), (5, 4), (5, 2)]


def test_tflite_matches_pytorch(tmp_path):
    model = tiny_model()
    path = str(tmp_path / "naive.tflite")
    export_naive_fp32(model, path)
    seg = Segment(path)
    for i in range(5):
        mel = torch.randn(1, N_MELS, N_FRAMES, generator=torch.Generator().manual_seed(i))
        with torch.no_grad():
            ref = model(mel)
        got = seg.run(mel.transpose(1, 2).numpy())  # tflite is channels last
        # the three heads can come out in any order, match by shape
        for r in ref:
            best = min(np.abs(g - r.numpy()).max() for g in got if g.shape == tuple(r.shape))
            assert best < 1e-3


@pytest.mark.gpu
def test_training_loss_decreases():
    dev = "cuda"
    model = tiny_model().to(dev)
    loader = DataLoader(FakeSLUData(256, N_MELS, N_FRAMES, 3, 4, 2), batch_size=32, shuffle=True)
    losses = train_steps(model, loader, 50, make_opt(model, 1e-3), dev)
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_training_loss_decreases_cpu():
    # same check as the gpu one but on cpu with a tiny model, so it runs on a laptop
    model = tiny_model()
    loader = DataLoader(FakeSLUData(128, N_MELS, N_FRAMES, 3, 4, 2), batch_size=32, shuffle=True)
    losses = train_steps(model, loader, 50, make_opt(model, 1e-3))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
