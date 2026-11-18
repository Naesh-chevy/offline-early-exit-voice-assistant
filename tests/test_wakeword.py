import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.runtime.tflite_runner import Segment
from src.wakeword.export import export_wakeword_int8
from src.wakeword.model import DSCNN, enable_qat, fold_bn, n_params

N_MELS, N_FRAMES = 80, 100
WIDTHS = (16, 16, 16)


def fake_batch(n, seed):
    """class 1 has a loud band in the low mel bins, class 0 is just noise."""
    g = np.random.default_rng(seed)
    x = g.normal(0, 0.3, (n, 1, N_MELS, N_FRAMES)).astype(np.float32)
    y = g.integers(0, 2, n)
    x[y == 1, :, 5:25, 20:80] += 1.0
    return torch.from_numpy(x), torch.from_numpy(y)


def train(model, steps, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for i in range(steps):
        x, y = fake_batch(32, i)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_forward_shape():
    out = DSCNN(WIDTHS)(torch.randn(3, 1, N_MELS, N_FRAMES))
    assert out.shape == (3, 2)


def test_default_model_under_param_limit():
    assert n_params(DSCNN((64, 64, 64, 64))) < 100_000


def test_fold_bn_keeps_outputs():
    torch.manual_seed(0)
    m = DSCNN(WIDTHS)
    train(m, 20)
    m.eval()
    x, _ = fake_batch(8, 99)
    before = m(x)
    after = fold_bn(m).eval()(x)
    assert torch.allclose(before, after, atol=1e-4)


@pytest.fixture(scope="module")
def int8_setup(tmp_path_factory):
    torch.manual_seed(0)
    m = DSCNN(WIDTHS)
    train(m, 150)
    m.eval()
    fold_bn(m)
    enable_qat(m)
    train(m, 100, lr=1e-3)   # qat fine tune
    m.eval()
    path = str(tmp_path_factory.mktemp("ww") / "ww_int8.tflite")
    calib = [fake_batch(1, 1000 + i)[0][0, 0].numpy() for i in range(60)]
    export_wakeword_int8(m, calib, path, N_MELS, N_FRAMES)
    return m, path


def test_exported_model_is_really_int8(int8_setup):
    _, path = int8_setup
    seg = Segment(path)
    assert seg.inp[0]["dtype"] == np.int8
    assert seg.out[0]["dtype"] == np.int8
    dtypes = {d["dtype"] for d in seg.interp.get_tensor_details()}
    assert np.float32 not in dtypes, dtypes


def test_fp32_and_int8_agree(int8_setup):
    model, path = int8_setup
    seg = Segment(path)
    x, _ = fake_batch(100, 5000)
    with torch.no_grad():
        # fp32 reference = same folded weights with fake quant switched off
        enable_qat(model, False)
        ref = model(x).argmax(1).numpy()
        enable_qat(model, True)
    got = np.array([seg.run(x[i, 0].numpy().reshape(1, N_MELS, N_FRAMES, 1))[0].argmax() for i in range(100)])
    assert (ref == got).mean() > 0.95


@pytest.mark.gpu
def test_loss_decreases_gpu():
    m = DSCNN(WIDTHS).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for i in range(50):
        x, y = fake_batch(32, i)
        loss = F.cross_entropy(m(x.cuda()), y.cuda())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_loss_decreases_cpu():
    losses = train(DSCNN(WIDTHS), 50)
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_noise_only_negatives_are_added(tmp_path):
    import csv
    import wave

    from src.wakeword.train import WakeDataset
    rows = []
    for i in range(10):
        p = tmp_path / f"{i}.wav"
        with wave.open(str(p), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes((np.random.randn(16000) * 3000).astype("<i2").tobytes())
        rows.append({"path": str(p), "is_wake": str(int(i < 2))})
    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, ["path", "is_wake"]); w.writeheader(); w.writerows(rows)
    cfg = {"audio": {"sample_rate": 16000, "window_s": 1.0, "n_mels": 80, "n_fft": 400, "hop": 160},
           "train": {"other_per_wake": 6, "noise_prob": 0.8, "noise_gain_max": 0.3, "noise_neg_frac": 0.5}}
    noise = [np.random.randn(32000).astype(np.float32) * 0.1]
    ds = WakeDataset(str(man), cfg, noise, True)
    assert len(ds) == int(10 * 1.5)          # 10 real clips + 50% noise only
    x, y = ds[len(ds) - 1]                   # the last ones are the noise only rows
    assert tuple(x.shape) == (1, 80, 100) and y == 0
    test_ds = WakeDataset(str(man), cfg, noise, False)
    assert len(test_ds) == 10                # test set stays untouched
