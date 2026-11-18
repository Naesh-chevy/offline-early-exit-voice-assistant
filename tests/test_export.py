import os

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from src.encoder.data import FakeSLUData
from src.encoder.exits import ExitModel
from src.encoder.loss import total_loss
from src.encoder.policy import run_policy
from src.encoder.student import WhisperEncoder
from src.export.segments import export_segments
from src.runtime.segmented import SegmentedEncoder
from src.runtime.tflite_runner import Segment

LOSS_CFG = {"reject_cost": 1.0, "continue_cost": 0.2, "exit_weights": [0.3, 0.3, 0.4],
            "early_distill_weight": 0.5, "decision_weight": 1.0}
SEG_CFG = yaml.safe_load(open("configs/export.yaml"))["segments"]
N_MELS, N_FRAMES = 80, 40


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    model = ExitModel(WhisperEncoder(32, 4, 2, 64, N_MELS, N_FRAMES), [2, 3, 4], 16, 3, 4, 2, 15)
    loader = DataLoader(FakeSLUData(512, N_MELS, N_FRAMES, 3, 4, 2, seed=1), batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step, (x, y) in enumerate(loader):
        for _ in range(2):
            loss, _ = total_loss(model(x), y, 2.0, LOSS_CFG)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    calib = FakeSLUData(100, N_MELS, N_FRAMES, 3, 4, 2, seed=11).x
    root = tmp_path_factory.mktemp("seg")
    int8 = export_segments(model, SEG_CFG, calib, str(root / "int8"), int8=True)
    fp32 = export_segments(model, SEG_CFG, calib, str(root / "fp32"), int8=False)
    held = FakeSLUData(100, N_MELS, N_FRAMES, 3, 4, 2, seed=99)
    return model, [p for p, _ in int8], [p for p, _ in fp32], held, int8


# thresholds that force going through every exit, the last one always acts
RUN_ALL = ([2.0, 2.0, 0.0], [2.0, 2.0, 0.0])
EXIT_AT_FIRST = ([0.0, 0.0, 0.0], [2.0, 2.0, 0.0])


def torch_final(model, x):
    with torch.no_grad():
        o = model(torch.from_numpy(x)[None])[-1]
    return {k: v[0].numpy() for k, v in o.items()}


def chain_final(enc, x):
    enc.start(x)
    return run_policy(enc.exit_fn, *RUN_ALL)


def test_fp32_segments_match_torch(setup):
    model, _, fp32, held, _ = setup
    enc = SegmentedEncoder(fp32)
    for i in range(10):
        enc.start(held.x[i])
        outs = [enc.exit_fn(k) for k in range(3)]
        with torch.no_grad():
            ref = model(torch.from_numpy(held.x[i])[None])
        for k in range(3):
            for name in ["action", "obj", "location", "slot"]:
                assert np.abs(outs[k][name] - ref[k][name][0].numpy()).max() < 1e-3


def test_int8_chain_agrees_with_torch_on_95_percent(setup):
    model, int8, _, held, _ = setup
    enc = SegmentedEncoder(int8)
    agree = 0
    for i in range(len(held)):
        ref = torch_final(model, held.x[i])
        r = chain_final(enc, held.x[i])
        agree += (r.action, r.obj, r.location) == (ref["action"].argmax(), ref["obj"].argmax(), ref["location"].argmax())
    print("int8 agreement", agree / len(held))
    assert agree / len(held) >= 0.95


def test_exit_at_layer_2_never_runs_later_segments(setup):
    _, int8, _, held, _ = setup
    enc = SegmentedEncoder(int8)
    for i in range(5):
        enc.start(held.x[i])
        r = run_policy(enc.exit_fn, *EXIT_AT_FIRST)
        assert r.exit_idx == 0
    assert enc.calls() == [5, 0, 0]     # seg2 and seg3 interpreters were never invoked


def test_running_all_exits_calls_every_segment_once_per_clip(setup):
    _, int8, _, held, _ = setup
    enc = SegmentedEncoder(int8)
    chain_final(enc, held.x[0])
    assert enc.calls() == [1, 1, 1]


def test_asking_for_exits_out_of_order_fails(setup):
    _, int8, _, held, _ = setup
    enc = SegmentedEncoder(int8)
    enc.start(held.x[0])
    with pytest.raises(AssertionError):
        enc.exit_fn(1)


def test_all_segment_files_are_int8_and_size_logged(setup):
    *_, int8 = setup
    total = 0
    for path, size in int8:
        seg = Segment(path)
        assert seg.inp[0]["dtype"] == np.int8
        assert all(o["dtype"] == np.int8 for o in seg.out)
        assert np.float32 not in {d["dtype"] for d in seg.interp.get_tensor_details()}
        total += size
    print(f"segment sizes: {[(os.path.basename(p), s // 1024) for p, s in int8]} total {total // 1024} KB")
    assert os.path.exists(os.path.join(os.path.dirname(int8[0][0]), "segments_size_int8.csv"))


@pytest.mark.device
def test_exit_at_layer_2_is_faster_than_all_layers_on_device():
    """Needs export/seg*_int8.tflite from a trained model. Run on the pi."""
    from src.measure.latency import measure_latency
    cfg = yaml.safe_load(open("configs/train_encoder.yaml"))
    dev = yaml.safe_load(open("configs/device.yaml"))
    paths = [f"export/{s['name']}_int8.tflite" for s in SEG_CFG]
    enc = SegmentedEncoder(paths, dev["n_threads"])
    frames = int(cfg["audio"]["window_s"] * 100)
    mel = np.random.default_rng(0).standard_normal((cfg["audio"]["n_mels"], frames)).astype(np.float32)

    def go(thresholds):
        enc.start(mel)
        run_policy(enc.exit_fn, *thresholds)

    early = measure_latency(lambda: go(EXIT_AT_FIRST), 5, 30)["p50_ms"]
    full = measure_latency(lambda: go(RUN_ALL), 5, 30)["p50_ms"]
    print(f"exit at layer 2: {early:.1f} ms, all layers: {full:.1f} ms")
    assert early < full
