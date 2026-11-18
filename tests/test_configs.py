import pathlib

import yaml

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs"

REQUIRED = {
    "device.yaml": ["target", "n_threads", "mic", "actuators", "energy"],
    "train_wakeword.yaml": ["seed", "wake_word", "audio", "model", "train", "eval", "out"],
    "train_encoder.yaml": ["seed", "audio", "student", "teacher", "heads", "distill", "loss", "policy", "train"],
    "export.yaml": ["segments", "calib", "parity"],
    "eval.yaml": ["seed", "latency", "memory", "lambdas"],
}


def load(name):
    with open(CFG / name) as f:
        return yaml.safe_load(f)


def test_all_configs_have_required_keys():
    for name, keys in REQUIRED.items():
        cfg = load(name)
        for k in keys:
            assert k in cfg, f"{name} is missing {k}"


def test_device_target_is_valid():
    assert load("device.yaml")["target"] in ("pi5", "pi4", "android")


def test_lambda_sweep_is_above_or_equal_one():
    lams = load("train_encoder.yaml")["loss"]["lambda_sweep"]
    assert all(l >= 1 for l in lams)


def test_exit_settings_line_up():
    enc = load("train_encoder.yaml")
    n_exits = len(enc["heads"]["exits"])
    assert len(enc["loss"]["exit_weights"]) == n_exits
    assert len(enc["policy"]["act_thresh"]) == n_exits
    assert len(enc["policy"]["reject_thresh"]) == n_exits
    assert len(enc["distill"]["teacher_layer_map"]) == n_exits
    assert max(enc["heads"]["exits"]) == enc["student"]["n_layers"]


def test_segments_cover_all_layers():
    layers = [l for s in load("export.yaml")["segments"] for l in s["layers"]]
    assert layers == [1, 2, 3, 4]
