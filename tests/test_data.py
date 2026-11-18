import csv
import pathlib

import numpy as np
import pytest

from src.audio.features import clip_features, log_mel, pad_or_crop
from src.audio.labels import build_label_map, invert, load_label_maps, save_label_maps

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared"


def sine(seconds, freq=440.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_log_mel_shape_and_dtype():
    feats = log_mel(sine(4.0))
    assert feats.shape == (80, 400)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_sine_energy_is_in_the_right_mel_bin():
    feats = log_mel(sine(1.0, freq=440.0))
    peak_bin = feats.mean(axis=1).argmax()
    # 440 Hz is low on the mel axis, should be in the bottom third
    assert peak_bin < 27


def test_pad_and_crop_lengths():
    assert len(pad_or_crop(sine(1.0), 64000)) == 64000
    assert len(pad_or_crop(sine(6.0), 64000)) == 64000
    short = pad_or_crop(np.ones(100, dtype=np.float32), 400)
    assert short[:100].sum() == 100 and short[100:].sum() == 0


def test_window_sizes():
    assert clip_features(sine(0.4), 1.0).shape == (80, 100)  # wake word window
    assert clip_features(sine(9.0), 4.0).shape == (80, 400)  # encoder window


def test_label_map_round_trip(tmp_path):
    maps = {"action": build_label_map(["increase", "activate", "increase", "bring"]),
            "location": build_label_map(["none", "kitchen"])}
    assert maps["action"] == {"activate": 0, "bring": 1, "increase": 2}
    p = tmp_path / "labels.json"
    save_label_maps(maps, str(p))
    back = load_label_maps(str(p))
    assert back == maps
    assert invert(back["action"])[2] == "increase"


def count_rows(name):
    p = PREP / name
    if not p.exists():
        pytest.skip(f"{name} not prepared, run data/download.py and data/prepare.py first")
    with open(p, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def test_fsc_split_sizes():
    # published fsc sizes
    assert count_rows("fsc_train.csv") == 23132
    assert count_rows("fsc_valid.csv") == 3118
    assert count_rows("fsc_test.csv") == 3793


def test_speech_commands_splits_not_empty():
    for split in ["train", "valid", "test"]:
        assert count_rows(f"sc_{split}.csv") > 1000


def test_slurp_iot_splits_not_empty():
    for split in ["train", "valid", "test"]:
        assert count_rows(f"slurp_iot_{split}.csv") > 100
