"""FSC dataset over the manifests from data/prepare.py."""
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audio.features import clip_features, load_wav
from src.audio.labels import load_label_maps

IGNORE = -100  # label for "no value", cross entropy skips it


class FSCDataset(Dataset):
    """Returns log mel (n_mels, frames) and labels (action, object, location, value).

    value is the thermostat setpoint class (setpoint - 16) from our own recordings, IGNORE for fsc.
    """

    def __init__(self, manifest: str, label_path: str, audio_cfg: dict):
        with open(manifest, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.maps = load_label_maps(label_path)
        self.cfg = audio_cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        c = self.cfg
        wave = load_wav(r["path"], c["sample_rate"])
        mel = clip_features(wave, c["window_s"], c["n_mels"], c["n_fft"], c["hop"], c["sample_rate"])
        value = int(r["value"]) - 16 if r.get("value") else IGNORE
        y = torch.tensor([self.maps["action"][r["action"]], self.maps["object"][r["object"]],
                          self.maps["location"][r["location"]], value])
        return torch.from_numpy(mel), y


class FakeSLUData(Dataset):
    """Synthetic data for tiny tests: labels are baked into the mel so a tiny model can learn them."""

    def __init__(self, n: int, n_mels=80, n_frames=40, n_action=3, n_object=4, n_location=2, seed=0):
        g = np.random.default_rng(seed)
        self.y = np.stack([g.integers(0, n_action, n), g.integers(0, n_object, n), g.integers(0, n_location, n),
                           np.full(n, IGNORE)], axis=1)
        self.x = g.normal(0, 1, (n, n_mels, n_frames)).astype(np.float32)
        for i in range(n):  # each label shifts its own band of mel rows, so all three are learnable
            for lab, (lo, hi) in zip(self.y[i, :3], [(0, 20), (20, 40), (40, 60)]):
                self.x[i, lo:hi] += 1.5 * lab

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])
