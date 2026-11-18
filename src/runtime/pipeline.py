"""On device pipeline: mic -> wake word -> segmented encoder + exit policy -> registry check -> actuator.

usage (on the pi): python -m src.runtime.pipeline --wake export/wakeword_int8.tflite --seg_dir export
"""
import argparse
import time

import numpy as np
import yaml

from src.actuators.registry import intent_to_command
from src.audio.features import clip_features
from src.audio.labels import invert, load_label_maps
from src.encoder.exits import ACT
from src.encoder.policy import run_policy
from src.runtime.segmented import SegmentedEncoder
from src.runtime.tflite_runner import Segment


class Pipeline:
    def __init__(self, wake: Segment | None, encoder: SegmentedEncoder, labels: dict, client, enc_cfg: dict,
                 audio_cfg: dict, wake_thresh: float = 0.5):
        self.wake, self.encoder, self.client = wake, encoder, client
        self.inv = {k: invert(v) for k, v in labels.items()}
        self.pol = enc_cfg["policy"]
        self.audio = audio_cfg
        self.wake_thresh = wake_thresh

    def wake_prob(self, wave_1s: np.ndarray) -> float:
        a = self.audio
        mel = clip_features(wave_1s, 1.0, a["n_mels"], a["n_fft"], a["hop"], a["sample_rate"])
        logits = self.wake.run(mel.reshape(1, *mel.shape, 1))[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())

    def handle_utterance(self, wave: np.ndarray) -> dict:
        """Run the encoder + policy on one command clip and, if it acts, send the command."""
        a = self.audio
        t0 = time.perf_counter()
        mel = clip_features(wave, a["window_s"], a["n_mels"], a["n_fft"], a["hop"], a["sample_rate"])
        t_feat = time.perf_counter()
        self.encoder.start(mel)
        res = run_policy(self.encoder.exit_fn, self.pol["act_thresh"], self.pol["reject_thresh"])
        t_enc = time.perf_counter()
        info = {"decision": "act" if res.decision == ACT else "reject", "exit": res.exit_idx + 1,
                "feat_ms": (t_feat - t0) * 1000, "encoder_ms": (t_enc - t_feat) * 1000,
                "command": None, "sent": False, "why": ""}
        if res.decision != ACT:
            info["why"] = "rejected, ask user to repeat"
            return info
        action, obj, loc = self.inv["action"][res.action], self.inv["object"][res.obj], self.inv["location"][res.location]
        cmd = intent_to_command(action, obj, loc, res.slot + 16)   # slot class 0 = 16 C
        if cmd is None:
            info["why"] = f"{action} {obj} {loc} is not a device we have"
            return info
        info["command"] = cmd
        info["sent"], info["why"] = self.client.send(cmd)   # registry check happens inside send
        return info

    def run_live(self, mic, max_utts: int | None = None, listen_s: float = 4.0) -> None:
        """Poll the ring buffer for the wake word, then record up to listen_s and handle it."""
        sr, n_done = self.audio["sample_rate"], 0
        while max_utts is None or n_done < max_utts:
            time.sleep(0.25)
            last = mic.ring.read_last(sr)
            if len(last) < sr or self.wake_prob(last) < self.wake_thresh:
                continue
            mic.sink = []
            t0 = time.time()
            while time.time() - t0 < listen_s:
                time.sleep(0.1)
                if time.time() - t0 > 1.0 and not mic.speech:   # speech over
                    break
            wave = np.concatenate(mic.sink) if mic.sink else np.zeros(1, dtype=np.float32)
            mic.sink = None
            print(self.handle_utterance(wave))
            n_done += 1


if __name__ == "__main__":
    import glob

    from src.actuators.client import from_config as client_from_config
    from src.audio.mic import MicStream

    ap = argparse.ArgumentParser()
    ap.add_argument("--wake", default="export/wakeword_int8.tflite")
    ap.add_argument("--seg_dir", default="export")
    ap.add_argument("--labels", default="data/prepared/fsc_labels.json")
    ap.add_argument("--enc_cfg", default="configs/train_encoder.yaml")
    ap.add_argument("--device_cfg", default="configs/device.yaml")
    args = ap.parse_args()
    dev = yaml.safe_load(open(args.device_cfg))
    enc_cfg = yaml.safe_load(open(args.enc_cfg))
    seg_paths = sorted(glob.glob(f"{args.seg_dir}/seg*_int8.tflite"))
    pipe = Pipeline(Segment(args.wake, n_threads=dev["n_threads"]), SegmentedEncoder(seg_paths, dev["n_threads"]),
                    load_label_maps(args.labels), client_from_config(dev), enc_cfg, enc_cfg["audio"])
    with MicStream(dev["mic"]) as mic:
        print("listening...")
        pipe.run_live(mic)
