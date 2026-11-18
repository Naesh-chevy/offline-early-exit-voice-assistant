"""Benchmark one tflite file: latency, peak rss, energy. Writes results/<stamp>_<run_name>/bench.csv

usage: python -m src.measure.bench --model export/naive_fp32.tflite --run_name naive_fp32
"""
import argparse

import numpy as np
import yaml

from src.measure.common import new_run_dir, pin_threads, write_csv
from src.measure.energy import GpioMarker, get_energy_logger
from src.measure.latency import measure_latency
from src.measure.memory import measure_peak_rss
from src.runtime.tflite_runner import Segment


def bench_model(model_path: str, dev_cfg: dict, eval_cfg: dict, label: str) -> dict:
    n = dev_cfg["n_threads"]
    pin_threads(n)
    rss_before = measure_peak_rss(lambda: None)["rss_before_mb"]
    seg = Segment(model_path, n_threads=n)
    rng = np.random.default_rng(eval_cfg["seed"])
    xs = [rng.standard_normal(d["shape"]).astype(np.float32) for d in seg.inp]
    marker = GpioMarker(dev_cfg["actuators"]["gpio"]["marker_pin"] if dev_cfg["target"] != "android" else None)
    energy = get_energy_logger(dev_cfg)

    def run():
        with marker:
            seg.run(*xs)

    mem = measure_peak_rss(run, repeat=eval_cfg["latency"]["runs"], sample_ms=eval_cfg["memory"]["sample_ms"])
    energy.start()
    lat = measure_latency(run, eval_cfg["latency"]["warmup"], eval_cfg["latency"]["runs"])
    e = energy.stop()
    return {"model": label, "rss_idle_mb": rss_before, **lat,
            "peak_rss_mb": mem["peak_rss_mb"], "rss_delta_mb": mem["delta_mb"], **e}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--device_cfg", default="configs/device.yaml")
    ap.add_argument("--eval_cfg", default="configs/eval.yaml")
    args = ap.parse_args()
    dev = yaml.safe_load(open(args.device_cfg))
    ev = yaml.safe_load(open(args.eval_cfg))
    row = bench_model(args.model, dev, ev, args.run_name)
    out = new_run_dir(args.run_name, ev["results_dir"]) / "bench.csv"
    write_csv(out, [row], dev["target"])
    print(row)
    print("wrote", out)
