import csv
import time

import pytest
import torch
import yaml

from src.baselines.naive import NaiveModel, export_naive_fp32
from src.encoder.student import WhisperEncoder
from src.measure.bench import bench_model
from src.measure.common import write_csv
from src.measure.energy import NOT_MEASURED, NoEnergy, get_energy_logger
from src.measure.latency import measure_latency
from src.measure.memory import measure_peak_rss


def test_latency_on_sleep():
    lat = measure_latency(lambda: time.sleep(0.01), warmup=2, runs=10)
    assert 9 < lat["p50_ms"] < 30
    assert lat["p50_ms"] <= lat["p90_ms"] <= lat["p99_ms"]
    assert lat["mean_ms"] > 9


def test_peak_rss_sees_an_allocation():
    def alloc():
        x = bytearray(80 * 2**20)
        x[::4096] = b"1" * len(x[::4096])  # touch pages so they really count
        time.sleep(0.03)
    mem = measure_peak_rss(alloc, sample_ms=2)
    assert mem["peak_rss_mb"] > mem["rss_before_mb"] + 40


def test_csv_has_header_and_git_hash(tmp_path):
    p = tmp_path / "x.csv"
    write_csv(p, [{"a": 1, "b": 2}], target="pi4")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["a"] == "1"
    for col in ["git_hash", "target", "machine", "governor", "temp_c"]:
        assert col in rows[0]


def test_energy_stub_says_not_measured():
    cfg = yaml.safe_load(open("configs/device.yaml"))
    logger = get_energy_logger(cfg)  # ina226_enabled is false in the config
    assert isinstance(logger, NoEnergy)
    logger.start()
    res = logger.stop()
    assert res["energy_mj"] == NOT_MEASURED and res["avg_power_mw"] == NOT_MEASURED


def tiny_naive_file(tmp_path):
    torch.manual_seed(0)
    enc = WhisperEncoder(32, 4, 2, 64, 80, 40)
    path = str(tmp_path / "tiny.tflite")
    export_naive_fp32(NaiveModel(enc, 3, 4, 2).eval(), path)
    return path


def test_bench_model_records_numbers(tmp_path):
    dev = yaml.safe_load(open("configs/device.yaml"))
    ev = yaml.safe_load(open("configs/eval.yaml"))
    ev["latency"] = {"warmup": 1, "runs": 5}
    row = bench_model(tiny_naive_file(tmp_path), dev, ev, "tiny")
    assert row["mean_ms"] > 0 and row["peak_rss_mb"] > 0
    assert row["energy_mj"] == NOT_MEASURED


@pytest.mark.device
def test_naive_baseline_on_device():
    """Needs export/naive_fp32.tflite from a trained model, run on the pi."""
    dev = yaml.safe_load(open("configs/device.yaml"))
    ev = yaml.safe_load(open("configs/eval.yaml"))
    row = bench_model("export/naive_fp32.tflite", dev, ev, "naive_fp32")
    assert row["mean_ms"] > 0 and row["peak_rss_mb"] > 0
