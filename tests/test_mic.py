import time

import numpy as np
import pytest
import yaml

from src.audio.mic import EnergyVAD, MicStream, RingBuffer


def test_ring_keeps_newest_in_order():
    ring = RingBuffer(10)
    ring.write(np.arange(7))
    assert ring.read_last().tolist() == list(range(7))
    ring.write(np.arange(7, 14))  # wraps, 0..3 get overwritten
    assert ring.read_last().tolist() == list(range(4, 14))
    assert ring.read_last(3).tolist() == [11, 12, 13]


def test_ring_chunk_bigger_than_buffer():
    ring = RingBuffer(5)
    ring.write(np.arange(12))
    assert ring.read_last().tolist() == [7, 8, 9, 10, 11]


def test_ring_many_small_frames():
    ring = RingBuffer(100)
    for i in range(50):
        ring.write(np.full(7, i))
    got = ring.read_last()
    assert len(got) == 100
    assert got[-1] == 49  # newest frame is last


def test_vad_energy_and_hangover():
    vad = EnergyVAD(thresh=0.1, hangover=2)
    quiet = np.zeros(480, dtype=np.float32)
    loud = np.full(480, 0.5, dtype=np.float32)
    assert vad.update(quiet) is False
    assert vad.update(loud) is True
    assert vad.update(quiet) is True   # hangover 1
    assert vad.update(quiet) is True   # hangover 2
    assert vad.update(quiet) is False


@pytest.mark.device
def test_mic_records_one_second():
    with open("configs/device.yaml") as f:
        cfg = yaml.safe_load(f)["mic"]
    with MicStream(cfg) as mic:
        time.sleep(1.2)
        audio = mic.ring.read_last(cfg["sample_rate"])
    rms = float(np.sqrt(np.mean(audio ** 2)))
    assert len(audio) == cfg["sample_rate"]
    assert rms > 1e-5  # noise floor, a dead/muted mic gives exact zeros
