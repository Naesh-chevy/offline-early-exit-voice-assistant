"""Latency harness: warmup, then N timed runs."""
import time

import numpy as np


def measure_latency(fn, warmup: int = 10, runs: int = 100) -> dict:
    """Time fn() with perf_counter, return mean/p50/p90/p99 in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"runs": runs, "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p90_ms": float(np.percentile(t, 90)), "p99_ms": float(np.percentile(t, 99))}
