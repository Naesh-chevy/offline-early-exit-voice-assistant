"""Peak RSS while a function runs. A thread polls rss because the peak can come and go fast."""
import threading
import time

import psutil


def measure_peak_rss(fn, repeat: int = 1, sample_ms: float = 5.0) -> dict:
    proc = psutil.Process()
    before = proc.memory_info().rss
    peak = before
    stop = threading.Event()

    def poll():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(sample_ms / 1000)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    for _ in range(repeat):
        fn()
    stop.set()
    th.join()
    peak = max(peak, proc.memory_info().rss)
    return {"rss_before_mb": before / 2**20, "peak_rss_mb": peak / 2**20, "delta_mb": (peak - before) / 2**20}
