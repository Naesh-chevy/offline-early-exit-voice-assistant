"""Shared bits for the measurement scripts: git hash, device info, csv writer."""
import csv
import datetime
import os
import platform
import subprocess
from pathlib import Path


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "no-git"
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def _read(path: str):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def cpu_governor() -> str:
    return _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "n/a"


def cpu_temp_c() -> float | None:
    raw = _read("/sys/class/thermal/thermal_zone0/temp")
    return int(raw) / 1000 if raw else None


def device_info(target: str = "unknown") -> dict:
    t = cpu_temp_c()
    return {"target": target, "machine": platform.machine(), "os": platform.platform(),
            "governor": cpu_governor(), "temp_c": "n/a" if t is None else round(t, 1)}


def pin_threads(n: int) -> None:
    """Pin the process to the first n cores where the os allows it (linux only)."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(range(n)))


def new_run_dir(run_name: str, root: str = "results") -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(root) / f"{stamp}_{run_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(path, rows: list, target: str = "unknown") -> None:
    """Header row, then one line per dict. git hash and device info get added to every row."""
    extra = {"git_hash": git_hash(), **device_info(target)}
    rows = [{**r, **extra} for r in rows]
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
