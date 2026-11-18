"""Live mic capture: ring buffer of the last ~2 s plus a simple energy VAD."""
import threading

import numpy as np


class RingBuffer:
    """Keeps the newest `size` samples. Old audio is overwritten."""

    def __init__(self, size: int):
        self.buf = np.zeros(size, dtype=np.float32)
        self.size = size
        self.pos = 0       # next write index
        self.filled = 0    # how many valid samples we have, up to size
        self.lock = threading.Lock()

    def write(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames, dtype=np.float32).ravel()
        if len(frames) >= self.size:  # chunk bigger than the buffer, only the tail matters
            frames = frames[-self.size:]
        n = len(frames)
        with self.lock:
            end = self.pos + n
            if end <= self.size:
                self.buf[self.pos:end] = frames
            else:
                k = self.size - self.pos
                self.buf[self.pos:] = frames[:k]
                self.buf[:n - k] = frames[k:]
            self.pos = end % self.size
            self.filled = min(self.size, self.filled + n)

    def read_last(self, n: int | None = None) -> np.ndarray:
        """Newest n samples in time order (oldest first). Fewer if we have not recorded n yet."""
        with self.lock:
            n = self.filled if n is None else min(n, self.filled)
            idx = (self.pos - n + np.arange(n)) % self.size
            return self.buf[idx].copy()


class EnergyVAD:
    """Speech if frame rms is over the threshold, stays on for a few frames after (hangover)."""

    def __init__(self, thresh: float = 0.01, hangover: int = 10):
        self.thresh = thresh
        self.hangover = hangover
        self.left = 0

    def update(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
        if rms > self.thresh:
            self.left = self.hangover
            return True
        if self.left > 0:
            self.left -= 1
            return True
        return False


class MicStream:
    """sounddevice input stream that feeds the ring buffer. Use as a context manager."""

    def __init__(self, cfg: dict):
        self.sr = cfg["sample_rate"]
        self.frame = int(self.sr * cfg["frame_ms"] / 1000)
        self.device = cfg["device_index"]
        self.ring = RingBuffer(int(self.sr * cfg["ring_seconds"]))
        self.vad = EnergyVAD(cfg["vad_energy_thresh"], cfg["vad_hangover_frames"])
        self.speech = False
        self.sink = None      # set to a list to also collect every frame (used to record a command after the wake word)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0]
        self.ring.write(mono)
        self.speech = self.vad.update(mono)
        if self.sink is not None:
            self.sink.append(mono.copy())

    def __enter__(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                     blocksize=self.frame, device=self.device, callback=self._callback)
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()
