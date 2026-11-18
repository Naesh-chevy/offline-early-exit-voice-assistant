"""Log mel features, whisper compatible (80 bins, 16k mono). Plain numpy so it runs anywhere."""
import numpy as np

SR = 16000


def load_wav(path: str, sr: int = SR) -> np.ndarray:
    """Read audio as mono float32 at sr. Resampling is plain linear interp, good enough for 16k sources."""
    import soundfile as sf
    wave, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if file_sr != sr:
        n_new = int(round(len(wave) * sr / file_sr))
        wave = np.interp(np.linspace(0, len(wave) - 1, n_new), np.arange(len(wave)), wave).astype(np.float32)
    return wave


def pad_or_crop(wave: np.ndarray, n_samples: int) -> np.ndarray:
    """Zero pad at the end or crop the end so the clip is exactly n_samples long."""
    if len(wave) >= n_samples:
        return wave[:n_samples].astype(np.float32)
    out = np.zeros(n_samples, dtype=np.float32)
    out[:len(wave)] = wave
    return out


def _hz_to_mel(f):
    # slaney scale, same as whisper's librosa filters: linear under 1k, log above
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    logstep = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / logstep, mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    hz = m * (200.0 / 3)
    logstep = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(logstep * (m - 15.0)), hz)


def mel_filters(n_mels: int = 80, n_fft: int = 400, sr: int = SR) -> np.ndarray:
    """Triangular slaney filterbank, shape (n_mels, n_fft // 2 + 1)."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / (mid - lo)
        down = (hi - fft_freqs) / (hi - mid)
        fb[i] = np.maximum(0, np.minimum(up, down))
        fb[i] *= 2.0 / (hi - lo)  # slaney area norm
    return fb.astype(np.float32)


_FB_CACHE: dict = {}


def log_mel(wave: np.ndarray, n_mels: int = 80, n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """Whisper style log mel, returns (n_mels, len(wave) // hop) float32."""
    key = (n_mels, n_fft)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = mel_filters(n_mels, n_fft)
    pad = n_fft // 2
    x = np.pad(wave.astype(np.float32), (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft + 1)[:-1].astype(np.float32)  # periodic hann like torch.hann_window
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (frames, bins)
    power = power[:-1]  # whisper drops the last frame
    mel = _FB_CACHE[key] @ power.T
    logm = np.log10(np.maximum(mel, 1e-10))
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).astype(np.float32)


def clip_features(wave: np.ndarray, window_s: float, n_mels: int = 80, n_fft: int = 400,
                  hop: int = 160, sr: int = SR) -> np.ndarray:
    """Pad or crop to window_s seconds, then log mel. 4s -> (80, 400), 1s -> (80, 100)."""
    return log_mel(pad_or_crop(wave, int(window_s * sr)), n_mels, n_fft, hop)
