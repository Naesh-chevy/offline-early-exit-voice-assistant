"""Thin wrapper over the tflite interpreter: float in, float out, handles int8 scales."""
import numpy as np


def make_interpreter(path: str | None = None, content: bytes | None = None, n_threads: int = 1):
    # tf 2.21 dropped tf.lite.Interpreter, litert is also what the pi runs
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=path, model_content=content, num_threads=n_threads)
    interp.allocate_tensors()
    return interp


class Segment:
    """One tflite model. Counts how many times it ran (the early exit test uses this)."""

    def __init__(self, path: str | None = None, content: bytes | None = None, n_threads: int = 1):
        self.interp = make_interpreter(path, content, n_threads)
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()
        self.n_calls = 0

    def run(self, *xs: np.ndarray) -> list:
        for d, x in zip(self.inp, xs):
            x = np.asarray(x, dtype=np.float32)
            if d["dtype"] != np.float32:  # quantize with the scale baked into the model
                scale, zp = d["quantization"]
                info = np.iinfo(d["dtype"])
                x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(d["dtype"])
            self.interp.set_tensor(d["index"], x)
        self.interp.invoke()
        self.n_calls += 1
        outs = []
        for d in self.out:
            y = self.interp.get_tensor(d["index"])
            if d["dtype"] != np.float32:
                scale, zp = d["quantization"]
                y = (y.astype(np.float32) - zp) * scale
            outs.append(y)
        return outs
