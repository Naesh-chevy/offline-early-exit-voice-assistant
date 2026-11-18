"""Runs the exported segments one at a time. A segment only runs when the policy asks for its exit."""
import numpy as np

from src.encoder.exits import CONTINUE, NO_OPTION
from src.runtime.tflite_runner import Segment

SLICES = ["action", "obj", "location", "slot", "decision"]


class SegmentedEncoder:
    def __init__(self, paths: list, n_threads: int = 1):
        self.segs = [Segment(p, n_threads=n_threads) for p in paths]
        self.next = 0
        self.hidden = None

    def start(self, mel: np.ndarray) -> None:
        """mel is (n_mels, frames). The tflite input is channels last (1, frames, n_mels)."""
        self.hidden = mel.T[None].astype(np.float32)
        self.next = 0

    def exit_fn(self, k: int) -> dict:
        """Run segment k and return that exit's logits. Exits must be asked for in order (0, 1, 2...)."""
        assert k == self.next, f"asked for exit {k} but next segment is {self.next}"
        last = k == len(self.segs) - 1
        outs = self.segs[k].run(self.hidden)
        if not last:
            self.hidden, outs = outs[0], outs[1:]
        self.next += 1
        res = {name: o[0] for name, o in zip(SLICES, outs)}
        if last:   # no CONTINUE at the last exit, same mask as ExitHead in torch
            res["decision"] = res["decision"].copy()
            res["decision"][CONTINUE] = NO_OPTION
        return res

    def calls(self) -> list:
        return [s.n_calls for s in self.segs]
