"""Folded DS-CNN -> TF graph -> full integer int8 tflite (int8 in and out)."""
import numpy as np
import tensorflow as tf

from src.export.tf_encoder import to_tflite
from src.wakeword.model import DSCNN


def _conv(x, m, depthwise=False):
    w = m.conv.weight.detach().numpy()   # torch (out, in/groups, kh, kw)
    b = m.conv.bias.detach().numpy()
    ph, pw = m.conv.padding
    sh, sw = m.conv.stride
    x = tf.pad(x, [[0, 0], [ph, ph], [pw, pw], [0, 0]])
    if depthwise:
        y = tf.nn.depthwise_conv2d(x, np.transpose(w, (2, 3, 0, 1)), [1, sh, sw, 1], "VALID")  # (kh,kw,ch,1)
    else:
        y = tf.nn.conv2d(x, np.transpose(w, (2, 3, 1, 0)), [1, sh, sw, 1], "VALID")
    return tf.nn.relu(y + b)


def export_wakeword_int8(model: DSCNN, calib_inputs: list, path: str, n_mels: int = 80, n_frames: int = 100) -> int:
    """calib_inputs: list of (n_mels, frames) float arrays. Returns file size in bytes."""
    model = model.eval()
    fc_w = model.fc.weight.detach().numpy()
    fc_b = model.fc.bias.detach().numpy()

    def fn(x):
        x = _conv(x, model.first)
        for m in model.blocks:
            x = _conv(x, m, depthwise=m.conv.groups > 1)
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.matmul(x, fc_w.T) + fc_b

    def calib():
        for c in calib_inputs:
            yield [c.reshape(1, n_mels, n_frames, 1).astype(np.float32)]

    data = to_tflite(fn, [tf.TensorSpec([1, n_mels, n_frames, 1], tf.float32)], calib)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
