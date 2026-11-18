"""TF mirror of the torch encoder pieces, used only to build TFLite files.

Weights come straight from the torch state dict (numpy), so torch and tflite share one set of
numbers. Everything is plain tf ops with fixed batch 1 so the converter has an easy time.
Layout here is channels last: mel comes in as (1, frames, n_mels).
"""
import numpy as np
import tensorflow as tf


def np_params(module) -> dict:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in module.state_dict().items()}


def linear(x, w, b=None):
    """x (1, T, in) or (1, in); torch weight is (out, in). 2D matmul so tflite makes a FullyConnected."""
    shape = x.shape
    flat = tf.reshape(x, [-1, shape[-1]])
    y = tf.matmul(flat, w.T)
    if b is not None:
        y = y + b
    return tf.reshape(y, list(shape[:-1]) + [w.shape[0]])


def layer_norm(x, w, b, eps=1e-5):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * tf.math.rsqrt(var + eps) * w + b


def gelu(x):
    return tf.nn.gelu(x, approximate=False)


def conv1d(x, w, b, stride=1):
    """torch conv1d with padding=1, k=3. Explicit pad because 'same' pads uneven when stride is 2."""
    x = tf.pad(x, [[0, 0], [1, 1], [0, 0]])
    k = np.transpose(w, (2, 1, 0))  # torch (out, in, k) -> tf (k, in, out)
    return tf.nn.conv1d(x, k, stride, "VALID") + b


def frontend(p, mel):
    x = gelu(conv1d(mel, p["conv1.weight"], p["conv1.bias"]))
    x = gelu(conv1d(x, p["conv2.weight"], p["conv2.bias"], stride=2))
    return x + p["embed_positions.weight"][: x.shape[1]]


def block(p, i, x, n_heads):
    pre = f"layers.{i}."
    t, d = x.shape[1], x.shape[2]
    dh = d // n_heads
    y = layer_norm(x, p[pre + "self_attn_layer_norm.weight"], p[pre + "self_attn_layer_norm.bias"])
    a = pre + "self_attn."

    def heads(z):  # (1, T, d) -> (1, h, T, dh)
        return tf.transpose(tf.reshape(z, [1, t, n_heads, dh]), [0, 2, 1, 3])

    q = heads(linear(y, p[a + "q_proj.weight"], p[a + "q_proj.bias"]))
    k = heads(linear(y, p[a + "k_proj.weight"]))
    v = heads(linear(y, p[a + "v_proj.weight"], p[a + "v_proj.bias"]))
    w = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * (1.0 / np.sqrt(dh)), axis=-1)
    o = tf.reshape(tf.transpose(tf.matmul(w, v), [0, 2, 1, 3]), [1, t, d])
    x = x + linear(o, p[a + "out_proj.weight"], p[a + "out_proj.bias"])
    y = layer_norm(x, p[pre + "final_layer_norm.weight"], p[pre + "final_layer_norm.bias"])
    return x + linear(gelu(linear(y, p[pre + "fc1.weight"], p[pre + "fc1.bias"])),
                      p[pre + "fc2.weight"], p[pre + "fc2.bias"])


def run_layers(p, x, first, last, n_heads):
    for i in range(first - 1, last):
        x = block(p, i, x, n_heads)
    return x


def to_tflite(fn, specs, calib=None) -> bytes:
    """Convert a tf function. calib is a generator function of input lists -> full int8 (io too)."""
    cf = tf.function(fn, input_signature=specs).get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    if calib is not None:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = calib
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()
