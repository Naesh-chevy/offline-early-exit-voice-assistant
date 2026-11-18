"""Whisper style encoder (tiny = student, base = teacher) written by hand, no decoder.

Same param names as the HF WhisperEncoder so we can load pretrained weights. Layers are
1-indexed in our API (layer 1 = layers[0]) to match the exit numbers in the configs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoids(n_pos: int, d: int) -> torch.Tensor:
    """Fixed sinusoidal pos emb, same formula as whisper."""
    log_inc = math.log(10000) / (d // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(d // 2))
    t = torch.arange(n_pos)[:, None] * inv[None, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn: int):
        super().__init__()
        self.n_heads = n_heads
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(d, d), "k_proj": nn.Linear(d, d, bias=False),
            "v_proj": nn.Linear(d, d), "out_proj": nn.Linear(d, d)})
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_heads
        y = self.self_attn_layer_norm(x)
        a = self.self_attn
        q = a["q_proj"](y).view(b, t, h, d // h).transpose(1, 2)
        k = a["k_proj"](y).view(b, t, h, d // h).transpose(1, 2)
        v = a["v_proj"](y).view(b, t, h, d // h).transpose(1, 2)
        w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // h), dim=-1)
        x = x + a["out_proj"]((w @ v).transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class WhisperEncoder(nn.Module):
    def __init__(self, d_model=384, n_layers=4, n_heads=6, ffn=1536, n_mels=80, n_frames=400):
        super().__init__()
        self.n_layers = n_layers
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)
        # whisper expects 30s (1500 positions), we crop to the window so pos emb has to be cut too
        self.n_pos = n_frames // 2
        self.embed_positions = nn.Embedding(self.n_pos, d_model)
        self.embed_positions.weight.data = sinusoids(self.n_pos, d_model)
        self.embed_positions.weight.requires_grad_(False)
        self.layers = nn.ModuleList([Block(d_model, n_heads, ffn) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def frontend(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B, n_mels, n_frames) -> (B, n_frames // 2, d)."""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        return x + self.embed_positions.weight[: x.shape[1]]

    def run_layers(self, x: torch.Tensor, first: int, last: int) -> torch.Tensor:
        """Run layers first..last (1-indexed, inclusive)."""
        for i in range(first - 1, last):
            x = self.layers[i](x)
        return x

    def forward(self, mel: torch.Tensor, return_hidden: bool = False):
        """Final output after the last layer norm. With return_hidden also the per layer states."""
        x = self.frontend(mel)
        hidden = []
        for blk in self.layers:
            x = blk(x)
            hidden.append(x)
        out = self.layer_norm(x)
        return (out, hidden) if return_hidden else out


def from_config(cfg: dict, which: str = "student", n_frames: int | None = None) -> WhisperEncoder:
    """Build encoder from train_encoder.yaml (random init). which is 'student' or 'teacher'."""
    m = cfg[which]
    if n_frames is None:
        n_frames = int(cfg["audio"]["window_s"] * cfg["audio"]["sample_rate"] / cfg["audio"]["hop"])
    return WhisperEncoder(m["d_model"], m["n_layers"], m["n_heads"], m["ffn"], cfg["audio"]["n_mels"], n_frames)


def load_pretrained(enc: WhisperEncoder, name: str) -> WhisperEncoder:
    """Copy weights from a huggingface whisper (needs internet the first time). Pos emb is cropped."""
    from transformers import WhisperModel
    sd = WhisperModel.from_pretrained(name).encoder.state_dict()
    # use hf's own pos emb (slightly rounded vs our formula), just the first n_pos rows
    enc.embed_positions.weight.data = sd.pop("embed_positions.weight")[: enc.n_pos].clone()
    # our attn projections live in a ModuleDict but the state dict names still match HF
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all("embed_positions" in k for k in missing), missing
    return enc
