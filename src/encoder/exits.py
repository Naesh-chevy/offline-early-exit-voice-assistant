"""Exit heads for the early exit encoder (written by us, no early exit library).

Each exit looks at the hidden state after one encoder layer and outputs
  intent logits (action, object, location), slot logits (thermostat setpoint class), and
  decision logits over {ACT, CONTINUE, REJECT}.
The last exit has no CONTINUE option (nothing left to run), we kill its logit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT, CONTINUE, REJECT = 0, 1, 2
NO_OPTION = -1e4   # logit for an option that is not allowed, softmax makes it ~0


class ExitHead(nn.Module):
    def __init__(self, d: int, hidden: int, n_action: int, n_object: int, n_location: int,
                 slot_bins: int, can_continue: bool = True):
        super().__init__()
        self.can_continue = can_continue
        self.norm = nn.LayerNorm(d)
        self.shared = nn.Linear(d, hidden)
        self.action = nn.Linear(hidden, n_action)
        self.obj = nn.Linear(hidden, n_object)
        self.location = nn.Linear(hidden, n_location)
        self.slot = nn.Linear(hidden, slot_bins)
        self.decision = nn.Linear(hidden, 3)

    def forward(self, h: torch.Tensor) -> dict:
        """h is (B, T, d) hidden state. Mean pool over time, then small shared layer, then the heads."""
        z = F.gelu(self.shared(self.norm(h.mean(dim=1))))
        dec = self.decision(z)
        if not self.can_continue:
            dec = torch.cat([dec[:, :CONTINUE], torch.full_like(dec[:, :1], NO_OPTION), dec[:, CONTINUE + 1:]], dim=1)
        return {"action": self.action(z), "obj": self.obj(z), "location": self.location(z),
                "slot": self.slot(z), "decision": dec}


class ExitModel(nn.Module):
    """Whisper style encoder + one ExitHead after each layer in `exits` (1-indexed layers)."""

    def __init__(self, enc, exits: list, hidden: int, n_action: int, n_object: int, n_location: int, slot_bins: int):
        super().__init__()
        self.enc = enc
        self.exits = list(exits)
        d = enc.layer_norm.normalized_shape[0]
        self.heads = nn.ModuleList([
            ExitHead(d, hidden, n_action, n_object, n_location, slot_bins, can_continue=(i < len(exits) - 1))
            for i in range(len(exits))])

    def forward(self, mel: torch.Tensor) -> list:
        """Runs everything and returns one output dict per exit. Used for training and eval, not on device."""
        x = self.enc.frontend(mel)
        outs, done = [], 0
        for head, layer in zip(self.heads, self.exits):
            x = self.enc.run_layers(x, done + 1, layer)
            done = layer
            outs.append(head(x))
        return outs


def from_config(cfg: dict, enc) -> ExitModel:
    h = cfg["heads"]
    return ExitModel(enc, h["exits"], h["head_hidden"], h["n_action"], h["n_object"], h["n_location"], h["slot_bins"])
