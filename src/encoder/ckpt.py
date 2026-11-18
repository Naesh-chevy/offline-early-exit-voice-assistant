"""Save and load resume files so a long run can continue after a colab disconnect."""
import os

import torch


def save_resume(path: str, epoch: int, opt, **modules) -> None:
    """epoch = number of epochs finished. Write to a temp file first so a crash mid save cannot break the old one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"epoch": epoch, "opt": opt.state_dict(), "modules": {k: m.state_dict() for k, m in modules.items()}}, tmp)
    os.replace(tmp, path)


def load_resume(path: str, opt, **modules) -> int:
    """Load weights and optimizer state, return how many epochs were already done."""
    ck = torch.load(path, map_location="cpu")
    for k, m in modules.items():
        m.load_state_dict(ck["modules"][k])
    opt.load_state_dict(ck["opt"])
    print(f"resumed from {path} at epoch {ck['epoch']}")
    return ck["epoch"]
