import torch
import yaml
from torch.utils.data import DataLoader

from src.baselines.naive import NaiveModel, make_opt, train_steps
from src.encoder.ckpt import load_resume, save_resume
from src.encoder.data import FakeSLUData
from src.encoder.student import WhisperEncoder


def tiny(seed):
    torch.manual_seed(seed)
    return NaiveModel(WhisperEncoder(32, 2, 2, 64, 80, 40), 3, 4, 2)


def test_resume_restores_weights_optimizer_and_epoch(tmp_path):
    path = str(tmp_path / "r.pt")
    model = tiny(0)
    opt = make_opt(model, 1e-3)
    loader = DataLoader(FakeSLUData(64, 80, 40, 3, 4, 2), batch_size=16)
    train_steps(model, loader, 4, opt)          # gives the optimizer some state
    save_resume(path, 3, opt, model=model)

    other = tiny(1)                              # different init on purpose
    other_opt = make_opt(other, 1e-3)
    assert load_resume(path, other_opt, model=other) == 3
    for a, b in zip(model.parameters(), other.parameters()):
        assert torch.equal(a, b)
    assert len(other_opt.state) == len(opt.state) > 0


def test_resumed_training_matches_uninterrupted(tmp_path):
    """3 steps, save, load into a new model, 3 more steps == 6 steps in a row (same batches)."""
    data = FakeSLUData(96, 80, 40, 3, 4, 2)
    batches = [DataLoader(data, batch_size=16, shuffle=False)] * 1

    def run(model, opt, n):
        train_steps(model, batches[0], n, opt)

    a = tiny(0)
    oa = make_opt(a, 1e-3)
    run(a, oa, 3)
    save_resume(str(tmp_path / "r.pt"), 1, oa, model=a)
    b = tiny(5)
    ob = make_opt(b, 1e-3)
    load_resume(str(tmp_path / "r.pt"), ob, model=b)
    run(a, oa, 3)
    run(b, ob, 3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)
