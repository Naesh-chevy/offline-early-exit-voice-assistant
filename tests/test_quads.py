import torch
from torch.utils.data import DataLoader

from src.baselines.naive import evaluate
from src.baselines.quads import KMeansQuantizer, kmeans_1d, quant_phase, recenter, run_mct
from src.encoder.data import FakeSLUData
from tests.test_baseline import N_FRAMES, N_MELS, tiny_model


def n_unique(p):
    return len(torch.unique(p.detach()))


def test_kmeans_error_goes_down_with_more_centroids():
    w = torch.randn(5000)
    err = []
    for k in [2, 4, 16, 64]:
        cent, idx = kmeans_1d(w, k)
        err.append(((cent[idx] - w) ** 2).mean().item())
    assert err == sorted(err, reverse=True) and err[-1] < err[0] / 10


def test_recenter_is_cluster_mean_and_keeps_empty_clusters():
    w = torch.tensor([1.0, 3.0, 10.0])
    idx = torch.tensor([0, 0, 2])
    cent = recenter(w, idx, torch.tensor([0.0, 5.0, 0.0]))
    assert cent.tolist() == [2.0, 5.0, 10.0]      # cluster 1 is empty so it keeps 5


def test_quantizer_limits_unique_values_and_only_touches_matrices():
    model = tiny_model()
    biases = {n: p.detach().clone() for n, p in model.named_parameters() if p.dim() == 1}
    q = KMeansQuantizer(model, bits=3)
    for n, p in model.named_parameters():
        if p.dim() > 1 and p.requires_grad:
            assert n_unique(p) <= 8, n
        elif p.dim() == 1:
            assert torch.equal(p, biases[n]), n   # biases and layer norms stay fp32
    assert q.size_mb(model) < sum(p.numel() for p in model.parameters()) * 4 / 1e6


def test_fewer_bits_means_smaller_model():
    m8, m4 = tiny_model(), tiny_model()
    assert KMeansQuantizer(m4, 4).size_mb(m4) < KMeansQuantizer(m8, 8).size_mb(m8)


def test_weights_stay_on_codebook_after_training_steps():
    model = tiny_model()
    q = KMeansQuantizer(model, bits=3)
    data = FakeSLUData(n=64, n_mels=N_MELS, n_frames=N_FRAMES)
    quant_phase(model, q, DataLoader(data, 16, shuffle=True), 0.01, "cpu")
    for n, p in model.named_parameters():
        if n in q.books:
            assert n_unique(p) <= 8, n
            assert torch.isfinite(p).all()


def test_mct_runs_the_schedule_and_learns_something():
    torch.manual_seed(0)
    model, teacher = tiny_model(1), tiny_model(2)
    proj = torch.nn.Linear(32, 32)
    data = FakeSLUData(n=640, n_mels=N_MELS, n_frames=N_FRAMES)
    dl = DataLoader(data, 32, shuffle=True)
    cfg = {"schedule": "dddqdqq", "l1_weight": 0.1, "enc_lr": 3e-3, "head_lr": 3e-3, "quant_lr": 0.01}
    q, log = run_mct(model, teacher, proj, dl, DataLoader(data, 64), 4, cfg, "cpu", evaluate)
    assert [r["kind"] for r in log] == list("dddqdqq")
    # chance for the 24 class intent is 0.04. accuracy dips after a distill epoch (re-snap) and the quant epochs bring it back
    assert log[-1]["val_intent"] > 0.25
    for n, p in model.named_parameters():
        if n in q.books:
            assert n_unique(p) <= 16, n
