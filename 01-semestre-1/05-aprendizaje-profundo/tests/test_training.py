import numpy as np
import pytest

torch = pytest.importorskip("torch")

from applied_dl.training import count_parameters, evaluate, fit  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from common import set_seed  # noqa: E402


def loaders(n=600, noise=0.0, seed=0):
    g = np.random.default_rng(seed)
    X = g.normal(size=(n, 10)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    flip = g.random(n) < noise
    y[flip] = 1 - y[flip]
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    tr, va = torch.utils.data.random_split(
        ds, [n // 2, n - n // 2], generator=torch.Generator().manual_seed(0)
    )
    return DataLoader(tr, batch_size=32, shuffle=True), DataLoader(va, batch_size=256)


def test_fit_learns_and_reports():
    set_seed(0)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
    tr, va = loaders()
    hist = fit(model, tr, va, torch.optim.Adam(model.parameters(), lr=1e-2), epochs=15)
    _, acc = evaluate(model, va, torch.device("cpu"))
    assert acc > 0.9
    assert hist.train_loss[-1] < hist.train_loss[0]
    assert count_parameters(model) == 10 * 16 + 16 + 16 * 2 + 2


def test_early_stopping_restores_best_weights():
    set_seed(0)
    model = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 2))
    tr, va = loaders(n=120, noise=0.3)  # pocos datos y ruidosos: sobreajusta pronto
    hist = fit(model, tr, va, torch.optim.Adam(model.parameters(), lr=1e-2), epochs=60, patience=5)
    assert len(hist.val_loss) < 60  # se detuvo antes
    final_val, _ = evaluate(model, va, torch.device("cpu"))
    assert final_val == pytest.approx(min(hist.val_loss), rel=1e-5)


def test_train_transform_applies_only_to_training_batches():
    set_seed(0)
    seen = {"calls": 0}

    def transform(xb):
        seen["calls"] += 1
        return xb

    model = nn.Sequential(nn.Linear(10, 2))
    tr, va = loaders(n=200)
    fit(
        model,
        tr,
        va,
        torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=2,
        train_transform=transform,
    )
    assert seen["calls"] == 2 * len(tr)
