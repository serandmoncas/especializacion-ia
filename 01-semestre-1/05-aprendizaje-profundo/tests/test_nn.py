import numpy as np
import pytest
from applied_dl import Tensor, cross_entropy
from applied_dl.nn import MLP, SGD, Adam, Dropout, init_weights

torch = pytest.importorskip("torch")


def _torch_mlp_like(ours: MLP, activation):
    layers = []
    for i, lin in enumerate(ours.layers):
        t = torch.nn.Linear(*lin.W.shape).double()
        with torch.no_grad():
            t.weight.copy_(torch.from_numpy(lin.W.data.T))
            t.bias.copy_(torch.from_numpy(lin.b.data))
        layers.append(t)
        if i < len(ours.layers) - 1:
            layers.append(activation())
    return torch.nn.Sequential(*layers)


@pytest.mark.parametrize(("act", "torch_act"), [("relu", "ReLU"), ("tanh", "Tanh")])
def test_mlp_forward_and_gradients_match_torch(act, torch_act):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(16, 5)), rng.integers(0, 3, 16)
    ours = MLP([5, 8, 6, 3], activation=act, rng=rng)
    ref = _torch_mlp_like(ours, getattr(torch.nn, torch_act))
    loss = cross_entropy(ours(Tensor(X)), y)
    loss.backward()
    ref_loss = torch.nn.functional.cross_entropy(ref(torch.from_numpy(X)), torch.from_numpy(y))
    ref_loss.backward()
    assert float(loss.data) == pytest.approx(ref_loss.item(), rel=1e-12)
    torch_linears = [m for m in ref if isinstance(m, torch.nn.Linear)]
    for lin, t in zip(ours.layers, torch_linears, strict=True):
        np.testing.assert_allclose(lin.W.grad, t.weight.grad.numpy().T, atol=1e-12)
        np.testing.assert_allclose(lin.b.grad, t.bias.grad.numpy(), atol=1e-12)


@pytest.mark.parametrize(
    ("ours_cls", "torch_cls", "kwargs"),
    [
        (SGD, "SGD", {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.01}),
        (Adam, "Adam", {"lr": 0.01, "weight_decay": 0.01}),
        (Adam, "AdamW", {"lr": 0.01, "weight_decay": 0.1, "decoupled": True}),
    ],
)
def test_optimizer_trajectory_matches_torch(ours_cls, torch_cls, kwargs):
    rng = np.random.default_rng(1)
    w0 = rng.normal(size=(4,))
    target = rng.normal(size=(4,))
    ours_p = Tensor(w0.copy(), requires_grad=True)
    torch_p = torch.tensor(w0.copy(), requires_grad=True)
    torch_kwargs = {k: v for k, v in kwargs.items() if k != "decoupled"}
    opt_ours, opt_torch = (
        ours_cls([ours_p], **kwargs),
        getattr(torch.optim, torch_cls)([torch_p], **torch_kwargs),
    )
    for _ in range(25):
        opt_ours.zero_grad()
        ((ours_p - Tensor(target)) ** 2).sum().backward()
        opt_ours.step()
        opt_torch.zero_grad()
        ((torch_p - torch.from_numpy(target)) ** 2).sum().backward()
        opt_torch.step()
    np.testing.assert_allclose(ours_p.data, torch_p.detach().numpy(), atol=1e-10)


def test_mlp_learns_digits():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    rng = np.random.default_rng(0)
    model = MLP([64, 64, 10], rng=rng)
    opt = Adam(model.parameters(), lr=0.01)
    for _ in range(150):
        opt.zero_grad()
        cross_entropy(model(Tensor(X)), y).backward()
        opt.step()
    acc = np.mean(model(Tensor(X)).data.argmax(1) == y)
    assert acc > 0.97


def test_init_preserves_variance_with_relu():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 256))
    for scheme, expect_stable in [("he", True), ("naive", False)]:
        h = x
        for _ in range(10):
            h = np.maximum(h @ init_weights(256, 256, scheme, rng), 0)
        ratio = h.var() / x.var()
        assert bool(0.1 < ratio < 10) == expect_stable, (scheme, ratio)


def test_dropout_train_vs_eval():
    d = Dropout(0.5, np.random.default_rng(0))
    x = Tensor(np.ones((1000, 50)))
    out = d(x).data
    assert out.mean() == pytest.approx(1.0, abs=0.02)  # escalado invertido preserva la media
    assert np.mean(out == 0) == pytest.approx(0.5, abs=0.02)
    d.train(False)
    np.testing.assert_array_equal(d(x).data, x.data)


def test_model_train_eval_propagates():
    m = MLP([3, 4, 2], dropout=0.5)
    m.eval()
    assert all(not d.training for d in m.drops)
    m.train()
    assert all(d.training for d in m.drops)
    assert len(m.parameters()) == 4
