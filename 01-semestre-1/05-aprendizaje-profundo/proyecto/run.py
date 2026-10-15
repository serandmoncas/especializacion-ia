"""Mini-proyecto: CNN en PyTorch sobre Fashion-MNIST, comparada con baselines serios.

Modelos: lineal (softmax), HistGradientBoosting sobre píxeles, MLP desde cero (autograd propio),
MLP en PyTorch, CNN y CNN con aumentación de datos. Evaluación única en test con IC de Wilson y
comparaciones pareadas por McNemar.

Uso:
    uv sync --extra base --extra dl
    uv run python 01-semestre-1/05-aprendizaje-profundo/data/download.py
    uv run python 01-semestre-1/05-aprendizaje-profundo/proyecto/run.py   # ~10 min (MPS/CPU)
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from applied_dl import Tensor, cross_entropy
from applied_dl.nn import MLP, Adam
from applied_dl.training import count_parameters, fit, get_device
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

from common import data_dir, save_fig, set_seed, setup_style

MODULE = "01-semestre-1/05-aprendizaje-profundo"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42
CLASSES = [
    "camiseta",
    "pantalón",
    "suéter",
    "vestido",
    "abrigo",
    "sandalia",
    "camisa",
    "tenis",
    "bolso",
    "botín",
]


def load_data() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    root = data_dir(MODULE)
    train = FashionMNIST(root, train=True, download=False)
    test = FashionMNIST(root, train=False, download=False)
    X, y = train.data.float() / 255, train.targets
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
    tr, va = perm[:55_000], perm[55_000:]
    mean, std = X[tr].mean(), X[tr].std()  # estadísticos solo de train
    norm = lambda t: ((t - mean) / std).unsqueeze(1)  # noqa: E731 - [N, 1, 28, 28]
    return {
        "train": (norm(X[tr]), y[tr]),
        "val": (norm(X[va]), y[va]),
        "test": (norm(test.data.float() / 255), test.targets),
    }


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Aumentación de datos sobre un lote [N, 1, 28, 28] ya en el dispositivo.

    Debe devolver un lote de la misma forma con transformaciones que *no cambien la clase*.
    """
    # TODO: implementar (5-10 líneas). Ver README del proyecto, sección "Tu turno".
    raise NotImplementedError


class CNN(nn.Module):
    """Dos bloques [conv-BN-ReLU ×2 → maxpool] (28→14→7) y un clasificador denso con dropout."""

    def __init__(self, channels: tuple[int, int] = (32, 64), dropout: float = 0.3):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        c1, c2 = channels
        self.features = nn.Sequential(block(1, c1), block(c1, c2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_torch(model: nn.Module, data, device, *, epochs: int, lr: float, transform=None) -> dict:
    set_seed(SEED)
    tr = DataLoader(
        TensorDataset(*data["train"]),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    va = DataLoader(TensorDataset(*data["val"]), batch_size=1024)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = fit(
        model,
        tr,
        va,
        opt,
        epochs=epochs,
        patience=6,
        device=device,
        scheduler=sched,
        train_transform=transform,
    )
    return {"history": hist, "params": count_parameters(model)}


@torch.no_grad()
def predict_torch(model: nn.Module, X: torch.Tensor, device) -> np.ndarray:
    model.eval()
    out = [model(xb.to(device)).argmax(1).cpu() for xb in torch.split(X, 2048)]
    return torch.cat(out).numpy()


def train_scratch_mlp(data, epochs: int = 6) -> tuple[MLP, float]:
    """MLP 784-256-10 con el autograd propio (NumPy, float64), Adam y minilotes."""
    rng = np.random.default_rng(SEED)
    X = data["train"][0].reshape(len(data["train"][0]), -1).numpy().astype(np.float64)
    y = data["train"][1].numpy()
    model = MLP([784, 256, 10], rng=rng)
    opt = Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(X)), len(X) // 128):
            opt.zero_grad()
            cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
            opt.step()
    return model, time.perf_counter() - t0


def wilson(correct: int, n: int) -> tuple[float, float]:
    ci = stats.binomtest(correct, n).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


def mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict:
    """McNemar exacta: solo cuentan los ejemplos donde *un* modelo acierta y el otro no."""
    a_ok, b_ok = pred_a == y, pred_b == y
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"solo_a_acierta": only_a, "solo_b_acierta": only_b, "p_valor": float(p)}


def main() -> None:
    set_seed(SEED)
    setup_style()
    OUT.mkdir(exist_ok=True)
    device = get_device()
    print(f"Dispositivo: {device}")
    data = load_data()
    Xte, yte = data["test"]
    y_test = yte.numpy()
    flat = {k: (v[0].reshape(len(v[0]), -1).numpy(), v[1].numpy()) for k, v in data.items()}
    preds, info = {}, {}

    # 1. Baselines clásicos
    t0 = time.perf_counter()
    hgb = HistGradientBoostingClassifier(max_iter=200, early_stopping=True, random_state=SEED)
    hgb.fit(*flat["train"])
    preds["hgb_pixeles"] = hgb.predict(flat["test"][0])
    info["hgb_pixeles"] = {"seconds": time.perf_counter() - t0, "params": None}
    print(
        f"HGB: {np.mean(preds['hgb_pixeles'] == y_test):.4f}"
        f" ({info['hgb_pixeles']['seconds']:.0f}s)"
    )

    # 2. MLP desde cero (autograd propio)
    scratch, secs = train_scratch_mlp(data)
    logits = scratch(Tensor(flat["test"][0].astype(np.float64))).data
    preds["mlp_desde_cero"] = logits.argmax(1)
    info["mlp_desde_cero"] = {
        "seconds": secs,
        "params": sum(p.data.size for p in scratch.parameters()),
    }
    print(f"MLP desde cero: {np.mean(preds['mlp_desde_cero'] == y_test):.4f} ({secs:.0f}s)")

    # 3. Modelos en PyTorch
    torch_models = {
        "lineal": (lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 10)), 20, 1e-3, None),
        "mlp_torch": (
            lambda: nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 10),
            ),
            30,
            1e-3,
            None,
        ),
        "cnn": (CNN, 20, 1e-3, None),
        "cnn_aumentada": (CNN, 30, 1e-3, augment),
    }
    curves = {}
    for name, (make, epochs, lr, transform) in torch_models.items():
        if transform is not None:
            try:
                transform(torch.zeros(2, 1, 28, 28))
            except NotImplementedError:
                print(f"{name}: omitido (augment() sin implementar)")
                continue
        set_seed(SEED)
        model = make()
        res = train_torch(model, data, device, epochs=epochs, lr=lr, transform=transform)
        hist = res["history"]
        preds[name] = predict_torch(model, Xte, device)
        info[name] = {
            "seconds": hist.seconds,
            "params": res["params"],
            "epochs_run": len(hist.val_loss),
            "best_epoch": hist.best_epoch + 1,
        }
        curves[name] = hist
        print(
            f"{name}: {np.mean(preds[name] == y_test):.4f}"
            f" ({hist.seconds:.0f}s, {res['params']:,} parámetros,"
            f" mejor época {hist.best_epoch + 1}/{len(hist.val_loss)})"
        )
        if name.startswith("cnn"):
            torch.save(model.state_dict(), OUT / f"{name}.pt")  # ignorado por git (*.pt)

    # 4. Evaluación
    results = {}
    for name, p in preds.items():
        correct = int(np.sum(p == y_test))
        results[name] = {
            "accuracy": correct / len(y_test),
            "ic95_wilson": wilson(correct, len(y_test)),
            **info[name],
        }
    best = max((n for n in results if n.startswith("cnn")), key=lambda n: results[n]["accuracy"])
    comparisons = {
        f"{best}_vs_{n}": mcnemar(preds[best], preds[n], y_test) for n in preds if n != best
    }

    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_test, preds[best]), 1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(
        (
            (int(off[i, j] + off[j, i]), CLASSES[i], CLASSES[j])
            for i in range(10)
            for j in range(i + 1, 10)
        ),
        reverse=True,
    )[:5]
    per_class = {CLASSES[i]: float(cm[i, i] / cm[i].sum()) for i in range(10)}

    report = {
        "device": str(device),
        "best": best,
        "test": results,
        "mcnemar": comparisons,
        "per_class_recall_best": per_class,
        "most_confused_pairs_best": pairs,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # 5. Figuras
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, hist in curves.items():
        ax.plot(range(1, len(hist.val_acc) + 1), hist.val_acc, label=name)
    ax.set(
        xlabel="época",
        ylabel="accuracy de validación",
        title="Curvas de aprendizaje",
        ylim=(0.8, 0.95),
    )
    ax.legend()
    save_fig(fig, OUT / "curvas.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > cm[i].sum() / 2 else "black",
            )
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set(xlabel="predicción", ylabel="verdad", title=f"Matriz de confusión ({best}, test)")
    ax.grid(False)
    save_fig(fig, OUT / "confusion.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = sorted(results, key=lambda n: results[n]["accuracy"])
    acc = [results[n]["accuracy"] for n in names]
    err = np.array(
        [
            [a - results[n]["ic95_wilson"][0], results[n]["ic95_wilson"][1] - a]
            for n, a in zip(names, acc, strict=True)
        ]
    ).T
    ax.barh(names, acc, xerr=err, color="C0")
    ax.set_xlim(0.8, 0.95)
    ax.set(xlabel="accuracy en test (IC 95 % Wilson)", title="Fashion-MNIST")
    save_fig(fig, OUT / "comparacion.png")

    wrong = np.flatnonzero(preds[best] != y_test)[:24]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for ax, i in zip(axes.flat, wrong, strict=False):
        ax.imshow(Xte[i, 0], cmap="gray_r")
        ax.set_title(f"{CLASSES[y_test[i]]}\n→ {CLASSES[preds[best][i]]}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Errores de la mejor CNN (verdad → predicción)")
    save_fig(fig, OUT / "errores.png")

    print(f"\nMejor: {best}. Pares más confundidos: {pairs[:3]}")
    for k, v in comparisons.items():
        print(f"  McNemar {k}: {v}")


if __name__ == "__main__":
    main()
