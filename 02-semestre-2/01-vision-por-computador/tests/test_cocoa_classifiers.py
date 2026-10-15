import numpy as np
import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.classifiers import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    CropDataset,
    build_resnet,
    eval_transform,
    extract_features,
    fine_tune,
    fit_logistic,
    predict_proba,
    train_augmentation,
)
from applied_cv.cocoa.features import color_texture_features  # noqa: E402
from applied_cv.image import rgb_to_hsv  # noqa: E402
from PIL import Image  # noqa: E402

RNG = np.random.default_rng(0)
GREEN, BROWN = (90, 160, 60), (110, 70, 30)


def _crop(color, size=32):
    arr = np.clip(np.array(color) + RNG.integers(-15, 15, (size, size, 3)), 0, 255)
    return arr.astype(np.uint8)


def test_color_texture_features_shape_and_separation():
    crops = [_crop(GREEN) for _ in range(20)] + [_crop(BROWN) for _ in range(20)]
    X = np.stack([color_texture_features(c) for c in crops])
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    assert X.shape == (40, 304)
    model = fit_logistic(X[::2], y[::2], X[1::2], y[1::2], n_classes=4)
    proba = model.predict_proba(X[1::2])
    assert proba.shape == (20, 4) and np.allclose(proba.sum(axis=1), 1)
    assert (proba.argmax(axis=1) == y[1::2]).mean() == 1.0


def test_train_augmentation_preserves_hue():
    # Regla de dominio: el café de la lesión es el síntoma; la aumentación no puede cambiar el tono.
    torch.manual_seed(0)
    image = Image.fromarray(np.full((64, 64, 3), BROWN, np.uint8))
    hue = rgb_to_hsv(np.array(BROWN) / 255)[0]
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    for _ in range(20):
        x = (train_augmentation()(image) * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        hsv = rgb_to_hsv(x)
        visible = hsv[..., 2] > 0.05  # ignora el relleno negro de rotaciones
        assert np.abs(hsv[..., 0][visible] - hue).max() < 0.02


def test_resnet_fine_tune_predict_and_features():
    torch.manual_seed(0)
    images = [Image.fromarray(_crop(c, 64)) for c in [GREEN] * 8 + [BROWN] * 8]
    labels = [0] * 8 + [1] * 8
    train_ds = CropDataset(images, labels, eval_transform())
    model = build_resnet(4, pretrained=False)
    history = fine_tune(
        model, train_ds, train_ds, epochs=2, lr=1e-3, device=torch.device("cpu"), batch_size=8
    )
    assert len(history.train_loss) == 2
    cpu, unlabeled = torch.device("cpu"), CropDataset(images, None, eval_transform())
    proba = predict_proba(model, unlabeled, cpu)
    assert proba.shape == (16, 4) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    feats = extract_features(model, unlabeled, cpu)
    assert feats.shape == (16, 512)
    assert isinstance(model.fc, torch.nn.Linear)  # extract_features no rompe el modelo
