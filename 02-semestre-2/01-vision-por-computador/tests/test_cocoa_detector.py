import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.detector import (  # noqa: E402
    DetectionDataset,
    build_detector,
    detect,
    train_detector,
)
from applied_cv.cocoa.prepare import prepare  # noqa: E402


def test_flip_mirrors_boxes(cocoa_raw, tmp_path):
    out = tmp_path / "p"
    prepare(cocoa_raw / "CocoaMoniliaDataSet", out, max_side=64, crop_size=32)
    path = out / "images" / "1728667861304.jpg"
    box = np.array([[16.0, 8.0, 28.0, 40.0]])
    plain = DetectionDataset([path], [box])[0]
    flipped = DetectionDataset([path], [box], flip_prob=1.0)[0]
    width = plain[0].shape[-1]
    torch.testing.assert_close(flipped[0], plain[0].flip(-1))
    torch.testing.assert_close(
        flipped[1]["boxes"], torch.tensor([[width - 28.0, 8, width - 16, 40]])
    )
    assert flipped[1]["labels"].tolist() == [1]


def test_train_and_detect_on_tiny_images(cocoa_raw, tmp_path):
    out = tmp_path / "p"
    prepare(cocoa_raw / "CocoaMoniliaDataSet", out, max_side=64, crop_size=32)
    pods = pd.read_csv(out / "pods.csv")
    names = sorted(pods.image.unique())[:4]
    paths = [out / "images" / n for n in names]
    boxes = [pods[pods.image == n][["x1", "y1", "x2", "y2"]].to_numpy() for n in names]
    model = build_detector(pretrained=False, min_size=64, max_size=96)
    losses = train_detector(
        model,
        DetectionDataset(paths, boxes, flip_prob=0.5),
        epochs=1,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert len(losses) == 1 and np.isfinite(losses[0])
    results = detect(model, paths, torch.device("cpu"), score_threshold=0.0)
    assert len(results) == 4
    assert all(b.shape[1:] == (4,) and len(b) == len(s) for b, s in results)
