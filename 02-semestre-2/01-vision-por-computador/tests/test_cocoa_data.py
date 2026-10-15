from datetime import datetime

import pandas as pd
import pytest
from applied_cv.cocoa.data import load_annotations, parse_filename
from applied_cv.cocoa.prepare import prepare, square_crop_box
from PIL import Image


@pytest.mark.parametrize(
    ("name", "camera", "timestamp", "seq"),
    [
        ("IMG_20240226_143016382_HDR.jpg", "IMG", datetime(2024, 2, 26, 14, 30, 16), None),
        ("20241011_110458.jpg", "fecha_hora", datetime(2024, 10, 11, 11, 4, 58), None),
        ("1728667861304.jpg", "epoch", datetime(2024, 10, 11, 17, 31, 1, 304000), None),
        ("IMG_E2361.jpg", "secuencia", None, 2361),
        ("IMG_2352.jpg", "secuencia", None, 2352),
        ("m0 (95).jpg", "otro", None, None),
    ],
)
def test_parse_filename_formats(name, camera, timestamp, seq):
    assert parse_filename(name) == (camera, timestamp, seq)


def test_load_annotations_dedupes_photos_and_pods(cocoa_raw):
    images, pods = load_annotations(cocoa_raw / "CocoaMoniliaDataSet")
    assert len(images) == 11 and images.image.is_unique
    shared = images.set_index("image").loc["IMG_20240227_090000000.jpg"]
    assert shared.folder == "h0" and shared.path == "cocoapods_images/h0/IMG_20240227_090000000.jpg"
    assert len(pods) == 12 and pods.pod_id.tolist() == list(range(12))
    assert pods.label.value_counts().to_dict() == {"h0": 4, "m1": 4, "m3": 2, "m2": 1, "m4": 1}
    first = pods[pods.image == "1728667861304.jpg"].iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == (40, 20, 70, 100)


def test_square_crop_box_is_square_and_clipped():
    # caja 30×80 → lado 96 centrado en (55, 60) → (7, 12, 103, 108), dentro de 160×120
    assert square_crop_box((40, 20, 70, 100), 160, 120, margin=0.1) == (7, 12, 103, 108)
    left, top, right, bottom = square_crop_box((0, 12, 16, 36), 48, 64, margin=0.1)
    assert (left, top) == (0, 10) and right <= 48 and bottom <= 64


def test_prepare_resizes_crops_and_reports_missing(cocoa_raw, tmp_path):
    out = tmp_path / "prep"
    summary = prepare(cocoa_raw / "CocoaMoniliaDataSet", out, max_side=64, crop_size=32)
    assert summary["missing"] == ["IMG_20240229_110000000.jpg"]
    assert summary["images"] == 10 and summary["pods"] == 11
    images = pd.read_csv(out / "images.csv", parse_dates=["timestamp"])
    pods = pd.read_csv(out / "pods.csv")
    assert len(images) == 10 and len(pods) == 11
    with Image.open(out / "images" / "1728667861304.jpg") as im:
        assert im.size == (64, 48)
    with Image.open(out / pods.crop.iloc[0]) as im:
        assert im.size == (32, 32)
    first = pods[pods.image == "1728667861304.jpg"].iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == pytest.approx((16, 8, 28, 40))


def test_prepare_reports_exif_and_size_mismatch(cocoa_raw, tmp_path):
    summary = prepare(cocoa_raw / "CocoaMoniliaDataSet", tmp_path / "p", max_side=64, crop_size=32)
    assert summary["size_mismatch"] == ["m0 (95).jpg"]
    assert summary["exif_orientations"] == {1: 10}
