import pytest

torch = pytest.importorskip("torch")
from applied_cv.gradcam import grad_cam  # noqa: E402


def test_grad_cam_of_known_network_is_the_normalized_input():
    # conv 1×1 con peso 1 → ReLU → promedio global → lineal con peso 1: el gradiente es uniforme,
    # así que Grad-CAM devuelve la activación (= la entrada) dividida por su máximo.
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 1)
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
        model[4].weight.fill_(1.0)
        model[4].bias.zero_()
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 1:3, 2:5] = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    cam = grad_cam(model, model[0], x, target=0)
    torch.testing.assert_close(cam[0], x[0, 0] / 6)


def test_grad_cam_on_resnet_shape_and_range():
    models = pytest.importorskip("torchvision.models")
    net = models.resnet18(weights=None, num_classes=3).eval()
    cam = grad_cam(net, net.layer4, torch.rand(2, 3, 64, 64), target=1)
    assert cam.shape == (2, 64, 64)
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0 + 1e-6
