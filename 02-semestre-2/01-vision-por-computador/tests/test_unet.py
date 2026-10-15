import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
from applied_cv.unet import UNet  # noqa: E402


def test_unet_output_shape_and_size_check():
    net = UNet(3, 2, base=4)
    assert net(torch.zeros(2, 3, 32, 32)).shape == (2, 2, 32, 32)
    with pytest.raises(ValueError, match="múltiplos de 4"):
        net(torch.zeros(1, 3, 30, 32))


def test_unet_overfits_one_example():
    torch.manual_seed(0)
    net = UNet(1, 2, base=4)
    x = torch.rand(1, 1, 16, 16)
    y = (x[0, 0] > 0.5).long()[None]
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
