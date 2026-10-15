import numpy as np
import pytest
from applied_dl import Tensor
from applied_dl.conv import col2im, conv2d, im2col

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(("stride", "padding"), [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_conv2d_forward_and_backward_match_torch(stride, padding):
    rng = np.random.default_rng(0)
    x, w, b = rng.normal(size=(2, 3, 9, 8)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    xt, wt, bt = (Tensor(a.copy(), requires_grad=True) for a in (x, w, b))
    out = conv2d(xt, wt, bt, stride=stride, padding=padding)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()

    xr, wr, br = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
    ref = torch.nn.functional.conv2d(xr, wr, br, stride=stride, padding=padding)
    (ref * torch.from_numpy(upstream)).sum().backward()

    np.testing.assert_allclose(out.data, ref.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(xt.grad, xr.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(wt.grad, wr.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(bt.grad, br.grad.numpy(), atol=1e-10)


def test_col2im_is_adjoint_of_im2col():
    # ⟨im2col(x), c⟩ = ⟨x, col2im(c)⟩: col2im es la transpuesta del operador lineal im2col
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 2, 7, 7))
    cols = im2col(x, 3, 3, 2)
    c = rng.normal(size=cols.shape)
    assert np.sum(cols * c) == pytest.approx(np.sum(x * col2im(c, x.shape, 3, 3, 2)))


def test_edge_detector():
    img = np.zeros((1, 1, 5, 5))
    img[..., :, 3:] = 1.0  # borde vertical
    sobel_x = np.array([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=float)
    out = conv2d(Tensor(img), Tensor(sobel_x)).data[0, 0]
    assert np.all(out[:, 1:] > 0) and np.all(out[:, 0] == 0)
