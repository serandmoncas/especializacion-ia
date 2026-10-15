import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common import data_dir, module_dir, repo_root, save_fig, set_seed, setup_style


def test_set_seed_is_reproducible():
    a = set_seed(7).normal(size=5)
    b = set_seed(7).normal(size=5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_fixes_global_numpy_state():
    set_seed(1)
    a = np.random.rand(3)  # noqa: NPY002
    set_seed(1)
    b = np.random.rand(3)  # noqa: NPY002
    np.testing.assert_array_equal(a, b)


def test_repo_root_contains_pyproject():
    assert (repo_root() / "pyproject.toml").is_file()


def test_module_and_data_dir():
    rel = "01-semestre-1/01-matematicas-aplicadas"
    assert module_dir(rel).name == "01-matematicas-aplicadas"
    assert data_dir(rel).name == "data"


def test_save_fig_creates_parents(tmp_path):
    setup_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_fig(fig, tmp_path / "a" / "b" / "fig.png")
    plt.close(fig)
    assert out.is_file()
