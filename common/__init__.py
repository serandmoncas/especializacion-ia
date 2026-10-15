"""Utilidades compartidas entre módulos: semillas, rutas de datos y estilo de gráficos."""

from common.paths import data_dir, module_dir, repo_root
from common.plotting import save_fig, setup_style
from common.seed import set_seed

__all__ = ["data_dir", "module_dir", "repo_root", "save_fig", "set_seed", "setup_style"]
