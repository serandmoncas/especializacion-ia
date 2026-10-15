"""`run.py`: ayuda y error claro si falta una clave (antes de gastar tiempo)."""

import importlib.util
from pathlib import Path

import pytest

RUN = Path(__file__).parents[1] / "proyecto" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("pln_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as err:
        _load().main(["--help"])
    assert err.value.code == 0


def test_missing_keys_fail_fast_with_instructions(monkeypatch, capsys):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _load().main([]) == 2
    assert "OPENALEX_API_KEY" in capsys.readouterr().out


def test_without_llm_only_openalex_key_is_required(monkeypatch, capsys):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _load().main(["--sin-llm"]) == 2
    assert "ANTHROPIC_API_KEY" not in capsys.readouterr().out
