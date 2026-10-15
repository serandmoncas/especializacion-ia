"""Servidor HTTP falso para los tests (respuestas grabadas a mano), expuesto como fixture."""

import json
import urllib.parse

import pytest


class FakeServer:
    """`routes` es una lista de (predicado(params) -> bool, respuesta | [respuestas]).

    Una respuesta es un dict (200 con ese JSON) o un int (código de error). Si hay una lista, se
    consume en orden (sirve para simular 429 y luego 200).
    """

    def __init__(self, routes):
        self.routes = [(pred, resp if isinstance(resp, list) else [resp]) for pred, resp in routes]
        self.calls: list[str] = []

    def __call__(self, url: str) -> tuple[int, bytes]:
        self.calls.append(url)
        params = {
            k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()
        }
        for pred, responses in self.routes:
            if pred(params):
                resp = responses.pop(0) if len(responses) > 1 else responses[0]
                if isinstance(resp, int):
                    return resp, b""
                return 200, json.dumps(resp).encode()
        return 404, b""


@pytest.fixture
def server_cls():
    return FakeServer
