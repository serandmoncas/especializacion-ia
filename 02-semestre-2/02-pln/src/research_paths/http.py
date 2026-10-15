"""GET con caché en disco y reintentos; las claves de API no entran a la caché ni a los errores."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Wikimedia responde 429 a clientes sin un User-Agent con forma de contacto.
USER_AGENT = (
    "especializacion-ia/0.1 "
    "(https://github.com/serandmoncas/especializacion-ia; proyecto de estudio)"
)
SECRET_PARAMS = frozenset({"api_key"})
RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 599})

Fetch = Callable[[str], tuple[int, bytes]]


class HTTPError(RuntimeError):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} en {url}")
        self.status = status


class MissingKeyError(RuntimeError):
    pass


def require_env(name: str, where: str) -> str:
    """Lee una clave del entorno o explica cómo conseguirla."""
    value = os.environ.get(name)
    if not value:
        raise MissingKeyError(
            f"Falta {name} en el entorno. Consíguela en {where} y expórtala: export {name}=..."
        )
    return value


class DiskCache:
    """JSON por clave en `root/<sha1[:2]>/<sha1>.json`; escritura atómica."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str):
        path = self._path(key)
        return json.loads(path.read_text()) if path.is_file() else None

    def set(self, key: str, value) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(path)


def _url(base_url: str, params: dict, *, public: bool) -> str:
    items = sorted((k, v) for k, v in params.items() if not (public and k in SECRET_PARAMS))
    return f"{base_url}?{urllib.parse.urlencode(items)}"


def urllib_fetch(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, b""
    except (urllib.error.URLError, TimeoutError):
        return 599, b""  # sin conexión: se reintenta como un 5xx


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        fetch: Fetch = urllib_fetch,
        max_retries: int = 5,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache = DiskCache(cache_dir)
        self.fetch, self.max_retries, self.base_delay, self.sleep = (
            fetch,
            max_retries,
            base_delay,
            sleep,
        )
        self.stats = {"network": 0, "cache": 0}

    def get_json(self, base_url: str, params: dict) -> dict:
        key = _url(base_url, params, public=True)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache"] += 1
            return hit
        full = _url(base_url, params, public=False)
        for attempt in range(self.max_retries):
            status, body = self.fetch(full)
            self.stats["network"] += 1
            if status == 200:
                data = json.loads(body)
                self.cache.set(key, data)
                return data
            if status not in RETRY_STATUS or attempt == self.max_retries - 1:
                raise HTTPError(status, key)
            self.sleep(self.base_delay * 2**attempt)
        raise AssertionError("inalcanzable")
