"""Juez LLM (Claude) para prerrequisitos: herramienta forzada, caché por respuesta y reintentos."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research_paths.http import DiskCache, require_env

SYSTEM = (
    "Eres un docente experto en diseñar rutas de aprendizaje. Decides si un concepto es "
    "prerrequisito de otro: si hay que entender B para poder entender A."
)
TOOL = {
    "name": "responder",
    "description": "Registra si B es prerrequisito de A.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prerrequisito": {"type": "boolean", "description": "true si B es prerrequisito de A"},
            "razon": {"type": "string", "description": "una frase"},
        },
        "required": ["prerrequisito", "razon"],
    },
}
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
KEY_URL = "https://console.anthropic.com/settings/keys"


def build_prompt(title_a: str, text_a: str, title_b: str, text_b: str, max_chars: int) -> str:
    return (
        f"Concepto A: {title_a}\n{text_a[:max_chars]}\n\n"
        f"Concepto B: {title_b}\n{text_b[:max_chars]}\n\n"
        "¿Hay que entender B antes de poder entender A?"
    )


def _retryable(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if status in RETRY_STATUS:
        return True
    name = type(err).__name__
    return name in {"APIConnectionError", "APITimeoutError"}


class LLMJudge:
    def __init__(
        self,
        cache_dir: Path,
        model: str = "claude-haiku-5-5",
        client=None,
        max_workers: int = 8,
        max_retries: int = 6,
        base_delay: float = 2.0,
        sleep=time.sleep,
        max_chars: int = 1200,
    ):
        self.cache = DiskCache(cache_dir)
        self.model, self._client = model, client
        self.max_workers, self.max_retries, self.base_delay = max_workers, max_retries, base_delay
        self.sleep, self.max_chars = sleep, max_chars
        self.stats = {"api": 0, "cache": 0, "unparsed": 0}
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            require_env("ANTHROPIC_API_KEY", KEY_URL)
            import anthropic

            self._client = anthropic.Anthropic(max_retries=0)  # los reintentos son nuestros
        return self._client

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _call(self, prompt: str):
        for attempt in range(self.max_retries):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=0,
                    system=SYSTEM,
                    tools=[TOOL],
                    tool_choice={"type": "tool", "name": "responder"},
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as err:
                if not _retryable(err) or attempt == self.max_retries - 1:
                    raise
                self.sleep(self.base_delay * 2**attempt)
        raise AssertionError("inalcanzable")

    @staticmethod
    def _parse(response) -> tuple[bool | None, str]:
        for block in getattr(response, "content", []):
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", "") == "responder"
            ):
                value = block.input.get("prerrequisito")
                if isinstance(value, bool):
                    return value, str(block.input.get("razon", ""))
        return None, ""

    def _ask(self, item: tuple[str, str, str, str]) -> bool | None:
        prompt = build_prompt(*item, max_chars=self.max_chars)
        key = f"{self.model}\n{SYSTEM}\n{prompt}"
        hit = self.cache.get(key)
        if hit is not None:
            self._count("cache")
            answer = hit["answer"]
        else:
            answer, reason = self._parse(self._call(prompt))
            self._count("api")
            self.cache.set(key, {"answer": answer, "reason": reason})
        if answer is None:
            self._count("unparsed")
        return answer

    def judge(self, items: list[tuple[str, str, str, str]]) -> list[bool | None]:
        if self.max_workers <= 1:
            return [self._ask(item) for item in items]
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._ask, items))
