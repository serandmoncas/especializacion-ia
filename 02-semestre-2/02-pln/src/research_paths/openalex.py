"""Cliente mínimo de OpenAlex sobre `CachedClient` (paginación por cursor)."""

from research_paths.http import CachedClient

BASE = "https://api.openalex.org"
WORK_FIELDS = (
    "id,title,type,publication_year,language,cited_by_count,referenced_works,"
    "abstract_inverted_index,primary_topic"
)
KEY_URL = "https://openalex.org/settings/api"


def short_id(value: str) -> str:
    """'https://openalex.org/W123' → 'W123'."""
    return value.rsplit("/", 1)[-1]


def abstract_text(inverted: dict | None) -> str:
    """Reconstruye el resumen desde el índice invertido {palabra: [posiciones]}."""
    if not inverted:
        return ""
    positions = {i: word for word, idxs in inverted.items() for i in idxs}
    return " ".join(positions[i] for i in sorted(positions))


class OpenAlex:
    def __init__(self, client: CachedClient, api_key: str | None):
        self.client, self.api_key = client, api_key

    def _paged(self, path: str, params: dict, max_results: int, per_page: int) -> list[dict]:
        out: list[dict] = []
        cursor = "*"
        while cursor and len(out) < max_results:
            page_params = {**params, "per-page": per_page, "cursor": cursor}
            if self.api_key:
                page_params["api_key"] = self.api_key
            data = self.client.get_json(f"{BASE}/{path}", page_params)
            results = data.get("results", [])
            out.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not results:
                break
        return out[:max_results]

    def works(
        self,
        *,
        filter: str,
        search: str | None = None,
        sort: str | None = None,
        max_results: int = 200,
        per_page: int = 200,
        select: str = WORK_FIELDS,
    ) -> list[dict]:
        params = {"filter": filter, "select": select}
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        return self._paged("works", params, max_results, min(per_page, max_results))

    def topics(self, *, filter: str) -> list[dict]:
        params = {"filter": filter, "select": "id,display_name,field,subfield,works_count"}
        return self._paged("topics", params, max_results=10_000, per_page=200)
