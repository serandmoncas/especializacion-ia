"""Cliente mínimo de la API de MediaWiki (Wikipedia) sobre `CachedClient`."""

from itertools import batched

from research_paths.http import CachedClient


class Wiki:
    def __init__(self, client: CachedClient, lang: str = "es"):
        self.client, self.lang = client, lang
        self.url = f"https://{lang}.wikipedia.org/w/api.php"

    def _query_all(self, **params) -> list[dict]:
        """Hace la consulta y sigue `continue` hasta agotarla."""
        base = {"action": "query", "format": "json", "formatversion": 2, **params}
        out, cont = [], {}
        while True:
            data = self.client.get_json(self.url, {**base, **cont})
            out.append(data)
            if "continue" not in data:
                return out
            cont = data["continue"]

    @staticmethod
    def _final_titles(responses: list[dict], titles: list[str]) -> dict[str, str]:
        norm, redir = {}, {}
        for data in responses:
            q = data.get("query", {})
            norm.update({x["from"]: x["to"] for x in q.get("normalized", [])})
            redir.update({x["from"]: x["to"] for x in q.get("redirects", [])})
        out = {}
        for title in titles:
            t, seen = norm.get(title, title), set()
            while t in redir and t not in seen:
                seen.add(t)
                t = redir[t]
            out[title] = t
        return out

    @staticmethod
    def _pages(responses: list[dict]) -> dict[str, dict]:
        """Une las páginas de varias respuestas; las propiedades de lista se concatenan."""
        pages: dict[str, dict] = {}
        for data in responses:
            for page in data.get("query", {}).get("pages", []):
                merged = pages.setdefault(page["title"], {})
                for key, value in page.items():
                    if isinstance(value, list):
                        merged.setdefault(key, []).extend(value)
                    else:
                        merged[key] = value
        return pages

    def _by_title(self, titles: list[str], size: int, **params) -> tuple[dict, dict]:
        finals, pages = {}, {}
        for chunk in batched(titles, size):
            responses = self._query_all(titles="|".join(chunk), redirects=1, **params)
            finals.update(self._final_titles(responses, list(chunk)))
            pages.update(self._pages(responses))
        return finals, pages

    def resolve(self, titles: list[str]) -> dict[str, str | None]:
        """Título canónico; None si no existe o es una página de desambiguación."""
        finals, pages = self._by_title(titles, 50, prop="pageprops", ppprop="disambiguation")
        out = {}
        for title in titles:
            page = pages.get(finals[title], {"missing": True})
            ok = not page.get("missing") and "disambiguation" not in page.get("pageprops", {})
            out[title] = finals[title] if ok else None
        return out

    def extracts(self, titles: list[str]) -> dict[str, str]:
        """Introducción en texto plano; "" si la página no existe."""
        finals, pages = self._by_title(
            titles, 20, prop="extracts", exintro=1, explaintext=1, exlimit=20
        )
        return {t: pages.get(finals[t], {}).get("extract", "") or "" for t in titles}

    def links(self, title: str) -> list[str]:
        """Enlaces salientes a artículos (espacio de nombres 0), en orden y sin repetir."""
        responses = self._query_all(
            titles=title, redirects=1, prop="links", plnamespace=0, pllimit="max"
        )
        seen: dict[str, None] = {}
        for page in self._pages(responses).values():
            for link in page.get("links", []):
                seen.setdefault(link["title"], None)
        return list(seen)

    def langlinks(self, titles: list[str], target: str) -> dict[str, str | None]:
        """Título equivalente en otro idioma; None si no existe."""
        finals, pages = self._by_title(titles, 50, prop="langlinks", lllang=target, lllimit="max")
        out = {}
        for title in titles:
            page = pages.get(finals[title], {})
            links = [x for x in page.get("langlinks", []) if x.get("lang") == target]
            out[title] = links[0]["title"] if links and not page.get("missing") else None
        return out

    def search(self, query: str, limit: int = 5) -> list[str]:
        data = self.client.get_json(
            self.url,
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
            },
        )
        return [x["title"] for x in data.get("query", {}).get("search", [])]
