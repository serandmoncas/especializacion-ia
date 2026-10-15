"""Ejercicio 6: RefD en un grafo de enlaces pequeño. Ver README.md."""


def refd(a: str, b: str, links: dict[str, set[str]], concepts: set[str]) -> float:
    def refers(x, y):
        out = links.get(x, set()) & concepts
        if not out:
            return 0.0
        return sum(1 for c in out if c == y or y in links.get(c, set())) / len(out)

    return refers(a, b) - refers(b, a)


def check() -> None:
    import numpy as np
    from research_paths.prereq.refd import refd as reference

    links = {"Varianza": {"Media", "Suma"}, "Media": {"Suma"}, "Suma": set()}
    concepts = set(links)
    assert refd("Varianza", "Media", links, concepts) == 0.5
    assert refd("Media", "Suma", links, concepts) == 1.0
    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(10)]
    for _ in range(20):
        g = {n: {m for m in names if m != n and rng.random() < 0.3} for n in names}
        for a in names:
            for b in names:
                assert refd(a, b, g, set(names)) == reference(a, b, g, set(names))
    print("ej06 ok — ejemplos a mano e igual a research_paths.prereq.refd")


if __name__ == "__main__":
    check()
