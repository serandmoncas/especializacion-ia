"""BPE (byte-pair encoding) desde cero, con el mismo desempate que `tokenizers` de Hugging Face."""

from collections import Counter
from collections.abc import Iterable
from itertools import pairwise

Merge = tuple[str, str]


def _apply(symbols: tuple[str, ...], pair: Merge) -> tuple[str, ...]:
    """Fusiona, de izquierda a derecha, cada aparición de `pair`."""
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(corpus: Iterable[str], num_merges: int) -> list[Merge]:
    """Aprende hasta `num_merges` fusiones; las palabras se separan por espacios.

    En cada paso fusiona el par adyacente más frecuente. En empate gana el de ids menores: el id de
    un símbolo es su posición en el alfabeto ordenado y, para los fusionados, su orden de creación
    (igual que `tokenizers.trainers.BpeTrainer`).
    """
    words = Counter(w for line in corpus for w in line.split())
    vocab: dict[tuple[str, ...], int] = {tuple(w): c for w, c in words.items()}
    ids = {ch: i for i, ch in enumerate(sorted({ch for w in words for ch in w}))}
    merges: list[Merge] = []
    for _ in range(num_merges):
        pairs: Counter[Merge] = Counter()
        for symbols, count in vocab.items():
            for pair in pairwise(symbols):
                pairs[pair] += count
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], -ids[kv[0][0]], -ids[kv[0][1]]))[0]
        merges.append(best)
        ids.setdefault(best[0] + best[1], len(ids))
        merged: dict[tuple[str, ...], int] = {}
        for symbols, count in vocab.items():
            new = _apply(symbols, best)
            merged[new] = merged.get(new, 0) + count
        vocab = merged
    return merges


def encode(text: str, merges: list[Merge]) -> list[str]:
    """Tokeniza cada palabra aplicando siempre la fusión de menor rango que esté presente."""
    rank = {pair: i for i, pair in enumerate(merges)}
    tokens: list[str] = []
    for word in text.split():
        symbols = tuple(word)
        while len(symbols) > 1:
            present = [rank[p] for p in pairwise(symbols) if p in rank]
            if not present:
                break
            symbols = _apply(symbols, merges[min(present)])
        tokens.extend(symbols)
    return tokens
