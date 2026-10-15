"""Ejercicio 1: WordPiece greedy (el más largo primero, continuaciones con ##). Ver README.md."""


def wordpiece(word: str, vocab: set[str], unk: str = "[UNK]", max_chars: int = 100) -> list[str]:
    """Parte `word` en piezas del vocabulario; si algún tramo no tiene pieza, devuelve [unk]."""
    if len(word) > max_chars:
        return [unk]
    pieces, start = [], 0
    while start < len(word):
        end, piece = len(word), None
        while start < end:
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in vocab:
                piece = sub
                break
            end -= 1
        if piece is None:
            return [unk]
        pieces.append(piece)
        start = end
    return pieces


def check() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = ["[UNK]", "un", "##afable", "##a", "##fable", "casa", "##s", "ca", "##sa", "in"]
    vocab += ["##vest", "##iga", "##ción", "##i", "##g", "##a"]
    ids = {v: i for i, v in enumerate(dict.fromkeys(vocab))}
    ref = Tokenizer(models.WordPiece(ids, unk_token="[UNK]"))
    ref.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    for w in ["unafable", "casas", "cas", "xyz", "ca", "investigación", "inv"]:
        assert wordpiece(w, set(ids)) == ref.encode(w).tokens, w
    assert wordpiece("a" * 101, set(ids)) == ["[UNK]"]
    print("ej01 ok — igual que tokenizers.models.WordPiece")


if __name__ == "__main__":
    check()
