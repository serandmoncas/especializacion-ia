"""BPE propio contra `tokenizers` de Hugging Face (mismas fusiones y misma tokenización)."""

import json
import random

import pytest
from research_paths.bpe import encode, train_bpe

WORDS = [
    "casa", "casas", "cosa", "cosas", "perro", "perros", "gato", "gatos", "camino", "caminos",
    "canto", "cantos", "tema", "temas", "investigación", "información", "aprendizaje",
]  # fmt: skip


def test_first_merge_is_the_most_frequent_pair():
    assert train_bpe(["ab ab ab aa"], 1) == [("a", "b")]


def test_ties_break_by_smallest_symbol_ids():
    # «aa» y «ab» aparecen 3 veces; gana (a, a) porque id(a) = 0 < id(b) = 1.
    assert train_bpe(["aaab aab ab"], 1) == [("a", "a")]


def test_stops_when_no_pairs_remain():
    assert train_bpe(["ab"], 10) == [("a", "b")]


def test_encode_reconstructs_each_word_and_compresses():
    merges = train_bpe([" ".join(WORDS * 3)], 30)
    text = "gatos caminos información"
    tokens = encode(text, merges)
    assert "".join(tokens) == text.replace(" ", "")
    assert len(tokens) < len(text.replace(" ", ""))


def _hf_bpe(corpus: list[str], num_merges: int):
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.BPE())
    tok.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    alphabet = {ch for line in corpus for ch in line if not ch.isspace()}
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=len(alphabet) + num_merges, min_frequency=0, show_progress=False
    )
    tok.train_from_iterator(corpus, trainer)
    merges = [tuple(m) for m in json.loads(tok.to_str())["model"]["merges"]]
    return tok, merges


def test_merges_and_tokens_match_huggingface_tokenizers():
    rng = random.Random(0)
    for _ in range(50):
        corpus = [" ".join(rng.choices(WORDS, k=rng.randint(20, 300)))]
        tok, ref = _hf_bpe(corpus, 25)
        merges = train_bpe(corpus, 25)
        assert merges == ref
        candidates = ["gatos", "cantan", "caminos", "de", "información"]
        text = " ".join(w for w in candidates if set(w) <= set(corpus[0]))
        assert encode(text, merges) == tok.encode(text).tokens
