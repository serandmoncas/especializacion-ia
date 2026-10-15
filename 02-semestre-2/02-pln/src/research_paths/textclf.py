"""Clasificador de texto afinado (XLM-R): un texto o un par (cross-encoder).

`tiny_model` construye un XLM-R diminuto con un tokenizador BPE entrenado en local: sirve para
tests y notebooks sin descargar nada. Para el proyecto se usa `load_pretrained("xlm-roberta-base")`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research_paths.evaluation import average_precision


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 256  # el par completo; se trunca `longest_first`
    epochs: int = 2
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def tiny_model(texts: list[str], vocab_size: int = 200, num_labels: int = 2, max_length: int = 64):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import (
        PreTrainedTokenizerFast,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
    )

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    special = ["<s>", "<pad>", "</s>", "<unk>"]
    tk.train_from_iterator(
        texts,
        trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special, show_progress=False),
    )
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=max_length,
    )
    torch.manual_seed(0)
    config = XLMRobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=max_length + 16,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
    )
    return XLMRobertaForSequenceClassification(config), tokenizer


def load_pretrained(name: str, num_labels: int = 2):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model, tokenizer


def _encode(tokenizer, texts_a, texts_b, max_length: int, device: str):
    # Nunca más largo de lo que el modelo admite (p. ej. 64 en el modelo diminuto, 512 en XLM-R).
    max_length = min(max_length, tokenizer.model_max_length)
    enc = tokenizer(
        list(texts_a),
        list(texts_b) if texts_b is not None else None,
        truncation="longest_first",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict_proba(
    model,
    tokenizer,
    texts_a,
    texts_b=None,
    *,
    max_length: int = 256,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """P(clase 1) para cada texto o par."""
    model.to(device).eval()
    out = []
    for s in range(0, len(texts_a), batch_size):
        b = texts_b[s : s + batch_size] if texts_b is not None else None
        logits = model(
            **_encode(tokenizer, texts_a[s : s + batch_size], b, max_length, device)
        ).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_classifier(
    texts_a,
    labels,
    val_a,
    val_labels,
    cfg: TrainConfig,
    *,
    texts_b=None,
    val_b=None,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    log=print,
):
    """AdamW + calentamiento lineal y pesos por clase; devuelve la mejor época (AP en val)."""
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if model is None:
        model, tokenizer = load_pretrained(cfg.model_name)
    model.to(device)
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(float)
    weight = torch.tensor(
        len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(texts_a) / cfg.batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    history, best_state, best_ap = [], None, -1.0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(texts_a))
        losses = []
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s : s + cfg.batch_size]
            b = [texts_b[i] for i in idx] if texts_b is not None else None
            enc = _encode(tokenizer, [texts_a[i] for i in idx], b, cfg.max_length, device)
            y = torch.tensor(labels[idx], device=device)
            loss = F.cross_entropy(model(**enc).logits.float(), y, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        p = predict_proba(model, tokenizer, val_a, val_b, max_length=cfg.max_length, device=device)
        ap = average_precision(val_labels, p)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_ap": ap})
        log(f"  época {epoch + 1}: pérdida {np.mean(losses):.4f} · AP val {ap:.3f}")
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tokenizer, history


def save_classifier(model, tokenizer, path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_classifier(path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return (
        AutoModelForSequenceClassification.from_pretrained(path),
        AutoTokenizer.from_pretrained(path),
    )
