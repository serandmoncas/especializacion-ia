"""Experimentos del motor de rutas: capa 1 (AL-CPL), capa 2 (OpenAlex) e hipótesis → resultados/.

Requiere `data/download.py` (AL-CPL) y las claves OPENALEX_API_KEY y ANTHROPIC_API_KEY en el
entorno (con --sin-llm no se usa Anthropic y H2 queda «no evaluada"). Wikipedia, OpenAlex y el
LLM se guardan en data/cache/: la segunda ejecución no usa red ni gasta presupuesto.

Uso:
    uv run python 02-semestre-2/02-pln/proyecto/run.py [--sin-llm] [--por-campo 15]
    uv run python 02-semestre-2/02-pln/proyecto/run.py --acuerdo-intro   # tras etiquetar la muestra
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from research_paths.http import CachedClient, MissingKeyError, require_env

from common import data_dir

MODULE = "02-semestre-2/02-pln"
OUT = Path(__file__).resolve().parent / "resultados"
LABELS = Path(__file__).resolve().parent / "intro_etiquetas.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sin-llm", action="store_true", help="no usar el juez LLM (H2 no evaluada)"
    )
    parser.add_argument("--por-campo", type=int, default=15, help="temas de OpenAlex por campo")
    parser.add_argument(
        "--acuerdo-intro", action="store_true",
        help="solo agrega a metrics.json el acuerdo con proyecto/intro_etiquetas.csv",
    )  # fmt: skip
    args = parser.parse_args(argv)

    if args.acuerdo_intro:
        from research_paths.experiments import intro_agreement

        metrics = json.loads((OUT / "metrics.json").read_text())
        sample = pd.read_csv(OUT / "intro_muestra.csv")
        metrics["capa2"]["intro"]["acuerdo_manual"] = intro_agreement(sample, pd.read_csv(LABELS))
        text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
        (OUT / "metrics.json").write_text(text + "\n")
        print(metrics["capa2"]["intro"]["acuerdo_manual"])
        return 0

    try:
        oa_key = require_env("OPENALEX_API_KEY", "https://openalex.org/settings/api")
        if not args.sin_llm:
            require_env("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys")
    except MissingKeyError as err:
        print(err)
        return 2

    from research_paths.experiments import Context, run_all
    from research_paths.openalex import OpenAlex
    from research_paths.prereq.llm import LLMJudge
    from research_paths.retrieval import e5_encoder
    from research_paths.textclf import TrainConfig, default_device, load_pretrained
    from research_paths.wiki import Wiki

    base = data_dir(MODULE)
    if not (base / "alcpl").is_dir():
        print(f"Falta AL-CPL: uv run python {MODULE}/data/download.py")
        return 2
    device = default_device()
    cache = base / "cache"
    ctx = Context(
        data_dir=base,
        out_dir=OUT,
        models_dir=base / "modelos",
        wiki_en=Wiki(CachedClient(cache / "wiki-en"), "en"),
        wiki_es=Wiki(CachedClient(cache / "wiki-es"), "es"),
        oa=OpenAlex(CachedClient(cache / "openalex"), oa_key),
        encoder=e5_encoder(device=device),
        make_model=lambda texts: load_pretrained("xlm-roberta-base"),
        train_cfg=TrainConfig(),
        judge=None if args.sin_llm else LLMJudge(cache / "llm"),
        device=device,
        per_field=args.por_campo,
    )
    metrics = run_all(ctx)
    for h, v in metrics["hipotesis"].items():
        print(h, v.get("descripcion"), {k: v[k] for k in ("delta", "p_holm", "se_apoya") if k in v})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
