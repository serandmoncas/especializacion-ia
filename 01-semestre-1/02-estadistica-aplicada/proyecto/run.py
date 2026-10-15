"""Mini-proyecto: análisis completo del experimento de email de Hillstrom (A/B/C).

Plan de análisis (fijado antes de mirar resultados):
1. Validez: SRM sobre tamaños de grupo y balance de covariables previas.
2. Métrica principal: gasto por cliente. Familia de 3 comparaciones, corrección de Holm.
   Métodos: Welch, bootstrap, permutación y Welch sobre CUPED (covariable: `history`).
3. Secundarias: visita y conversión (z para proporciones + Beta–Binomial), corrección BH.
4. Exploratorio: efecto por segmento de historial de compra (no confirmatorio).

Uso:
    uv run python 01-semestre-1/02-estadistica-aplicada/data/download.py
    uv run python 01-semestre-1/02-estadistica-aplicada/proyecto/run.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from applied_stats import (
    benjamini_hochberg,
    beta_posterior,
    bootstrap_diff_ci,
    cuped,
    holm,
    permutation_test,
    prob_b_beats_a,
    sample_ratio_mismatch,
    two_proportion_ztest,
    welch_ttest,
)
from scipy import stats

from common import data_dir, save_fig, set_seed, setup_style

MODULE = "01-semestre-1/02-estadistica-aplicada"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42
CONTROL = "No E-Mail"
ARMS = {"mens": "Mens E-Mail", "womens": "Womens E-Mail", "control": CONTROL}
COMPARISONS = [("mens", "control"), ("womens", "control"), ("mens", "womens")]
COVARIATES = ["recency", "history", "mens", "womens", "newbie"]
N_RESAMPLES = 5_000
COST_PER_EMAIL = 0.10  # USD por email enviado: supuesto de negocio, no viene en los datos


def load() -> pd.DataFrame:
    csv = data_dir(MODULE) / "hillstrom.csv"
    if not csv.exists():
        raise SystemExit(f"Falta {csv}. Ejecutar primero data/download.py")
    df = pd.read_csv(csv)
    df["arm"] = df["segment"].map({v: k for k, v in ARMS.items()})
    return df


def validity_checks(df: pd.DataFrame) -> dict:
    counts = df["arm"].value_counts().reindex(list(ARMS))
    chi2, p_srm = sample_ratio_mismatch(counts.to_numpy())
    # Diferencia de medias estandarizada: |SMD| < 0.1 se considera balanceado.
    smd = {}
    for cov in COVARIATES:
        c = df.loc[df.arm == "control", cov]
        smd[cov] = {
            arm: float((df.loc[df.arm == arm, cov].mean() - c.mean()) / df[cov].std())
            for arm in ("mens", "womens")
        }
    return {"counts": counts.to_dict(), "srm_chi2": chi2, "srm_pvalue": p_srm, "smd": smd}


def analyze_spend(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    out = {}
    for b, a in COMPARISONS:
        sub = df[df.arm.isin([a, b])]
        ya, yb = (sub.loc[sub.arm == g, "spend"].to_numpy() for g in (a, b))
        welch = welch_ttest(ya, yb)
        lo, hi, boot = bootstrap_diff_ci(ya, yb, n_boot=N_RESAMPLES, rng=rng)
        _, p_perm = permutation_test(ya, yb, n_perm=N_RESAMPLES, rng=rng)
        # θ de CUPED se estima con ambos grupos juntos para no sesgar la diferencia.
        adj = cuped(sub["spend"].to_numpy(), sub["history"].to_numpy())
        is_b = (sub.arm == b).to_numpy()
        cup = welch_ttest(adj[~is_b], adj[is_b])
        out[f"{b}_vs_{a}"] = {
            "diff": welch.diff,
            "control_mean": float(ya.mean()),
            "welch_ci": welch.ci,
            "welch_p": welch.pvalue,
            "bootstrap_ci": (lo, hi),
            "permutation_p": p_perm,
            "cuped_diff": cup.diff,
            "cuped_ci": cup.ci,
            "cuped_p": cup.pvalue,
            "cuped_var_reduction": float(
                1 - (cup.ci[1] - cup.ci[0]) ** 2 / (welch.ci[1] - welch.ci[0]) ** 2
            ),
            "_boot": boot,
        }
    adjusted = holm([r["welch_p"] for r in out.values()])
    for r, p in zip(out.values(), adjusted, strict=True):
        r["welch_p_holm"] = float(p)
    return out


def analyze_binary(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    out = {}
    for metric in ("visit", "conversion"):
        for b, a in COMPARISONS:
            sa, na = int(df.loc[df.arm == a, metric].sum()), int((df.arm == a).sum())
            sb, nb = int(df.loc[df.arm == b, metric].sum()), int((df.arm == b).sum())
            z = two_proportion_ztest(sa, na, sb, nb)
            out[f"{metric}:{b}_vs_{a}"] = {
                "rate_a": sa / na,
                "rate_b": sb / nb,
                "diff": z.diff,
                "ci": z.ci,
                "p": z.pvalue,
                "prob_b_beats_a": prob_b_beats_a(
                    beta_posterior(sa, na), beta_posterior(sb, nb), rng=rng
                ),
            }
    adjusted = benjamini_hochberg([r["p"] for r in out.values()])
    for r, p in zip(out.values(), adjusted, strict=True):
        r["p_bh"] = float(p)
    return out


def analyze_segments(df: pd.DataFrame) -> dict:
    """Exploratorio: efecto en gasto según qué tipo de mercancía compró antes el cliente."""
    segments = {
        "solo_hombre": (df.mens == 1) & (df.womens == 0),
        "solo_mujer": (df.mens == 0) & (df.womens == 1),
        "ambos": (df.mens == 1) & (df.womens == 1),
    }
    out = {}
    for seg, mask in segments.items():
        d = df[mask]
        ctrl = d.loc[d.arm == "control", "spend"]
        for arm in ("mens", "womens"):
            r = welch_ttest(ctrl, d.loc[d.arm == arm, "spend"])
            out[f"{seg}:{arm}"] = {"n": int(mask.sum()), "diff": r.diff, "ci": r.ci, "p": r.pvalue}
    adjusted = benjamini_hochberg([r["p"] for r in out.values()])
    for r, p in zip(out.values(), adjusted, strict=True):
        r["p_bh"] = float(p)
    return out


def choose_campaign(spend: dict, cost_per_email: float) -> str:
    """Decide qué email enviar a la base completa: "mens", "womens" o "ninguno".

    `spend["mens_vs_control"]` y `spend["womens_vs_control"]` traen, entre otros:
    diff (USD incremental por cliente), cuped_ci, welch_p_holm.
    Un email es rentable si el gasto incremental supera `cost_per_email`.
    """
    # TODO: implementar la regla de decisión (5-10 líneas)
    raise NotImplementedError


def plot_forest(spend: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = list(spend)
    for i, name in enumerate(names):
        r = spend[name]
        for offset, key, label, color in [
            (0.12, "welch_ci", "Welch", "C0"),
            (-0.12, "cuped_ci", "CUPED", "C2"),
        ]:
            lo, hi = r[key]
            center = r["diff"] if key == "welch_ci" else r["cuped_diff"]
            ax.plot(
                [lo, hi], [i + offset] * 2, color=color, lw=2.5, label=label if i == 0 else None
            )
            ax.plot(center, i + offset, "o", color=color)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(COST_PER_EMAIL, color="C3", ls="--", lw=1, label=f"costo email (${COST_PER_EMAIL})")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("diferencia en gasto por cliente (USD), IC 95 %")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_bootstrap(spend: dict) -> plt.Figure:
    r = spend["mens_vs_control"]
    boot = r["_boot"]
    fig, ax = plt.subplots()
    ax.hist(boot, bins=60, density=True, color="C0", alpha=0.7, label="bootstrap")
    g = np.linspace(boot.min(), boot.max(), 200)
    se = (r["welch_ci"][1] - r["welch_ci"][0]) / (2 * 1.96)
    ax.plot(g, stats.norm.pdf(g, r["diff"], se), "C3", label="normal (Welch)")
    ax.set_xlabel("diferencia en gasto, mens − control (USD)")
    ax.set_title(f"Asimetría bootstrap = {stats.skew(boot):.2f}")
    ax.legend()
    return fig


def plot_posteriors(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    g = np.linspace(0.003, 0.016, 500)
    for arm in ARMS:
        d = df[df.arm == arm]
        post = beta_posterior(int(d.conversion.sum()), len(d))
        ax.plot(g, stats.beta.pdf(g, *post), label=ARMS[arm])
    ax.set_xlabel("tasa de conversión")
    ax.set_title("Posteriores Beta (prior uniforme)")
    ax.legend()
    return fig


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, tuple | list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> None:
    rng = set_seed(SEED)
    setup_style()
    df = load()

    validity = validity_checks(df)
    spend = analyze_spend(df, rng)
    binary = analyze_binary(df, rng)
    segments = analyze_segments(df)
    try:
        decision = choose_campaign(spend, COST_PER_EMAIL)
    except NotImplementedError:
        decision = "pendiente (implementar choose_campaign)"

    OUT.mkdir(exist_ok=True)
    report = {
        "validity": validity,
        "spend": spend,
        "binary": binary,
        "segments_exploratory": segments,
        "cost_per_email": COST_PER_EMAIL,
        "decision": decision,
    }
    (OUT / "metrics.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    save_fig(plot_forest(spend), OUT / "forest_gasto.png")
    save_fig(plot_bootstrap(spend), OUT / "bootstrap_gasto.png")
    save_fig(plot_posteriors(df), OUT / "posteriores_conversion.png")

    print(f"SRM: χ² = {validity['srm_chi2']:.2f}, p = {validity['srm_pvalue']:.3f}")
    max_smd = max(abs(v) for d in validity["smd"].values() for v in d.values())
    print(f"balance: max |SMD| = {max_smd:.3f}")
    print("\nGasto por cliente (principal):")
    for name, r in spend.items():
        print(
            f"  {name:<18} Δ = {r['diff']:+.3f}"
            f"  Welch IC ({r['welch_ci'][0]:+.3f}, {r['welch_ci'][1]:+.3f})"
            f"  boot IC ({r['bootstrap_ci'][0]:+.3f}, {r['bootstrap_ci'][1]:+.3f})"
            f"  p_holm = {r['welch_p_holm']:.4f}  p_perm = {r['permutation_p']:.4f}"
            f"  CUPED Δ = {r['cuped_diff']:+.3f} (var −{r['cuped_var_reduction']:.1%})"
        )
    print("\nSecundarias:")
    for name, r in binary.items():
        print(
            f"  {name:<28} {r['rate_a']:.4f} → {r['rate_b']:.4f}  p_bh = {r['p_bh']:.2e}"
            f"  P(B>A) = {r['prob_b_beats_a']:.3f}"
        )
    print("\nSegmentos (exploratorio, gasto):")
    for name, r in segments.items():
        print(f"  {name:<18} n = {r['n']:>6}  Δ = {r['diff']:+.3f}  p_bh = {r['p_bh']:.3f}")
    print(f"\nDecisión: {decision}")


if __name__ == "__main__":
    main()
