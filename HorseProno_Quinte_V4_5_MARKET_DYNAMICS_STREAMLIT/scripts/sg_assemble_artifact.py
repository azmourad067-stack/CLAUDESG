"""Assemble l'artefact Simple Gagnant à partir des estimations de validation.

Entrées :
- l'artefact de l'étape 1 (sg_train_stage1.py) ;
- la grille de log-vraisemblance (α, γ) calculée dans Supabase sur la période
  de validation (sg_forward_sql.py --mode grid), format "alpha:gamma:loglik" ;
- le résumé des rapports définitifs / cotes snapshot des gagnants
  (sg_forward_sql.py --mode payout), au format JSON.

L'optimum (α, γ) et sa covariance sont obtenus par ajustement d'une surface
quadratique autour du maximum (approximation de Laplace).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_gagnant.model import MarketCombination, PayoutFactor, SimpleGagnantModel  # noqa: E402


def fit_quadratic(grid_text: str, alpha_max: float | None = None, gamma_abs_max: float | None = None):
    pts = np.array([[float(x) for x in item.split(":")] for item in grid_text.split()])
    a, g, ll = pts.T
    mask = np.ones(len(a), dtype=bool)
    if alpha_max is not None:
        mask &= a <= alpha_max
    if gamma_abs_max is not None:
        mask &= np.abs(g) <= gamma_abs_max
    X = np.column_stack([np.ones(mask.sum()), a[mask], g[mask], a[mask] ** 2, g[mask] ** 2, a[mask] * g[mask]])
    c = np.linalg.lstsq(X, ll[mask], rcond=None)[0]
    H = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
    if np.any(np.linalg.eigvalsh(H) >= 0):
        raise ValueError("Surface non concave : élargir ou recentrer la grille.")
    opt = np.linalg.solve(H, -c[1:3])
    cov = np.linalg.inv(-H)
    ll_opt = float(c @ np.array([1, opt[0], opt[1], opt[0] ** 2, opt[1] ** 2, opt[0] * opt[1]]))
    market = [row for row in pts if abs(row[0] - 1.0) < 1e-12 and abs(row[1]) < 1e-12]
    ll_market = float(market[0][2]) if market else float("nan")
    resid = float(np.abs(X @ c - ll[mask]).max())
    return opt, cov, ll_opt, ll_market, resid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--payout", required=True)
    ap.add_argument("--races", type=int, required=True, help="nombre de courses de la grille")
    ap.add_argument("--validation", required=True, help="AAAA-MM-JJ:AAAA-MM-JJ")
    ap.add_argument("--test", required=True, help="AAAA-MM-JJ:AAAA-MM-JJ (réservée, non utilisée ici)")
    ap.add_argument("--alpha-max", type=float, default=1.15)
    ap.add_argument("--gamma-abs-max", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = SimpleGagnantModel.load(args.stage1)
    opt, cov, ll_opt, ll_market, resid = fit_quadratic(
        Path(args.grid).read_text(encoding="utf-8"), args.alpha_max, args.gamma_abs_max
    )
    model.combination = MarketCombination(
        alpha=float(opt[0]),
        gamma=float(opt[1]),
        cov=[[float(v) for v in row] for row in cov],
        n_races=args.races,
        fit_period=args.validation,
        odds_timing="snapshot PMU pré-course T-30..T-10 (médiane ≈ T-24)",
    )
    summary = json.loads(Path(args.payout).read_text(encoding="utf-8"))
    model.payout = PayoutFactor.from_summary(
        summary, source=f"rapports définitifs e-Simple Gagnant / cote snapshot, validation {args.validation}"
    )
    model.metadata.update({
        "validation_period": args.validation,
        "test_period_reserved": args.test,
        "stage2_loglik_gain_vs_market": ll_opt - ll_market,
        "stage2_loglik_gain_per_race": (ll_opt - ll_market) / max(args.races, 1),
        "stage2_quadratic_fit_max_residual": resid,
        "stage2_validation_races": args.races,
    })
    fp = model.save(args.out)
    print(json.dumps({
        "out": args.out, "fingerprint": fp,
        "alpha": opt[0], "gamma": opt[1], "se": np.sqrt(np.diag(cov)).tolist(),
        "loglik_gain": ll_opt - ll_market,
        "payout_factors": model.payout.factors, "payout_low_factors": model.payout.low_factors,
    }, indent=2))


if __name__ == "__main__":
    main()
