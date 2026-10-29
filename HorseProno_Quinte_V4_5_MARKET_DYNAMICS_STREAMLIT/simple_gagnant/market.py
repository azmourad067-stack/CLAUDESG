"""Probabilités de marché à partir des cotes PMU (pari mutuel).

- probabilité implicite brute : 1 / cote ;
- somme des probabilités brutes (booksum) : 1 + marge ; au PMU la marge
  reflète le prélèvement sur la masse (≈ 18 % observé sur l'historique) ;
- probabilité proportionnelle : brute / booksum ;
- méthode « puissance » : p_i = (1/O_i)^k avec k tel que Σ p_i = 1. Elle
  attribue davantage de marge aux outsiders (biais favori-outsider).

Les cotes affichées avant le départ ne sont pas le rapport payé : le rapport
définitif PMU dépend de la masse finale. Voir model.py (facteur de dérive).
"""
from __future__ import annotations

import math

import numpy as np


def clean_odds(odds) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    o = np.where(np.isfinite(o) & (o > 1.0), o, np.nan)
    return o


def booksum(odds) -> float:
    o = clean_odds(odds)
    if np.isnan(o).any() or len(o) == 0:
        return float("nan")
    return float(np.sum(1.0 / o))


def raw_implied(odds) -> np.ndarray:
    return 1.0 / clean_odds(odds)


def proportional_probabilities(odds) -> np.ndarray:
    raw = raw_implied(odds)
    if np.isnan(raw).any():
        return np.full_like(raw, np.nan)
    return raw / raw.sum()


def power_probabilities(odds, tol: float = 1e-10, max_iter: int = 200) -> tuple[np.ndarray, float]:
    """Retourne (probabilités, exposant k). k = 1 si la marge est nulle ou négative."""
    raw = raw_implied(odds)
    if np.isnan(raw).any() or len(raw) == 0:
        return np.full_like(raw, np.nan), float("nan")
    total = raw.sum()
    if total <= 1.0 + 1e-12:
        return raw / total, 1.0
    lo, hi = 1.0, 1.0
    while np.sum(raw ** hi) > 1.0:
        hi *= 2.0
        if hi > 1e4:
            return raw / total, float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if np.sum(raw ** mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = 0.5 * (lo + hi)
    p = raw ** k
    return p / p.sum(), float(k)


def fair_odds(probability: float) -> float:
    if probability is None or not math.isfinite(probability) or probability <= 0:
        return float("nan")
    return 1.0 / probability


def expected_value(probability: float, odds: float, payout_factor: float = 1.0) -> float:
    """EV par euro misé : p × cote × facteur − 1.

    `payout_factor` corrige l'écart attendu entre la cote observée à la décision
    et le rapport définitif effectivement payé en cas de victoire.
    """
    if probability is None or odds is None:
        return float("nan")
    if not (math.isfinite(probability) and math.isfinite(odds)) or odds <= 1.0:
        return float("nan")
    return probability * odds * payout_factor - 1.0


def break_even_odds(probability: float, payout_factor: float = 1.0) -> float:
    """Cote minimale observée à la décision pour que l'EV soit positive."""
    if probability is None or not math.isfinite(probability) or probability <= 0 or payout_factor <= 0:
        return float("nan")
    return 1.0 / (probability * payout_factor)
