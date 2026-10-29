"""Sélection des courses et décision Simple Gagnant.

Pour chaque course, dans cet ordre :
1. portes de qualité (données, marché, discipline, horaire) → sinon NO BET ;
2. choix du meilleur candidat EV parmi les partants éligibles (probabilité
   minimale, cote maximale) — affiché même s'il n'est pas recommandé ;
3. échelle de décision mesurable :
   - EV < ev_watch                                   → NO BET
   - ev_watch ≤ EV < ev_test, ou cote trop précoce   → SURVEILLANCE
   - EV ≥ ev_test sans toutes les preuves            → CANDIDAT À TESTER
   - EV ≥ ev_validated, EV prudente (paramètres défavorables) ≥ 0,
     EV sous dérive défavorable de rapport ≥ 0, fenêtre horaire respectée,
     famille de discipline validée hors échantillon → OPPORTUNITÉ VALIDÉE

EV = p_modèle × cote observée × facteur de rapport − 1 (voir model.py).
Aucune course n'est jouée par défaut.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    STATUS_NO_BET,
    STATUS_TEST,
    STATUS_VALIDATED,
    STATUS_WATCH,
    SelectorConfig,
)

REASONS = {
    "COTES_INCOMPLETES": "Au moins un partant n'a pas de cote exploitable : marché incomplet.",
    "PELOTON_HORS_PERIMETRE": "Nombre de partants hors du périmètre validé.",
    "MARCHE_ANORMAL": "Somme des probabilités implicites anormale (cotes incohérentes ou figées).",
    "DISCIPLINE_NON_COUVERTE": "Discipline sans modèle fondamental entraîné.",
    "FORME_INSUFFISANTE": "Trop de partants sans musique exploitable : estimation fondamentale peu fiable.",
    "COURSE_PARTIE": "Course déjà partie : les cotes visibles ne sont plus des cotes pré-course.",
    "HORAIRE_INCONNU": "Heure de départ inconnue : impossible de dater la cote par rapport au départ.",
    "AUCUN_ELIGIBLE": "Aucun partant ne respecte les critères de probabilité et de cote.",
    "AVANTAGE_NON_DEMONTRE": "Espérance corrigée du rapport attendu négative ou nulle.",
    "EV_SOUS_SEUIL": "Espérance positive mais inférieure au seuil de test.",
    "COTE_PRECOCE": "Cote relevée trop tôt : elle peut encore évoluer fortement avant le départ.",
    "HORS_FENETRE": "Lecture hors de la fenêtre de décision validée (T-30 à T-2).",
    "EV_PRUDENTE_NEGATIVE": "L'avantage disparaît avec des paramètres de modèle défavorables (incertitude).",
    "DERIVE_DEFAVORABLE": "L'avantage disparaît si le rapport définitif baisse comme chez un quart des gagnants.",
    "SEGMENT_NON_VALIDE": "Aucune preuve hors échantillon suffisante pour cette famille de discipline.",
    "MODELE_NON_VALIDE": "Le modèle n'a pas démontré d'avantage hors échantillon : statut plafonné.",
}


@dataclass
class RaceDecision:
    race_id: str
    status: str
    reasons: list[str] = field(default_factory=list)
    candidate: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    arguments_for: list[str] = field(default_factory=list)
    arguments_against: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return " ".join(REASONS.get(r, r) for r in self.reasons)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reason_text"] = self.reason_text
        return d


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def race_metrics(r: pd.DataFrame) -> dict[str, Any]:
    p = pd.to_numeric(r.get("p_model"), errors="coerce").to_numpy(dtype=float)
    pm = pd.to_numeric(r.get("p_market"), errors="coerce").to_numpy(dtype=float)
    n = len(r)
    out: dict[str, Any] = {
        "runners": n,
        "odds_coverage": float(pd.to_numeric(r["decision_odds"], errors="coerce").notna().mean()) if n else 0.0,
        "musique_coverage": float(1.0 - pd.to_numeric(r.get("mus_missing", 0), errors="coerce").fillna(1).mean()) if n else 0.0,
        "booksum": _f(r["booksum"].iloc[0]) if n and "booksum" in r else float("nan"),
        "margin_pct": float("nan"),
    }
    if math.isfinite(out["booksum"]):
        out["margin_pct"] = 100.0 * (out["booksum"] - 1.0)
    if np.isfinite(p).all() and n:
        ps = np.sort(p)[::-1]
        out["entropy_norm"] = float(-(p * np.log(p)).sum() / math.log(n)) if n > 1 else 0.0
        out["top1_prob"] = float(ps[0])
        out["gap_top2"] = float(ps[0] - ps[1]) if n > 1 else float("nan")
        if np.isfinite(pm).all():
            out["kl_model_market"] = float((p * np.log(p / pm)).sum())
            out["max_abs_disagreement"] = float(np.abs(p - pm).max())
            out["rank_agreement_top1"] = bool(np.argmax(p) == np.argmax(pm))
    return out


def _explain(cand: pd.Series, race: pd.DataFrame) -> tuple[list[str], list[str]]:
    pro, con = [], []
    p, pm = _f(cand.get("p_model")), _f(cand.get("p_market"))
    if math.isfinite(p) and math.isfinite(pm):
        if p > pm:
            pro.append(f"Probabilité modèle {100*p:.1f} % > marché {100*pm:.1f} %.")
        else:
            con.append(f"Probabilité modèle {100*p:.1f} % ≤ marché {100*pm:.1f} %.")
    o, be = _f(cand.get("decision_odds")), _f(cand.get("break_even_odds"))
    if math.isfinite(o) and math.isfinite(be):
        if o > be:
            pro.append(f"Cote observée {o:.2f} au-dessus de la cote d'équilibre {be:.2f} (rapport attendu corrigé).")
        else:
            con.append(f"Cote observée {o:.2f} sous la cote d'équilibre {be:.2f}.")
    fac = _f(cand.get("payout_factor"))
    if math.isfinite(fac) and fac < 1.0:
        con.append(f"Les gagnants de cette tranche de cote ont été payés en moyenne {100*(1-fac):.0f} % sous la cote snapshot.")
    elif math.isfinite(fac) and fac > 1.0:
        pro.append(f"Les gagnants de cette tranche de cote ont été payés en moyenne {100*(fac-1):.0f} % au-dessus de la cote snapshot.")
    ms = pd.to_numeric(race.get("mus_score5"), errors="coerce")
    cms = _f(cand.get("mus_score5"))
    if math.isfinite(cms) and ms.notna().any():
        rank = int((ms > cms).sum()) + 1
        if _f(cand.get("mus_missing")) == 1.0:
            con.append("Musique absente : forme inconnue (débutant ou données manquantes).")
        elif rank <= max(2, len(race) // 4):
            pro.append(f"Forme récente (musique) parmi les meilleures du peloton (rang {rank}/{len(race)}).")
        elif rank > len(race) // 2:
            con.append(f"Forme récente (musique) dans la moitié basse du peloton (rang {rank}/{len(race)}).")
    if _f(cand.get("mus_inc5")) >= 0.35:
        con.append("Incidents fréquents dans les dernières courses (disqualifications/arrêts).")
    mr = _f(cand.get("market_rank"))
    if math.isfinite(mr):
        pro.append("Favori du marché." if mr == 1 else f"Rang marché : {int(mr)}.")
    el, ed = _f(cand.get("ev_low")), _f(cand.get("ev_drift"))
    if math.isfinite(el):
        (pro if el >= 0 else con).append(f"EV prudente (incertitude des paramètres) : {100*el:+.1f} %.")
    if math.isfinite(ed):
        (pro if ed >= 0 else con).append(f"EV si dérive défavorable du rapport : {100*ed:+.1f} %.")
    return pro, con


def evaluate_race(
    race: pd.DataFrame,
    cfg: SelectorConfig,
    minutes_to_start: float | None,
    model_validated: bool | None = None,
) -> RaceDecision:
    """`race` : sortie de SimpleGagnantModel.predict pour une seule course."""
    r = race.reset_index(drop=True)
    rid = str(r["race_id"].iloc[0]) if len(r) else ""
    dec = RaceDecision(race_id=rid, status=STATUS_NO_BET)
    dec.metrics = race_metrics(r)
    family = str(r["family"].iloc[0]) if len(r) and "family" in r else "AUTRE"
    dec.metrics["family"] = family
    dec.metrics["minutes_to_start"] = None if minutes_to_start is None else float(minutes_to_start)

    gates: list[str] = []
    if not len(r) or not bool(r["market_complete"].all()):
        gates.append("COTES_INCOMPLETES")
    if not (cfg.min_runners <= len(r) <= cfg.max_runners):
        gates.append("PELOTON_HORS_PERIMETRE")
    bk = dec.metrics.get("booksum", float("nan"))
    if "COTES_INCOMPLETES" not in gates and not (cfg.booksum_min <= bk <= cfg.booksum_max):
        gates.append("MARCHE_ANORMAL")
    if family not in cfg.supported_families:
        gates.append("DISCIPLINE_NON_COUVERTE")
    missing_share = 1.0 - dec.metrics.get("musique_coverage", 0.0)
    if missing_share > cfg.max_missing_musique_share:
        gates.append("FORME_INSUFFISANTE")
    if minutes_to_start is not None and minutes_to_start < 0:
        gates.append("COURSE_PARTIE")
    if minutes_to_start is None:
        dec.alerts.append(REASONS["HORAIRE_INCONNU"])

    # Meilleur candidat (affiché même si la course est écartée).
    elig = r[np.isfinite(pd.to_numeric(r.get("ev"), errors="coerce"))].copy()
    elig_strict = elig[(elig["p_model"] >= cfg.p_min) & (elig["decision_odds"] <= cfg.odds_max)]
    pool = elig_strict if not elig_strict.empty else elig
    cand = None
    if not pool.empty:
        cand = pool.sort_values(["ev", "p_model"], ascending=[False, False]).iloc[0]
        keys = [
            "horse_number", "horse_name", "decision_odds", "p_model", "p_low", "p_high", "p_market",
            "p_market_raw", "p_fund", "fair_odds", "break_even_odds", "payout_factor", "payout_factor_low",
            "ev_naive", "ev", "ev_low", "ev_drift", "edge", "model_rank", "market_rank",
        ]
        dec.candidate = {k: (cand[k].item() if hasattr(cand[k], "item") else cand[k]) for k in keys if k in cand.index}
        dec.candidate["eligible"] = bool(not elig_strict.empty and cand.name in elig_strict.index)
        dec.arguments_for, dec.arguments_against = _explain(cand, r)
        # Stabilité : le candidat reste-t-il le meilleur EV sous les scénarios de paramètres ?
        dec.metrics["candidate_is_model_top1"] = bool(_f(cand.get("model_rank")) == 1)

    if gates:
        dec.status = STATUS_NO_BET
        dec.reasons = gates
        return dec
    if cand is None or elig_strict.empty:
        dec.status = STATUS_NO_BET
        dec.reasons = ["AUCUN_ELIGIBLE"]
        return dec

    ev = _f(cand["ev"])
    if not (ev >= cfg.ev_watch) or ev <= 0:
        dec.status = STATUS_NO_BET
        dec.reasons = ["AVANTAGE_NON_DEMONTRE"]
        return dec

    early = minutes_to_start is not None and minutes_to_start > cfg.decision_window_max
    if ev < cfg.ev_test or early:
        dec.status = STATUS_WATCH
        dec.reasons = ["EV_SOUS_SEUIL"] if ev < cfg.ev_test else []
        if early:
            dec.reasons.append("COTE_PRECOCE")
        return dec

    missing: list[str] = []
    if minutes_to_start is None or not (cfg.decision_window_min <= minutes_to_start <= cfg.decision_window_max):
        missing.append("HORS_FENETRE")
    if cfg.require_ev_low_positive and not (_f(cand.get("ev_low")) >= 0):
        missing.append("EV_PRUDENTE_NEGATIVE")
    if cfg.require_ev_drift_positive and not (_f(cand.get("ev_drift")) >= 0):
        missing.append("DERIVE_DEFAVORABLE")
    if ev < cfg.ev_validated:
        missing.append("EV_SOUS_SEUIL")
    if family not in cfg.validated_segments:
        missing.append("SEGMENT_NON_VALIDE")
    if model_validated is False:
        missing.append("MODELE_NON_VALIDE")

    if missing:
        dec.status = STATUS_TEST
        dec.reasons = missing
    else:
        dec.status = STATUS_VALIDATED
        dec.reasons = []
    return dec


def evaluate_card(
    predictions: pd.DataFrame,
    cfg: SelectorConfig,
    minutes_by_race: dict[str, float | None] | None = None,
    model_validated: bool | None = None,
) -> list[RaceDecision]:
    out = []
    minutes_by_race = minutes_by_race or {}
    for rid, g in predictions.groupby("race_id", sort=False):
        out.append(evaluate_race(g, cfg, minutes_by_race.get(str(rid)), model_validated=model_validated))
    return out
