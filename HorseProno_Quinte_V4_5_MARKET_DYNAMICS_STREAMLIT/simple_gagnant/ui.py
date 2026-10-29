"""Mise en forme des résultats pour l'interface (aucune dépendance Streamlit)."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .config import STATUS_NO_BET, STATUS_TEST, STATUS_VALIDATED, STATUS_WATCH

STATUS_ICON = {
    STATUS_NO_BET: "⛔",
    STATUS_WATCH: "👀",
    STATUS_TEST: "🧪",
    STATUS_VALIDATED: "✅",
}


def _pct(x) -> float | None:
    try:
        v = float(x)
        return round(100.0 * v, 1) if math.isfinite(v) else None
    except Exception:
        return None


def _num(x, nd=2) -> float | None:
    try:
        v = float(x)
        return round(v, nd) if math.isfinite(v) else None
    except Exception:
        return None


def start_hhmm(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return pd.Timestamp(iso).tz_convert("Europe/Paris").strftime("%H:%M")
    except Exception:
        try:
            return pd.Timestamp(iso).strftime("%H:%M")
        except Exception:
            return ""


def decisions_table(analyses: Iterable) -> pd.DataFrame:
    rows = []
    for a in analyses:
        meta = a.meta or {}
        d = a.decision
        c = d.candidate if d else {}
        rows.append({
            "Heure": start_hhmm(meta.get("scheduled_start")),
            "Course": f"R{meta.get('meeting_number')}C{meta.get('race_number')}",
            "Hippodrome": meta.get("hippodrome"),
            "Discipline": meta.get("discipline"),
            "Partants": len(a.card) if a.card is not None else None,
            "T- (min)": _num(a.minutes_to_start, 0),
            "Statut": (STATUS_ICON.get(d.status, "") + " " + d.status) if d else "⚠️ ERREUR",
            "Candidat": (f"N°{c.get('horse_number')} {c.get('horse_name') or ''}".strip() if c else ""),
            "Cote obs.": _num(c.get("decision_odds")) if c else None,
            "P modèle %": _pct(c.get("p_model")) if c else None,
            "P marché %": _pct(c.get("p_market")) if c else None,
            "Cote juste": _num(c.get("fair_odds")) if c else None,
            "Cote d'équilibre": _num(c.get("break_even_odds")) if c else None,
            "EV %": _pct(c.get("ev")) if c else None,
            "EV prudente %": _pct(c.get("ev_low")) if c else None,
            "Motif": (d.reason_text if d else a.error) or "",
        })
    return pd.DataFrame(rows)


def runner_table(pred: pd.DataFrame) -> pd.DataFrame:
    if pred is None or pred.empty:
        return pd.DataFrame()
    p = pred.copy()
    out = pd.DataFrame({
        "N°": p["horse_number"].astype(int),
        "Cheval": p.get("horse_name", ""),
        "Musique": p.get("recent_form", ""),
        "Cote obs.": p["decision_odds"].round(2),
        "Cote réf.": pd.to_numeric(p.get("odds_reference"), errors="coerce").round(2) if "odds_reference" in p else np.nan,
        "P brute %": (100 * p["p_market_raw"]).round(1),
        "P marché (sans marge) %": (100 * p["p_market"]).round(1),
        "P fondamentale %": (100 * p["p_fund"]).round(1),
        "P modèle %": (100 * p["p_model"]).round(1),
        "P prudente %": (100 * p["p_low"]).round(1),
        "Écart modèle-marché pts": (100 * p["edge"]).round(1),
        "Cote juste": p["fair_odds"].round(2),
        "Facteur rapport": p["payout_factor"].round(3),
        "EV naïve %": (100 * p["ev_naive"]).round(1),
        "EV corrigée %": (100 * p["ev"]).round(1),
        "EV dérive défavorable %": (100 * p["ev_drift"]).round(1),
        "Rang modèle": p["model_rank"],
        "Rang marché": p["market_rank"],
    })
    return out.sort_values("P modèle %", ascending=False, na_position="last").reset_index(drop=True)


def odds_sensitivity(p_model: float, payout_factor: float, odds_grid: Iterable[float]) -> pd.DataFrame:
    """EV en fonction de la cote observée : sensibilité de l'opportunité."""
    rows = [{"Cote observée": o, "EV corrigée %": round(100 * (p_model * o * payout_factor - 1), 1)} for o in odds_grid]
    return pd.DataFrame(rows)


def status_summary(analyses: Iterable) -> dict[str, int]:
    out = {STATUS_NO_BET: 0, STATUS_WATCH: 0, STATUS_TEST: 0, STATUS_VALIDATED: 0, "ERREUR": 0}
    for a in analyses:
        if a.decision is None:
            out["ERREUR"] += 1
        else:
            out[a.decision.status] = out.get(a.decision.status, 0) + 1
    return out
