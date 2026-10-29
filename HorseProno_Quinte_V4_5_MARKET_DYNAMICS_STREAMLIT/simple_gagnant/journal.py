"""Journal d'audit des prédictions Simple Gagnant.

Règles :
- une photographie est écrite AVANT le départ et n'est jamais modifiée ;
- le règlement (rapport définitif PMU) est stocké dans un enregistrement
  séparé, relié par `snapshot_id` ;
- stockage local CSV en ajout seul + (option) Supabase via la Data API REST
  avec une clé serveur. Une clé de lecture n'est jamais utilisée pour écrire.

Tables Supabase attendues : voir supabase/sg_simple_gagnant_migration.sql.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .config import STATUS_TEST, STATUS_VALIDATED

PARIS = ZoneInfo("Europe/Paris")
BET_STATUSES = (STATUS_TEST, STATUS_VALIDATED)
DRIFT_MIN_RACES = 300

PREDICTION_COLUMNS = [
    "snapshot_id", "captured_at", "race_date", "meeting_number", "race_number", "race_id",
    "hippodrome", "discipline", "family", "scheduled_start", "minutes_to_start", "runners",
    "status", "reasons", "reason_text", "candidate_number", "candidate_name", "candidate_odds",
    "p_model", "p_low", "p_market", "fair_odds", "break_even_odds", "payout_factor",
    "ev", "ev_low", "ev_drift", "edge", "booksum", "entropy_norm", "kl_model_market",
    "arguments_for", "arguments_against", "alerts", "non_runners",
    "model_fingerprint", "model_version", "selector_json", "engine_version", "mode",
]

RUNNER_COLUMNS = [
    "snapshot_id", "horse_number", "horse_name", "decision_odds", "odds_reference", "odds_observed_at",
    "p_model", "p_low", "p_market", "p_fund", "ev", "model_rank", "market_rank",
    "recent_form", "jockey", "trainer", "weight", "draw", "age", "sex",
    "career_starts", "career_wins", "career_places", "career_earnings", "handicap_distance",
]

SETTLEMENT_COLUMNS = [
    "snapshot_id", "settled_at", "race_id", "result_available", "refunded", "winners",
    "candidate_number", "candidate_won", "candidate_dividend", "simulated_bet", "stake",
    "gross_return", "profit", "p_model_winner", "p_market_winner", "settlement_note",
]


def _clean(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, ensure_ascii=False, default=str)
    if v is pd.NaT:
        return None
    return v


def snapshot_id(race_id: str, captured_at: str, fingerprint: str) -> str:
    return hashlib.sha256(f"{race_id}|{captured_at}|{fingerprint}".encode()).hexdigest()[:24]


def build_snapshot(
    decision,
    race_pred: pd.DataFrame,
    card: pd.DataFrame,
    model_fingerprint: str,
    model_version: str,
    selector_cfg: dict,
    engine_version: str,
    minutes: float | None,
    non_runners: list[dict] | None = None,
    captured_at: datetime | None = None,
    mode: str = "OBSERVATION",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    captured_at = captured_at or datetime.now(PARIS)
    cap = captured_at.isoformat(timespec="seconds")
    first = card.iloc[0]
    sid = snapshot_id(decision.race_id, cap, model_fingerprint)
    c = decision.candidate or {}
    m = decision.metrics or {}
    row = {
        "snapshot_id": sid, "captured_at": cap,
        "race_date": str(first.get("race_date")), "meeting_number": int(first.get("meeting_number", 0) or 0),
        "race_number": int(first.get("race_number", 0) or 0), "race_id": decision.race_id,
        "hippodrome": first.get("hippodrome"), "discipline": first.get("discipline"), "family": m.get("family"),
        "scheduled_start": first.get("scheduled_start"), "minutes_to_start": minutes, "runners": m.get("runners"),
        "status": decision.status, "reasons": "|".join(decision.reasons), "reason_text": decision.reason_text,
        "candidate_number": c.get("horse_number"), "candidate_name": c.get("horse_name"),
        "candidate_odds": c.get("decision_odds"), "p_model": c.get("p_model"), "p_low": c.get("p_low"),
        "p_market": c.get("p_market"), "fair_odds": c.get("fair_odds"), "break_even_odds": c.get("break_even_odds"),
        "payout_factor": c.get("payout_factor"), "ev": c.get("ev"), "ev_low": c.get("ev_low"),
        "ev_drift": c.get("ev_drift"), "edge": c.get("edge"), "booksum": m.get("booksum"),
        "entropy_norm": m.get("entropy_norm"), "kl_model_market": m.get("kl_model_market"),
        "arguments_for": decision.arguments_for, "arguments_against": decision.arguments_against,
        "alerts": decision.alerts, "non_runners": non_runners or [],
        "model_fingerprint": model_fingerprint, "model_version": model_version,
        "selector_json": selector_cfg, "engine_version": engine_version, "mode": mode,
    }
    merged = race_pred.merge(
        card[[c for c in card.columns if c in RUNNER_COLUMNS or c == "horse_number"]].drop_duplicates("horse_number"),
        on="horse_number", how="left", suffixes=("", "_card"),
    )
    runners = []
    for _, r in merged.iterrows():
        runners.append({k: _clean(r.get(k)) for k in RUNNER_COLUMNS if k != "snapshot_id"} | {"snapshot_id": sid})
    return {k: _clean(row.get(k)) for k in PREDICTION_COLUMNS}, runners


def settle_snapshot(snapshot: dict[str, Any], sg_result: dict[str, Any], runner_rows: list[dict] | None = None,
                    settled_at: datetime | None = None, candidate_non_runner: bool = False) -> dict[str, Any]:
    """Règlement au rapport définitif e-Simple Gagnant (mise simulée de 1 €).

    Remboursement (mise rendue, profit nul) si le PMU rembourse le pari ou si
    le cheval sélectionné est devenu non-partant après la photographie.
    """
    settled_at = settled_at or datetime.now(PARIS)
    cand = snapshot.get("candidate_number")
    cand = int(float(cand)) if cand is not None and not (isinstance(cand, float) and math.isnan(cand)) else None
    winners = {int(k): float(v) for k, v in (sg_result.get("winners") or {}).items()}
    available = bool(sg_result.get("available"))
    refunded = bool(sg_result.get("refunded")) or candidate_non_runner
    is_bet = snapshot.get("status") in BET_STATUSES and cand is not None
    won = (cand in winners) if (cand is not None and available and not refunded) else None
    dividend = winners.get(cand) if won else None
    stake = 1.0 if is_bet else 0.0
    gross = profit = None
    if not available:
        note = "Rapport définitif indisponible : règlement en attente."
    elif refunded:
        note = "Non-partant après la photographie : mise remboursée." if candidate_non_runner else "Pari remboursé par le PMU."
        if is_bet:
            gross, profit = stake, 0.0
    elif is_bet:
        gross = dividend if won else 0.0
        profit = gross - stake
        note = "Gagnant : rapport définitif officiel." if won else "Sélection non gagnante."
    else:
        note = "Aucune mise simulée (statut non joué) ; résultat enregistré pour l'évaluation."
    pmw = pmk = None
    if runner_rows and winners and len(winners) == 1:
        w = next(iter(winners))
        for r in runner_rows:
            if int(r.get("horse_number") or -1) == w:
                pmw, pmk = r.get("p_model"), r.get("p_market")
    return {
        "snapshot_id": snapshot["snapshot_id"], "settled_at": settled_at.isoformat(timespec="seconds"),
        "race_id": snapshot.get("race_id"), "result_available": available, "refunded": refunded,
        "winners": json.dumps(winners), "candidate_number": cand, "candidate_won": won,
        "candidate_dividend": dividend, "simulated_bet": is_bet, "stake": stake, "gross_return": gross,
        "profit": profit, "p_model_winner": pmw, "p_market_winner": pmk, "settlement_note": note,
    }


# --------------------------------------------------------------------------- stockage local
class LocalJournal:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pred_path = self.root / "sg_predictions.csv"
        self.run_path = self.root / "sg_prediction_runners.csv"
        self.set_path = self.root / "sg_settlements.csv"

    @staticmethod
    def _append(path: Path, rows: list[dict], columns: list[str]) -> int:
        if not rows:
            return 0
        df = pd.DataFrame(rows).reindex(columns=columns)
        if path.exists():
            existing = pd.read_csv(path, usecols=["snapshot_id"] + (["horse_number"] if "horse_number" in columns else []),
                                   dtype=str)
            if "horse_number" in columns:
                keys = set(zip(existing["snapshot_id"], existing["horse_number"]))
                df = df[[(str(a), str(b)) not in keys for a, b in zip(df["snapshot_id"], df["horse_number"])]]
            else:
                df = df[~df["snapshot_id"].astype(str).isin(set(existing["snapshot_id"]))]
        if df.empty:
            return 0
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        return len(df)

    def write_snapshot(self, prediction: dict, runners: list[dict]) -> int:
        n = self._append(self.pred_path, [prediction], PREDICTION_COLUMNS)
        if n:
            self._append(self.run_path, runners, RUNNER_COLUMNS)
        return n

    def write_settlement(self, settlement: dict) -> int:
        return self._append(self.set_path, [settlement], SETTLEMENT_COLUMNS)

    def predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.pred_path) if self.pred_path.exists() else pd.DataFrame(columns=PREDICTION_COLUMNS)

    def runners(self) -> pd.DataFrame:
        return pd.read_csv(self.run_path) if self.run_path.exists() else pd.DataFrame(columns=RUNNER_COLUMNS)

    def settlements(self) -> pd.DataFrame:
        return pd.read_csv(self.set_path) if self.set_path.exists() else pd.DataFrame(columns=SETTLEMENT_COLUMNS)


# --------------------------------------------------------------------------- Supabase (REST)
class SupabaseJournal:
    """Écriture/lecture via la Data API. Écriture uniquement avec une clé serveur."""

    def __init__(self, url: str, read_key: str | None, write_key: str | None, timeout: int = 20):
        self.url = url.rstrip("/")
        self.read_key = read_key or write_key
        self.write_key = write_key
        self.timeout = timeout

    @classmethod
    def from_env(cls, getter=os.getenv) -> "SupabaseJournal | None":
        url = getter("SUPABASE_URL")
        read = getter("SUPABASE_PUBLISHABLE_KEY") or getter("SUPABASE_KEY")
        write = getter("SUPABASE_SECRET_KEY") or getter("SUPABASE_SERVICE_KEY")
        if not url or not (read or write):
            return None
        return cls(url, read, write)

    def _headers(self, key: str) -> dict[str, str]:
        return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "Accept": "application/json"}

    def _insert(self, table: str, rows: list[dict], on_conflict: str) -> int:
        if not rows:
            return 0
        if not self.write_key:
            raise PermissionError("Écriture Supabase refusée : aucune clé serveur configurée.")
        h = self._headers(self.write_key)
        h["Prefer"] = "resolution=ignore-duplicates,return=minimal"
        r = requests.post(f"{self.url}/rest/v1/{table}", headers=h, params={"on_conflict": on_conflict},
                          data=json.dumps(rows, default=str), timeout=self.timeout)
        r.raise_for_status()
        return len(rows)

    @staticmethod
    def _as_json(row: dict, keys: tuple[str, ...]) -> dict:
        out = dict(row)
        for k in keys:
            v = out.get(k)
            if isinstance(v, str):
                try:
                    out[k] = json.loads(v)
                except ValueError:
                    pass
        return out

    def write_snapshot(self, prediction: dict, runners: list[dict]) -> int:
        pred = self._as_json(prediction, ("arguments_for", "arguments_against", "alerts", "non_runners", "selector_json"))
        self._insert("sg_predictions", [pred], "snapshot_id")
        self._insert("sg_prediction_runners", runners, "snapshot_id,horse_number")
        return 1

    def write_settlement(self, settlement: dict) -> int:
        return self._insert("sg_settlements", [self._as_json(settlement, ("winners",))], "snapshot_id")

    def select(self, table: str, params: dict[str, str]) -> pd.DataFrame:
        if not self.read_key:
            return pd.DataFrame()
        r = requests.get(f"{self.url}/rest/v1/{table}", headers=self._headers(self.read_key),
                         params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data if isinstance(data, list) else [])


# --------------------------------------------------------------------------- suivi
def monitoring_summary(predictions: pd.DataFrame, settlements: pd.DataFrame) -> dict[str, Any]:
    """Indicateurs prospectifs : statuts, ROI simulé, log-loss modèle vs marché."""
    if predictions.empty:
        return {"snapshots": 0}
    p = predictions.copy()
    out: dict[str, Any] = {"snapshots": int(len(p)), "races": int(p["race_id"].nunique()),
                           "status_counts": p["status"].value_counts().to_dict()}
    if settlements.empty:
        out["settled"] = 0
        return out
    s = settlements.drop_duplicates("snapshot_id", keep="last")
    j = p.merge(s, on="snapshot_id", how="inner", suffixes=("", "_s"))
    j = j[j["result_available"].astype(str).str.lower().isin(["true", "1"])]
    out["settled"] = int(len(j))
    bets = j[j["simulated_bet"].astype(str).str.lower().isin(["true", "1"])]
    if len(bets):
        stake = pd.to_numeric(bets["stake"], errors="coerce").sum()
        profit = pd.to_numeric(bets["profit"], errors="coerce").sum()
        out["simulated_bets"] = int(len(bets))
        out["simulated_profit"] = float(profit)
        out["simulated_roi"] = float(profit / stake) if stake else float("nan")
        out["by_status"] = {
            st: {"bets": int(len(g)), "profit": float(pd.to_numeric(g["profit"], errors="coerce").sum())}
            for st, g in bets.groupby("status")
        }
    w = j.dropna(subset=["p_model_winner", "p_market_winner"])
    if len(w):
        pm = pd.to_numeric(w["p_model_winner"], errors="coerce").clip(lower=1e-9)
        pk = pd.to_numeric(w["p_market_winner"], errors="coerce").clip(lower=1e-9)
        gain = np.log(pm) - np.log(pk)
        out["log_loss_model"] = float(-np.log(pm).mean())
        out["log_loss_market"] = float(-np.log(pk).mean())
        out["log_loss_gain_per_race"] = float(gain.mean())
        out["log_loss_gain_se"] = float(gain.std(ddof=1) / math.sqrt(len(gain))) if len(gain) > 1 else float("nan")
        out["log_loss_races"] = int(len(w))
        # Alerte de dérive : le modèle fait moins bien que le marché sur un échantillon suffisant.
        out["drift_alert"] = bool(len(w) >= DRIFT_MIN_RACES and gain.mean() < 0)
    cw = j[j["candidate_won"].astype(str).str.lower().isin(["true", "false", "1", "0"])]
    if len(cw):
        won = cw["candidate_won"].astype(str).str.lower().isin(["true", "1"])
        div = pd.to_numeric(cw["candidate_dividend"], errors="coerce").fillna(0.0)
        out["shadow_all_candidates"] = {"races": int(len(cw)), "wins": int(won.sum()),
                                        "roi_if_all_played": float((np.where(won, div, 0.0) - 1.0).mean())}
    return out
