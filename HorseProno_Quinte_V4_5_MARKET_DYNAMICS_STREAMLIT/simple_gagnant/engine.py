"""Orchestration du moteur Simple Gagnant : PMU → fiche → modèle → décision → journal."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from . import pmu
from .config import SelectorConfig
from .journal import build_snapshot
from .model import SimpleGagnantModel
from .selector import RaceDecision, evaluate_race

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "sg_model_v1.json"
DEFAULT_SELECTOR = ROOT / "models" / "sg_selector_v1.json"


@dataclass
class RaceAnalysis:
    meta: dict[str, Any]
    card: pd.DataFrame
    predictions: pd.DataFrame
    decision: RaceDecision | None
    minutes_to_start: float | None
    non_runners: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class SimpleGagnantEngine:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, selector_path: str | Path = DEFAULT_SELECTOR):
        self.model_path = Path(model_path)
        raw = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.model = SimpleGagnantModel.from_dict(raw)
        self.fingerprint = raw.get("fingerprint") or self.model.fingerprint()
        if self.fingerprint != self.model.fingerprint():
            raise ValueError("Empreinte de l'artefact incohérente : fichier modifié après sa création.")
        self.selector = SelectorConfig.load(selector_path) if Path(selector_path).exists() else SelectorConfig()
        self.model_version = f"{self.model_path.stem}@{self.fingerprint[:12]}"

    def analyse_card(self, card: pd.DataFrame, minutes: float | None, meta: dict[str, Any] | None = None,
                     non_runners: list[dict] | None = None) -> RaceAnalysis:
        meta = meta or {}
        if card.empty:
            return RaceAnalysis(meta, card, pd.DataFrame(), None, minutes, non_runners or [],
                                error="Aucun partant exploitable dans le flux PMU.")
        pred = self.model.predict(card, decision_odds_col="odds")
        decision = evaluate_race(pred, self.selector, minutes)
        if non_runners:
            decision.alerts.append(
                "Non-partant(s) : " + ", ".join(f"N°{x['horse_number']}" for x in non_runners)
                + " — retirés du calcul ; le marché affiché peut encore s'ajuster.")
        missing_w = card["weight"].isna().mean() if "weight" in card else 1.0
        if meta.get("discipline") in ("PLAT", "HAIES", "STEEPLECHASE", "CROSS_COUNTRY") and missing_w > 0:
            decision.alerts.append(f"Poids manquant pour {missing_w:.0%} des partants.")
        return RaceAnalysis(meta, card, pred, decision, minutes, non_runners or [])

    def analyse_race(self, day: str, meta: dict[str, Any], now: datetime | None = None, session=None) -> RaceAnalysis:
        minutes = pmu.minutes_to_start(meta.get("scheduled_start"), now)
        try:
            payload = pmu.fetch_participants(day, meta["meeting_number"], meta["race_number"], session)
            card, nr = pmu.participants_card(payload, meta, observed_at=now)
        except Exception as exc:  # noqa: BLE001 - erreur isolée par course
            return RaceAnalysis(meta, pd.DataFrame(), pd.DataFrame(), None, minutes, [], error=f"{type(exc).__name__}: {exc}")
        return self.analyse_card(card, minutes, meta, nr)

    def snapshot_rows(self, analysis: RaceAnalysis, captured_at: datetime | None = None, mode: str = "OBSERVATION"):
        if analysis.decision is None or analysis.card.empty:
            return None, []
        return build_snapshot(
            analysis.decision, analysis.predictions, analysis.card, self.fingerprint, self.model_version,
            self.selector.to_dict(), __version__, analysis.minutes_to_start, analysis.non_runners,
            captured_at=captured_at, mode=mode,
        )

    def describe(self) -> dict[str, Any]:
        md = self.model.metadata
        return {
            "model_version": self.model_version,
            "fingerprint": self.fingerprint,
            "alpha": self.model.combination.alpha,
            "gamma": self.model.combination.gamma,
            "combination_fit": self.model.combination.fit_period,
            "odds_timing": self.model.combination.odds_timing,
            "payout_factors": dict(zip(["<3", "3-6", "6-12", "12-25", "25+"], self.model.payout.factors)),
            "stage1_train": f"{md.get('stage1_train_start')} → {md.get('stage1_train_end_exclusive')} (exclu)",
            "validation_period": md.get("validation_period"),
            "test_period": md.get("test_period_reserved"),
            "validated_segments": list(self.selector.validated_segments),
            "evidence_note": self.selector.evidence_note,
        }
