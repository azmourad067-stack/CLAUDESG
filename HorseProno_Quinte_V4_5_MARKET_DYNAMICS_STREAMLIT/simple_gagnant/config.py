"""Configuration documentée du moteur Simple Gagnant.

Tous les seuils sont mesurables et versionnés. Les valeurs par défaut des
seuils de décision ont été sélectionnées sur la période de validation
(12–30 septembre 2026, cotes snapshot pré-course, rapports définitifs PMU)
puis gelées avant l'évaluation de la période de test (1er–23 octobre 2026).
Voir reports/SG_RAPPORT_VALIDATION.md.

`validated_segments` liste les familles de disciplines pour lesquelles le test
hors échantillon a apporté une preuve suffisante (borne basse de l'IC à 90 %
du ROI > 0 avec au moins `min_test_bets` paris). Tant que cette liste est vide,
le statut « OPPORTUNITÉ VALIDÉE PAR LE MODÈLE » n'est jamais attribué.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUS_NO_BET = "NO BET — COURSE ÉCARTÉE"
STATUS_WATCH = "SURVEILLANCE"
STATUS_TEST = "CANDIDAT À TESTER"
STATUS_VALIDATED = "OPPORTUNITÉ VALIDÉE PAR LE MODÈLE"
STATUS_ORDER = [STATUS_NO_BET, STATUS_WATCH, STATUS_TEST, STATUS_VALIDATED]


@dataclass
class SelectorConfig:
    # Portes de qualité (course écartée si l'une échoue)
    min_runners: int = 5
    max_runners: int = 20
    booksum_min: float = 1.08
    booksum_max: float = 1.40
    max_missing_musique_share: float = 0.40
    supported_families: tuple[str, ...] = ("PLAT", "TROT", "OBSTACLE")

    # Fenêtre temporelle de décision (minutes avant le départ)
    decision_window_min: float = 2.0
    decision_window_max: float = 30.0
    early_limit: float = 90.0

    # Éligibilité du candidat
    p_min: float = 0.05
    odds_max: float = 30.0

    # Échelle de décision (EV par euro, rapport attendu corrigé de la dérive)
    ev_watch: float = 0.0
    ev_test: float = 0.10
    ev_validated: float = 0.10
    require_ev_low_positive: bool = True
    require_ev_drift_positive: bool = True

    # Preuves hors échantillon
    validated_segments: tuple[str, ...] = ()
    min_test_bets: int = 100
    evidence_note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["supported_families"] = list(self.supported_families)
        d["validated_segments"] = list(self.validated_segments)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "SelectorConfig":
        if not d:
            return cls()
        kwargs = dict(d)
        for key in ("supported_families", "validated_segments"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in kwargs.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "SelectorConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class EngineInfo:
    model_path: str = "models/sg_model_v1.json"
    selector_path: str = "models/sg_selector_v1.json"
    journal_path: str = "data/sg_journal.csv"
    notes: list[str] = field(default_factory=list)
