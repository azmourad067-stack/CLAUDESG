"""Lecture de la musique PMU.

Format observé dans les données (validated_history.csv et flux PMU) :
- chaque performance = <résultat><discipline>, la plus récente en premier ;
  résultat : 1..9 = place, 0 = au-delà de la 9e place, lettre = incident
  (D disqualifié, A arrêté, T tombé, R rétrogradé, J dérobé, ...) ;
  discipline : a attelé, m monté, p plat, h haies, s steeple, c cross ;
- "(25)" marque le début des performances de la saison précédente.

La musique est publiée avant la course : elle ne contient jamais le résultat
de la course analysée (contrôle effectué sur l'historique : le premier jeton
d'une course correspond à la place de la course précédente du cheval dans 92 %
des cas, et ne révèle pas la place de la course courante).

Les taux sont régularisés (lissage bayésien) : un taux calculé sur une ou deux
courses est ramené vers une valeur de population, conformément à la règle
"ne jamais considérer un taux sur deux courses comme une preuve solide".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

TOKEN_RE = re.compile(r"\(\d{2,4}\)|(\d{1,2}|[A-Za-z])([A-Za-z])")

# Fenêtre de forme récente et lissage (valeurs figées, reproduites en SQL).
WINDOW = 5
PRIOR_STRENGTH = 2.0
PRIOR_SCORE = 0.30
PRIOR_WIN = 0.10
PRIOR_PLACE = 0.30
PRIOR_INCIDENT = 0.10

FAMILY_LETTERS = {
    "PLAT": {"p"},
    "TROT": {"a", "m"},
    "OBSTACLE": {"h", "s", "c"},
}

DISCIPLINE_FAMILY = {
    "PLAT": "PLAT",
    "ATTELE_AUTOSTART": "TROT",
    "ATTELE_VOLTE": "TROT",
    "TROT_MONTE": "TROT",
    "HAIES": "OBSTACLE",
    "STEEPLECHASE": "OBSTACLE",
    "CROSS_COUNTRY": "OBSTACLE",
}

MUSIQUE_FEATURES = [
    "mus_n",
    "mus_cur",
    "mus_missing",
    "mus_score5",
    "mus_last",
    "mus_win5",
    "mus_place5",
    "mus_inc5",
    "mus_samefam5",
]


@dataclass(frozen=True)
class Performance:
    result: str        # "1".."9", "0" ou lettre d'incident en majuscule
    discipline: str    # lettre en minuscule
    season: int        # 0 = saison en cours, 1 = précédente, ...

    @property
    def position(self) -> int | None:
        if self.result.isdigit():
            return int(self.result)
        return None

    @property
    def is_incident(self) -> bool:
        return not self.result.isdigit()

    @property
    def score(self) -> float:
        """1.0 pour une victoire, décroissant jusqu'à 1/9 pour une 9e place, 0 sinon."""
        pos = self.position
        if pos is None or pos <= 0 or pos >= 10:
            return 0.0
        return (10.0 - pos) / 9.0


def discipline_family(discipline: str | None) -> str:
    return DISCIPLINE_FAMILY.get(str(discipline or "").strip().upper(), "AUTRE")


def parse_musique(value: object) -> list[Performance]:
    text = "" if value is None else str(value)
    if text.lower() in {"nan", "none"}:
        text = ""
    text = re.sub(r"\s+", "", text)
    season = 0
    out: list[Performance] = []
    for match in TOKEN_RE.finditer(text):
        if match.group(1) is None:
            season += 1
            continue
        result = match.group(1).upper()
        if result.isdigit():
            result = str(int(result))
        out.append(Performance(result=result, discipline=match.group(2).lower(), season=season))
    return out


def _smoothed(total: float, count: int, prior: float) -> float:
    return (total + PRIOR_STRENGTH * prior) / (count + PRIOR_STRENGTH)


def musique_features(value: object, discipline: str | None) -> dict[str, float]:
    perfs = parse_musique(value)
    recent = perfs[:WINDOW]
    k = len(recent)
    fam_letters = FAMILY_LETTERS.get(discipline_family(discipline), set())
    if k == 0:
        return {
            "mus_n": 0.0,
            "mus_cur": 0.0,
            "mus_missing": 1.0,
            "mus_score5": PRIOR_SCORE,
            "mus_last": PRIOR_SCORE,
            "mus_win5": PRIOR_WIN,
            "mus_place5": PRIOR_PLACE,
            "mus_inc5": PRIOR_INCIDENT,
            "mus_samefam5": 0.0,
        }
    scores = [p.score for p in recent]
    wins = sum(1 for p in recent if p.position == 1)
    places = sum(1 for p in recent if p.position is not None and 1 <= p.position <= 3)
    incidents = sum(1 for p in recent if p.is_incident)
    same = sum(1 for p in recent if p.discipline in fam_letters)
    return {
        "mus_n": min(len(perfs), 10) / 10.0,
        "mus_cur": min(sum(1 for p in perfs if p.season == 0), 10) / 10.0,
        "mus_missing": 0.0,
        "mus_score5": _smoothed(sum(scores), k, PRIOR_SCORE),
        "mus_last": recent[0].score,
        "mus_win5": _smoothed(float(wins), k, PRIOR_WIN),
        "mus_place5": _smoothed(float(places), k, PRIOR_PLACE),
        "mus_inc5": _smoothed(float(incidents), k, PRIOR_INCIDENT),
        "mus_samefam5": same / k,
    }


def musique_frame(values: Iterable[object], disciplines: Iterable[object]):
    import pandas as pd

    rows = [musique_features(v, d) for v, d in zip(values, disciplines)]
    return pd.DataFrame(rows, columns=MUSIQUE_FEATURES)
