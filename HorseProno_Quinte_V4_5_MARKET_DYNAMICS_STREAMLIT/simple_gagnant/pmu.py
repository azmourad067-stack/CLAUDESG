"""Collecte PMU pour le moteur Simple Gagnant (sans dépendance Streamlit).

Endpoints utilisés (mêmes bases que streamlit_app.py ; format vérifié) :
- /programme/{JJMMAAAA}                          programme de la journée
- /programme/{JJMMAAAA}/R{r}/C{c}/participants   partants, musique, cotes
- /programme/{JJMMAAAA}/R{r}/C{c}/rapports-definitifs
      rapports officiels ; E_SIMPLE_GAGNANT → combinaison + dividendePourUnEuro
      (en centimes : 3120 = 31,20 € pour 1 €). Format vérifié le 29/10/2026 sur
      R1C1 du 20/10/2026 et sur les 2 421 réponses stockées dans Supabase.

Robustesse : chaque champ est lu par plusieurs clés possibles ; les valeurs
absentes restent manquantes (jamais inventées) ; les non-partants sont exclus
de la course et signalés.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
HEADERS = {"User-Agent": "HorsePronoSimpleGagnant/1.0", "Accept": "application/json"}
TIMEOUT = 25
PARIS = ZoneInfo("Europe/Paris")


# --------------------------------------------------------------------------- HTTP
def _get(path: str, session: requests.Session | None = None) -> Any:
    s = session or requests
    r = s.get(f"{PMU_BASE_URL}{path}", headers=HEADERS, params={"specialisation": "INTERNET"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _d(day: date | str) -> str:
    if isinstance(day, str):
        day = date.fromisoformat(day)
    return day.strftime("%d%m%Y")


def fetch_programme(day, session=None) -> Any:
    return _get(f"/programme/{_d(day)}", session)


def fetch_participants(day, reunion: int, course: int, session=None) -> Any:
    return _get(f"/programme/{_d(day)}/R{int(reunion)}/C{int(course)}/participants", session)


def fetch_rapports_definitifs(day, reunion: int, course: int, session=None) -> Any:
    return _get(f"/programme/{_d(day)}/R{int(reunion)}/C{int(course)}/rapports-definitifs", session)


# --------------------------------------------------------------------------- utilitaires
def _safe_int(v) -> int | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(float(v))
    except Exception:
        return None


def _safe_float(v) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _iter_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _collect_strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _collect_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _collect_strings(v)]
    return []


def _person(v) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        for k in ("nom", "nomCourt", "libelle", "libelleCourt", "name"):
            if v.get(k):
                return str(v[k]).strip()
    return ""


# --------------------------------------------------------------------------- programme
def _find_reunions(obj) -> list[dict]:
    if isinstance(obj, dict):
        if isinstance(obj.get("reunions"), list):
            return [x for x in obj["reunions"] if isinstance(x, dict)]
        for v in obj.values():
            found = _find_reunions(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_reunions(v)
            if found:
                return found
    return []


def start_datetime(course_obj: dict) -> datetime | None:
    for key in ("heureDepart", "heureDepartCourse", "dateHeureDepart"):
        v = course_obj.get(key)
        x = _safe_float(v)
        if x is not None and x > 10_000_000_000:
            return datetime.fromtimestamp(x / 1000.0, tz=PARIS)
        if isinstance(v, str) and v.strip():
            try:
                ts = pd.Timestamp(v)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(PARIS)
                return ts.to_pydatetime()
            except Exception:
                continue
    return None


def normalize_discipline(obj: Any) -> str:
    text = " ".join(_collect_strings(obj)).upper()
    if "CROSS" in text:
        return "CROSS_COUNTRY"
    if "STEEPLE" in text:
        return "STEEPLECHASE"
    if "HAIE" in text:
        return "HAIES"
    if "MONTE" in text or "MONTÉ" in text:
        return "TROT_MONTE"
    if "ATTELE" in text or "ATTELÉ" in text:
        return "ATTELE_AUTOSTART" if ("AUTOSTART" in text or "AUTO-START" in text) else "ATTELE_VOLTE"
    if "PLAT" in text:
        return "PLAT"
    return str(obj.get("discipline") or "INCONNU").upper() if isinstance(obj, dict) else "INCONNU"


EXPLICIT_DISCIPLINES = {
    "PLAT": "PLAT", "HAIE": "HAIES", "HAIES": "HAIES", "STEEPLECHASE": "STEEPLECHASE",
    "STEEPLE": "STEEPLECHASE", "CROSS": "CROSS_COUNTRY", "CROSS_COUNTRY": "CROSS_COUNTRY",
    "MONTE": "TROT_MONTE", "TROT_MONTE": "TROT_MONTE",
}


def _course_discipline(course_obj: dict) -> str:
    """Le champ explicite `discipline` prime sur le texte libre (un libellé comme
    « Prix du Plateau » ne doit pas classer la course en plat)."""
    explicit = str(course_obj.get("discipline") or "").strip().upper()
    if explicit in EXPLICIT_DISCIPLINES:
        return EXPLICIT_DISCIPLINES[explicit]
    if explicit.startswith("ATTELE"):
        text = " ".join(_collect_strings(course_obj)).upper()
        return "ATTELE_AUTOSTART" if ("AUTOSTART" in text or "AUTO-START" in text) else "ATTELE_VOLTE"
    return normalize_discipline(course_obj)


def programme_frame(programme: Any, day) -> pd.DataFrame:
    root = programme.get("programme", programme) if isinstance(programme, dict) else programme
    rows = []
    for reunion_obj in _find_reunions(root):
        r = _safe_int(reunion_obj.get("numOfficiel") or reunion_obj.get("numReunion") or reunion_obj.get("numero"))
        if r is None:
            continue
        hip = reunion_obj.get("hippodrome")
        hip = (hip.get("libelleCourt") or hip.get("libelleLong") or "INCONNU") if isinstance(hip, dict) else (hip or "INCONNU")
        for c_obj in reunion_obj.get("courses") or []:
            if not isinstance(c_obj, dict):
                continue
            c = _safe_int(c_obj.get("numOrdre") or c_obj.get("numCourse") or c_obj.get("numero"))
            if c is None:
                continue
            start = start_datetime(c_obj)
            dist = _safe_int(c_obj.get("distance"))
            rows.append({
                "race_date": str(day), "meeting_number": r, "race_number": c, "hippodrome": str(hip),
                "label": str(c_obj.get("libelle") or c_obj.get("libelleCourt") or f"Course {c}"),
                "discipline": _course_discipline(c_obj),
                "distance": dist if dist and 500 <= dist <= 10000 else None,
                "declared_runners": _safe_int(c_obj.get("nombreDeclaresPartants")),
                "scheduled_start": start.isoformat() if start else None,
                "race_status": str(c_obj.get("statut") or c_obj.get("categorieStatut") or ""),
                "results_available": bool(c_obj.get("rapportsDefinitifsDisponibles")),
            })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).drop_duplicates(["race_date", "meeting_number", "race_number"])
            .sort_values(["meeting_number", "race_number"]).reset_index(drop=True))


# --------------------------------------------------------------------------- partants
def _report(p: dict, key: str) -> dict[str, Any]:
    v = p.get(key)
    if not isinstance(v, dict):
        return {"odds": _safe_float(v) if _safe_float(v) and _safe_float(v) > 1 else None, "trend": "", "date": None}
    odds = None
    for k in ("rapport", "cote", "valeur", "value"):
        x = _safe_float(v.get(k))
        if x is not None and x > 1:
            odds = x
            break
    return {"odds": odds, "trend": str(v.get("indicateurTendance") or ""), "date": v.get("dateRapport")}


def _weight(p: dict) -> float | None:
    for key in ("handicapPoids", "poidsConditionMonte", "poidsMontureSurchargeDecharge", "poids"):
        x = _safe_float(p.get(key))
        if x is None or x <= 0:
            continue
        if x > 200:
            x /= 10.0
        if 30 <= x <= 90:
            return x
    return None


def _sex(v) -> str:
    t = str(v or "").strip().upper()
    if t in {"F", "FEMELLE", "FEMELLES", "JUMENT"}:
        return "FEMELLES"
    if t in {"H", "HONGRE", "HONGRES"}:
        return "HONGRES"
    if t in {"M", "MALE", "MÂLE", "MALES", "MÂLES", "ENTIER"}:
        return "MALES"
    return t


def is_non_runner(p: dict) -> bool:
    if p.get("nonPartant") is True:
        return True
    status = str(p.get("statut") or "").upper()
    return any(tok in status for tok in ("NON_PARTANT", "NON PARTANT", "NON-PARTANT"))


def _find_participants(payload) -> list[dict]:
    if isinstance(payload, dict):
        if isinstance(payload.get("participants"), list):
            return [p for p in payload["participants"] if isinstance(p, dict)]
        for v in payload.values():
            found = _find_participants(v)
            if found:
                return found
    elif isinstance(payload, list):
        for v in payload:
            found = _find_participants(v)
            if found:
                return found
    return []


def participants_card(
    payload: Any,
    race_meta: dict[str, Any],
    observed_at: datetime | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fiche de course normalisée (partants seulement) + liste des non-partants.

    `race_meta` : race_date, meeting_number, race_number, discipline, distance,
    hippodrome, scheduled_start (ISO). Les cotes sont horodatées par `observed_at`.
    """
    observed_at = observed_at or datetime.now(PARIS)
    rid = f"R{int(race_meta['meeting_number'])}C{int(race_meta['race_number'])}_{race_meta['race_date']}"
    rows, non_runners = [], []
    for p in _find_participants(payload):
        num = _safe_int(p.get("numPmu") or p.get("numero"))
        if num is None:
            continue
        direct, ref = _report(p, "dernierRapportDirect"), _report(p, "dernierRapportReference")
        row = {
            "race_id": rid,
            "race_date": race_meta["race_date"],
            "meeting_number": int(race_meta["meeting_number"]),
            "race_number": int(race_meta["race_number"]),
            "hippodrome": race_meta.get("hippodrome"),
            "discipline": race_meta.get("discipline"),
            "distance": race_meta.get("distance"),
            "scheduled_start": race_meta.get("scheduled_start"),
            "horse_number": num,
            "horse_name": str(p.get("nom") or f"N°{num}").strip(),
            "jockey": _person(p.get("driver")) or _person(p.get("jockey")),
            "trainer": _person(p.get("entraineur")),
            "draw": _safe_int(p.get("placeCorde")),
            "weight": _weight(p),
            "age": _safe_int(p.get("age")),
            "sex": _sex(p.get("sexe")),
            "recent_form": str(p.get("musique") or ""),
            "odds": direct["odds"] or ref["odds"],
            "odds_direct": direct["odds"],
            "odds_reference": ref["odds"],
            "trend_indicator": direct["trend"] or ref["trend"],
            "odds_observed_at": observed_at.isoformat(),
            # Variables pré-course collectées pour de futurs modèles (absentes de l'historique actuel).
            "career_starts": _safe_int(p.get("nombreCourses")),
            "career_wins": _safe_int(p.get("nombreVictoires")),
            "career_places": _safe_int(p.get("nombrePlaces")),
            "career_earnings": _safe_float((p.get("gainsParticipant") or {}).get("gainsCarriere")
                                           if isinstance(p.get("gainsParticipant"), dict) else p.get("gainsParticipant")),
            "handicap_distance": _safe_int(p.get("handicapDistance")),
            "blinkers": str(p.get("oeilleres") or ""),
            "driver_change": bool(p.get("driverChange")) if p.get("driverChange") is not None else None,
            "first_time_out": bool(p.get("indicateurInedit")) if p.get("indicateurInedit") is not None else None,
        }
        if is_non_runner(p):
            non_runners.append({"horse_number": num, "horse_name": row["horse_name"], "statut": p.get("statut")})
            continue
        rows.append(row)
    card = pd.DataFrame(rows)
    if not card.empty:
        if card["horse_number"].duplicated().any():
            raise ValueError("Numéro de partant en double dans le flux PMU.")
        card["field_size"] = len(card)
        card = card.sort_values("horse_number").reset_index(drop=True)
    return card, non_runners


# --------------------------------------------------------------------------- rapports
def simple_gagnant_result(payload: Any) -> dict[str, Any]:
    """Résultat officiel Simple Gagnant.

    Retour : {"available": bool, "refunded": bool, "winners": {numéro: dividende €/1€}}.
    Plusieurs gagnants = dead-heat (chaque cheval a son propre rapport).
    """
    out = {"available": False, "refunded": False, "winners": {}}
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if not isinstance(item, dict) or str(item.get("typePari")) != "E_SIMPLE_GAGNANT":
            continue
        out["available"] = True
        out["refunded"] = bool(item.get("rembourse"))
        for r in item.get("rapports") or []:
            comb = r.get("combinaison")
            num = comb[0] if isinstance(comb, list) and comb else comb
            n = _safe_int(num) if str(num).strip().isdigit() else None
            div = _safe_float(r.get("dividendePourUnEuro"))
            if n is not None and div is not None:
                out["winners"][n] = div / 100.0
    return out


def minutes_to_start(scheduled_start: str | None, now: datetime | None = None) -> float | None:
    if not scheduled_start:
        return None
    try:
        start = pd.Timestamp(scheduled_start)
        if start.tzinfo is None:
            start = start.tz_localize(PARIS)
        now = now or datetime.now(PARIS)
        return (start.to_pydatetime() - now).total_seconds() / 60.0
    except Exception:
        return None


def races_in_window(programme: pd.DataFrame, lo_minutes: float, hi_minutes: float,
                    now: datetime | None = None) -> pd.DataFrame:
    if programme.empty:
        return programme
    mts = programme["scheduled_start"].map(lambda s: minutes_to_start(s, now))
    return programme.assign(minutes_to_start=mts)[(mts >= lo_minutes) & (mts <= hi_minutes)]


def iter_cards(day, races: Iterable[dict[str, Any]], session=None):
    """Itère (race_meta, card, non_runners, erreur) en isolant les erreurs par course."""
    for meta in races:
        try:
            payload = fetch_participants(day, meta["meeting_number"], meta["race_number"], session)
            card, nr = participants_card(payload, meta)
            yield meta, card, nr, None
        except Exception as exc:  # noqa: BLE001 - une course en erreur ne bloque pas les autres
            yield meta, pd.DataFrame(), [], f"{type(exc).__name__}: {exc}"
