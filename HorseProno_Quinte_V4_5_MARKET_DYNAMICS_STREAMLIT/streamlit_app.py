from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

from quinte_v2 import load_artifact
from quinte_v3 import load_ranker, V3_CONFIG
from quinte_v42 import V42_CONFIG
from quinte_v43 import load_outsider_artifact
from quinte_v44 import rank_quinte_v44
from quinte_v45 import apply_v45_market_dynamics, V45_CONFIG

# =============================================================================
# CONFIG
# =============================================================================

APP_NAME = "HorseProno Quinté V4.5 — Market Dynamics"
PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
REQUEST_TIMEOUT = 25
HEADERS = {
    "User-Agent": "HorsePronoQuinteV45/4.5 (+https://streamlit.io)",
    "Accept": "application/json",
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "quinte_v2_artifact.json"
RANKER_PATH = BASE_DIR / "quinte_v2_ranker.txt"
V4_REPORT_PATH = BASE_DIR / "quinte_v42_candidate_report.json"
OUTSIDER_ARTIFACT_PATH = BASE_DIR / "v43_outsider_artifact.json"
BASE_HISTORY_PATH = BASE_DIR / "validated_history.csv"

HISTORY_COLUMNS = [
    "race_id", "race_date", "discipline", "hippodrome", "distance",
    "terrain", "field_size", "horse_number", "horse_name", "jockey",
    "trainer", "odds", "draw", "weight", "age", "sex", "recent_form",
    "finish_position", "is_non_runner",
]

st.set_page_config(page_title=APP_NAME, page_icon="🏇", layout="wide")
st.title("🏇 HorseProno Quinté V4.5 — Market Dynamics")
st.caption(
    "Le programme PMU complet de la journée est chargé automatiquement. "
    "V4.5 conserve exactement le Top 7 V4.2/V4.3. En PLAT, elle part du classement Winner Top3 V4.4 "
    "et ajoute un coefficient expérimental de dynamique des cotes pour projeter le marché vers T-2. "
    "Aucune cote post-course n'est utilisée pour calibrer le signal live."
)


# =============================================================================
# OUTILS GENERAUX
# =============================================================================

def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _secret(name: str) -> str | None:
    try:
        if name in st.secrets and st.secrets[name]:
            return str(st.secrets[name])
    except Exception:
        pass
    value = os.getenv(name)
    return str(value) if value else None


def _iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_dicts(value)


def _collect_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_collect_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_collect_strings(value))
    return out


def _person_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("nom", "nomCourt", "libelle", "libelleCourt", "name"):
            if value.get(key):
                return str(value[key]).strip()
    return ""


# =============================================================================
# PMU : PROGRAMME COMPLET + SELECTION MANUELLE
# =============================================================================

@st.cache_data(ttl=10 * 60, show_spinner=False)
def get_programme(day_iso: str) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")
    response = requests.get(
        f"{PMU_BASE_URL}/programme/{d}",
        headers=HEADERS,
        params={"specialisation": "INTERNET"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10 * 60, show_spinner=False)
def get_course_detail(day_iso: str, reunion: int, course: int) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")
    response = requests.get(
        f"{PMU_BASE_URL}/programme/{d}/R{reunion}/C{course}",
        headers=HEADERS,
        params={"specialisation": "INTERNET"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=5 * 60, show_spinner=False)
def get_participants(day_iso: str, reunion: int, course: int) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")
    response = requests.get(
        f"{PMU_BASE_URL}/programme/{d}/R{reunion}/C{course}/participants",
        headers=HEADERS,
        params={"specialisation": "INTERNET"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _find_reunions(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        reunions = obj.get("reunions")
        if isinstance(reunions, list):
            return [x for x in reunions if isinstance(x, dict)]
        for value in obj.values():
            found = _find_reunions(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_reunions(value)
            if found:
                return found
    return []


def _meeting_number(reunion_obj: dict) -> int | None:
    return _safe_int(
        reunion_obj.get("numOfficiel")
        or reunion_obj.get("numReunion")
        or reunion_obj.get("numReunionProgramme")
        or reunion_obj.get("numero")
    )


def _course_number(course_obj: dict) -> int | None:
    return _safe_int(
        course_obj.get("numOrdre")
        or course_obj.get("numCourse")
        or course_obj.get("numOfficiel")
        or course_obj.get("numero")
    )


def _hippodrome(reunion_obj: dict) -> str:
    value = reunion_obj.get("hippodrome")
    if isinstance(value, dict):
        return str(
            value.get("libelleCourt")
            or value.get("libelleLong")
            or value.get("nom")
            or "INCONNU"
        )
    return str(value or "INCONNU")


def _course_label(course_obj: dict, course: int) -> str:
    return str(
        course_obj.get("libelle")
        or course_obj.get("nom")
        or course_obj.get("libelleCourt")
        or f"Course {course}"
    )


def exact_quinte_codes(course_obj: dict) -> list[str]:
    """Détection stricte du support Quinté officiel PMU."""
    found: set[str] = set()
    for obj in _iter_dicts(course_obj):
        for key in ("codePari", "typePari", "code"):
            value = obj.get(key)
            if value is None:
                continue
            code = str(value).strip().upper()
            if code == "E_QUINTE_PLUS":
                found.add(code)
    return sorted(found)


def _course_start_time(course_obj: dict) -> str:
    """Retourne une heure HH:MM quand le programme PMU la fournit."""
    for key in (
        "heureDepart",
        "heureDepartCourse",
        "dateHeureDepart",
        "heure",
    ):
        value = course_obj.get(key)
        if value is None:
            continue

        # Le flux PMU peut fournir un timestamp en millisecondes.
        numeric = _safe_float(value)
        if numeric is not None and numeric > 10_000_000_000:
            try:
                dt = datetime.fromtimestamp(numeric / 1000.0, tz=ZoneInfo("Europe/Paris"))
                return dt.strftime("%H:%M")
            except Exception:
                pass

        if isinstance(value, str):
            text = value.strip()
            # ISO / date-heure.
            try:
                dt = pd.to_datetime(text, errors="raise")
                if not pd.isna(dt):
                    return dt.strftime("%H:%M")
            except Exception:
                pass
            # Valeur déjà sous forme HH:MM[:SS].
            if len(text) >= 5 and ":" in text:
                return text[:5]

    return ""


def _course_program_distance(course_obj: dict) -> int | None:
    for key in ("distance", "distanceMetres", "distanceM"):
        value = _safe_int(course_obj.get(key))
        if value is not None and 500 <= value <= 10000:
            return value
    return None


def _course_program_field_size(course_obj: dict) -> int | None:
    for key in (
        "nombreDeclaresPartants",
        "nombrePartants",
        "nbPartants",
        "fieldSize",
    ):
        value = _safe_int(course_obj.get(key))
        if value is not None and value >= 0:
            return value
    return None


@st.cache_data(ttl=10 * 60, show_spinner=False)
def discover_daily_program(day_iso: str) -> pd.DataFrame:
    """Charge toutes les réunions et toutes les courses du programme PMU."""
    programme = get_programme(day_iso)
    root = programme.get("programme") if isinstance(programme, dict) else programme
    if not isinstance(root, dict):
        root = programme

    rows: list[dict] = []
    for reunion_obj in _find_reunions(root):
        reunion = _meeting_number(reunion_obj)
        if reunion is None:
            continue

        hippodrome = _hippodrome(reunion_obj)
        courses = reunion_obj.get("courses")
        if not isinstance(courses, list):
            continue

        for course_obj in courses:
            if not isinstance(course_obj, dict):
                continue
            course = _course_number(course_obj)
            if course is None:
                continue

            quinte_codes = exact_quinte_codes(course_obj)
            rows.append(
                {
                    "race_date": day_iso,
                    "meeting_number": reunion,
                    "race_number": course,
                    "hippodrome": hippodrome,
                    "label": _course_label(course_obj, course),
                    "start_time": _course_start_time(course_obj),
                    "distance_m": _course_program_distance(course_obj),
                    "field_size_pmu": _course_program_field_size(course_obj),
                    "is_quinte": bool(quinte_codes),
                    "quinte_codes": " | ".join(quinte_codes),
                }
            )

    if not rows:
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows)
        .drop_duplicates(["race_date", "meeting_number", "race_number"])
        .sort_values(["meeting_number", "race_number"])
        .reset_index(drop=True)
    )
    return result


# =============================================================================
# PMU : CONVERSION DES PARTANTS VERS LE FORMAT DU MODELE
# =============================================================================

def _extract_odds(participant: dict) -> float | None:
    for key in (
        "dernierRapportDirect",
        "dernierRapportReference",
        "rapportDirect",
        "rapportReference",
    ):
        value = participant.get(key)
        if isinstance(value, dict):
            for subkey in ("rapport", "cote", "valeur", "value"):
                odds = _safe_float(value.get(subkey))
                if odds is not None and odds > 1:
                    return odds
        else:
            odds = _safe_float(value)
            if odds is not None and odds > 1:
                return odds

    for key in ("odds", "cote", "rapport"):
        odds = _safe_float(participant.get(key))
        if odds is not None and odds > 1:
            return odds
    return None


def _extract_report_info(participant: dict, key: str) -> dict[str, Any]:
    value = participant.get(key)
    if not isinstance(value, dict):
        return {"odds": None, "trend": "", "trend_strength": None, "date": None}
    odds = None
    for subkey in ("rapport", "cote", "valeur", "value"):
        odds = _safe_float(value.get(subkey))
        if odds is not None and odds > 1:
            break
        odds = None
    trend = str(value.get("indicateurTendance") or value.get("tendance") or "").strip()
    strength = _safe_float(value.get("nombreIndicateurTendance") or value.get("forceTendance"))
    report_date = value.get("dateRapport") or value.get("date")
    return {"odds": odds, "trend": trend, "trend_strength": strength, "date": report_date}


def _extract_weight(participant: dict) -> float | None:
    for key in (
        "handicapPoids",
        "poidsConditionMonte",
        "poidsMontureSurchargeDecharge",
        "poids",
    ):
        value = _safe_float(participant.get(key))
        if value is None or value <= 0:
            continue
        # Certains flux historiques expriment les poids au dixième de kg.
        if value > 200:
            value /= 10.0
        if 30 <= value <= 90:
            return value
    return None


def _normalize_sex(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"F", "FEMELLE", "FEMELLES", "JUMENT"}:
        return "FEMELLES"
    if text in {"H", "HONGRE", "HONGRES"}:
        return "HONGRES"
    if text in {"M", "MALE", "MÂLE", "MALES", "MÂLES", "ENTIER"}:
        return "MALES"
    return text


def _is_non_runner(participant: dict) -> bool:
    status = str(participant.get("statut") or "").strip().upper()
    explicit = participant.get("nonPartant")
    if explicit is True:
        return True
    return any(token in status for token in ("NON_PARTANT", "NON PARTANT", "NON-PARTANT"))


def normalize_discipline(course_detail: dict) -> str:
    text = " ".join(_collect_strings(course_detail)).upper()

    if "CROSS" in text:
        return "CROSS_COUNTRY"
    if "STEEPLE" in text:
        return "STEEPLECHASE"
    if "HAIE" in text:
        return "HAIES"
    if "MONTE" in text or "MONTÉ" in text:
        return "TROT_MONTE"
    if "ATTELE" in text or "ATTELÉ" in text:
        if "AUTOSTART" in text or "AUTO-START" in text:
            return "ATTELE_AUTOSTART"
        return "ATTELE_VOLTE"
    if "PLAT" in text:
        return "PLAT"

    discipline = str(course_detail.get("discipline") or "").strip().upper()
    return discipline or "INCONNU"


def _course_distance(course_detail: dict) -> int | None:
    for obj in _iter_dicts(course_detail):
        for key in ("distance", "distanceMetres", "distanceM"):
            value = _safe_int(obj.get(key))
            if value is not None and 500 <= value <= 10000:
                return value
    return None


def _find_participant_list(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        participants = payload.get("participants")
        if isinstance(participants, list):
            return [p for p in participants if isinstance(p, dict)]
        for value in payload.values():
            found = _find_participant_list(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_participant_list(value)
            if found:
                return found
    return []


def participants_to_frame(
    day_iso: str,
    reunion: int,
    course: int,
    hippodrome: str,
    course_detail: dict,
    participant_payload: dict,
) -> pd.DataFrame:
    participants = _find_participant_list(participant_payload)
    if not participants:
        raise RuntimeError("Le flux PMU ne contient aucun participant exploitable.")

    discipline = normalize_discipline(course_detail)
    distance = _course_distance(course_detail)
    race_id = f"R{reunion}C{course}_{day_iso}"

    rows: list[dict] = []
    for p in participants:
        number = _safe_int(p.get("numPmu") or p.get("numero") or p.get("num"))
        if number is None:
            continue

        horse_name = str(p.get("nom") or p.get("nomCheval") or "").strip()
        jockey = _person_name(p.get("driver")) or _person_name(p.get("jockey"))
        trainer = _person_name(p.get("entraineur")) or _person_name(p.get("trainer"))
        direct_report = _extract_report_info(p, "dernierRapportDirect")
        reference_report = _extract_report_info(p, "dernierRapportReference")
        odds_live = direct_report["odds"] or reference_report["odds"] or _extract_odds(p)

        rows.append(
            {
                "race_id": race_id,
                "race_date": day_iso,
                "discipline": discipline,
                "hippodrome": hippodrome,
                "distance": distance,
                "terrain": "INCONNU",
                "field_size": len(participants),
                "horse_number": number,
                "horse_name": horse_name or f"N°{number}",
                "jockey": jockey,
                "trainer": trainer,
                "odds": odds_live,
                "odds_direct": direct_report["odds"],
                "odds_reference": reference_report["odds"],
                "trend_indicator": direct_report["trend"] or reference_report["trend"],
                "trend_strength": direct_report["trend_strength"] if direct_report["trend_strength"] is not None else reference_report["trend_strength"],
                "report_direct_date": direct_report["date"],
                "report_reference_date": reference_report["date"],
                "draw": _safe_int(p.get("placeCorde")),
                "weight": _extract_weight(p),
                "age": _safe_int(p.get("age")),
                "sex": _normalize_sex(p.get("sexe")),
                "recent_form": str(p.get("musique") or ""),
                "finish_position": np.nan,
                "is_non_runner": _is_non_runner(p),
            }
        )

    race = pd.DataFrame(rows)
    if race.empty:
        raise RuntimeError("Aucun partant PMU n'a pu être converti.")

    # Le field_size utilisé par le modèle doit correspondre aux partants réels.
    race = race[~race["is_non_runner"].fillna(False).astype(bool)].copy()
    race["field_size"] = len(race)

    if race["horse_number"].duplicated().any():
        raise RuntimeError("Le flux PMU contient un numéro de cheval en double.")
    if len(race) < 5:
        raise RuntimeError("Moins de cinq partants exploitables après retrait des non-partants.")

    return race.sort_values("horse_number").reset_index(drop=True)


def _scheduled_start_datetime(day_iso: str, start_time: Any) -> datetime | None:
    if not start_time or str(start_time).strip() in {"", "—", "nan", "None"}:
        return None
    try:
        hh, mm = [int(x) for x in str(start_time).strip()[:5].split(":")]
        d = date.fromisoformat(day_iso)
        return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ZoneInfo("Europe/Paris"))
    except Exception:
        return None


def _minutes_to_start(scheduled_start: datetime | None, now: datetime | None = None) -> float | None:
    if scheduled_start is None:
        return None
    current = now or datetime.now(ZoneInfo("Europe/Paris"))
    return (scheduled_start - current).total_seconds() / 60.0


def load_saved_market_history(day_iso: str, reunion: int, course: int) -> dict[int, list[dict[str, Any]]]:
    """Recharge les snapshots pré-course persistés dans Supabase avec une clé de lecture."""
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_KEY") or _secret("SUPABASE_PUBLISHABLE_KEY") or _secret("SUPABASE_SECRET_KEY") or _secret("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return {}
    try:
        endpoint = f"{url.rstrip('/')}/rest/v1/quinte_market_snapshots"
        params = {
            "select": "horse_number,captured_at,odds,minutes_to_start",
            "race_date": f"eq.{day_iso}",
            "meeting_number": f"eq.{int(reunion)}",
            "race_number": f"eq.{int(course)}",
            "order": "captured_at.asc",
            "limit": "500",
        }
        response = requests.get(endpoint, headers=_supabase_headers(key), params=params, timeout=15)
        response.raise_for_status()
        payload = response.json() if response.content else []
        out: dict[int, list[dict[str, Any]]] = {}
        for item in payload if isinstance(payload, list) else []:
            num = _safe_int(item.get("horse_number"))
            odds = _safe_float(item.get("odds"))
            mb = _safe_float(item.get("minutes_to_start"))
            # Les lectures post-départ ne participent jamais à la prédiction.
            if num is None or odds is None or odds <= 1 or (mb is not None and mb < 0):
                continue
            out.setdefault(num, []).append({"captured_at": item.get("captured_at"), "odds": odds})
        return out
    except Exception:
        return {}


def _merge_market_histories(*histories: dict[int, list[dict[str, Any]]]) -> dict[int, list[dict[str, Any]]]:
    merged: dict[int, list[dict[str, Any]]] = {}
    for hist in histories:
        for num, seq in (hist or {}).items():
            merged.setdefault(int(num), []).extend(seq or [])
    for num, seq in merged.items():
        unique = {}
        for obs in seq:
            key = f"{obs.get('captured_at')}|{obs.get('odds')}"
            unique[key] = obs
        merged[num] = sorted(unique.values(), key=lambda x: str(x.get("captured_at") or ""))[-8:]
    return merged


def _session_market_history(
    race: pd.DataFrame,
    race_key: str,
    allow_predictive_capture: bool,
) -> dict[int, list[dict[str, Any]]]:
    """Stocke les snapshots de la session navigateur sans persistance externe.

    Les observations post-départ peuvent être enregistrées en base pour analyse future,
    mais elles ne sont jamais ajoutées à l'historique utilisé pour la prédiction live.
    """
    store = st.session_state.setdefault("v45_market_history", {})
    history = store.setdefault(race_key, {})
    if not allow_predictive_capture:
        return history
    now = datetime.now(ZoneInfo("Europe/Paris"))
    stamp = now.isoformat()
    for row in race.itertuples(index=False):
        odds = _safe_float(getattr(row, "odds_direct", None)) or _safe_float(getattr(row, "odds", None))
        if odds is None or odds <= 1:
            continue
        num = int(row.horse_number)
        seq = history.setdefault(num, [])
        # Evite de multiplier les doublons lors de reruns Streamlit rapprochés.
        if seq:
            prev_t = datetime.fromisoformat(str(seq[-1]["captured_at"]))
            if (now - prev_t).total_seconds() < 20 and abs(float(seq[-1]["odds"]) - odds) < 1e-9:
                continue
        seq.append({"captured_at": stamp, "odds": odds})
        history[num] = seq[-8:]
    store[race_key] = history
    st.session_state["v45_market_history"] = store
    return history


# =============================================================================
# HISTORIQUE : BASE FIGEE + COMPLEMENT SUPABASE OPTIONNEL
# =============================================================================

@st.cache_data(show_spinner=False)
def load_base_history(path: str) -> pd.DataFrame:
    history = pd.read_csv(path)
    history["race_date"] = pd.to_datetime(history["race_date"], errors="coerce").dt.normalize()
    return history


def _supabase_headers(key: str) -> dict[str, str]:
    # Les nouvelles clés sb_publishable_/sb_secret_ se passent dans `apikey`.
    # Les anciennes clés JWT fonctionnent aussi ainsi pour la Data API.
    return {
        "apikey": key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _supabase_get(url: str, key: str, table: str, params: dict[str, str]) -> list[dict]:
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    response = requests.get(
        endpoint,
        headers=_supabase_headers(key),
        params=params,
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Réponse Supabase inattendue pour {table}.")
    return payload


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_incremental_history(start_iso: str, end_iso: str) -> pd.DataFrame:
    """Charge uniquement les courses postérieures à la base figée et antérieures à la course cible."""
    url = _secret("SUPABASE_URL")
    # Préférer une clé de lecture limitée si elle est configurée.
    key = (
        _secret("SUPABASE_PUBLISHABLE_KEY")
        or _secret("SUPABASE_KEY")
        or _secret("SUPABASE_SECRET_KEY")
        or _secret("SUPABASE_SERVICE_KEY")
    )
    if not url or not key:
        return pd.DataFrame()

    start_day = date.fromisoformat(start_iso)
    end_day = date.fromisoformat(end_iso)
    if start_day >= end_day:
        return pd.DataFrame()

    race_rows: list[dict] = []
    offset = 0
    page_size = 1000
    while True:
        batch = _supabase_get(
            url,
            key,
            "races",
            {
                "select": (
                    "id,race_date,meeting_number,race_number,hippodrome,discipline,"
                    "distance_m,terrain,field_size,status"
                ),
                "and": f"(race_date.gte.{start_day.isoformat()},race_date.lt.{end_day.isoformat()})",
                "order": "id.asc",
                "limit": str(page_size),
                "offset": str(offset),
            },
        )
        race_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not race_rows:
        return pd.DataFrame()

    races = pd.DataFrame(race_rows)
    races["id"] = pd.to_numeric(races["id"], errors="coerce")
    races = races[races["id"].notna()].copy()
    races["id"] = races["id"].astype(int)
    race_ids = races["id"].tolist()

    participant_rows: list[dict] = []
    for i in range(0, len(race_ids), 30):
        chunk = race_ids[i : i + 30]
        ids = ",".join(str(int(x)) for x in chunk)
        participant_rows.extend(
            _supabase_get(
                url,
                key,
                "participants",
                {
                    "select": (
                        "id,race_id,horse_name,horse_number,jockey_name,trainer_name,odds,"
                        "weight_kg,draw,age,sex,recent_form,finish_position,is_non_runner"
                    ),
                    "race_id": f"in.({ids})",
                },
            )
        )

    if not participant_rows:
        return pd.DataFrame()

    participants = pd.DataFrame(participant_rows)
    merged = participants.merge(races, left_on="race_id", right_on="id", how="inner", suffixes=("", "_race"))

    merged["race_id_model"] = (
        "R" + merged["meeting_number"].astype(int).astype(str)
        + "C" + merged["race_number"].astype(int).astype(str)
        + "_" + pd.to_datetime(merged["race_date"]).dt.strftime("%Y-%m-%d")
    )

    out = pd.DataFrame(
        {
            "race_id": merged["race_id_model"],
            "race_date": pd.to_datetime(merged["race_date"], errors="coerce").dt.normalize(),
            "discipline": merged["discipline"],
            "hippodrome": merged["hippodrome"],
            "distance": merged["distance_m"],
            "terrain": merged["terrain"],
            "field_size": merged["field_size"],
            "horse_number": merged["horse_number"],
            "horse_name": merged["horse_name"],
            "jockey": merged["jockey_name"],
            "trainer": merged["trainer_name"],
            "odds": merged["odds"],
            "draw": merged["draw"],
            "weight": merged["weight_kg"],
            "age": merged["age"],
            "sex": merged["sex"],
            "recent_form": merged["recent_form"],
            "finish_position": merged["finish_position"],
            "is_non_runner": merged["is_non_runner"],
        }
    )

    # Reproduit la règle de validation de l'historique d'origine :
    # non-partants exclus + top 5 complet et unique + au moins 7 partants.
    out = out[~out["is_non_runner"].fillna(False).astype(bool)].copy()
    valid_ids: list[str] = []
    for rid, group in out.groupby("race_id", sort=False):
        if len(group) < 7 or group["horse_number"].duplicated().any():
            continue
        pos = pd.to_numeric(group["finish_position"], errors="coerce")
        top5 = sorted(pos[pos.between(1, 5)].astype(int).tolist())
        if top5 == [1, 2, 3, 4, 5]:
            valid_ids.append(rid)

    return out[out["race_id"].isin(valid_ids)].reset_index(drop=True)


def build_history(target_day: date) -> tuple[pd.DataFrame, str]:
    base = load_base_history(str(BASE_HISTORY_PATH)).copy()
    if base.empty:
        raise RuntimeError("L'historique validé embarqué est vide.")

    base_max = pd.to_datetime(base["race_date"], errors="coerce").max()
    if pd.isna(base_max):
        raise RuntimeError("Date maximale de l'historique embarqué illisible.")

    next_day = base_max.date() + timedelta(days=1)
    source = f"historique embarqué jusqu'au {base_max.date().isoformat()}"

    if next_day < target_day:
        incremental = load_incremental_history(next_day.isoformat(), target_day.isoformat())
        if not incremental.empty:
            base = pd.concat([base, incremental], ignore_index=True, sort=False)
            base = base.drop_duplicates(["race_id", "horse_number"], keep="last")
            inc_max = pd.to_datetime(incremental["race_date"]).max().date()
            source += f" + Supabase jusqu'au {inc_max.isoformat()}"
        else:
            source += " ; complément Supabase indisponible ou sans course validée"

    # Le modèle n'a jamais le droit de voir le jour cible ou le futur.
    base["race_date"] = pd.to_datetime(base["race_date"], errors="coerce").dt.normalize()
    base = base[base["race_date"] < pd.Timestamp(target_day)].copy()
    return base, source


# =============================================================================
# MODELE
# =============================================================================

@st.cache_data(show_spinner=False)
def load_model(path: str) -> dict:
    return load_artifact(path)


@st.cache_resource(show_spinner=False)
def load_v4_ranker(path: str):
    return load_ranker(path)


@st.cache_data(show_spinner=False)
def load_v43_outsider(path: str) -> dict:
    return load_outsider_artifact(path)


def run_prediction(
    history: pd.DataFrame,
    race: pd.DataFrame,
    snapshot_history: dict[int, list[dict[str, Any]]] | None = None,
    minutes_to_start: float | None = None,
) -> pd.DataFrame:
    artifact = load_model(str(MODEL_PATH))
    ranker = load_v4_ranker(str(RANKER_PATH))
    outsider = load_v43_outsider(str(OUTSIDER_ARTIFACT_PATH))
    base = rank_quinte_v44(history, race, artifact, ranker, outsider)
    return apply_v45_market_dynamics(base, snapshot_history=snapshot_history, minutes_to_start=minutes_to_start)


def save_market_snapshot(
    ranked: pd.DataFrame,
    day_iso: str,
    reunion: int,
    course: int,
    scheduled_start: datetime | None = None,
    minutes_to_start: float | None = None,
) -> tuple[int, str]:
    """Enregistre un snapshot uniquement avec une clé serveur Supabase.

    Une clé anon/de lecture n'est jamais utilisée pour écrire. Si aucune service key
    n'est configurée, la prédiction continue normalement sans snapshot.
    """
    url = _secret("SUPABASE_URL")
    service_key = _secret("SUPABASE_SECRET_KEY") or _secret("SUPABASE_SERVICE_KEY")
    if not url or not service_key:
        return 0, "Snapshot non enregistré : SUPABASE_SECRET_KEY/SUPABASE_SERVICE_KEY non configurée."
    try:
        rows = []
        for row in ranked.itertuples(index=False):
            odds = _safe_float(getattr(row, "odds", None))
            market_probability = _safe_float(getattr(row, "market_probability", None))
            rows.append({
                "race_date": day_iso,
                "meeting_number": int(reunion),
                "race_number": int(course),
                "horse_number": int(row.horse_number),
                "odds": odds,
                "market_probability": market_probability,
                "source": "PMU_ONLINE_V45",
                "scheduled_start": scheduled_start.isoformat() if scheduled_start else None,
                "minutes_to_start": minutes_to_start,
                "odds_direct": _safe_float(getattr(row, "odds_direct", None)),
                "odds_reference": _safe_float(getattr(row, "odds_reference", None)),
                "trend_indicator": str(getattr(row, "trend_indicator", "") or ""),
                "trend_strength": _safe_float(getattr(row, "trend_strength", None)),
                "projected_odds_t2": _safe_float(getattr(row, "v45_projected_odds_t2", None)),
                "predicted_log_move": _safe_float(getattr(row, "v45_predicted_log_move", None)),
                "volatility_score": _safe_float(getattr(row, "v45_volatility", None)),
                "dynamics_confidence": _safe_float(getattr(row, "v45_dynamics_confidence", None)),
                "dynamics_score": _safe_float(getattr(row, "v45_dynamics_score", None)),
                "model_rank": _safe_int(getattr(row, "quinte_rank", None)),
                "model_name": "HorseProno_V4_5",
            })
        if not rows:
            return 0, "Aucune cote à enregistrer."

        endpoint = f"{url.rstrip('/')}/rest/v1/quinte_market_snapshots"
        headers = _supabase_headers(service_key)
        headers["Prefer"] = "return=representation"
        response = requests.post(endpoint, headers=headers, json=rows, timeout=25)
        response.raise_for_status()
        payload = response.json() if response.content else []
        count = len(payload) if isinstance(payload, list) and payload else len(rows)
        return int(count), f"Snapshot de marché enregistré ({count} partants)."
    except Exception as exc:
        return 0, f"Snapshot non enregistré : {exc}"


# =============================================================================
# INTERFACE
# =============================================================================

paris_today = datetime.now(ZoneInfo("Europe/Paris")).date()
selected_day = st.date_input("📅 Programme à analyser", value=paris_today)

if st.button("🔄 Actualiser le programme", use_container_width=False):
    get_programme.clear()
    get_course_detail.clear()
    get_participants.clear()
    discover_daily_program.clear()

try:
    with st.spinner("Chargement du programme PMU complet…"):
        programme_df = discover_daily_program(selected_day.isoformat())

    if programme_df.empty:
        st.warning(
            f"Aucune course détectée dans le programme PMU du {selected_day.strftime('%d/%m/%Y')}."
        )
        st.stop()

    meeting_count = int(programme_df["meeting_number"].nunique())
    race_count = int(len(programme_df))
    quinte_count = int(programme_df["is_quinte"].fillna(False).astype(bool).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Réunions", meeting_count)
    m2.metric("Courses", race_count)
    m3.metric("Supports Quinté+", quinte_count)

    with st.expander("📋 Voir le programme complet de la journée"):
        programme_display = programme_df.copy()
        programme_display["Course"] = (
            "R" + programme_display["meeting_number"].astype(int).astype(str)
            + "C" + programme_display["race_number"].astype(int).astype(str)
        )
        programme_display["Quinté+"] = programme_display["is_quinte"].map(
            lambda x: "⭐ Oui" if bool(x) else ""
        )
        programme_display = programme_display[
            [
                "Course", "start_time", "hippodrome", "label",
                "distance_m", "field_size_pmu", "Quinté+",
            ]
        ].rename(
            columns={
                "start_time": "Heure",
                "hippodrome": "Hippodrome",
                "label": "Course / Prix",
                "distance_m": "Distance (m)",
                "field_size_pmu": "Partants annoncés",
            }
        )
        st.dataframe(programme_display, use_container_width=True, hide_index=True)

    st.subheader("🎯 Choisir la course à analyser")

    meeting_rows = (
        programme_df[["meeting_number", "hippodrome"]]
        .drop_duplicates()
        .sort_values("meeting_number")
    )
    meeting_options: list[tuple[int, str]] = []
    for row in meeting_rows.itertuples(index=False):
        meeting_no = int(row.meeting_number)
        race_n = int((programme_df["meeting_number"] == meeting_no).sum())
        meeting_options.append(
            (meeting_no, f"R{meeting_no} — {row.hippodrome} ({race_n} courses)")
        )

    selected_meeting_label = st.selectbox(
        "Réunion",
        [label for _, label in meeting_options],
    )
    reunion = next(
        number for number, label in meeting_options if label == selected_meeting_label
    )

    meeting_program = programme_df[
        programme_df["meeting_number"] == reunion
    ].copy().sort_values("race_number")

    course_options: list[tuple[int, str]] = []
    for row in meeting_program.itertuples(index=False):
        time_text = f"{row.start_time} — " if str(row.start_time or "").strip() else ""
        distance_text = (
            f" — {int(row.distance_m)} m"
            if pd.notna(row.distance_m)
            else ""
        )
        field_text = (
            f" — {int(row.field_size_pmu)} partants"
            if pd.notna(row.field_size_pmu)
            else ""
        )
        quinte_text = " ⭐ QUINTÉ+" if bool(row.is_quinte) else ""
        label = (
            f"C{int(row.race_number)} — {time_text}{row.label}"
            f"{distance_text}{field_text}{quinte_text}"
        )
        course_options.append((int(row.race_number), label))

    selected_course_label = st.selectbox(
        "Course",
        [label for _, label in course_options],
    )
    course = next(
        number for number, label in course_options if label == selected_course_label
    )

    selected_row = meeting_program[
        meeting_program["race_number"] == course
    ].iloc[0]
    hippodrome = str(selected_row["hippodrome"])
    label = str(selected_row["label"])
    is_quinte = bool(selected_row["is_quinte"])
    scheduled_start = _scheduled_start_datetime(selected_day.isoformat(), selected_row.get("start_time"))
    minutes_before = _minutes_to_start(scheduled_start)

    if is_quinte:
        st.success(
            f"Course sélectionnée : R{reunion}C{course} — {hippodrome} — {label} · ⭐ Support Quinté+"
        )
    else:
        st.info(
            f"Course sélectionnée : R{reunion}C{course} — {hippodrome} — {label}"
        )
        st.warning(
            "La recette Quinté V4.2/V4.3 a été étudiée en rétrospective sur les supports Quinté officiels disponibles. "
            "Il peut calculer un classement sur cette course, mais ses performances hors Quinté "
            "n'ont pas été validées par le backtest spécialisé."
        )

    if st.button("🏇 Analyser cette course", type="primary"):
        with st.spinner("Récupération des partants PMU…"):
            detail = get_course_detail(selected_day.isoformat(), reunion, course)
            participant_payload = get_participants(selected_day.isoformat(), reunion, course)
            race = participants_to_frame(
                selected_day.isoformat(),
                reunion,
                course,
                hippodrome,
                detail,
                participant_payload,
            )

        race_key = f"{selected_day.isoformat()}_R{reunion}C{course}"
        # Les snapshots de session ne sont utilisables comme information prédictive que tant que la course n'est pas partie.
        pre_start = minutes_before is None or minutes_before >= 0
        session_history = _session_market_history(race, race_key, allow_predictive_capture=pre_start)
        persisted_history = load_saved_market_history(selected_day.isoformat(), reunion, course) if is_quinte else {}
        market_history = _merge_market_histories(persisted_history, session_history)

        with st.spinner("Préparation de l'historique et calcul du pronostic…"):
            history, history_source = build_history(selected_day)
            ranked = run_prediction(history, race, snapshot_history=market_history, minutes_to_start=minutes_before)

        snapshot_count, snapshot_message = (0, "")
        if is_quinte:
            snapshot_count, snapshot_message = save_market_snapshot(
                ranked, selected_day.isoformat(), reunion, course,
                scheduled_start=scheduled_start, minutes_to_start=minutes_before
            )

        top7 = ranked.head(7)

        a, b, c, d = st.columns(4)
        a.metric("Course", f"R{reunion}C{course}")
        b.metric("Hippodrome", hippodrome)
        c.metric("Discipline", str(ranked.iloc[0]["discipline"]))
        d.metric("Partants", len(ranked))

        st.subheader("🎯 Sélection V4.5 — Top 7 stable + projection marché T-2")
        mode = str(ranked.iloc[0].get("v42_mode", "V4_GENERAL"))
        if mode == "PLAT_CONTEXTUEL":
            spread = ranked.iloc[0].get("v42_weight_spread")
            mw = ranked.iloc[0].get("v42_market_weight")
            bucket = str(ranked.iloc[0].get("v42_context_bucket", ""))
            spread_txt = "indisponible" if pd.isna(spread) else f"{float(spread):.1f} kg"
            st.success(
                f"🏇 Mode V4.2 PLAT contextuel activé · écart de poids : {spread_txt} · "
                f"coefficient marché : {float(mw):.2f} · profil : {bucket}."
            )
            dist = pd.to_numeric(ranked.iloc[0].get("distance"), errors="coerce")
            n_partants = len(ranked)
            if n_partants >= 17:
                st.warning("⚠️ Zone de confiance faible : 17 partants ou plus. L'historique PLAT disponible est encore trop réduit pour calibrer ce segment.")
            elif pd.notna(dist) and 1500 <= float(dist) <= 2000:
                st.caption("📊 Segment 1500–2000 m : zone PLAT historiquement la plus régulière dans le benchmark actuel.")
            elif pd.notna(dist) and float(dist) <= 1400:
                st.caption("📊 Sprint ≤1400 m : segment plus instable dans le contrôle temporel; interpréter la shortlist avec davantage de prudence.")
            elif pd.notna(dist) and float(dist) >= 2400:
                st.caption("📊 2400 m+ : performance correcte, mais la corde extérieure reste un profil historiquement plus difficile à récupérer.")
        else:
            st.info("Mode V4 général conservé pour cette discipline.")
        st.markdown(
            "## " + " - ".join(top7["horse_number"].astype(int).astype(str).tolist())
        )

        if str(ranked.iloc[0].get("v45_mode", "")) == "MARKET_DYNAMICS_PLAT":
            top3 = ranked.head(3)
            st.success(
                "🏆 **V4.5 — Winner Top 3 projeté T-2 :** "
                + " - ".join(top3["horse_number"].astype(int).astype(str).tolist())
                + "  ·  Top7 inchangé"
            )
            if minutes_before is not None:
                if minutes_before < 0:
                    st.error(
                        "⛔ Course déjà partie : les cotes visibles maintenant sont post-départ et ne doivent pas être interprétées comme un pronostic. "
                        "Elles sont enregistrées uniquement pour l'analyse future de la dynamique des cotes."
                    )
                else:
                    st.caption(f"⏱️ Lecture effectuée à environ T-{minutes_before:.1f} min.")
            st.caption(
                "V4.5 ajoute à l'ordre V4.4 un ajustement de dynamique borné à ±0,10 : contraction projetée de cote = bonus ; "
                "dérive ou forte volatilité = pénalité. Avec un seul point, le différentiel cote directe/référence sert de proxy ; "
                "avec plusieurs lectures, la pente réelle des cotes prend le relais."
            )
        else:
            st.info("V4.5 : hors PLAT, l'ordre V4.4/V4.3 est conservé à l'identique.")

        consensus = ranked[ranked["shortlist_role"].eq("CONSENSUS_V2_V3")].head(7)
        arbitration = ranked[ranked["shortlist_role"].eq("ARBITRAGE_EDGE_V42")].head(7)

        def _nums(df: pd.DataFrame) -> str:
            return " - ".join(df["horse_number"].astype(int).astype(str).tolist()) if not df.empty else "—"

        c1, c2 = st.columns(2)
        c1.success(f"🤝 Consensus V2 + V3\n\n**{_nums(consensus)}**")
        c2.info(f"🧠 Arbitrage V4.2 des désaccords\n\n**{_nums(arbitration)}**")
        st.caption(
            f"Consensus : {len(consensus)} cheval(aux) retenu(s) par V2 et V3. "
            f"La V4.2 remplit {len(arbitration)} place(s) parmi les désaccords avec son score edge. "
            "V4.2/V4.3 construit la shortlist de 7; V4.4 construit l’ordre sportif et V4.5 ajoute seulement la dynamique de marché sur le PLAT."
        )

        # V4.3 : outsider hors V2 ET V3 en shadow mode.
        outsider = ranked[ranked.get("v43_outsider_candidate", False).astype(bool)] if "v43_outsider_candidate" in ranked.columns else ranked.iloc[0:0]
        if str(ranked.iloc[0].get("v43_mode", "")) == "OUTSIDER_SHADOW" and not outsider.empty:
            out = outsider.iloc[0]
            prob = float(out.get("v43_outsider_probability", 0.0))
            repl_num = ranked.iloc[0].get("v43_shadow_replacement_number")
            repl_txt = "—" if pd.isna(repl_num) else str(int(float(repl_num)))
            threshold_met = bool(out.get("v43_shadow_threshold_met", False))
            st.subheader("🕵️ V4.3 — Outsider Detector (shadow mode)")
            o1, o2, o3 = st.columns(3)
            o1.metric("Outsider détecté", f"N° {int(out['horse_number'])}")
            o2.metric("Score outsider", f"{100*prob:.1f} %")
            o3.metric("Remplacement théorique", f"N° {repl_txt}")
            if threshold_met:
                st.warning(
                    "⚠️ Signal V4.3 ≥ 50 %. En mode expérimental, ce cheval aurait remplacé le cheval indiqué, "
                    "mais le Top 7 officiel reste volontairement celui de V4.2 jusqu'à validation forward suffisante."
                )
            else:
                st.info(
                    "V4.3 s'abstient : le meilleur outsider reste sous le seuil de 50 %. "
                    "Aucun changement n'est proposé au Top 7 V4.2."
                )

        display = ranked[
            [
                "quinte_rank", "horse_number", "horse_name", "shortlist_role",
                "v45_winner_score", "v45_adjustment", "v45_projected_odds_t2", "v45_projected_change_pct",
                "v45_dynamics_score", "v45_dynamics_confidence", "v45_signal_source",
                "v44_winner_score", "v45_original_rank", "v42_edge_score", "v43_outsider_probability", "v43_outsider_candidate",
                "in_v2_top7", "in_v3_top7", "top5_probability", "market_probability", "market_rank",
                "model_top5_rank", "ranker_rank", "context_score",
                "odds", "draw", "weight", "age", "jockey", "trainer",
            ]
        ].copy()
        display["top5_probability"] = (100 * display["top5_probability"]).round(1)
        display["market_probability"] = (100 * display["market_probability"]).round(1)
        display["context_score"] = (100 * display["context_score"]).round(1)
        display["v45_winner_score"] = pd.to_numeric(display["v45_winner_score"], errors="coerce").round(3)
        display["v45_adjustment"] = pd.to_numeric(display["v45_adjustment"], errors="coerce").round(3)
        display["v45_projected_odds_t2"] = pd.to_numeric(display["v45_projected_odds_t2"], errors="coerce").round(2)
        display["v45_projected_change_pct"] = pd.to_numeric(display["v45_projected_change_pct"], errors="coerce").round(1)
        display["v45_dynamics_score"] = pd.to_numeric(display["v45_dynamics_score"], errors="coerce").round(3)
        display["v45_dynamics_confidence"] = (100 * pd.to_numeric(display["v45_dynamics_confidence"], errors="coerce")).round(0)
        display["v44_winner_score"] = pd.to_numeric(display["v44_winner_score"], errors="coerce").round(3)
        display["v45_original_rank"] = pd.to_numeric(display["v45_original_rank"], errors="coerce").astype("Int64")
        display["v42_edge_score"] = display["v42_edge_score"].round(3)
        display["v43_outsider_probability"] = (100 * pd.to_numeric(display["v43_outsider_probability"], errors="coerce")).round(1)
        display["v43_outsider_candidate"] = display["v43_outsider_candidate"].map({True: "🕵️", False: ""})
        display["in_v2_top7"] = display["in_v2_top7"].map({True: "✅", False: ""})
        display["in_v3_top7"] = display["in_v3_top7"].map({True: "✅", False: ""})
        display["shortlist_role"] = display["shortlist_role"].map({
            "CONSENSUS_V2_V3": "🤝 Consensus V2/V3",
            "ARBITRAGE_EDGE_V42": "🧠 Arbitrage V4.2",
            "HORS_TOP7": "Hors Top7",
        }).fillna(display["shortlist_role"])
        display = display.rename(
            columns={
                "quinte_rank": "Rang shortlist",
                "horse_number": "N°",
                "horse_name": "Cheval",
                "shortlist_role": "Rôle",
                "v45_winner_score": "Score Winner V4.5",
                "v45_adjustment": "Coef dynamique",
                "v45_projected_odds_t2": "Cote projetée T-2",
                "v45_projected_change_pct": "Δ cote projeté %",
                "v45_dynamics_score": "Dynamique marché",
                "v45_dynamics_confidence": "Confiance dyn. %",
                "v45_signal_source": "Source dynamique",
                "v44_winner_score": "Score sportif V4.4",
                "v45_original_rank": "Rang V4.4",
                "v42_edge_score": "Score edge V4.2",
                "v43_outsider_probability": "Score outsider V4.3 %",
                "v43_outsider_candidate": "V4.3",
                "in_v2_top7": "V2",
                "in_v3_top7": "V3",
                "top5_probability": "P(Top5) calibrée %",
                "market_probability": "P(marché) %",
                "market_rank": "Rang marché",
                "model_top5_rank": "Rang modèle Top5",
                "ranker_rank": "Rang LambdaMART",
                "context_score": "Contexte %",
                "odds": "Cote PMU",
                "draw": "Corde",
                "weight": "Poids",
                "age": "Âge",
                "jockey": "Jockey / Driver",
                "trainer": "Entraîneur",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        missing_odds = int(pd.to_numeric(ranked["odds"], errors="coerce").isna().sum())
        if missing_odds:
            st.warning(
                f"{missing_odds} partant(s) sans cote PMU exploitable au moment de la lecture. "
                "Le modèle utilise alors sa valeur de repli prévue pour les cotes manquantes."
            )

        st.caption(
            f"Historique utilisé : {history_source} · {len(history):,} lignes historiques."
        )
        if is_quinte and snapshot_message:
            if snapshot_count:
                st.caption("📸 " + snapshot_message)
            else:
                st.caption("ℹ️ " + snapshot_message)

        st.download_button(
            "⬇️ Télécharger le pronostic CSV",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"pronostic_quinte_v3_{selected_day.isoformat()}_R{reunion}C{course}.csv"
            ),
            mime="text/csv",
        )

        with st.expander("🔎 Données PMU récupérées"):
            st.dataframe(
                race[
                    [
                        "horse_number", "horse_name", "odds", "odds_direct", "odds_reference",
                        "trend_indicator", "trend_strength", "draw", "weight",
                        "age", "sex", "jockey", "trainer", "recent_form",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("ℹ️ Informations modèle"):
            artifact = load_model(str(MODEL_PATH))
            st.json(
                {
                    "model_name": artifact.get("model_name"),
                    "status": artifact.get("status"),
                    "target": artifact.get("target"),
                    "training_supports": artifact.get("training_supports"),
                    "training_end": artifact.get("training_end"),
                    "v3_strategy": V3_CONFIG,
                    "v45_market_dynamics": V45_CONFIG,
                    "base_top5_model": artifact.get("model_name"),
                    "walk_forward_races_base": artifact.get("backtest", {}).get("walk_forward_races"),
                    "course_selection": "programme PMU complet + sélection manuelle",
                    "selected_course_is_quinte_plus": is_quinte,
                }
            )
            # Correctif : V3_REPORT_PATH n'existait pas (NameError après chaque analyse).
            if V4_REPORT_PATH.exists():
                try:
                    report = json.loads(V4_REPORT_PATH.read_text(encoding="utf-8"))
                    st.caption("Rapport candidat V4.2 embarqué")
                    st.json(report)
                except Exception:
                    pass

except requests.HTTPError as exc:
    st.error(f"Erreur HTTP PMU : {exc}")
except requests.RequestException as exc:
    st.error(f"Impossible de joindre le flux PMU : {exc}")
except Exception as exc:
    st.error(f"Erreur : {exc}")
    st.exception(exc)
