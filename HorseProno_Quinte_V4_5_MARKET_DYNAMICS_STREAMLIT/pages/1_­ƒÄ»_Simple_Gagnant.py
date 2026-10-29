"""Page Streamlit — moteur Simple Gagnant (isolé de la stratégie Quinté)."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_gagnant import pmu  # noqa: E402
from simple_gagnant.config import STATUS_NO_BET, STATUS_TEST, STATUS_VALIDATED, STATUS_WATCH  # noqa: E402
from simple_gagnant.engine import SimpleGagnantEngine  # noqa: E402
from simple_gagnant.journal import LocalJournal, SupabaseJournal, monitoring_summary  # noqa: E402
from simple_gagnant.ui import decisions_table, odds_sensitivity, runner_table, status_summary  # noqa: E402

PARIS = ZoneInfo("Europe/Paris")
VALIDATION_SUMMARY = ROOT / "reports" / "forward_eval" / "sg_validation_summary.json"
LOCAL_JOURNAL = Path(os.getenv("SG_JOURNAL_DIR") or (ROOT / "data" / "journal"))

st.set_page_config(page_title="Simple Gagnant — HorseProno", page_icon="🎯", layout="wide")
st.title("🎯 Moteur Simple Gagnant — mode observation")
st.caption(
    "Analyse course par course : probabilité du modèle, cote observée, rapport attendu corrigé de la dérive, "
    "espérance et décision documentée. Aucune course n'est jouée par défaut ; ce module ne place aucun pari."
)


def _secret(name: str) -> str | None:
    try:
        if name in st.secrets and st.secrets[name]:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name)


@st.cache_resource(show_spinner=False)
def get_engine() -> SimpleGagnantEngine:
    return SimpleGagnantEngine()


@st.cache_data(ttl=10 * 60, show_spinner=False)
def load_programme(day_iso: str) -> pd.DataFrame:
    return pmu.programme_frame(pmu.fetch_programme(day_iso), day_iso)


@st.cache_data(ttl=3 * 60, show_spinner=False)
def load_participants(day_iso: str, reunion: int, course: int):
    return pmu.fetch_participants(day_iso, reunion, course)


def analyse(engine: SimpleGagnantEngine, day_iso: str, meta: dict):
    now = datetime.now(PARIS)
    minutes = pmu.minutes_to_start(meta.get("scheduled_start"), now)
    try:
        payload = load_participants(day_iso, int(meta["meeting_number"]), int(meta["race_number"]))
        card, nr = pmu.participants_card(payload, meta, observed_at=now)
    except Exception as exc:  # noqa: BLE001
        from simple_gagnant.engine import RaceAnalysis
        return RaceAnalysis(meta, pd.DataFrame(), pd.DataFrame(), None, minutes, [], error=f"{type(exc).__name__}: {exc}")
    return engine.analyse_card(card, minutes, meta, nr)


def journal_store():
    remote = SupabaseJournal.from_env(_secret)
    return remote, LocalJournal(LOCAL_JOURNAL)


engine = get_engine()
desc = engine.describe()
if not desc["validated_segments"]:
    st.warning(
        "Aucun segment n'a démontré d'avantage hors échantillon : le statut « OPPORTUNITÉ VALIDÉE PAR LE MODÈLE » "
        "est désactivé. Les « candidats à tester » servent uniquement à l'observation prospective."
    )

tab_day, tab_race, tab_journal, tab_model = st.tabs(
    ["📋 Courses du jour", "🔎 Détail d'une course", "🗂️ Journal & résultats", "📊 Validation du modèle"]
)

# ----------------------------------------------------------------------------- courses du jour
with tab_day:
    day = st.date_input("Journée", value=datetime.now(PARIS).date(), key="sg_day")
    day_iso = day.isoformat()
    try:
        programme = load_programme(day_iso)
    except requests.RequestException as exc:
        st.error(f"Programme PMU indisponible : {exc}")
        programme = pd.DataFrame()
    if programme.empty:
        st.info("Aucune course trouvée pour cette journée.")
    else:
        only_upcoming = st.checkbox("Seulement les courses non parties", value=True, key="sg_upcoming")
        todo = programme.copy()
        if only_upcoming:
            mts = todo["scheduled_start"].map(lambda s: pmu.minutes_to_start(s))
            todo = todo[(mts.isna()) | (mts >= 0)]
        st.caption(f"{len(programme)} courses au programme · {len(todo)} à analyser.")
        if st.button("Analyser les courses", type="primary", key="sg_run"):
            results = []
            bar = st.progress(0.0)
            metas = todo.to_dict("records")
            for i, meta in enumerate(metas):
                results.append(analyse(engine, day_iso, meta))
                bar.progress((i + 1) / max(len(metas), 1))
            st.session_state["sg_results"] = results
            st.session_state["sg_results_day"] = day_iso
        results = st.session_state.get("sg_results", [])
        if results and st.session_state.get("sg_results_day") == day_iso:
            counts = status_summary(results)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Courses écartées", counts[STATUS_NO_BET])
            c2.metric("Surveillance", counts[STATUS_WATCH])
            c3.metric("Candidats à tester", counts[STATUS_TEST])
            c4.metric("Opportunités validées", counts[STATUS_VALIDATED])
            c5.metric("Erreurs de collecte", counts["ERREUR"])
            st.dataframe(decisions_table(results), use_container_width=True, hide_index=True)
            st.caption(
                "Le meilleur candidat est affiché même lorsque la course est écartée. EV = probabilité modèle × cote "
                "observée × facteur de rapport − 1 ; le facteur corrige l'écart mesuré entre cote pré-course et rapport "
                "définitif des gagnants. Une cote relevée maintenant peut encore évoluer jusqu'au départ."
            )
            remote, local = journal_store()
            if st.button("📸 Enregistrer ces analyses dans le journal", key="sg_save"):
                saved = 0
                for a in results:
                    if a.decision is None or a.minutes_to_start is None or a.minutes_to_start < 0:
                        continue
                    pred, runners = engine.snapshot_rows(a, mode="MANUEL")
                    if pred is None:
                        continue
                    local.write_snapshot(pred, runners)
                    if remote and remote.write_key:
                        try:
                            remote.write_snapshot(pred, runners)
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Supabase : {exc}")
                    saved += 1
                st.success(f"{saved} photographie(s) pré-course enregistrée(s) (courses non parties uniquement).")

# ----------------------------------------------------------------------------- détail course
with tab_race:
    results = st.session_state.get("sg_results", [])
    if not results:
        st.info("Lancez d'abord l'analyse dans l'onglet « Courses du jour ».")
    else:
        labels = [f"R{a.meta.get('meeting_number')}C{a.meta.get('race_number')} — {a.meta.get('hippodrome')} — "
                  f"{a.meta.get('label', '')}" for a in results]
        idx = st.selectbox("Course", range(len(labels)), format_func=lambda i: labels[i], key="sg_race_pick")
        a = results[idx]
        if a.error:
            st.error(a.error)
        elif a.decision is not None:
            d = a.decision
            st.subheader(d.status)
            if d.reasons:
                st.write("**Motif :** " + d.reason_text)
            c = d.candidate
            if c:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Candidat", f"N°{c.get('horse_number')}")
                m2.metric("Cote observée", f"{c.get('decision_odds', float('nan')):.2f}")
                m3.metric("P modèle", f"{100 * c.get('p_model', float('nan')):.1f} %")
                m4.metric("Cote juste", f"{c.get('fair_odds', float('nan')):.2f}")
                m5.metric("EV corrigée", f"{100 * c.get('ev', float('nan')):+.1f} %")
            left, right = st.columns(2)
            left.markdown("**Arguments favorables**\n\n" + ("\n".join(f"- {x}" for x in d.arguments_for) or "—"))
            right.markdown("**Arguments défavorables**\n\n" + ("\n".join(f"- {x}" for x in d.arguments_against) or "—"))
            for alert in d.alerts:
                st.warning(alert)
            st.dataframe(runner_table(a.predictions), use_container_width=True, hide_index=True)
            if c and c.get("p_model"):
                o = float(c.get("decision_odds"))
                grid = sorted({round(o * f, 2) for f in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)})
                st.markdown("**Sensibilité à la cote** (même probabilité, facteur de rapport de la tranche actuelle)")
                st.dataframe(odds_sensitivity(float(c["p_model"]), float(c.get("payout_factor") or 1.0), grid),
                             hide_index=True)
            with st.expander("Métriques de la course"):
                st.json(d.metrics)
            if "odds_reference" in a.predictions:
                mv = a.predictions[["horse_number", "odds_reference", "decision_odds"]].copy()
                mv["Variation %"] = (100 * (mv["decision_odds"] / mv["odds_reference"] - 1)).round(1)
                with st.expander("Mouvements de cote (référence PMU → cote actuelle)"):
                    st.caption("Signal descriptif à tester historiquement, pas une information certaine sur le résultat.")
                    st.dataframe(mv.rename(columns={"horse_number": "N°", "odds_reference": "Cote de référence",
                                                    "decision_odds": "Cote actuelle"}), hide_index=True)

# ----------------------------------------------------------------------------- journal
with tab_journal:
    remote, local = journal_store()
    source = "Supabase" if remote else "journal local"
    try:
        if remote:
            j = remote.select("sg_journal", {"select": "*", "order": "captured_at.desc", "limit": "2000"})
            preds, sets = j, j.dropna(subset=["settled_at"]) if not j.empty else j
        else:
            preds, sets = local.predictions(), local.settlements()
            j = preds.merge(sets, on="snapshot_id", how="left", suffixes=("", "_s")) if not preds.empty else preds
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lecture du journal impossible ({source}) : {exc}")
        preds = sets = j = pd.DataFrame()
    st.caption(f"Source : {source}. Les photographies sont immuables ; les règlements sont des enregistrements séparés.")
    if preds.empty:
        st.info("Journal vide : activez le workflow d'observation ou enregistrez des analyses.")
    else:
        summary = monitoring_summary(preds, sets)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Photographies", summary.get("snapshots", 0))
        k2.metric("Réglées", summary.get("settled", 0))
        k3.metric("Mises simulées", summary.get("simulated_bets", 0))
        roi = summary.get("simulated_roi")
        k4.metric("ROI simulé", "—" if roi is None else f"{100 * roi:+.1f} %")
        if "log_loss_gain_per_race" in summary:
            st.caption(
                f"Log-loss modèle {summary['log_loss_model']:.4f} vs marché {summary['log_loss_market']:.4f} "
                f"sur {summary['log_loss_races']} courses (gain {summary['log_loss_gain_per_race']:+.4f} ± "
                f"{summary['log_loss_gain_se']:.4f}). Une dégradation durable signale une dérive du modèle."
            )
        cols = [c for c in ["captured_at", "race_id", "hippodrome", "discipline", "minutes_to_start", "status",
                            "candidate_number", "candidate_odds", "p_model", "ev", "reason_text", "candidate_won",
                            "candidate_dividend", "profit", "settlement_note"] if c in j.columns]
        st.dataframe(j[cols], use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------- validation
with tab_model:
    st.json(desc)
    if VALIDATION_SUMMARY.exists():
        v = json.loads(VALIDATION_SUMMARY.read_text(encoding="utf-8"))
        st.subheader("Évaluation prospective gelée")
        st.write(v.get("conclusion", ""))
        t = v.get("test", {})
        a1, a2, a3 = st.columns(3)
        a1.metric("Courses de test", t.get("races"))
        a2.metric("Log-loss marché → modèle", f"{t.get('log_loss_per_race', {}).get('market')} → "
                                            f"{t.get('log_loss_per_race', {}).get('model')}")
        a3.metric("ROI favori (rapports officiels)", f"{100 * t.get('benchmark_favourite', {}).get('roi', 0):.1f} %")
        with st.expander("Détails (validation, test, expériences existantes re-réglées)"):
            st.json(v)
