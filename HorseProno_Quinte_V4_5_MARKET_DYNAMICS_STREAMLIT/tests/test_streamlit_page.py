import json
import runpy
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.fake_streamlit import make_fake

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


def _programme(now):
    ms = lambda dt: int(dt.timestamp() * 1000)  # noqa: E731
    return {"programme": {"reunions": [{"numOfficiel": 3, "hippodrome": {"libelleCourt": "CHANTILLY"}, "courses": [
        {"numOrdre": 1, "libelle": "PRIX TEST", "discipline": "PLAT", "heureDepart": ms(now + timedelta(minutes=20)),
         "distance": 1600, "nombreDeclaresPartants": 6},
        {"numOrdre": 2, "libelle": "PRIX TEST 2", "discipline": "PLAT", "heureDepart": ms(now + timedelta(minutes=50)),
         "distance": 2000, "nombreDeclaresPartants": 6}]}]}}


def test_simple_gagnant_page_runs_end_to_end(monkeypatch, tmp_path):
    from simple_gagnant import pmu

    now = datetime.now(ZoneInfo("Europe/Paris"))
    participants = json.loads((FIX / "pmu_participants_synthetic.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(pmu, "fetch_programme", lambda day, session=None: _programme(now))
    monkeypatch.setattr(pmu, "fetch_participants", lambda day, r, c, session=None: participants)
    for k in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SG_JOURNAL_DIR", str(tmp_path / "journal"))
    fake = make_fake(buttons=("Analyser les courses", "Enregistrer ces analyses"))
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    page = next((ROOT / "pages").glob("*Simple_Gagnant.py"))
    # Journal local redirigé vers un dossier temporaire.
    g = runpy.run_path(str(page), init_globals={}, run_name="__page__")
    log = fake._log
    names = [n for n, *_ in log]
    assert "set_page_config" in names and "dataframe" in names
    results = fake.session_state["sg_results"]
    assert len(results) == 2 and all(r.decision is not None for r in results)
    errors = [a for n, a, k in log if n == "error"]
    assert not errors, errors
    # Les deux courses ont été analysées et le tableau récapitulatif contient le statut et le candidat.
    tables = [a[0] for n, a, k in log if n == "dataframe"]
    summary = tables[0]
    assert {"Statut", "Candidat", "EV %", "Motif"} <= set(summary.columns)
    saved = (tmp_path / "journal" / "sg_predictions.csv")
    assert saved.exists()
    import pandas as pd
    j = pd.read_csv(saved)
    assert len(j) == 2 and set(j["mode"]) == {"MANUEL"} and j["model_fingerprint"].nunique() == 1
