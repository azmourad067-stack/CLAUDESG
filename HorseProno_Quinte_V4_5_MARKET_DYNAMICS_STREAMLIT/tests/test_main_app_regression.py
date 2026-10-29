"""Régression : l'application Quinté principale ne doit plus lever NameError après l'analyse."""
import json
import runpy
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from tests.fake_streamlit import make_fake

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.content = b"x"

    def json(self):
        return self._p

    def raise_for_status(self):
        return None


def test_main_quinte_app_analysis_has_no_error(monkeypatch):
    now = datetime.now(ZoneInfo("Europe/Paris"))
    course = {"numOrdre": 1, "libelle": "PRIX TEST", "discipline": "PLAT", "specialite": "PLAT",
              "heureDepart": int((now + timedelta(hours=2)).timestamp() * 1000), "distance": 1600,
              "nombreDeclaresPartants": 5, "paris": [{"codePari": "E_QUINTE_PLUS"}]}
    programme = {"programme": {"reunions": [{"numOfficiel": 1, "hippodrome": {"libelleCourt": "CHANTILLY"},
                                             "courses": [course]}]}}
    participants = json.loads((FIX / "pmu_participants_synthetic.json").read_text(encoding="utf-8"))
    # La recette Quinté (V3) suppose au moins 7 partants : on complète le champ synthétique à 10 partants.
    base = participants["participants"][0]
    for n in range(7, 12):
        participants["participants"].append({**base, "numPmu": n, "nom": f"EXTRA {n}", "placeCorde": n,
                                             "dernierRapportDirect": {"rapport": 5.0 + n}})

    def fake_get(url, *a, **k):
        if url.endswith("/participants"):
            return _Resp(participants)
        if "/R1/C1" in url:
            return _Resp(course)
        if "/programme/" in url:
            return _Resp(programme)
        raise requests.ConnectionError("réseau désactivé en test")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_get)
    for k in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY"):
        monkeypatch.delenv(k, raising=False)
    fake = make_fake(buttons=("Analyser cette course",))
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    monkeypatch.chdir(ROOT)
    runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="__app__")
    names = [n for n, *_ in fake._log]
    errors = [a for n, a, k in fake._log if n in ("error", "exception")]
    assert not errors, errors
    assert "download_button" in names  # l'analyse est allée jusqu'au bout
    assert any(n == "json" for n in names)  # l'expander « Informations modèle » s'affiche
