import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


def _load_script():
    spec = importlib.util.spec_from_file_location("sg_observe", ROOT / "scripts" / "sg_observe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capture_then_settle_cycle(monkeypatch, tmp_path):
    obs = _load_script()
    from simple_gagnant import pmu
    from simple_gagnant.engine import SimpleGagnantEngine

    for k in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY"):
        monkeypatch.delenv(k, raising=False)
    now = datetime(2026, 10, 29, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    ms = lambda dt: int(dt.timestamp() * 1000)  # noqa: E731
    programme = {"programme": {"reunions": [{"numOfficiel": 3, "hippodrome": {"libelleCourt": "CHANTILLY"}, "courses": [
        {"numOrdre": 1, "libelle": "DANS LA FENETRE", "discipline": "PLAT", "heureDepart": ms(now + timedelta(minutes=20))},
        {"numOrdre": 2, "libelle": "TROP TOT", "discipline": "PLAT", "heureDepart": ms(now + timedelta(minutes=95))},
        {"numOrdre": 3, "libelle": "PARTIE", "discipline": "PLAT", "heureDepart": ms(now - timedelta(minutes=5))}]}]}}
    participants = json.loads((FIX / "pmu_participants_synthetic.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(pmu, "fetch_programme", lambda day, session=None: programme)
    monkeypatch.setattr(pmu, "fetch_participants", lambda day, r, c, session=None: participants)
    rapports = [{"typePari": "E_SIMPLE_GAGNANT", "rembourse": False,
                 "rapports": [{"combinaison": [4], "dividendePourUnEuro": 310}]}]
    monkeypatch.setattr(pmu, "fetch_rapports_definitifs", lambda day, r, c, session=None: rapports)

    store = obs.Store(tmp_path)
    engine = SimpleGagnantEngine()
    rep = obs.capture(engine, store, now)
    assert rep["in_window"] == 1 and rep["captured"] == 1 and not rep["errors"]
    # Deuxième passage dans la même phase : aucune nouvelle photographie.
    rep2 = obs.capture(engine, store, now + timedelta(minutes=2))
    assert rep2["captured"] == 0 and rep2["skipped"] == 1
    # Avant la fin de course : le règlement attend.
    assert obs.settle(store, now + timedelta(minutes=25))["settled"] == 0
    rep3 = obs.settle(store, now + timedelta(minutes=60))
    assert rep3["settled"] == 1
    s = store.local.settlements()
    assert len(s) == 1 and bool(s["result_available"].iloc[0])
    preds = store.local.predictions()
    assert len(preds) == 1  # la photographie initiale n'a pas été modifiée
    assert obs.settle(store, now + timedelta(minutes=70))["settled"] == 0  # pas de double règlement
