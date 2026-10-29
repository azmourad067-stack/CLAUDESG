import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


def _mod():
    spec = importlib.util.spec_from_file_location("sg_export_forward", ROOT / "scripts" / "sg_export_forward.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_forward_runners_matches_sql_rules():
    m = _mod()
    races = pd.DataFrame([{"id": 10, "race_date": "2026-10-20", "meeting_number": 1, "race_number": 1,
                           "hippodrome": "X", "discipline": "PLAT", "distance_m": 1600},
                          {"id": 11, "race_date": "2026-10-20", "meeting_number": 1, "race_number": 2,
                           "hippodrome": "X", "discipline": "PLAT", "distance_m": 1600}])
    parts = pd.DataFrame([
        {"id": 1, "race_id": 10, "horse_name": "A", "horse_number": 5, "jockey_name": "J", "trainer_name": "T",
         "weight_kg": 57, "draw": 1, "age": 4, "sex": "MALES", "recent_form": "1p", "is_non_runner": False},
        {"id": 2, "race_id": 10, "horse_name": "B", "horse_number": 6, "jockey_name": "K", "trainer_name": "U",
         "weight_kg": 56, "draw": 2, "age": 5, "sex": "FEMELLES", "recent_form": "2p", "is_non_runner": False},
        {"id": 3, "race_id": 10, "horse_name": "C", "horse_number": 7, "jockey_name": "L", "trainer_name": "V",
         "weight_kg": 55, "draw": 3, "age": 5, "sex": "HONGRES", "recent_form": "", "is_non_runner": True},
        {"id": 4, "race_id": 11, "horse_name": "D", "horse_number": 1, "jockey_name": "J", "trainer_name": "T",
         "weight_kg": 57, "draw": 1, "age": 4, "sex": "MALES", "recent_form": "1p", "is_non_runner": False},
    ])
    snaps = pd.DataFrame([
        {"participant_id": 1, "race_id": 10, "model_version_id": 9, "odds_snapshot": 4.5,
         "predicted_at": "2026-10-20T11:00:00Z", "scheduled_start": "2026-10-20T11:25:00Z"},
        {"participant_id": 1, "race_id": 10, "model_version_id": 8, "odds_snapshot": 4.5,
         "predicted_at": "2026-10-20T11:00:00Z", "scheduled_start": "2026-10-20T11:25:00Z"},
        {"participant_id": 2, "race_id": 10, "model_version_id": 8, "odds_snapshot": 2.1,
         "predicted_at": "2026-10-20T11:00:00Z", "scheduled_start": "2026-10-20T11:25:00Z"},
        {"participant_id": 4, "race_id": 11, "model_version_id": 8, "odds_snapshot": 3.0,
         "predicted_at": "2026-10-20T12:00:00Z", "scheduled_start": "2026-10-20T12:20:00Z"},
    ])
    payload = json.loads((FIX / "pmu_rapports_R1C1_20261020.json").read_text(encoding="utf-8"))
    payouts = pd.DataFrame([{"race_date": "2026-10-20", "meeting_number": 1, "race_number": 1, "payload": payload},
                            {"race_date": "2026-10-20", "meeting_number": 1, "race_number": 2,
                             "payload": [{"typePari": "E_SIMPLE_GAGNANT", "rembourse": True, "rapports": []}]}])
    out = m.build_forward_runners(races, parts, snaps, payouts)
    assert out["race_id"].unique().tolist() == ["R1C1_2026-10-20"]  # course remboursée exclue
    assert out["horse_number"].tolist() == [5, 6]                     # non-partant exclu
    assert out.loc[out.horse_number == 5, "finish_position"].item() == 1.0
    assert out.loc[out.horse_number == 5, "dividend"].item() == 31.2
    assert out["minutes_before"].tolist() == [25.0, 25.0]
