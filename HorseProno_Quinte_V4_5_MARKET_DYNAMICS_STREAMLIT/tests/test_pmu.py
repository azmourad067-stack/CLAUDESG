import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from simple_gagnant import pmu

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_simple_gagnant_result_real_payload():
    r = pmu.simple_gagnant_result(_load("pmu_rapports_R1C1_20261020.json"))
    assert r == {"available": True, "refunded": False, "winners": {5: 31.2}}


def test_simple_gagnant_result_dead_heat_and_missing():
    payload = [{"typePari": "E_SIMPLE_GAGNANT", "rembourse": False, "rapports": [
        {"combinaison": [3], "dividendePourUnEuro": 250}, {"combinaison": [7], "dividendePourUnEuro": 410}]}]
    assert pmu.simple_gagnant_result(payload)["winners"] == {3: 2.5, 7: 4.1}
    assert pmu.simple_gagnant_result([])["available"] is False


def test_participants_card_excludes_non_runners_and_reads_fields():
    meta = {"race_date": "2026-10-29", "meeting_number": 3, "race_number": 2, "discipline": "PLAT",
            "distance": 1600, "hippodrome": "CHANTILLY", "scheduled_start": "2026-10-29T15:10:00+01:00"}
    t = datetime(2026, 10, 29, 14, 50, tzinfo=ZoneInfo("Europe/Paris"))
    card, nr = pmu.participants_card(_load("pmu_participants_synthetic.json"), meta, observed_at=t)
    assert [x["horse_number"] for x in nr] == [3]
    assert card["horse_number"].tolist() == [1, 2, 4, 5, 6]
    row1 = card.iloc[0]
    assert row1["weight"] == 57.5 and row1["odds"] == 3.4 and row1["odds_reference"] == 4.0
    assert row1["jockey"] == "A. JOCKEY" and row1["career_starts"] == 12
    assert card.loc[card["horse_number"] == 2, "jockey"].item() == "B. JOCKEY"
    assert card.loc[card["horse_number"] == 4, "odds"].item() == 2.9  # repli sur la cote de référence
    assert (card["field_size"] == 5).all()
    assert card["odds_observed_at"].iloc[0].startswith("2026-10-29T14:50")


def test_duplicate_numbers_raise():
    payload = {"participants": [{"numPmu": 1, "nom": "A"}, {"numPmu": 1, "nom": "B"}]}
    with pytest.raises(ValueError):
        pmu.participants_card(payload, {"race_date": "2026-10-29", "meeting_number": 1, "race_number": 1})


def test_programme_frame_disciplines_and_start_times():
    prog = pmu.programme_frame(_load("pmu_programme_synthetic.json"), "2026-10-29")
    d = {(r.meeting_number, r.race_number): r.discipline for r in prog.itertuples()}
    assert d[(1, 1)] == "ATTELE_VOLTE" and d[(1, 2)] == "ATTELE_AUTOSTART"
    assert d[(3, 1)] == "HAIES"   # le libellé « Plateau » ne doit pas classer la course en plat
    assert d[(3, 2)] == "PLAT"
    start = prog.loc[(prog.meeting_number == 1) & (prog.race_number == 1), "scheduled_start"].item()
    assert start.startswith("2026-10-29T")
    now = datetime.fromisoformat(start).astimezone(ZoneInfo("Europe/Paris"))
    win = pmu.races_in_window(prog, 0, 10, now=now)
    assert (1, 1) in set(zip(win.meeting_number, win.race_number))
