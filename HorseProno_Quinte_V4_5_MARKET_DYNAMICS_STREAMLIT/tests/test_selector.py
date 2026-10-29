import numpy as np
import pandas as pd

from simple_gagnant.config import (
    STATUS_NO_BET, STATUS_TEST, STATUS_VALIDATED, STATUS_WATCH, SelectorConfig,
)
from simple_gagnant.selector import evaluate_race


def _race(ev_best=0.2, ev_low=0.05, ev_drift=0.02, n=8, booksum=1.18, family="PLAT", complete=True, missing=0.0):
    rows = []
    for i in range(n):
        best = i == 0
        rows.append({
            "race_id": "R1C1_2026-10-29", "horse_number": i + 1, "horse_name": f"H{i+1}", "family": family,
            "decision_odds": 6.0 if best else 8.0 + i, "market_complete": complete, "booksum": booksum,
            "p_model": 0.20 if best else 0.8 / (n - 1), "p_low": 0.18 if best else 0.1,
            "p_market": 0.15 if best else 0.85 / (n - 1), "p_market_raw": 0.17, "p_fund": 1 / n,
            "ev": ev_best if best else -0.3, "ev_low": ev_low if best else -0.4, "ev_drift": ev_drift if best else -0.5,
            "ev_naive": ev_best + 0.1, "fair_odds": 5.0, "break_even_odds": 5.5, "payout_factor": 0.89,
            "payout_factor_low": 0.7, "edge": 0.05, "model_rank": 1 if best else i + 1, "market_rank": 2 if best else 1,
            "mus_missing": missing, "mus_score5": 0.6 if best else 0.4, "mus_inc5": 0.05,
        })
    return pd.DataFrame(rows)


def test_quality_gates_reject_race():
    cfg = SelectorConfig()
    assert evaluate_race(_race(complete=False), cfg, 20).status == STATUS_NO_BET
    d = evaluate_race(_race(booksum=1.6), cfg, 20)
    assert d.status == STATUS_NO_BET and "MARCHE_ANORMAL" in d.reasons
    d = evaluate_race(_race(missing=1.0), cfg, 20)
    assert "FORME_INSUFFISANTE" in d.reasons
    d = evaluate_race(_race(), cfg, -3)
    assert "COURSE_PARTIE" in d.reasons
    d = evaluate_race(_race(n=3), cfg, 20)
    assert "PELOTON_HORS_PERIMETRE" in d.reasons
    # Le meilleur candidat reste affiché même si la course est écartée.
    assert d.candidate.get("horse_number") == 1


def test_status_ladder():
    cfg = SelectorConfig(ev_test=0.05, ev_validated=0.10, validated_segments=("PLAT",))
    assert evaluate_race(_race(ev_best=-0.02), cfg, 20).status == STATUS_NO_BET
    assert evaluate_race(_race(ev_best=0.03), cfg, 20).status == STATUS_WATCH
    assert evaluate_race(_race(ev_best=0.2), cfg, 60).status == STATUS_WATCH  # cote trop précoce
    d = evaluate_race(_race(ev_best=0.2, ev_low=-0.01), cfg, 20)
    assert d.status == STATUS_TEST and "EV_PRUDENTE_NEGATIVE" in d.reasons
    d = evaluate_race(_race(ev_best=0.2, ev_drift=-0.01), cfg, 20)
    assert d.status == STATUS_TEST and "DERIVE_DEFAVORABLE" in d.reasons
    assert evaluate_race(_race(ev_best=0.2), cfg, 20).status == STATUS_VALIDATED


def test_no_validated_status_without_out_of_sample_evidence():
    cfg = SelectorConfig(ev_test=0.05, ev_validated=0.10, validated_segments=())
    d = evaluate_race(_race(ev_best=0.5), cfg, 20)
    assert d.status == STATUS_TEST and "SEGMENT_NON_VALIDE" in d.reasons


def test_favourite_is_neither_forced_nor_excluded():
    cfg = SelectorConfig(ev_test=0.05, ev_validated=0.10, validated_segments=("PLAT",))
    r = _race(ev_best=0.2)
    r.loc[0, "market_rank"] = 1
    assert evaluate_race(r, cfg, 20).status == STATUS_VALIDATED
    r.loc[0, ["ev", "ev_low", "ev_drift"]] = [-0.1, -0.2, -0.2]
    assert evaluate_race(r, cfg, 20).status == STATUS_NO_BET


def test_ineligible_longshot_not_selected():
    cfg = SelectorConfig(odds_max=30.0)
    r = _race(ev_best=0.2)
    r.loc[0, "decision_odds"] = 45.0
    d = evaluate_race(r, cfg, 20)
    assert d.status == STATUS_NO_BET and d.candidate["horse_number"] != 1
    r["decision_odds"] = 45.0
    d = evaluate_race(r, cfg, 20)
    assert d.status == STATUS_NO_BET and d.reasons == ["AUCUN_ELIGIBLE"]
