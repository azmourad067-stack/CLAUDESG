import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simple_gagnant.features import build_features, entity_features
from simple_gagnant.model import SimpleGagnantModel

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "forward_parity_sample.csv"
MODEL = ROOT / "models" / "sg_model_v1.json"


def _card():
    c = pd.read_csv(FIX)
    c["race_id"] = c["race_id"].astype(str)
    c["horse_name"] = [f"H{i}" for i in range(len(c))]
    return c


def test_python_matches_sql_feature_pipeline():
    """Parité avec les valeurs calculées dans Supabase par scripts/sg_forward_sql.py."""
    c = _card()
    m = SimpleGagnantModel.load(ROOT / "models" / "sg_stage1_until_20260911.json")
    u = m.fundamental_utility(c).to_numpy()
    assert np.abs(u - c["u_fund_sql"].to_numpy()).max() < 1e-5
    f = build_features(c)
    for col in ("mus_score5", "mus_inc5", "weight_rel", "draw_rel", "age_rel"):
        assert np.abs(f[col].to_numpy() - c[f"{col}_sql"].to_numpy()).max() < 1e-5


def test_artifact_fingerprint_and_predictions_are_coherent():
    raw = json.loads(MODEL.read_text(encoding="utf-8"))
    model = SimpleGagnantModel.from_dict(raw)
    assert raw["fingerprint"] == model.fingerprint()
    pred = model.predict(_card(), decision_odds_col="odds")
    sums = pred.groupby("race_id")["p_model"].sum()
    assert np.allclose(sums, 1.0)
    assert (pred["p_low"] <= pred["p_model"] + 1e-12).all()
    assert np.allclose(pred["ev"], pred["p_model"] * pred["decision_odds"] * pred["payout_factor"] - 1)


def test_incomplete_market_gives_no_combined_probability():
    c = _card()
    rid = c["race_id"].iloc[0]
    c.loc[c.index[0], "odds"] = np.nan
    pred = SimpleGagnantModel.load(MODEL).predict(c, decision_odds_col="odds")
    assert pred.loc[pred["race_id"] == rid, "p_model"].isna().all()
    assert pred.loc[pred["race_id"] != rid, "p_model"].notna().all()


def test_features_never_read_current_results():
    c = _card()
    a = build_features(c.assign(finish_position=np.arange(len(c)) % 9 + 1))
    b = build_features(c.assign(finish_position=np.nan))
    pd.testing.assert_frame_equal(a, b)


def test_entity_stats_use_strictly_earlier_dates_only():
    hist = pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r2"],
        "race_date": ["2026-09-01", "2026-09-01", "2026-09-02", "2026-09-02"],
        "horse_number": [1, 2, 1, 2],
        "jockey": ["A", "B", "A", "B"], "trainer": ["T", "U", "T", "U"],
        "finish_position": [1, 2, 1, 2],
    })
    card = pd.DataFrame({"race_id": ["r2", "r2"], "race_date": ["2026-09-02", "2026-09-02"],
                         "horse_number": [1, 2], "jockey": ["A", "B"], "trainer": ["T", "U"]})
    e = entity_features(hist, card)
    # Le jour même (r2) ne doit pas compter : une seule course antérieure pour chaque jockey.
    assert np.isclose(e.loc[0, "jockey_exp"], np.log1p(1) / 6.0)
    assert e.loc[0, "jockey_win"] > e.loc[1, "jockey_win"]
