from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simple_gagnant.backtest import bootstrap_roi, max_drawdown, walk_forward
from simple_gagnant.config import SelectorConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def history():
    h = pd.read_csv(ROOT / "validated_history.csv", low_memory=False)
    h["race_date"] = pd.to_datetime(h["race_date"])
    return h[h["race_date"] >= "2026-08-10"].copy()


def test_drawdown_and_bootstrap():
    assert max_drawdown(pd.Series([1, -2, -3, 4, -1])) == 5
    bets = pd.DataFrame({"race_date": ["d1", "d1", "d2"], "stake": [1, 1, 1], "pnl": [-1, 2, -1]})
    ci = bootstrap_roi(bets, n_boot=200)
    assert ci["roi_ci_low"] <= ci["roi_ci_high"]


def test_walk_forward_runs_and_reports(history):
    res = walk_forward(history, "2026-09-01", "2026-09-16", step_days=7, selector=SelectorConfig(),
                       odds_timing="test", min_stage2_races=100, n_boot=100)
    s = res.summary
    assert s["races_analysed"] > 0 and s["odds_timing"] == "test"
    assert s["settlement"].startswith("cote de décision")  # aucun rapport définitif fourni
    assert set(res.benchmarks) >= {"favori_toutes_courses_retenues", "log_loss_par_course"}
    assert all("status" in f for f in res.folds)


def test_walk_forward_does_not_peek_at_evaluated_results(history):
    """Changer les arrivées de la dernière semaine ne doit modifier aucune prédiction de cette semaine."""
    a = walk_forward(history, "2026-09-01", "2026-09-16", step_days=7, min_stage2_races=100, n_boot=50,
                     odds_timing="test")
    h2 = history.copy()
    last = h2["race_date"] >= "2026-09-15"
    rng = np.random.default_rng(0)
    for rid, idx in h2[last].groupby("race_id").groups.items():
        pos = h2.loc[idx, "finish_position"].to_numpy()
        h2.loc[idx, "finish_position"] = rng.permutation(pos)
    b = walk_forward(h2, "2026-09-01", "2026-09-16", step_days=7, min_stage2_races=100, n_boot=50,
                     odds_timing="test")
    pa = a.predictions[a.predictions["race_date"] >= "2026-09-15"].sort_values(["race_id", "horse_number"])
    pb = b.predictions[b.predictions["race_date"] >= "2026-09-15"].sort_values(["race_id", "horse_number"])
    assert len(pa) > 0
    assert np.allclose(pa["p_model"].to_numpy(), pb["p_model"].to_numpy(), equal_nan=True)
    da = a.decisions[a.decisions["race_date"] >= "2026-09-15"].sort_values("race_id")["status"].tolist()
    db = b.decisions[b.decisions["race_date"] >= "2026-09-15"].sort_values("race_id")["status"].tolist()
    assert da == db
