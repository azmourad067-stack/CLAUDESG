import pandas as pd

from simple_gagnant.config import STATUS_NO_BET, STATUS_TEST
from simple_gagnant.journal import LocalJournal, monitoring_summary, settle_snapshot

SNAP = {"snapshot_id": "abc", "race_id": "R1C1_2026-10-20", "status": STATUS_TEST, "candidate_number": 5}
RUNNERS = [{"horse_number": 5, "p_model": 0.1, "p_market": 0.08}, {"horse_number": 2, "p_model": 0.3, "p_market": 0.3}]


def test_settlement_win_loss_refund_nonrunner():
    win = settle_snapshot(SNAP, {"available": True, "refunded": False, "winners": {5: 31.2}}, RUNNERS)
    assert win["candidate_won"] and win["gross_return"] == 31.2 and win["profit"] == 30.2
    assert win["p_model_winner"] == 0.1 and win["p_market_winner"] == 0.08
    loss = settle_snapshot(SNAP, {"available": True, "refunded": False, "winners": {2: 3.0}})
    assert loss["candidate_won"] is False and loss["profit"] == -1.0
    ref = settle_snapshot(SNAP, {"available": True, "refunded": True, "winners": {}})
    assert ref["profit"] == 0.0 and ref["gross_return"] == 1.0
    nr = settle_snapshot(SNAP, {"available": True, "refunded": False, "winners": {5: 31.2}}, candidate_non_runner=True)
    assert nr["profit"] == 0.0 and nr["candidate_won"] is None
    pending = settle_snapshot(SNAP, {"available": False, "winners": {}})
    assert pending["result_available"] is False and pending["profit"] is None


def test_no_bet_is_not_staked_but_outcome_recorded():
    s = settle_snapshot({**SNAP, "status": STATUS_NO_BET}, {"available": True, "refunded": False, "winners": {5: 4.0}})
    assert s["simulated_bet"] is False and s["stake"] == 0.0 and s["profit"] is None and s["candidate_won"] is True


def test_local_journal_is_append_only_and_deduplicated(tmp_path):
    j = LocalJournal(tmp_path)
    pred = {"snapshot_id": "abc", "race_id": "R1C1_2026-10-20", "status": STATUS_TEST, "candidate_number": 5}
    assert j.write_snapshot(pred, [{**r, "snapshot_id": "abc"} for r in RUNNERS]) == 1
    assert j.write_snapshot({**pred, "status": STATUS_NO_BET}, []) == 0  # même photographie : refusée
    assert len(j.predictions()) == 1 and j.predictions()["status"].iloc[0] == STATUS_TEST
    s = settle_snapshot(pred, {"available": True, "refunded": False, "winners": {5: 31.2}}, RUNNERS)
    assert j.write_settlement(s) == 1 and j.write_settlement(s) == 0
    summ = monitoring_summary(j.predictions(), j.settlements())
    assert summ["simulated_bets"] == 1 and abs(summ["simulated_profit"] - 30.2) < 1e-9
