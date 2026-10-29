"""Backtest walk-forward du moteur Simple Gagnant sur une table de partants.

Exemples :
    # Historique embarqué (cotes stockées après import, proches de la clôture)
    python scripts/sg_backtest.py --runners validated_history.csv \
        --start 2026-07-06 --end 2026-09-16 --odds-col odds \
        --odds-timing "cotes stockées après import (proches de la clôture)" \
        --out reports/history_backtest

    # Export prospectif (cotes snapshot pré-course + rapports définitifs)
    python scripts/sg_backtest.py --runners data/sg_forward_runners.csv \
        --start 2026-10-01 --end 2026-10-23 --odds-col odds_snapshot \
        --dividend-col dividend --minutes-col minutes_before \
        --odds-timing "snapshot T-30..T-10" --out reports/forward_backtest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_gagnant.backtest import walk_forward  # noqa: E402
from simple_gagnant.config import SelectorConfig  # noqa: E402


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runners", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--odds-col", default="odds")
    ap.add_argument("--dividend-col", default=None)
    ap.add_argument("--minutes-col", default=None)
    ap.add_argument("--odds-timing", required=True)
    ap.add_argument("--selector", default=str(ROOT / "models" / "sg_selector_v1.json"))
    ap.add_argument("--min-stage2-races", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runners = pd.read_csv(args.runners, low_memory=False)
    cfg = SelectorConfig.load(args.selector) if Path(args.selector).exists() else SelectorConfig()
    res = walk_forward(
        runners, args.start, args.end, step_days=args.step_days, selector=cfg,
        decision_odds_col=args.odds_col, dividend_col=args.dividend_col, minutes_col=args.minutes_col,
        odds_timing=args.odds_timing, min_stage2_races=args.min_stage2_races,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.decisions.to_csv(out / "decisions.csv", index=False)
    res.bets.to_csv(out / "bets.csv", index=False)
    for name, t in res.segments.items():
        t.to_csv(out / f"segment_{name}.csv", index=False)
    for name, t in res.calibration.items():
        t.to_csv(out / f"calibration_{name}.csv", index=False)
    report = {"summary": res.summary, "benchmarks": res.benchmarks, "folds": res.folds,
              "selector": cfg.to_dict(), "runners_file": Path(args.runners).name}
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
                                      encoding="utf-8")
    print(json.dumps({"summary": res.summary, "benchmarks": res.benchmarks}, indent=2, ensure_ascii=False,
                     default=_json_default))


if __name__ == "__main__":
    main()
