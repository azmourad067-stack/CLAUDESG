"""Entraîne le modèle fondamental Simple Gagnant (étape 1) et écrit l'artefact.

Usage :
    python scripts/sg_train_stage1.py --history validated_history.csv \
        --train-end 2026-09-12 --out models/sg_stage1.json

Seules les courses de date strictement antérieure à --train-end sont utilisées.
L'étape 2 (combinaison marché) et le facteur de rapport restent neutres
(α = 1, γ = 0, facteur = 1) : ils doivent être estimés ensuite sur des cotes
horodatées pré-course (voir sg_forward_sql.py / sg_fit_forward.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_gagnant import __version__  # noqa: E402
from simple_gagnant.model import MarketCombination, PayoutFactor, SimpleGagnantModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(ROOT / "validated_history.csv"))
    ap.add_argument("--train-end", required=True, help="date exclue (AAAA-MM-JJ)")
    ap.add_argument("--train-start", default=None, help="date incluse (fenêtre glissante), optionnelle")
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    history = pd.read_csv(args.history, low_memory=False)
    if args.train_start:
        history = history[pd.to_datetime(history["race_date"], errors="coerce") >= pd.Timestamp(args.train_start)]
    stage1 = SimpleGagnantModel.fit_stage1(history, args.train_end, l2=args.l2)
    dates = pd.to_datetime(history["race_date"], errors="coerce")
    used = history[dates < pd.Timestamp(args.train_end)]
    model = SimpleGagnantModel(
        stage1=stage1,
        combination=MarketCombination(alpha=1.0, gamma=0.0, fit_period="neutre", odds_timing="aucune"),
        payout=PayoutFactor(),
        metadata={
            "package_version": __version__,
            "stage1_trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage1_train_end_exclusive": args.train_end,
            "stage1_train_start": str(pd.to_datetime(used["race_date"]).min().date()),
            "stage1_races": int(used["race_id"].nunique()),
            "stage1_l2": args.l2,
            "stage1_source": Path(args.history).name,
            "target": "victoire (finish_position == 1), logit conditionnel par famille",
        },
    )
    fp = model.save(args.out)
    summary = {fam: {"races": m.n_races_, "rows": m.n_rows_, "converged": m.converged_} for fam, m in stage1.items()}
    print(json.dumps({"out": args.out, "fingerprint": fp, "families": summary}, indent=2))


if __name__ == "__main__":
    main()
