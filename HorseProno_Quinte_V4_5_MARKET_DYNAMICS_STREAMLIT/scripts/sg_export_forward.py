"""Exporte depuis Supabase la table « partants » prospective (cotes snapshot + rapports).

Sortie : un CSV directement utilisable par scripts/sg_backtest.py, avec
- `odds_snapshot`  : cote PMU horodatée pré-course (forward_model_predictions),
- `minutes_before` : minutes entre la photographie et le départ prévu,
- `dividend`       : rapport définitif e-Simple Gagnant du gagnant (€ pour 1 €),
- `finish_position`: 1 pour le gagnant officiel, vide sinon (seule information
  de résultat utilisée par le moteur).
Option --with-history : ajoute validated_history.csv (odds_snapshot vide) pour
l'entraînement de l'étape 1 ; ces courses ne servent jamais à estimer l'étape 2.

Clés : SUPABASE_URL + SUPABASE_SECRET_KEY (la table quinte_payouts_raw n'a pas
de politique de lecture publique).

    python scripts/sg_export_forward.py --start 2026-09-12 --end 2026-10-23 \
        --with-history --out data/sg_forward_runners.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_gagnant.pmu import simple_gagnant_result  # noqa: E402

PAGE = 1000


def _get_all(url: str, key: str, table: str, params: dict[str, str]) -> pd.DataFrame:
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}
    rows, offset = [], 0
    while True:
        p = dict(params, limit=str(PAGE), offset=str(offset))
        r = requests.get(f"{url.rstrip('/')}/rest/v1/{table}", headers=headers, params=p, timeout=60)
        r.raise_for_status()
        batch = r.json()
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(rows)


def build_forward_runners(races: pd.DataFrame, participants: pd.DataFrame, snapshots: pd.DataFrame,
                          payouts: pd.DataFrame) -> pd.DataFrame:
    """Transformation pure (testée) : mêmes règles que scripts/sg_forward_sql.py."""
    snap = (snapshots.sort_values(["participant_id", "model_version_id"])
            .drop_duplicates("participant_id"))
    race_snap = snap.groupby("race_id").agg(predicted_at=("predicted_at", "min"),
                                            scheduled_start=("scheduled_start", "min")).reset_index()
    res = []
    for row in payouts.itertuples(index=False):
        sg = simple_gagnant_result(row.payload)
        if not sg["available"] or sg["refunded"] or len(sg["winners"]) != 1:
            continue
        (w, d), = sg["winners"].items()
        res.append({"race_date": str(row.race_date), "meeting_number": int(row.meeting_number),
                    "race_number": int(row.race_number), "winner": w, "dividend": d})
    results = pd.DataFrame(res, columns=["race_date", "meeting_number", "race_number", "winner", "dividend"])
    r = races.rename(columns={"id": "race_pk"}).merge(race_snap, left_on="race_pk", right_on="race_id", how="inner")
    r["race_date"] = r["race_date"].astype(str)
    r = r.merge(results, on=["race_date", "meeting_number", "race_number"], how="inner")
    p = participants[~participants["is_non_runner"].fillna(False).astype(bool)]
    out = p.merge(r, left_on="race_id", right_on="race_pk", how="inner", suffixes=("", "_race"))
    out = out.merge(snap[["participant_id", "odds_snapshot"]], left_on="id", right_on="participant_id", how="left")
    out["minutes_before"] = (pd.to_datetime(out["scheduled_start"], utc=True)
                             - pd.to_datetime(out["predicted_at"], utc=True)).dt.total_seconds() / 60.0
    out["finish_position"] = (out["horse_number"] == out["winner"]).map({True: 1.0, False: float("nan")})
    out["race_id"] = ("R" + out["meeting_number"].astype(int).astype(str) + "C"
                      + out["race_number"].astype(int).astype(str) + "_" + out["race_date"])
    keep = {
        "race_id": "race_id", "race_date": "race_date", "meeting_number": "meeting_number",
        "race_number": "race_number", "hippodrome": "hippodrome", "discipline": "discipline",
        "distance_m": "distance", "horse_number": "horse_number", "horse_name": "horse_name",
        "jockey_name": "jockey", "trainer_name": "trainer", "draw": "draw", "weight_kg": "weight",
        "age": "age", "sex": "sex", "recent_form": "recent_form", "odds_snapshot": "odds_snapshot",
        "minutes_before": "minutes_before", "dividend": "dividend", "finish_position": "finish_position",
    }
    out = out[list(keep)].rename(columns=keep)
    # Course gardée seulement si le gagnant officiel figure parmi les partants.
    has_winner = out.groupby("race_id")["finish_position"].transform(lambda s: s.eq(1).sum() == 1)
    return out[has_winner].sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--with-history", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise SystemExit("SUPABASE_URL et SUPABASE_SECRET_KEY sont nécessaires.")
    period = {"and": f"(race_date.gte.{args.start},race_date.lte.{args.end})"}
    races = _get_all(url, key, "races", {"select": "id,race_date,meeting_number,race_number,hippodrome,discipline,distance_m", **period})
    ids = races["id"].astype(int).tolist()
    parts, snaps = [], []
    for i in range(0, len(ids), 150):
        chunk = ",".join(map(str, ids[i:i + 150]))
        parts.append(_get_all(url, key, "participants", {
            "select": "id,race_id,horse_name,horse_number,jockey_name,trainer_name,weight_kg,draw,age,sex,recent_form,is_non_runner",
            "race_id": f"in.({chunk})"}))
        snaps.append(_get_all(url, key, "forward_model_predictions", {
            "select": "participant_id,race_id,model_version_id,odds_snapshot,predicted_at,scheduled_start",
            "race_id": f"in.({chunk})"}))
    payouts = _get_all(url, key, "quinte_payouts_raw", {"select": "race_date,meeting_number,race_number,payload", **period})
    out = build_forward_runners(races, pd.concat(parts), pd.concat(snaps), payouts)
    if args.with_history:
        h = pd.read_csv(ROOT / "validated_history.csv", low_memory=False)
        h = h[pd.to_datetime(h["race_date"]) < pd.Timestamp(args.start)].assign(odds_snapshot=float("nan"))
        out = pd.concat([h, out], ignore_index=True, sort=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{out['race_id'].nunique()} courses, {len(out)} partants → {args.out}")


if __name__ == "__main__":
    main()
