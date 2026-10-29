"""Mode observation Simple Gagnant (aucun pari réel).

    python scripts/sg_observe.py capture   # photographie des courses entre T-30 et T-2
    python scripts/sg_observe.py settle    # règlement des photographies au rapport définitif
    python scripts/sg_observe.py summary   # indicateurs prospectifs

Stockage : Supabase si SUPABASE_URL + clé serveur (SUPABASE_SECRET_KEY ou
SUPABASE_SERVICE_KEY) sont présents, sinon journal local data/journal/.
Une photographie au plus par course, par phase (T-30/T-15, T-15/T-5, T-5/T-2)
et par version de modèle. Les photographies existantes ne sont jamais modifiées.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_gagnant import pmu  # noqa: E402
from simple_gagnant.engine import SimpleGagnantEngine  # noqa: E402
from simple_gagnant.journal import LocalJournal, SupabaseJournal, monitoring_summary, settle_snapshot  # noqa: E402

PARIS = ZoneInfo("Europe/Paris")
PHASES = [("T30", 15.0, 30.0), ("T15", 5.0, 15.0), ("T5", 2.0, 5.0)]


def phase_of(minutes: float | None) -> str | None:
    if minutes is None:
        return None
    for name, lo, hi in PHASES:
        if lo <= minutes < hi or (name == "T30" and minutes == hi):
            return name
    return None


class Store:
    def __init__(self, local_dir: Path):
        self.remote = SupabaseJournal.from_env()
        if self.remote is not None and not self.remote.write_key:
            self.remote = None
        self.local = LocalJournal(local_dir)

    @property
    def label(self) -> str:
        return "Supabase" if self.remote else f"local ({self.local.root})"

    def predictions(self, since: str) -> pd.DataFrame:
        if self.remote:
            return self.remote.select("sg_predictions", {
                "select": "snapshot_id,captured_at,race_date,meeting_number,race_number,race_id,status,"
                          "candidate_number,minutes_to_start,model_fingerprint,scheduled_start",
                "race_date": f"gte.{since}", "order": "captured_at.asc", "limit": "5000"})
        p = self.local.predictions()
        return p[p["race_date"].astype(str) >= since] if not p.empty else p

    def settled_ids(self, since: str) -> set[str]:
        if self.remote:
            s = self.remote.select("sg_settlements", {"select": "snapshot_id", "settled_at": f"gte.{since}",
                                                      "limit": "10000"})
        else:
            s = self.local.settlements()
        return set(s["snapshot_id"].astype(str)) if not s.empty else set()

    def runners(self, snapshot_id: str) -> list[dict]:
        if self.remote:
            r = self.remote.select("sg_prediction_runners", {"select": "horse_number,p_model,p_market",
                                                             "snapshot_id": f"eq.{snapshot_id}"})
        else:
            r = self.local.runners()
            r = r[r["snapshot_id"].astype(str) == snapshot_id] if not r.empty else r
        return r.to_dict("records") if not r.empty else []

    def write_snapshot(self, pred: dict, runners: list[dict]) -> None:
        self.local.write_snapshot(pred, runners)
        if self.remote:
            self.remote.write_snapshot(pred, runners)

    def write_settlement(self, s: dict) -> None:
        self.local.write_settlement(s)
        if self.remote:
            self.remote.write_settlement(s)


def capture(engine: SimpleGagnantEngine, store: Store, now: datetime) -> dict:
    day = now.date().isoformat()
    session = requests.Session()
    prog = pmu.programme_frame(pmu.fetch_programme(day, session), day)
    window = pmu.races_in_window(prog, engine.selector.decision_window_min, engine.selector.decision_window_max, now)
    existing = store.predictions(day)
    done = set()
    if not existing.empty:
        for row in existing.itertuples(index=False):
            if str(row.model_fingerprint) == engine.fingerprint:
                done.add((str(row.race_id), phase_of(pd.to_numeric(row.minutes_to_start, errors="coerce"))))
    report = {"day": day, "in_window": int(len(window)), "captured": 0, "skipped": 0, "errors": [], "statuses": {}}
    for meta in window.to_dict("records"):
        rid = f"R{int(meta['meeting_number'])}C{int(meta['race_number'])}_{day}"
        ph = phase_of(meta.get("minutes_to_start"))
        if (rid, ph) in done:
            report["skipped"] += 1
            continue
        analysis = engine.analyse_race(day, meta, now=now, session=session)
        if analysis.error:
            report["errors"].append(f"{rid}: {analysis.error}")
            continue
        pred, runners = engine.snapshot_rows(analysis, captured_at=now)
        if pred is None:
            continue
        store.write_snapshot(pred, runners)
        report["captured"] += 1
        report["statuses"][pred["status"]] = report["statuses"].get(pred["status"], 0) + 1
    return report


def settle(store: Store, now: datetime, lookback_days: int = 3, min_delay_minutes: float = 20.0) -> dict:
    since = (now.date() - timedelta(days=lookback_days)).isoformat()
    preds = store.predictions(since)
    done = store.settled_ids(since)
    report = {"pending": 0, "settled": 0, "waiting": 0, "errors": []}
    if preds.empty:
        return report
    session = requests.Session()
    cache: dict[tuple, dict] = {}
    for row in preds.to_dict("records"):
        sid = str(row["snapshot_id"])
        if sid in done:
            continue
        report["pending"] += 1
        mts = pmu.minutes_to_start(row.get("scheduled_start"), now)
        if mts is not None and mts > -min_delay_minutes:
            report["waiting"] += 1
            continue
        key = (str(row["race_date"]), int(row["meeting_number"]), int(row["race_number"]))
        try:
            if key not in cache:
                payload = pmu.fetch_rapports_definitifs(key[0], key[1], key[2], session)
                result = pmu.simple_gagnant_result(payload)
                nr_numbers: set[int] = set()
                try:
                    part = pmu.fetch_participants(key[0], key[1], key[2], session)
                    _, nr = pmu.participants_card(part, {"race_date": key[0], "meeting_number": key[1],
                                                         "race_number": key[2]})
                    nr_numbers = {int(x["horse_number"]) for x in nr}
                except Exception:  # noqa: BLE001 - le statut non-partant reste inconnu
                    pass
                cache[key] = {"result": result, "non_runners": nr_numbers}
            entry = cache[key]
            if not entry["result"]["available"]:
                report["waiting"] += 1
                continue
            cand = row.get("candidate_number")
            cand_nr = cand is not None and pd.notna(cand) and int(float(cand)) in entry["non_runners"]
            s = settle_snapshot(row, entry["result"], store.runners(sid), settled_at=now, candidate_non_runner=cand_nr)
            store.write_settlement(s)
            report["settled"] += 1
        except requests.HTTPError as exc:
            report["errors"].append(f"{sid}: HTTP {exc.response.status_code if exc.response is not None else '?'}")
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{sid}: {type(exc).__name__}: {exc}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["capture", "settle", "summary"])
    ap.add_argument("--journal-dir", default=str(ROOT / "data" / "journal"))
    ap.add_argument("--model", default=str(ROOT / "models" / "sg_model_v1.json"))
    ap.add_argument("--selector", default=str(ROOT / "models" / "sg_selector_v1.json"))
    args = ap.parse_args()
    now = datetime.now(PARIS)
    store = Store(Path(args.journal_dir))
    if os.getenv("GITHUB_ACTIONS") == "true" and store.remote is None and args.action != "summary":
        raise SystemExit("Supabase non configuré (SUPABASE_URL + SUPABASE_SECRET_KEY) : "
                         "le journal d'un runner GitHub ne serait pas conservé.")
    if args.action == "capture":
        engine = SimpleGagnantEngine(args.model, args.selector)
        out = capture(engine, store, now)
    elif args.action == "settle":
        out = settle(store, now)
    else:
        if store.remote:
            preds = store.remote.select("sg_journal", {"select": "*", "limit": "20000"})
            out = monitoring_summary(preds, preds.dropna(subset=["settled_at"]) if not preds.empty else preds)
        else:
            out = monitoring_summary(store.local.predictions(), store.local.settlements())
    out["storage"] = store.label
    out["run_at"] = now.isoformat(timespec="seconds")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
