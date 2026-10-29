"""Backtest chronologique (walk-forward) du moteur Simple Gagnant.

Principe, pour chaque bloc temporel [t, t + pas) :
1. étape 1 (fondamentale) entraînée uniquement sur les courses < t ;
2. étape 2 (α, γ) estimée sur les utilités hors échantillon des blocs
   précédents (jamais sur le bloc évalué) ;
3. facteur de rapport estimé sur les gagnants des blocs précédents lorsque
   des rapports définitifs sont fournis ; sinon neutre (et signalé) ;
4. décision par course avec le sélecteur ; règlement au rapport fourni.

Le jeu de données d'entrée est une table « partants » : une ligne par partant
avec la fiche de course, la cote utilisée pour décider (`decision_odds_col`)
et, si disponible, le rapport définitif du gagnant (`dividend_col`).
L'étiquette `odds_timing` est reportée dans tous les résultats : un backtest
réalisé avec des cotes proches de la clôture n'est pas une mesure de
rentabilité exploitable à la décision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .clogit import ConditionalLogit, softmax_by_race
from .config import STATUS_NO_BET, STATUS_TEST, STATUS_VALIDATED, STATUS_WATCH, SelectorConfig
from .features import build_features, feature_list
from .model import FAMILIES, MarketCombination, PayoutFactor, SimpleGagnantModel
from .selector import evaluate_race

BET_STATUSES = (STATUS_TEST, STATUS_VALIDATED)


# --------------------------------------------------------------------------- métriques
def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    cum = pnl.cumsum()
    peak = cum.cummax().clip(lower=0.0)
    return float((peak - cum).max())


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 7, block: str = "race_date") -> dict[str, float]:
    """IC du ROI par bootstrap par blocs journaliers (corrélation intra-journée)."""
    if bets.empty:
        return {"roi_ci_low": float("nan"), "roi_ci_high": float("nan"), "p_roi_positive": float("nan")}
    rng = np.random.default_rng(seed)
    days = bets.groupby(bets[block].astype(str))
    stakes = days["stake"].sum().to_numpy()
    profits = days["pnl"].sum().to_numpy()
    k = len(stakes)
    idx = rng.integers(0, k, size=(n_boot, k))
    roi = profits[idx].sum(axis=1) / np.maximum(stakes[idx].sum(axis=1), 1e-9)
    return {
        "roi_ci_low": float(np.quantile(roi, 0.05)),
        "roi_ci_high": float(np.quantile(roi, 0.95)),
        "p_roi_positive": float((roi > 0).mean()),
    }


def summarize_bets(bets: pd.DataFrame, n_boot: int = 2000) -> dict[str, Any]:
    if bets.empty:
        return {"bets": 0, "wins": 0, "win_rate": float("nan"), "stake": 0.0, "profit": 0.0,
                "roi": float("nan"), "avg_odds": float("nan"), "median_odds": float("nan"),
                "max_drawdown": 0.0, "roi_ci_low": float("nan"), "roi_ci_high": float("nan"),
                "p_roi_positive": float("nan")}
    b = bets.sort_values(["race_date", "race_id"])
    out = {
        "bets": int(len(b)),
        "wins": int(b["won"].sum()),
        "win_rate": float(b["won"].mean()),
        "stake": float(b["stake"].sum()),
        "profit": float(b["pnl"].sum()),
        "roi": float(b["pnl"].sum() / b["stake"].sum()),
        "avg_odds": float(b["decision_odds"].mean()),
        "median_odds": float(b["decision_odds"].median()),
        "max_drawdown": max_drawdown(b["pnl"]),
    }
    out.update(bootstrap_roi(b, n_boot=n_boot))
    return out


def segment_table(bets: pd.DataFrame, by: str) -> pd.DataFrame:
    if bets.empty or by not in bets.columns:
        return pd.DataFrame()
    rows = []
    for key, g in bets.groupby(by, dropna=False, observed=True):
        rows.append({by: key, "bets": len(g), "wins": int(g["won"].sum()), "profit": float(g["pnl"].sum()),
                     "roi": float(g["pnl"].sum() / g["stake"].sum()), "avg_odds": float(g["decision_odds"].mean())})
    return pd.DataFrame(rows).sort_values("bets", ascending=False)


def calibration_table(pred: pd.DataFrame, prob_col: str, bins=(0, .02, .05, .1, .15, .2, .3, .45, .6, 1.0)) -> pd.DataFrame:
    p = pred[[prob_col, "won"]].dropna()
    cut = pd.cut(p[prob_col], bins=list(bins), include_lowest=True)
    t = p.groupby(cut, observed=True).agg(n=("won", "size"), predicted=(prob_col, "mean"), observed=("won", "mean"))
    return t.reset_index().rename(columns={prob_col: "bucket"})


def log_loss_per_race(pred: pd.DataFrame, prob_col: str) -> float:
    w = pred[pred["won"] == 1]
    p = pd.to_numeric(w[prob_col], errors="coerce").clip(lower=1e-12)
    return float(-np.log(p).mean()) if len(p) else float("nan")


# --------------------------------------------------------------------------- walk-forward
@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    decisions: pd.DataFrame
    bets: pd.DataFrame
    folds: list[dict[str, Any]]
    summary: dict[str, Any]
    odds_timing: str
    benchmarks: dict[str, Any] = field(default_factory=dict)
    segments: dict[str, pd.DataFrame] = field(default_factory=dict)
    calibration: dict[str, pd.DataFrame] = field(default_factory=dict)


def _bucket_odds(o: pd.Series) -> pd.Series:
    return pd.cut(o, [1, 3, 6, 12, 25, 1e9], right=False, labels=["<3", "3-6", "6-12", "12-25", "25+"])


def walk_forward(
    runners: pd.DataFrame,
    start: str,
    end: str,
    step_days: int = 7,
    selector: SelectorConfig | None = None,
    decision_odds_col: str = "odds",
    dividend_col: str | None = None,
    minutes_col: str | None = None,
    odds_timing: str = "non précisé",
    l2: float = 5.0,
    min_stage2_races: int = 300,
    n_boot: int = 2000,
) -> BacktestResult:
    """`runners` : une ligne par partant (non-partants exclus) avec finish_position.

    Les courses avant `start` ne servent qu'à l'apprentissage.
    """
    cfg = selector or SelectorConfig()
    df = runners.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.normalize()
    if "is_non_runner" in df.columns:
        df = df[~df["is_non_runner"].fillna(False).astype(bool)]
    df = df.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    df["won"] = pd.to_numeric(df["finish_position"], errors="coerce").eq(1).astype(int)
    winners_per_race = df.groupby("race_id")["won"].transform("sum")
    df = df[winners_per_race == 1].reset_index(drop=True)

    feats = build_features(df)
    fam = feats["family"].to_numpy()
    df["family"] = fam
    y = df["won"].to_numpy(dtype=float)

    t0, t_end = pd.Timestamp(start), pd.Timestamp(end)
    edges = list(pd.date_range(t0, t_end + pd.Timedelta(days=1), freq=f"{step_days}D"))
    if edges[-1] <= t_end:
        edges.append(t_end + pd.Timedelta(days=1))

    oos_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.DataFrame] = []
    dec_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        train = (df["race_date"] < lo).to_numpy()
        test = ((df["race_date"] >= lo) & (df["race_date"] < hi)).to_numpy()
        if not test.any():
            continue
        stage1: dict[str, ConditionalLogit] = {}
        for f in FAMILIES:
            cols = feature_list(f)
            m_tr = train & (fam == f)
            if df.loc[m_tr, "race_id"].nunique() < 50:
                continue
            stage1[f] = ConditionalLogit(cols, l2=l2).fit(
                feats.loc[m_tr, cols].to_numpy(float), df.loc[m_tr, "race_id"].to_numpy(), y[m_tr]
            )
        block = df.loc[test].copy()
        u = np.zeros(test.sum())
        bfam = fam[test]
        bfeats = feats.loc[test]
        for f, m in stage1.items():
            mask = bfam == f
            if mask.any():
                u[mask] = m.utility(bfeats.loc[mask, m.features].to_numpy(float))
        block["u_fund"] = u
        odds = pd.to_numeric(block[decision_odds_col], errors="coerce").where(lambda s: s > 1)
        inv = 1.0 / odds
        complete = inv.notna().groupby(block["race_id"]).transform("all")
        block["lpm"] = np.log(inv / inv.groupby(block["race_id"]).transform("sum")).where(complete)

        # Étape 2 et facteur de rapport : uniquement à partir des blocs précédents.
        history_oos = pd.concat(oos_parts) if oos_parts else pd.DataFrame()
        fold_info: dict[str, Any] = {"start": str(lo.date()), "end": str((hi - pd.Timedelta(days=1)).date()),
                                     "train_races": int(df.loc[train, "race_id"].nunique()),
                                     "test_races": int(block["race_id"].nunique())}
        usable = history_oos.dropna(subset=["lpm"]) if not history_oos.empty else history_oos
        if usable.empty or usable["race_id"].nunique() < min_stage2_races:
            fold_info["status"] = "étape 2 non estimable (historique hors échantillon insuffisant) : bloc non joué"
            oos_parts.append(block)
            folds.append(fold_info)
            continue
        comb = MarketCombination.fit(usable["lpm"], usable["u_fund"], usable["race_id"], usable["won"],
                                     fit_period=f"< {lo.date()}", odds_timing=odds_timing)
        if dividend_col and dividend_col in usable.columns:
            w = usable[usable["won"] == 1]
            payout = PayoutFactor.fit(w[decision_odds_col], w[dividend_col], source=f"blocs < {lo.date()}")
        else:
            payout = PayoutFactor(source="neutre : aucun rapport définitif dans les données")
        model = SimpleGagnantModel(stage1=stage1, combination=comb, payout=payout)
        fold_info.update({"alpha": comb.alpha, "gamma": comb.gamma, "stage2_races": comb.n_races,
                          "payout_factors": payout.factors, "status": "évalué"})
        pred = model.predict(block, decision_odds_col=decision_odds_col)
        for col in ("won", "finish_position", "race_date", "hippodrome", "distance", "discipline"):
            if col in block.columns:
                pred[col] = block[col].to_numpy()
        if dividend_col and dividend_col in block.columns:
            pred["dividend"] = block[dividend_col].to_numpy()
        if minutes_col and minutes_col in block.columns:
            pred["minutes_to_start"] = block[minutes_col].to_numpy()
        pred_parts.append(pred)
        for rid, g in pred.groupby("race_id", sort=False):
            mts = float(g["minutes_to_start"].iloc[0]) if "minutes_to_start" in g.columns else None
            d = evaluate_race(g, cfg, mts if mts is not None and math.isfinite(mts) else None)
            row = {
                "race_id": rid, "race_date": g["race_date"].iloc[0], "discipline": g["discipline"].iloc[0],
                "family": g["family"].iloc[0], "hippodrome": g.get("hippodrome", pd.Series([""])).iloc[0],
                "distance": g.get("distance", pd.Series([np.nan])).iloc[0], "runners": len(g),
                "status": d.status, "reasons": "|".join(d.reasons),
            }
            c = d.candidate
            if c:
                hn = c.get("horse_number")
                won = int(g.loc[g["horse_number"] == hn, "won"].max()) if hn is not None else 0
                row.update({"horse_number": hn, "decision_odds": c.get("decision_odds"), "p_model": c.get("p_model"),
                            "p_market": c.get("p_market"), "ev": c.get("ev"), "ev_low": c.get("ev_low"),
                            "ev_drift": c.get("ev_drift"), "won": won})
                if "dividend" in g.columns:
                    row["dividend"] = g.loc[g["won"] == 1, "dividend"].max()
            fav = g.sort_values(["decision_odds", "horse_number"]).iloc[0]
            row.update({"fav_won": int(fav["won"]), "fav_odds": fav["decision_odds"],
                        "fav_dividend": fav.get("dividend", np.nan)})
            dec_rows.append(row)
        oos_parts.append(block)
        folds.append(fold_info)

    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    decisions = pd.DataFrame(dec_rows)
    bets = pd.DataFrame()
    benchmarks: dict[str, Any] = {}
    if not decisions.empty:
        settle_div = "dividend" in decisions.columns
        bets = decisions[decisions["status"].isin(BET_STATUSES)].copy()
        bets["stake"] = 1.0
        if not bets.empty:
            pay = bets["dividend"] if settle_div else bets["decision_odds"]
            bets["pnl"] = np.where(bets["won"] == 1, pay - 1.0, -1.0)
            bets["odds_bucket"] = _bucket_odds(pd.to_numeric(bets["decision_odds"], errors="coerce"))
            bets["field_bucket"] = pd.cut(bets["runners"], [0, 8, 12, 16, 30], labels=["≤8", "9-12", "13-16", "17+"])
            bets["distance_bucket"] = pd.cut(pd.to_numeric(bets["distance"], errors="coerce"),
                                             [0, 1400, 2000, 2800, 10000], labels=["≤1400", "1401-2000", "2001-2800", "2800+"])
        gated = decisions[~decisions["reasons"].fillna("").str.contains("COTES_INCOMPLETES|PELOTON|MARCHE_ANORMAL|FORME|DISCIPLINE")]
        fav = gated.copy()
        fav["stake"] = 1.0
        fav_pay = fav["fav_dividend"] if settle_div else fav["fav_odds"]
        fav["pnl"] = np.where(fav["fav_won"] == 1, fav_pay - 1.0, -1.0)
        fav["won"] = fav["fav_won"]
        fav["decision_odds"] = fav["fav_odds"]
        benchmarks["favori_toutes_courses_retenues"] = summarize_bets(fav, n_boot=n_boot)
        if not predictions.empty:
            p = predictions.dropna(subset=["p_model"])
            benchmarks["log_loss_par_course"] = {
                "marche": log_loss_per_race(p, "p_market"),
                "modele": log_loss_per_race(p, "p_model"),
                "fondamental_seul": log_loss_per_race(p, "p_fund"),
                "courses": int(p["race_id"].nunique()),
            }
    summary = {
        "odds_timing": odds_timing,
        "settlement": "rapport définitif fourni" if dividend_col else "cote de décision (proxy, pas un rapport PMU)",
        "races_analysed": int(len(decisions)),
        "status_counts": decisions["status"].value_counts().to_dict() if not decisions.empty else {},
        "races_excluded": int((decisions["status"] == STATUS_NO_BET).sum()) if not decisions.empty else 0,
        "watch": int((decisions["status"] == STATUS_WATCH).sum()) if not decisions.empty else 0,
    }
    summary.update(summarize_bets(bets, n_boot=n_boot) if not bets.empty else summarize_bets(pd.DataFrame()))
    segments = {}
    if not bets.empty:
        for col in ("discipline", "hippodrome", "distance_bucket", "field_bucket", "odds_bucket"):
            segments[col] = segment_table(bets, col)
        bets["month"] = pd.to_datetime(bets["race_date"]).dt.to_period("M").astype(str)
        segments["month"] = segment_table(bets, "month")
    calib = {}
    if not predictions.empty:
        calib["modele"] = calibration_table(predictions, "p_model")
        calib["marche"] = calibration_table(predictions, "p_market")
    return BacktestResult(predictions=predictions, decisions=decisions, bets=bets, folds=folds,
                          summary=summary, odds_timing=odds_timing, benchmarks=benchmarks,
                          segments=segments, calibration=calib)
