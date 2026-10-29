"""Modèle Simple Gagnant en deux étapes (approche de type Benter).

Étape 1 — modèle fondamental : logit conditionnel par famille de discipline
(PLAT, TROT, OBSTACLE) sur les variables de la fiche de course, sans les cotes.
Il produit une utilité u_fund par partant.

Étape 2 — combinaison avec le marché, à l'heure de la décision :
    u_i = α · log(p_marché_i) + γ · u_fund_i ;  p_i = softmax_course(u)
α corrige le biais favori-outsider ; γ mesure l'information fondamentale
non déjà intégrée dans la cote. α et γ doivent être estimés sur des cotes
observées au même moment que celles utilisées en production (snapshots
pré-course), jamais sur des cotes de clôture.

Rapport payé : la cote observée à la décision n'est pas le rapport définitif.
`payout_factor` = E[rapport définitif / cote observée | victoire], estimé par
tranche de cote sur la période de validation. EV = p · cote · facteur − 1.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clogit import ConditionalLogit, softmax_by_race
from .features import build_features, feature_list
from .musique import discipline_family

FAMILIES = ("PLAT", "TROT", "OBSTACLE")
DEFAULT_ODDS_BUCKETS = [1.0, 3.0, 6.0, 12.0, 25.0, 1e9]
EXPLAIN_FEATURES = [
    "mus_missing", "mus_score5", "mus_last", "mus_win5", "mus_place5", "mus_inc5",
    "weight_rel", "draw_rel", "age_rel",
]


@dataclass
class PayoutFactor:
    """Facteur multiplicatif cote observée → rapport définitif, par tranche de cote."""

    edges: list[float] = field(default_factory=lambda: list(DEFAULT_ODDS_BUCKETS))
    factors: list[float] = field(default_factory=lambda: [1.0] * (len(DEFAULT_ODDS_BUCKETS) - 1))
    low_factors: list[float] = field(default_factory=lambda: [1.0] * (len(DEFAULT_ODDS_BUCKETS) - 1))
    counts: list[int] = field(default_factory=lambda: [0] * (len(DEFAULT_ODDS_BUCKETS) - 1))
    source: str = "neutre (aucune estimation)"

    def _bucket(self, odds: float) -> int:
        for i in range(len(self.edges) - 1):
            if self.edges[i] <= odds < self.edges[i + 1]:
                return i
        return len(self.edges) - 2

    def factor(self, odds: float) -> float:
        if odds is None or not np.isfinite(odds):
            return float("nan")
        return float(self.factors[self._bucket(float(odds))])

    def low_factor(self, odds: float) -> float:
        if odds is None or not np.isfinite(odds):
            return float("nan")
        return float(self.low_factors[self._bucket(float(odds))])

    @classmethod
    def fit(cls, decision_odds, dividends, edges=None, shrink: float = 30.0, source: str = "") -> "PayoutFactor":
        """Estimation sur les gagnants : moyenne géométrique de dividende/cote par tranche.

        Lissage vers la moyenne globale (pseudo-effectif `shrink`) ; facteur bas =
        quantile 25 % des ratios de la tranche (scénario défavorable de dérive).
        """
        edges = list(edges or DEFAULT_ODDS_BUCKETS)
        o = np.asarray(decision_odds, dtype=float)
        d = np.asarray(dividends, dtype=float)
        ok = np.isfinite(o) & np.isfinite(d) & (o > 1) & (d > 0)
        o, d = o[ok], d[ok]
        lr = np.log(d / o)
        global_mean = float(lr.mean()) if len(lr) else 0.0
        global_q25 = float(np.quantile(lr, 0.25)) if len(lr) else 0.0
        factors, lows, counts = [], [], []
        for i in range(len(edges) - 1):
            m = (o >= edges[i]) & (o < edges[i + 1])
            n = int(m.sum())
            mean = (lr[m].sum() + shrink * global_mean) / (n + shrink) if n else global_mean
            q25 = float(np.quantile(lr[m], 0.25)) if n >= 20 else global_q25
            factors.append(float(np.exp(mean)))
            lows.append(float(np.exp(min(q25, mean))))
            counts.append(n)
        return cls(edges=edges, factors=factors, low_factors=lows, counts=counts, source=source)

    @classmethod
    def from_summary(cls, summary: dict[str, Any], shrink: float = 30.0, source: str = "") -> "PayoutFactor":
        """Même estimateur que `fit`, à partir des statistiques agrégées par tranche
        (moyenne et quantile 25 % de log(dividende/cote)) calculées dans la base."""
        edges = [float(e) for e in summary["edges"]]
        g_mean = float(summary["global"]["mean_lr"])
        g_q25 = float(summary["global"]["q25_lr"])
        by = {int(b["bucket"]): b for b in summary["buckets"]}
        factors, lows, counts = [], [], []
        for i in range(len(edges) - 1):
            b = by.get(i)
            n = int(b["n"]) if b else 0
            mean = (float(b["mean_lr"]) * n + shrink * g_mean) / (n + shrink) if n else g_mean
            q25 = float(b["q25_lr"]) if n >= 20 else g_q25
            factors.append(float(np.exp(mean)))
            lows.append(float(np.exp(min(q25, mean))))
            counts.append(n)
        return cls(edges=edges, factors=factors, low_factors=lows, counts=counts, source=source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": [float(e) for e in self.edges],
            "factors": [float(f) for f in self.factors],
            "low_factors": [float(f) for f in self.low_factors],
            "counts": [int(c) for c in self.counts],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PayoutFactor":
        if not d:
            return cls()
        return cls(
            edges=list(d["edges"]),
            factors=list(d["factors"]),
            low_factors=list(d.get("low_factors", d["factors"])),
            counts=list(d.get("counts", [0] * len(d["factors"]))),
            source=str(d.get("source", "")),
        )


@dataclass
class MarketCombination:
    alpha: float = 1.0
    gamma: float = 0.0
    cov: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    n_races: int = 0
    fit_period: str = ""
    odds_timing: str = ""

    def utility(self, log_pm: np.ndarray, u_fund: np.ndarray, alpha=None, gamma=None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        g = self.gamma if gamma is None else gamma
        return a * np.asarray(log_pm, dtype=float) + g * np.asarray(u_fund, dtype=float)

    def parameter_corners(self, z: float = 1.645) -> list[tuple[float, float]]:
        """Scénarios (α, γ) aux bornes d'un intervalle de confiance ≈ 90 %."""
        cov = np.asarray(self.cov, dtype=float)
        se_a = float(np.sqrt(max(cov[0, 0], 0.0)))
        se_g = float(np.sqrt(max(cov[1, 1], 0.0)))
        corners = [(self.alpha, self.gamma)]
        for da in (-1, 1):
            for dg in (-1, 1):
                corners.append((self.alpha + da * z * se_a, self.gamma + dg * z * se_g))
        return corners

    @classmethod
    def fit(cls, log_pm, u_fund, groups, y, fit_period: str = "", odds_timing: str = "") -> "MarketCombination":
        x = np.column_stack([np.asarray(log_pm, float), np.asarray(u_fund, float)])
        m = ConditionalLogit(["log_pm", "u_fund"], l2=1e-4, standardize=False).fit(x, groups, y)
        return cls(
            alpha=float(m.coef_[0]),
            gamma=float(m.coef_[1]),
            cov=[[float(v) for v in row] for row in m.cov_],
            n_races=int(m.n_races_),
            fit_period=fit_period,
            odds_timing=odds_timing,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha, "gamma": self.gamma, "cov": self.cov,
            "n_races": self.n_races, "fit_period": self.fit_period, "odds_timing": self.odds_timing,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MarketCombination":
        if not d:
            return cls()
        return cls(
            alpha=float(d["alpha"]), gamma=float(d["gamma"]),
            cov=[[float(v) for v in row] for row in d.get("cov", [[0, 0], [0, 0]])],
            n_races=int(d.get("n_races", 0)), fit_period=str(d.get("fit_period", "")),
            odds_timing=str(d.get("odds_timing", "")),
        )


@dataclass
class SimpleGagnantModel:
    stage1: dict[str, ConditionalLogit]
    combination: MarketCombination
    payout: PayoutFactor
    use_entities: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- training
    @classmethod
    def fit_stage1(
        cls,
        history: pd.DataFrame,
        train_end_exclusive: str,
        l2: float = 5.0,
        use_entities: bool = False,
        min_races: int = 50,
    ) -> dict[str, ConditionalLogit]:
        h = history.copy()
        h["race_date"] = pd.to_datetime(h["race_date"], errors="coerce").dt.normalize()
        h = h[h["race_date"] < pd.Timestamp(train_end_exclusive)].sort_values(["race_date", "race_id", "horse_number"])
        if "is_non_runner" in h.columns:
            h = h[~h["is_non_runner"].fillna(False).astype(bool)]
        h = h.reset_index(drop=True)
        feats = build_features(h, history=h if use_entities else None, use_entities=use_entities)
        y = pd.to_numeric(h["finish_position"], errors="coerce").eq(1).astype(float).to_numpy()
        models: dict[str, ConditionalLogit] = {}
        for fam in FAMILIES:
            cols = feature_list(fam, use_entities)
            mask = (feats["family"] == fam).to_numpy()
            if h.loc[mask, "race_id"].nunique() < min_races:
                continue
            models[fam] = ConditionalLogit(cols, l2=l2).fit(
                feats.loc[mask, cols].to_numpy(dtype=float), h.loc[mask, "race_id"].to_numpy(), y[mask]
            )
        return models

    # ------------------------------------------------------------ inference
    def fundamental_utility(self, card: pd.DataFrame, history: pd.DataFrame | None = None, feats: pd.DataFrame | None = None) -> pd.Series:
        c = card.reset_index(drop=True)
        if feats is None:
            feats = build_features(c, history=history, use_entities=self.use_entities)
        u = pd.Series(0.0, index=c.index)
        known = pd.Series(False, index=c.index)
        for fam, m in self.stage1.items():
            mask = feats["family"] == fam
            if mask.any():
                u[mask] = m.utility(feats.loc[mask, m.features].to_numpy(dtype=float))
                known[mask] = True
        u[~known] = 0.0
        return u

    def predict(
        self,
        card: pd.DataFrame,
        decision_odds_col: str = "odds",
        history: pd.DataFrame | None = None,
        n_uncertainty: int = 0,
    ) -> pd.DataFrame:
        """Probabilités par partant pour une ou plusieurs courses.

        Les courses dont une cote manque ne reçoivent pas de probabilité combinée
        (le marché n'est pas complet) : seules les colonnes fondamentales sont
        renseignées, la décision sera NO BET.
        """
        c = card.reset_index(drop=True).copy()
        c["race_id"] = c["race_id"].astype(str)
        odds = pd.to_numeric(c[decision_odds_col], errors="coerce").where(lambda s: s > 1.0)
        c["decision_odds"] = odds
        inv = 1.0 / odds
        complete = inv.notna().groupby(c["race_id"]).transform("all")
        c["market_complete"] = complete
        c["booksum"] = inv.groupby(c["race_id"]).transform("sum").where(complete)
        c["p_market_raw"] = inv
        c["p_market"] = (inv / c["booksum"]).where(complete)
        feats = build_features(c, history=history, use_entities=self.use_entities)
        for col in EXPLAIN_FEATURES:
            if col in feats.columns:
                c[col] = feats[col].to_numpy()
        c["u_fund"] = self.fundamental_utility(c, feats=feats).to_numpy()
        c["p_fund"] = softmax_by_race(c["u_fund"].to_numpy(), c["race_id"].to_numpy())
        c["family"] = [discipline_family(d) for d in c.get("discipline", pd.Series([""] * len(c)))]

        c["p_model"] = np.nan
        c["p_low"] = np.nan
        c["p_high"] = np.nan
        ok = c["market_complete"].to_numpy() & np.isfinite(c["p_market"].to_numpy())
        if ok.any():
            sub = c.loc[ok]
            lpm = np.log(sub["p_market"].to_numpy())
            uf = sub["u_fund"].to_numpy()
            groups = sub["race_id"].to_numpy()
            c.loc[ok, "p_model"] = softmax_by_race(self.combination.utility(lpm, uf), groups)
            corners = [softmax_by_race(self.combination.utility(lpm, uf, a, g), groups)
                       for a, g in self.combination.parameter_corners()]
            stack = np.vstack(corners)
            c.loc[ok, "p_low"] = stack.min(axis=0)
            c.loc[ok, "p_high"] = stack.max(axis=0)
        c["payout_factor"] = [self.payout.factor(o) if np.isfinite(o) else np.nan for o in c["decision_odds"].fillna(np.nan)]
        c["payout_factor_low"] = [self.payout.low_factor(o) if np.isfinite(o) else np.nan for o in c["decision_odds"].fillna(np.nan)]
        c["fair_odds"] = 1.0 / c["p_model"]
        c["ev_naive"] = c["p_model"] * c["decision_odds"] - 1.0
        c["ev"] = c["p_model"] * c["decision_odds"] * c["payout_factor"] - 1.0
        c["ev_low"] = c["p_low"] * c["decision_odds"] * c["payout_factor"] - 1.0
        c["ev_drift"] = c["p_model"] * c["decision_odds"] * c["payout_factor_low"] - 1.0
        c["edge"] = c["p_model"] - c["p_market"]
        c["break_even_odds"] = 1.0 / (c["p_model"] * c["payout_factor"])
        c["model_rank"] = c.groupby("race_id")["p_model"].rank(ascending=False, method="first")
        c["market_rank"] = c.groupby("race_id")["p_market_raw"].rank(ascending=False, method="first")
        return c

    # ------------------------------------------------------------- artefact
    def to_dict(self) -> dict[str, Any]:
        body = {
            "engine": "HorseProno_SimpleGagnant",
            "stage1": {fam: m.to_dict() for fam, m in self.stage1.items()},
            "combination": self.combination.to_dict(),
            "payout": self.payout.to_dict(),
            "use_entities": self.use_entities,
            "metadata": self.metadata,
        }
        return body

    def fingerprint(self) -> str:
        payload = json.dumps(
            {k: v for k, v in self.to_dict().items() if k != "metadata"}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> str:
        d = self.to_dict()
        d["fingerprint"] = self.fingerprint()
        Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        return d["fingerprint"]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimpleGagnantModel":
        return cls(
            stage1={fam: ConditionalLogit.from_dict(v) for fam, v in d.get("stage1", {}).items()},
            combination=MarketCombination.from_dict(d.get("combination")),
            payout=PayoutFactor.from_dict(d.get("payout")),
            use_entities=bool(d.get("use_entities", False)),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleGagnantModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
