"""Logit conditionnel régularisé (softmax intra-course), en numpy pur.

P(i gagne | course r) = exp(u_i) / Σ_{j ∈ r} exp(u_j),  u_i = x_i·β

- cohérence garantie : les probabilités d'une course somment à 1 ;
- régularisation L2 (a priori gaussien) sur les coefficients standardisés ;
- ajustement par Newton-Raphson avec recherche linéaire (problème convexe) ;
- covariance de Laplace (inverse de la hessienne pénalisée) pour quantifier
  l'incertitude des probabilités.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _group_index(groups) -> tuple[np.ndarray, np.ndarray]:
    """Codes de groupe 0..R-1 dans l'ordre d'apparition + nombre de groupes."""
    groups = np.asarray(groups)
    _, first_idx, codes = np.unique(groups, return_index=True, return_inverse=True)
    # Renumérote selon l'ordre d'apparition pour la stabilité des sorties.
    order = np.argsort(first_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[codes], len(order)


def group_softmax(utility: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    u = np.asarray(utility, dtype=float)
    gmax = np.full(n_groups, -np.inf)
    np.maximum.at(gmax, codes, u)
    e = np.exp(u - gmax[codes])
    denom = np.bincount(codes, weights=e, minlength=n_groups)
    return e / denom[codes]


def softmax_by_race(utility, groups) -> np.ndarray:
    codes, n_groups = _group_index(groups)
    return group_softmax(np.asarray(utility, dtype=float), codes, n_groups)


@dataclass
class ConditionalLogit:
    features: list[str]
    l2: float = 1.0
    max_iter: int = 100
    tol: float = 1e-8
    standardize: bool = True
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    cov_: np.ndarray | None = None
    n_races_: int = 0
    n_rows_: int = 0
    loglik_: float = float("nan")
    converged_: bool = False
    history_: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------ fit
    def _design(self, X) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return (x - self.mean_) / self.scale_

    def fit(self, X, groups, y) -> "ConditionalLogit":
        x = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if np.isnan(x).any():
            raise ValueError("Valeurs manquantes dans la matrice de variables.")
        codes, n_groups = _group_index(groups)
        winners = np.bincount(codes, weights=y, minlength=n_groups)
        keep_group = winners == 1.0
        if not keep_group.all():
            keep = keep_group[codes]
            x, y = x[keep], y[keep]
            codes, n_groups = _group_index(np.asarray(groups)[keep])

        if self.standardize:
            self.mean_ = x.mean(axis=0)
            sd = x.std(axis=0)
            self.scale_ = np.where(sd < 1e-9, 1.0, sd)
        else:
            self.mean_ = np.zeros(x.shape[1])
            self.scale_ = np.ones(x.shape[1])
        z = (x - self.mean_) / self.scale_
        d = z.shape[1]
        beta = np.zeros(d)
        lam = float(self.l2)
        win_x = (z * y[:, None]).sum(axis=0)

        def objective(b: np.ndarray) -> float:
            u = z @ b
            gmax = np.full(n_groups, -np.inf)
            np.maximum.at(gmax, codes, u)
            lse = gmax + np.log(np.bincount(codes, weights=np.exp(u - gmax[codes]), minlength=n_groups))
            ll = float((u * y).sum() - lse.sum())
            return -ll + 0.5 * lam * float(b @ b)

        f_old = objective(beta)
        self.history_ = [f_old]
        self.converged_ = False
        for _ in range(self.max_iter):
            p = group_softmax(z @ beta, codes, n_groups)
            mean_x = np.zeros((n_groups, d))
            np.add.at(mean_x, codes, z * p[:, None])
            grad = -(win_x - mean_x.sum(axis=0)) + lam * beta
            zp = z * np.sqrt(p)[:, None]
            hess = zp.T @ zp - mean_x.T @ mean_x + lam * np.eye(d)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            t = 1.0
            while t > 1e-6:
                cand = beta - t * step
                f_new = objective(cand)
                if f_new <= f_old - 1e-4 * t * float(grad @ step):
                    break
                t *= 0.5
            beta = cand
            self.history_.append(f_new)
            if abs(f_old - f_new) < self.tol * max(1.0, abs(f_old)):
                f_old = f_new
                self.converged_ = True
                break
            f_old = f_new

        p = group_softmax(z @ beta, codes, n_groups)
        mean_x = np.zeros((n_groups, d))
        np.add.at(mean_x, codes, z * p[:, None])
        zp = z * np.sqrt(p)[:, None]
        hess = zp.T @ zp - mean_x.T @ mean_x + lam * np.eye(d)
        self.coef_ = beta
        self.cov_ = np.linalg.pinv(hess)
        self.n_races_ = int(n_groups)
        self.n_rows_ = int(len(z))
        self.loglik_ = -(f_old - 0.5 * lam * float(beta @ beta))
        return self

    # -------------------------------------------------------------- predict
    def utility(self, X) -> np.ndarray:
        return self._design(X) @ self.coef_

    def predict_proba(self, X, groups) -> np.ndarray:
        return softmax_by_race(self.utility(X), groups)

    def utility_draws(self, X, n_draws: int = 200, seed: int = 0) -> np.ndarray:
        """Tirages des utilités selon la covariance de Laplace (n_draws × n)."""
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(self.coef_, self.cov_, size=n_draws, method="eigh")
        return draws @ self._design(X).T

    # ------------------------------------------------------------ artefact
    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "l2": float(self.l2),
            "mean": [float(v) for v in self.mean_],
            "scale": [float(v) for v in self.scale_],
            "coef": [float(v) for v in self.coef_],
            "cov": [[float(v) for v in row] for row in self.cov_],
            "n_races": int(self.n_races_),
            "n_rows": int(self.n_rows_),
            "loglik": float(self.loglik_),
            "converged": bool(self.converged_),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConditionalLogit":
        m = cls(features=list(d["features"]), l2=float(d.get("l2", 1.0)))
        m.mean_ = np.asarray(d["mean"], dtype=float)
        m.scale_ = np.asarray(d["scale"], dtype=float)
        m.coef_ = np.asarray(d["coef"], dtype=float)
        m.cov_ = np.asarray(d.get("cov") or np.zeros((len(m.coef_), len(m.coef_))), dtype=float)
        m.n_races_ = int(d.get("n_races", 0))
        m.n_rows_ = int(d.get("n_rows", 0))
        m.loglik_ = float(d.get("loglik", float("nan")))
        m.converged_ = bool(d.get("converged", True))
        return m

    def coefficients_table(self) -> list[dict[str, float]]:
        se = np.sqrt(np.clip(np.diag(self.cov_), 0, None))
        return [
            {"feature": f, "coef_std": float(b), "se": float(s), "z": float(b / s) if s > 0 else float("nan")}
            for f, b, s in zip(self.features, self.coef_, se)
        ]
