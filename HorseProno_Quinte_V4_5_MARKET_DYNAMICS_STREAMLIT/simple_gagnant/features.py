"""Variables pré-course par partant pour le moteur Simple Gagnant.

Règles anti-fuite appliquées ici :
- aucune colonne de résultat de la course analysée n'est lue (finish_position,
  rapports) ; seules les données de la fiche de course sont utilisées ;
- les statistiques jockey/entraîneur sont calculées uniquement sur des courses
  de dates strictement antérieures (merge_asof, allow_exact_matches=False) ;
- les cotes ne sont pas utilisées par le modèle fondamental : elles entrent
  uniquement dans la combinaison marché (model.py), à l'heure de la décision.

Le logit conditionnel compare les partants d'une même course : seules les
variables qui varient à l'intérieur d'une course comptent. Les variables
physiques sont donc exprimées relativement à la course.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .musique import MUSIQUE_FEATURES, discipline_family, musique_frame

CARD_COLUMNS = [
    "race_id", "race_date", "discipline", "distance", "horse_number",
    "horse_name", "jockey", "trainer", "draw", "weight", "age", "sex", "recent_form",
]

PHYSICAL_FEATURES = [
    "weight_rel",
    "weight_missing",
    "draw_rel",
    "draw_missing",
    "age_rel",
    "age_missing",
    "sex_female",
    "sex_gelding",
]

ENTITY_FEATURES = [
    "jockey_win",
    "jockey_place",
    "jockey_exp",
    "trainer_win",
    "trainer_place",
    "trainer_exp",
]

# Variables du modèle fondamental par famille de discipline. Les variables
# physiques sans variation dans une famille (poids au trot attelé, corde en
# obstacle) sont exclues pour éviter des coefficients non identifiables.
FAMILY_FEATURES: dict[str, list[str]] = {
    "PLAT": MUSIQUE_FEATURES + [
        "weight_rel", "draw_rel", "age_rel", "age_missing", "sex_female", "sex_gelding",
    ],
    "TROT": MUSIQUE_FEATURES + [
        "draw_rel", "age_rel", "age_missing", "sex_female", "sex_gelding",
    ],
    "OBSTACLE": MUSIQUE_FEATURES + [
        "weight_rel", "age_rel", "age_missing", "sex_female", "sex_gelding",
    ],
}

ENTITY_PRIOR_STRENGTH = 20.0


def _key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def normalise_card(card: pd.DataFrame) -> pd.DataFrame:
    out = card.copy()
    for col in CARD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["race_id"] = out["race_id"].astype(str)
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce").dt.normalize()
    out["discipline"] = out["discipline"].fillna("").astype(str).str.strip().str.upper()
    for col in ("distance", "horse_number", "draw", "weight", "age"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def physical_features(card: pd.DataFrame) -> pd.DataFrame:
    c = normalise_card(card).reset_index(drop=True)
    g = c.groupby("race_id", sort=False)
    out = pd.DataFrame(index=c.index)

    weight = c["weight"].where(c["weight"].between(30, 90))
    med = weight.groupby(c["race_id"]).transform("median")
    out["weight_rel"] = ((weight - med) / 5.0).fillna(0.0)
    out["weight_missing"] = weight.isna().astype(float)

    n = g["horse_number"].transform("size").astype(float)
    draw = c["draw"].where(c["draw"] >= 1)
    rel = (draw - 1.0) / (n - 1.0).clip(lower=1.0)
    autostart_or_flat = c["discipline"].isin(["PLAT", "ATTELE_AUTOSTART"])
    out["draw_rel"] = rel.where(autostart_or_flat).fillna(0.0).clip(0.0, 1.5)
    out["draw_missing"] = (draw.isna() & autostart_or_flat).astype(float)

    age = c["age"].where(c["age"].between(2, 20))
    mean_age = age.groupby(c["race_id"]).transform("mean")
    out["age_rel"] = (age - mean_age).fillna(0.0).clip(-6, 6)
    out["age_missing"] = age.isna().astype(float)

    sex = c["sex"].fillna("").astype(str).str.upper()
    out["sex_female"] = sex.str.startswith("F").astype(float)
    out["sex_gelding"] = sex.str.startswith("H").astype(float)
    return out


def placed_flag(finish_position: pd.Series, field_size: pd.Series) -> pd.Series:
    """Règle PMU du Simple Placé : 3 places à partir de 8 partants, 2 sinon."""
    pos = pd.to_numeric(finish_position, errors="coerce")
    n = pd.to_numeric(field_size, errors="coerce").fillna(8)
    limit = np.where(n >= 8, 3, 2)
    return (pos.between(1, 3) & (pos <= limit)).astype(float)


def entity_features(history: pd.DataFrame | None, card: pd.DataFrame) -> pd.DataFrame:
    """Taux de victoire/place jockey et entraîneur, strictement antérieurs.

    `history` : courses terminées (finish_position renseignée pour les arrivants,
    non-partants exclus). Lissage : (succès + m·p0) / (courses + m), m = 20.
    """
    c = normalise_card(card).reset_index(drop=True)
    out = pd.DataFrame(index=c.index)
    if history is None or history.empty:
        for col in ENTITY_FEATURES:
            out[col] = 0.0
        return _entity_defaults(out)

    h = history.copy()
    h["race_date"] = pd.to_datetime(h["race_date"], errors="coerce").dt.normalize()
    h = h[h["race_date"].notna()]
    if "is_non_runner" in h.columns:
        h = h[~h["is_non_runner"].fillna(False).astype(bool)]
    n_field = h.groupby("race_id")["horse_number"].transform("size")
    pos = pd.to_numeric(h["finish_position"], errors="coerce")
    h["_win"] = pos.eq(1).astype(float)
    h["_place"] = placed_flag(pos, n_field)
    h["_expected_win"] = 1.0 / n_field.clip(lower=1)

    for role, col in (("jockey", "jockey"), ("trainer", "trainer")):
        events = pd.DataFrame({
            "entity": _key(h[col]),
            "race_date": h["race_date"],
            "starts": 1.0,
            "wins": h["_win"],
            "places": h["_place"],
            "exp_wins": h["_expected_win"],
        })
        events = events[events["entity"].ne("")]
        daily = (
            events.groupby(["entity", "race_date"])[["starts", "wins", "places", "exp_wins"]]
            .sum()
            .groupby(level=0)
            .cumsum()
            .reset_index()
            .sort_values("race_date")
        )
        left = pd.DataFrame({"entity": _key(c[col]), "race_date": c["race_date"], "_idx": c.index})
        left = left.sort_values("race_date")
        if daily.empty:
            merged = left.assign(starts=0.0, wins=0.0, places=0.0, exp_wins=0.0)
        else:
            merged = pd.merge_asof(
                left, daily, on="race_date", by="entity", allow_exact_matches=False,
            )
        merged = merged.set_index("_idx").reindex(c.index)
        starts = merged["starts"].fillna(0.0)
        wins = merged["wins"].fillna(0.0)
        places = merged["places"].fillna(0.0)
        exp_wins = merged["exp_wins"].fillna(0.0)
        m = ENTITY_PRIOR_STRENGTH
        # Taux de victoire relatif à l'attendu "uniforme" (corrige la taille des pelotons).
        out[f"{role}_win"] = np.log((wins + m * 0.1) / (exp_wins + m * 0.1))
        out[f"{role}_place"] = (places + m * 0.3) / (starts + m) - 0.3
        out[f"{role}_exp"] = np.log1p(starts) / 6.0
    return out


def _entity_defaults(out: pd.DataFrame) -> pd.DataFrame:
    for col in ENTITY_FEATURES:
        out[col] = 0.0
    return out


def build_features(
    card: pd.DataFrame,
    history: pd.DataFrame | None = None,
    use_entities: bool = False,
) -> pd.DataFrame:
    """Matrice de variables alignée ligne à ligne sur `card` (index réinitialisé)."""
    c = normalise_card(card).reset_index(drop=True)
    mus = musique_frame(c["recent_form"].tolist(), c["discipline"].tolist())
    phys = physical_features(c)
    parts = [mus, phys]
    if use_entities:
        parts.append(entity_features(history, c))
    feats = pd.concat(parts, axis=1)
    feats.index = c.index
    feats.insert(0, "family", [discipline_family(d) for d in c["discipline"]])
    feats.insert(0, "race_id", c["race_id"].values)
    return feats


def feature_list(family: str, use_entities: bool = False) -> list[str]:
    base = list(FAMILY_FEATURES.get(family, FAMILY_FEATURES["PLAT"]))
    if use_entities:
        base += ENTITY_FEATURES
    return base
