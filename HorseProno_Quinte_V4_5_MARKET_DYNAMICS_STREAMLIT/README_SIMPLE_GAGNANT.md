# HorseProno — Moteur Simple Gagnant (v1.0, mode observation)

Module isolé de la stratégie Quinté (`simple_gagnant/`). Il analyse chaque course, choisit le
meilleur candidat Simple Gagnant au regard de sa cote, et **s'abstient par défaut**.
Aucun pari n'est placé.

> État au 29/10/2026 : sur la période de test gelée (1er–23/10/2026, 600 courses, cotes
> pré-course horodatées, rapports définitifs officiels), le modèle ne démontre **aucun avantage
> exploitable**. Le statut « OPPORTUNITÉ VALIDÉE PAR LE MODÈLE » est donc désactivé
> (`validated_segments` vide). Détails : `reports/SG_RAPPORT_VALIDATION.md`.

## Utilisation

| Besoin | Commande / accès |
|---|---|
| Interface | Streamlit : page **🎯 Simple Gagnant** (menu latéral de `streamlit_app.py`) |
| Observation planifiée | `.github/workflows/sg_observation.yml` (racine du dépôt) |
| Capture manuelle | `python scripts/sg_observe.py capture` |
| Règlement officiel | `python scripts/sg_observe.py settle` |
| Indicateurs prospectifs | `python scripts/sg_observe.py summary` |
| Backtest walk-forward | `python scripts/sg_backtest.py --runners … --odds-timing "…" --out …` |
| Export prospectif Supabase | `python scripts/sg_export_forward.py --start … --end … --with-history --out …` |
| Évaluation SQL (lecture seule) | `python scripts/sg_forward_sql.py --artifact … --mode grid|payout|policy|test|logloss` |
| Tests | `pip install -r requirements-dev.txt && python -m pytest tests -q` |

Aucune dépendance d'exécution nouvelle : numpy, pandas, requests, streamlit (déjà présents).

## Architecture

```
PMU (programme, participants, rapports-definitifs)
   └─ simple_gagnant/pmu.py        collecte + normalisation, non-partants exclus, cotes horodatées
        └─ features.py / musique.py variables pré-course (aucun résultat de la course lue)
             └─ model.py            étape 1 fondamentale (logit conditionnel par famille)
                                    étape 2 combinaison marché  u = α·log p_marché + γ·u_fond
                                    facteur de rapport (cote observée → rapport définitif)
                  └─ selector.py    portes de qualité → candidat → statut mesurable
                       └─ journal.py photographie immuable + règlement séparé + suivi de dérive
backtest.py   validation chronologique (walk-forward), bootstrap par jour, segments
engine.py     orchestration ; ui.py mise en forme ; config.py seuils versionnés
```

Artefacts : `models/sg_model_v1.json` (empreinte SHA-256 vérifiée au chargement),
`models/sg_selector_v1.json` (seuils gelés), `models/sg_stage1_until_20260911.json` (étape 1 seule).

## Probabilités, cotes et espérance

Pour chaque partant : probabilité implicite brute `1/cote`, probabilité de marché sans marge
(proportionnelle ; la méthode « puissance » est disponible dans `market.py`), probabilité
fondamentale, probabilité du modèle, cote juste `1/p`, écart modèle − marché, et :

```
EV = p_modèle × cote_observée × facteur_rapport(tranche de cote) − 1
EV prudente      : p minimale sur les scénarios (α ± 1,645σ, γ ± 1,645σ)
EV dérive défav. : facteur de rapport au quartile bas des gagnants
```

Le facteur de rapport est indispensable au PMU (pari mutuel) : sur la validation, le rapport
définitif des gagnants est en moyenne **10 % sous** la cote snapshot T-24 (−11 % à −18 % entre
3 et 25 contre 1), mais **+13 %** pour les cotes < 3.

## Statuts (critères mesurables, `models/sg_selector_v1.json`)

1. **NO BET — COURSE ÉCARTÉE** si une porte échoue : marché incomplet, 5–20 partants,
   somme des 1/cote ∉ [1,08 ; 1,40], > 40 % de musiques absentes, course partie,
   discipline non couverte ; ou aucun candidat (p ≥ 5 %, cote ≤ 30) ; ou EV ≤ 0.
2. **SURVEILLANCE** : 0 < EV < 5 %, ou cote relevée à plus de T-30.
3. **CANDIDAT À TESTER** : EV ≥ 5 % sans toutes les preuves ci-dessous.
4. **OPPORTUNITÉ VALIDÉE PAR LE MODÈLE** : EV ≥ 10 %, EV prudente ≥ 0, EV sous dérive
   défavorable ≥ 0, lecture entre T-30 et T-2, **et** famille de discipline listée dans
   `validated_segments` (borne basse de l'IC 90 % du ROI hors échantillon > 0 avec ≥ 100 paris).

Le favori n'est ni imposé ni exclu : seul le rapport probabilité/cote compte.

## Règles anti-fuite

- Aucune variable ne lit le résultat de la course (test `test_features_never_read_current_results`).
- Musique : publiée avant la course (vérifié : le 1er jeton = place de la course *précédente* dans 92 % des cas).
- Statistiques d'entités : dates strictement antérieures (`merge_asof`, `allow_exact_matches=False`).
- α, γ et le facteur de rapport sont estimés sur des cotes **pré-course horodatées** ; les cotes
  stockées de l'historique (proches de la clôture) ne servent jamais à les estimer.
- Seuils choisis sur la validation (12–30/09) puis gelés avant le test (01–23/10).
- Walk-forward : changer les arrivées d'un bloc ne modifie aucune prédiction de ce bloc (test dédié).
- Photographies immuables (trigger SQL) ; règlement dans une table séparée.

## Réentraînement (pas après chaque course)

- **Étape 1** : mensuelle, fenêtre glissante de 4 mois,
  `sg_train_stage1.py --train-start <J-120> --train-end <1er du mois>` (historique de résultats à jour requis).
- **Étape 2 + facteur de rapport** : après 3 semaines de photographies pré-course réglées, sur
  ces seules courses (`sg_forward_sql.py --mode grid/payout` ou `sg_backtest.py` sur l'export).
- **Promotion** : une nouvelle version n'est utilisée qu'après une période de test gelée d'au moins
  4 semaines, comparée au marché (log-loss) et au favori (ROI). Les versions coexistent dans le
  journal grâce à `model_fingerprint`.
- **Alerte de dérive** (`monitoring_summary` → `drift_alert`) : gain de log-loss modèle − marché
  négatif sur au moins 300 courses réglées. Le ROI simulé par statut est suivi en parallèle.

## Déploiement

1. Supabase : appliquer `supabase/sg_simple_gagnant_migration.sql` (tables additives
   `sg_predictions`, `sg_prediction_runners`, `sg_settlements`, vue `sg_journal`).
2. GitHub → Settings → Secrets : `SUPABASE_URL`, `SUPABASE_SECRET_KEY` (et `SUPABASE_PUBLISHABLE_KEY`).
3. Streamlit Cloud : rien à changer (`streamlit_app.py` reste le fichier principal ; la page est
   détectée automatiquement dans `pages/`). Secrets Streamlit facultatifs pour lire le journal.

## Limites connues

- Cotes pré-course : un seul instantané par course (T-30 à T-10) dans les données existantes ;
  la dynamique des cotes ne peut pas encore être testée historiquement (la table
  `quinte_market_snapshots` est vide). Le workflow d'observation capture désormais 3 phases.
- Variables de carrière PMU (`nombreCourses`, gains, `handicapDistance`…) collectées dans le
  journal mais absentes de l'historique : non utilisées tant qu'elles ne sont pas historisées.
- Tests exécutés avec pandas 3.0.5 / numpy 2.5 ; `requirements.txt` impose pandas < 3 :
  le code n'utilise pas d'API propre à pandas 3, mais cette combinaison n'a pas pu être exécutée ici.
- Streamlit n'a pas pu être installé dans l'environnement de développement : les pages ont été
  exécutées de bout en bout avec un substitut de Streamlit (`tests/fake_streamlit.py`).
