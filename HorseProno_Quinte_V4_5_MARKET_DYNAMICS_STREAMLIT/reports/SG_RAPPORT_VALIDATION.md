# Moteur Simple Gagnant — rapport des étapes B à H (29/10/2026)

Résumé : le modèle apporte une information très légèrement supérieure au marché PMU à T-24
(log-loss −0,0052 par course, ≈ 1,9 erreur-type, sur 600 courses de test), mais cet écart est
sans commune mesure avec le prélèvement (marge ≈ 18 %) et la dérive défavorable des rapports
(gagnants payés ≈ 10 % sous la cote snapshot). **Aucune configuration n'est validée** ; le moteur
s'abstient sur 97 % des courses de test et n'attribue jamais « OPPORTUNITÉ VALIDÉE ».

---

## Étape B — Cible, référence de marché, prévention des fuites

| Élément | Choix | Raison |
|---|---|---|
| Cible | victoire (finish_position = 1 / gagnant officiel e-Simple Gagnant) | objectif du pari |
| Probabilités | logit conditionnel (softmax par course) | somme = 1 par course, comparaisons intra-course |
| Référence | marché : 1/cote normalisée ; α libre ⇒ équivaut à la dé-margination « puissance » | benchmark systématique |
| Décision | cote snapshot pré-course horodatée (T-30 à T-10) | seule cote réellement disponible à la décision |
| Règlement | rapport définitif PMU e-Simple Gagnant | ce qui est réellement payé |
| Périodes | apprentissage étape 1 : 01/06 → 11/09 ; validation : 12 → 30/09 ; test gelé : 01 → 23/10 | séparation stricte |

Fichiers : `simple_gagnant/features.py`, `simple_gagnant/musique.py`, `simple_gagnant/market.py`.
Tests : `test_features_never_read_current_results`, `test_entity_stats_use_strictly_earlier_dates_only`,
`test_walk_forward_does_not_peek_at_evaluated_results`, `tests/test_musique.py`, `tests/test_market.py` — réussis.

## Étape C — Modèle probabiliste

**Étape 1 (fondamentale)** — `simple_gagnant/clogit.py`, `model.py`, `scripts/sg_train_stage1.py`.
Logit conditionnel L2 (Newton, numpy pur) par famille PLAT / TROT / OBSTACLE sur la musique
(forme lissée sur 5 courses, victoires, places, incidents, discipline des courses récentes, activité),
le poids et la corde relatifs, l'âge relatif et le sexe. 4 873 courses (2 155 PLAT, 2 524 TROT, 194 OBSTACLE).

Comparaisons sur l'historique (walk-forward hebdomadaire, cotes stockées proches de la clôture) :

| Variante | Log-loss fondamental seul | Log-loss combiné vs marché |
|---|---:|---:|
| Musique + physique | 2,2439 | 1,8733 vs 1,8778 |
| + statistiques jockey/entraîneur lissées | 2,1635 | 1,8742 vs 1,8778 |

Les statistiques d'entités améliorent le modèle fondamental mais **pas** le modèle combiné : elles
sont déjà intégrées dans la cote. Elles sont disponibles (`use_entities=True`) mais non retenues
(plus simple, et l'historique de résultats n'est plus alimenté depuis septembre).
Un gradient boosting n'a pas été retenu : LightGBM n'est pas installable dans l'environnement de
travail et, surtout, le volume (≈ 5 000 courses) et le gain nul des variables supplémentaires ne le
justifient pas au regard du critère « hors échantillon ».

**Étape 2 (combinaison marché)** estimée dans Supabase sur la validation (591 courses, snapshots) :
α = 0,958 (σ 0,063), γ = 0,148 (σ 0,086) ; gain de log-vraisemblance +1,5 (0,0026 par course).
Sur cotes proches de la clôture (historique), on obtient au contraire α ≈ 1,15 et γ ≈ −0,10 : la
relation forme/marché dépend de l'heure de la cote, d'où l'interdiction d'estimer l'étape 2 sur
des cotes de clôture.

**Facteur de rapport** (gagnants de la validation, rapport définitif / cote snapshot) :

| Cote snapshot | < 3 | 3–6 | 6–12 | 12–25 | ≥ 25 |
|---|---:|---:|---:|---:|---:|
| Gagnants | 112 | 160 | 192 | 97 | 30 |
| Moyenne géométrique brute | 1,128 | 0,890 | 0,823 | 0,876 | 0,800 |
| Facteur retenu (lissé) | 1,076 | 0,892 | 0,833 | 0,881 | 0,849 |
| Facteur défavorable (quartile bas) | 0,991 | 0,717 | 0,626 | 0,662 | 0,520 |

Artefacts : `models/sg_stage1_until_20260911.json` (empreinte 96b057bc…), `models/sg_model_v1.json`
(empreinte 9f773de2…) — **reconstruits à l'identique** par `sg_train_stage1.py` puis
`sg_assemble_artifact.py` (vérifié). Parité Python ↔ SQL de l'utilité fondamentale : écart maximal
5 × 10⁻⁷ sur 53 partants (`tests/fixtures/forward_parity_sample.csv`).

## Étape D — Sélection des courses et NO BET

Fichiers : `simple_gagnant/selector.py`, `simple_gagnant/config.py`, `models/sg_selector_v1.json`.
Portes (marché complet, 5–20 partants, Σ 1/cote ∈ [1,08 ; 1,40], ≤ 40 % de musiques absentes,
course non partie), candidat = meilleure EV parmi p ≥ 5 % et cote ≤ 30, échelle EV, EV prudente
(incertitude α, γ), EV sous dérive défavorable, fenêtre T-30/T-2, segment validé. Chaque abstention
porte un code et un texte. Indicateurs de course : entropie, écart top 2, KL modèle/marché,
désaccord maximal, accord des favoris.

Sélection des seuils sur la **validation uniquement** (288 combinaisons p_min × cote max × seuil EV ×
robustesse) : 563 courses retenues, 29 candidats à EV > 0 (10 gagnants, −7,8 u), 10 à EV ≥ 5 %
(1 gagnant, −7,7 u), 3 à EV ≥ 10 %, **aucun candidat robuste**. Aucune configuration n'ayant de ROI
positif, la configuration prudente a été gelée et `validated_segments` laissé vide
(fichier `reports/forward_eval/policy_grid_validation_20260912_20260930.txt`).
Tests : `tests/test_selector.py` (portes, échelle, favori ni imposé ni exclu, outsider inéligible) — réussis.

## Étape E — Backtest chronologique

**E1. Test prospectif gelé (01–23/10/2026, 600 courses)** — décision sur snapshot, règlement officiel :

| Indicateur | Valeur |
|---|---|
| Courses analysées / écartées | 600 / 582 (24 portes qualité, 558 EV ≤ 0) |
| Surveillance / candidats à tester / validées | 14 / 4 / 0 |
| Candidats à tester (non joués en réel) | 4 paris, 2 gagnants, −0,2 u (échantillon non interprétable) |
| Log-loss par course : uniforme / fondamental / marché / modèle | 2,3963 / 2,1974 / 1,9723 / **1,9671** |
| Gain vs marché | +0,0052 ± 0,0028 par course |
| Modèle M8 existant, mêmes courses | 1,9614 (gain +0,0110 ± 0,0066) |
| Favori de chaque course | 576 paris, 180 gagnants, −64,0 u, ROI −11,1 % |
| Premier du modèle dans chaque course | 576 paris, 181 gagnants, −68,7 u, ROI −11,9 % |

Par discipline (gain de log-loss vs marché, erreur-type) : PLAT +0,0064 ± 0,0036 (274) ;
attelé volté +0,0053 ± 0,0065 (128) ; autostart +0,0083 ± 0,0086 (99) ; haies +0,0094 ± 0,0095 (37) ;
monté −0,0090 ± 0,0149 (35) ; steeple −0,0055 ± 0,0093 (24). Aucun segment n'est significatif.

**E2. Walk-forward sur l'historique embarqué** (`scripts/sg_backtest.py`, 11 blocs hebdomadaires,
06/07 → 16/09 ; ⚠ cotes stockées proches de la clôture, règlement au proxy de la cote, non
exploitable comme rentabilité) : 2 468 courses ; 2 415 NO BET, 40 surveillance, 13 candidats
(1 gagnant, −9,6 u) ; log-loss modèle 1,8649 vs marché 1,8703 ; favori : ROI −13,2 %
(IC 90 % bootstrap par jour −17,4 % à −8,7 %). Calibration du modèle conforme par tranche
(ex. 10–15 % : prédit 12,2 %, observé 12,5 %). Sorties : `reports/history_backtest/`
(décisions, paris, segments discipline/hippodrome/distance/partants/tranche de cote/mois, calibration).

Interprétation : l'information fondamentale apporte ≈ 0,005 nat par course. Un pari n'est positif
que si p_modèle > p_marché × Σ(1/cote) / facteur, soit ≈ +10 % relatifs pour une cote < 3
(1,18 / 1,076) et ≈ +32 % à +42 % entre 3 et 12 contre 1 (1,18 / 0,89 ; 1,18 / 0,83). Le modèle
ne s'écarte presque jamais autant du marché, et ces rares écarts n'ont pas été rentables.

## Étape F — Intégration

| Fichier | Rôle |
|---|---|
| `pages/1_🎯_Simple_Gagnant.py` | Page Streamlit : courses du jour, statut, meilleur candidat même non recommandé, p / cote / cote juste / EV, arguments pour/contre, alertes, motifs, sensibilité à la cote, mouvements de cote, journal, validation |
| `simple_gagnant/engine.py`, `ui.py`, `pmu.py` | Orchestration, mise en forme, collecte PMU (sans dépendance Streamlit) |
| `simple_gagnant/journal.py` | Photographies immuables, règlement séparé, suivi de dérive |
| `supabase/sg_simple_gagnant_migration.sql` | Tables additives + trigger d'immuabilité + vue `sg_journal` (**non appliquée**) |
| `streamlit_app.py` | Seule modification : correctif `V3_REPORT_PATH` → `V4_REPORT_PATH` |

## Étape G — Tests, dépendances, documentation

`python -m pytest tests -q` → **35 tests réussis** (unitaires, parité SQL, anti-fuite, backtest,
PMU sur rapports réels et fixtures, journal, cycle capture/règlement, page Streamlit et application
Quinté exécutées de bout en bout avec un substitut de Streamlit). Les 6 smoke tests Quinté
existants passent toujours. Aucune dépendance d'exécution ajoutée ; `requirements-dev.txt`
(pytest) et `requirements-observe.txt` (CI). Documentation : `README_SIMPLE_GAGNANT.md`,
`reports/SG_AUDIT.md`, ce rapport.

Non vérifiable ici : appels réseau réels vers PMU/Supabase depuis le code Python (bloqués dans
l'environnement de travail ; formats vérifiés par lecture des endpoints et des données stockées),
exécution sous Streamlit réel, pandas 2.x, exécution du workflow GitHub.

## Étape H — Observation contrôlée

`.github/workflows/sg_observation.yml` (racine du dépôt) : toutes les 10 min de 8 h à 22 h UTC,
`sg_observe.py capture` (une photographie par course et par phase T-30/T-15/T-5) puis `settle`
(rapport définitif, non-partant → remboursement). Refuse de tourner en CI sans Supabase pour ne
pas perdre le journal. Prérequis : appliquer la migration et ajouter les secrets.
Critère de réévaluation proposé : après 1 500 courses réglées, ré-estimer α, γ et le facteur de
rapport sur les seules photographies, puis geler un nouveau test de 4 semaines.
