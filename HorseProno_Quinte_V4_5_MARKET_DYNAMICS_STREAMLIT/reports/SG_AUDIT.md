# Étape A — Audit du dépôt et inventaire des données (29/10/2026)

Périmètre examiné : `azmourad067-stack/GENSPARK`, dossier `HorseProno_Quinte_V4_5_MARKET_DYNAMICS_STREAMLIT`
(commit 635fe55, unique commit « Add files via upload ») et le projet Supabase connecté
`gwnpqsbtkubtumrfnspd` (lecture seule, aucune écriture effectuée).

## 1. Architecture du code

| Fichier | Rôle vérifié dans le code |
|---|---|
| `streamlit_app.py` | Application unique : programme PMU, choix réunion/course, pronostic Quinté, snapshots Supabase. Code d'interface exécuté au niveau module (non importable). |
| `quinte_candidate.py` | `prepare()` : variables pré-course (cotes, poids, corde, âge, sexe, taux cheval/jockey/entraîneur lissés, forme). Jointures strictement antérieures (`merge_asof`, `allow_exact_matches=False`). |
| `quinte_v2.py` | Logistique Top 5 calibrée (Platt) + variables relatives ; shortlist 6 marché + 1 challenger. |
| `quinte_v3.py` | Score consensus marché/logit/LambdaMART/contexte ; 4 noyaux + 3 challengers. |
| `quinte_v42.py` | Fusion V2/V3 ; pondération marché PLAT selon l'écart de poids. |
| `quinte_v43.py` | Détecteur d'outsider (logistique, mode « shadow »). |
| `quinte_v44.py` / `quinte_v45.py` | Réordonnancement du Top 7 PLAT ; ajustement « dynamique de marché » borné ±0,10. |
| `portable_lgbm.py` | Lecteur Python pur du modèle LightGBM (`quinte_v2_ranker.txt`). |
| `train_v45_market_dynamics.py` | Régression ridge T-30→T-2 ; exige ≥ 50 paires de snapshots. |
| `validated_history.csv` | 57 892 partants, 5 023 courses, 01/06 → 16/09/2026, toutes disciplines. |

Tous les modèles existants ciblent le **Top 5 / Quinté**. Aucun ne produit de probabilité de
victoire cohérente par course ni de décision d'abstention : la stratégie Simple Gagnant devait
être construite à part (module `simple_gagnant/`).

## 2. Sources et variables

- PMU `online.turfinfo.api.pmu.fr/rest/client/1` : programme, détail de course, participants
  (formats vérifiés le 29/10/2026 ; l'endpoint `rapports-definitifs` également).
- Historique embarqué : `odds`, `draw`, `weight`, `age`, `sex`, `recent_form` (musique),
  `jockey`, `trainer`, `finish_position` (positions complètes, 1 à 5 garanties), terrain toujours
  `INCONNU`. Cotes manquantes : 20 % des lignes ; 3 471 courses ont un marché complet.
- **Nature des cotes historiques** : comparées aux rapports définitifs e-Simple Gagnant des
  100 gagnants Quinté disponibles, l'écart moyen |log| est 0,126 (contre 0,36 pour un snapshot
  à T-24) → cotes **proches de la clôture, non horodatées**. Elles ne peuvent pas servir à
  mesurer une rentabilité au moment de la décision.
- **Musique** : publiée avant la course (le 1er jeton reproduit la place de la course précédente
  du cheval dans 92 % des cas ; aucune fuite du résultat courant). Elle n'est **pas utilisée**
  par les modèles Quinté (`music_score` est défini mais jamais appelé).

## 3. Supabase (inspecté, non modifié)

| Table | Contenu constaté |
|---|---|
| `races` / `participants` | 7 935 courses. 5 399 « finished » (≤ 08/09) ; 2 536 « scheduled » (11/09 → 25/10) dont seulement 387 avec arrivée renseignée. |
| `forward_model_predictions` | 56 302 lignes, 2 532 courses (12/09 → 25/10) : **un snapshot de cote par course entre T-30 et T-10 (médiane T-24)**, probabilités des modèles #8 et #9. Seule source de cotes pré-course horodatées. |
| `quinte_payouts_raw` | 2 421 réponses `rapports-definitifs` complètes (malgré le nom : 2 322 courses de la période prospective). Pas de politique de lecture publique. |
| `forward_bets` / `forward_experiments` | 4 expériences Simple Gagnant prospectives (voir §5). |
| `quinte_market_snapshots`, `market_snapshots`, `predictions`, `timing_*`, `v51_*`, `m8_2_*`, `multi_v6_*` | **0 ligne** : la persistance des snapshots V4.5 n'a jamais fonctionné en production ; le script `train_v45_market_dynamics.py` ne peut donc pas s'exécuter. |
| `forward_scheduler_runs` | pg_cron → Edge Function `horseprono-github-dispatch` → GitHub. **Depuis le 25/10, 596 déclenchements « snapshot » échouent** (« workflow désactivé ») ; le règlement fonctionne encore. Les workflows sont dans un autre dépôt, non fourni : non audités. |

Index en double sans conséquence fonctionnelle : `participants(race_id)` ×2, unicité
`(race_id, horse_number)` ×2, `races(external_id)` ×3.

## 4. Anomalies et limites du code existant

1. **Corrigé** — `streamlit_app.py` référençait `V3_REPORT_PATH` (inexistant) : chaque analyse se
   terminait par `NameError` affiché à l'écran. Remplacé par `V4_REPORT_PATH` (3 lignes) ; test
   de non-régression `tests/test_main_app_regression.py`.
2. **Non modifié (à décider)** — `quinte_v43._recent_features` : l'expression régulière
   `r"(?<!\\d)(\\d{1,2})p"` est doublement échappée ; elle ne reconnaît jamais une musique.
   Les 9 variables `rf_*` valent toujours 0 en production alors que l'artefact a été appris avec
   des coefficients non nuls → décalage apprentissage/production du détecteur V4.3 (mode shadow).
3. **Non modifié** — `rank_quinte_v3` lève `IndexError` sur une course de 5 ou 6 partants alors que
   l'application accepte dès 5 partants.
4. **Historique figé** : `load_incremental_history` n'ajoute que des courses avec Top 5 complet ;
   les courses prospectives n'ayant pas d'arrivée stockée, l'historique effectif s'arrête au 16/09.
5. Le dépôt ne contient ni `packages.txt`, ni `.streamlit/`, ni `.gitignore` cités par les README
   et le manifeste ; aucun workflow GitHub Actions.
6. Limites statistiques des versions Quinté : 58 courses en walk-forward, corpus réutilisé de V2 à
   V4.5 (biais de sélection reconnu dans les README), intervalles bootstrap larges.

## 5. Expériences Simple Gagnant existantes : re-règlement officiel

Les gains de l'expérience #1 ont été réglés à la **cote snapshot** et non au rapport définitif.
Recalcul sur les rapports officiels `quinte_payouts_raw` :

| Expérience | Paris réglables | Profit enregistré (mêmes paris) | Profit officiel | ROI officiel |
|---|---:|---:|---:|---:|
| #1 M8 écart ≥ 2 pts | 439 | +46,1 | **−45,4** | **−10,3 %** |
| #3 R1 cote 3–5 | 47 | −13,4 | −11,7 | −24,9 % |
| #5 trot rang 2 cote 5–8 | 49 | +20,5 | +20,5 | +41,8 % (9 gagnants ; règle issue de 33 paris historiques, IC très large) |

## 6. Conséquences retenues pour la conception

- Cible binaire « gagne », probabilités cohérentes par course → logit conditionnel.
- Étape fondamentale apprise sur l'historique (sans cote) ; combinaison avec le marché et
  facteur de rapport estimés **uniquement** sur les snapshots pré-course + rapports officiels.
- Validation 12–30/09, test gelé 01–23/10 ; règlement exclusivement au rapport définitif.
- Nouveau module isolé, nouvelles tables additives, aucune modification des modèles Quinté.
