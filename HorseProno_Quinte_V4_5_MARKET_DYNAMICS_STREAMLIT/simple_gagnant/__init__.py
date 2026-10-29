"""Moteur Simple Gagnant PMU — module isolé de la stratégie Quinté.

Le package ne dépend que de numpy/pandas à l'exécution (compatible Streamlit
Community Cloud). L'entraînement utilise scipy lorsqu'il est disponible.

Organisation :
- musique.py   : lecture de la musique PMU (forme pré-course).
- market.py    : probabilités implicites, marge, méthodes de dé-margination.
- features.py  : variables pré-course par partant (aucun résultat de la course).
- clogit.py    : logit conditionnel (softmax intra-course) régularisé.
- model.py     : modèle en deux étapes (fondamental puis combinaison marché).
- selector.py  : évaluation de la course, choix du candidat, statut de décision.
- backtest.py  : validation chronologique, métriques, intervalles bootstrap.
- pmu.py       : collecte PMU (programme, partants, rapports définitifs).
- journal.py   : photographie horodatée des prédictions et règlement officiel.
"""

__version__ = "1.0.0"
ENGINE_NAME = "HorseProno_SimpleGagnant"
