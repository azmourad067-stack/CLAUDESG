import math

import numpy as np

from simple_gagnant.market import (
    booksum, break_even_odds, expected_value, fair_odds, power_probabilities, proportional_probabilities,
)


def test_example_from_brief():
    # Cheval A : cote 2,20 / 38 % ; B : 5,00 / 25 % ; C : 8,00 / 10 %
    assert math.isclose(expected_value(0.38, 2.2), -0.164, abs_tol=1e-9)
    assert math.isclose(expected_value(0.25, 5.0), 0.25, abs_tol=1e-9)
    assert math.isclose(expected_value(0.10, 8.0), -0.20, abs_tol=1e-9)
    assert math.isclose(fair_odds(0.25), 4.0)
    # Avec un rapport attendu à 89 % de la cote observée, l'avantage de B diminue.
    assert math.isclose(expected_value(0.25, 5.0, 0.89), 0.1125, abs_tol=1e-9)
    assert math.isclose(break_even_odds(0.25, 0.89), 1 / (0.25 * 0.89))


def test_margin_removal_methods_sum_to_one():
    odds = [1.8, 3.5, 6.0, 8.0, 12.0]  # somme des 1/cote ≈ 1,22 (marge PMU typique)
    assert booksum(odds) > 1
    p = proportional_probabilities(odds)
    q, k = power_probabilities(odds)
    assert math.isclose(p.sum(), 1.0) and math.isclose(q.sum(), 1.0, rel_tol=1e-9)
    assert k > 1.0
    assert q[0] > p[0] and q[-1] < p[-1]  # la méthode puissance retire plus de marge aux outsiders


def test_missing_odds_propagate_nan():
    assert np.isnan(proportional_probabilities([2.0, None, 4.0])).all()
    assert math.isnan(booksum([2.0, 1.0]))
    assert math.isnan(expected_value(0.3, 1.0))
