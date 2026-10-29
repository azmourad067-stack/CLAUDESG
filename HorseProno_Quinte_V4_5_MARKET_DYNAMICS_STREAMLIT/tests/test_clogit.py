import numpy as np

from simple_gagnant.clogit import ConditionalLogit, softmax_by_race


def _simulate(beta, n_races=3000, seed=1):
    rng = np.random.default_rng(seed)
    X, G, Y = [], [], []
    for r in range(n_races):
        n = rng.integers(5, 15)
        x = rng.normal(size=(n, len(beta)))
        u = x @ beta
        p = np.exp(u - u.max()); p /= p.sum()
        y = np.zeros(n); y[rng.choice(n, p=p)] = 1
        X.append(x); G += [r] * n; Y.append(y)
    return np.vstack(X), np.array(G), np.concatenate(Y)


def test_recovers_coefficients_and_probabilities_sum_to_one():
    beta = np.array([0.8, -0.4])
    X, G, Y = _simulate(beta)
    m = ConditionalLogit(["a", "b"], l2=1e-6, standardize=False).fit(X, G, Y)
    assert m.converged_
    assert np.allclose(m.coef_, beta, atol=0.08)
    p = m.predict_proba(X, G)
    assert np.allclose(np.bincount(G, weights=p), 1.0)


def test_serialisation_roundtrip():
    X, G, Y = _simulate(np.array([0.5]), n_races=500)
    m = ConditionalLogit(["a"], l2=1.0).fit(X, G, Y)
    m2 = ConditionalLogit.from_dict(m.to_dict())
    assert np.allclose(m.utility(X), m2.utility(X))


def test_softmax_is_invariant_to_race_constant():
    u = np.array([1.0, 2.0, 3.0, 0.0, 0.5])
    g = np.array(["a", "a", "a", "b", "b"])
    assert np.allclose(softmax_by_race(u, g), softmax_by_race(u + np.array([5, 5, 5, -2, -2]), g))
