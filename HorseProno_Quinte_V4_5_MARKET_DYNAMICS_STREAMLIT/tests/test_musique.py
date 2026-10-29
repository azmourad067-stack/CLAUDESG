import math

from simple_gagnant.musique import PRIOR_SCORE, musique_features, parse_musique


def test_parse_tokens_years_and_incidents():
    perfs = parse_musique("1a Da 0a(25)3m 12p")
    assert [p.result for p in perfs] == ["1", "D", "0", "3", "12"]
    assert [p.discipline for p in perfs] == ["a", "a", "a", "m", "p"]
    assert [p.season for p in perfs] == [0, 0, 0, 1, 1]
    assert perfs[0].score == 1.0 and perfs[1].score == 0.0 and perfs[2].score == 0.0
    assert perfs[4].score == 0.0  # au-delà de la 9e place


def test_empty_or_nan_musique_uses_priors():
    for v in (None, "", "nan", float("nan")):
        f = musique_features(v, "PLAT")
        assert f["mus_missing"] == 1.0 and f["mus_n"] == 0.0
        assert f["mus_score5"] == PRIOR_SCORE


def test_smoothing_shrinks_small_samples():
    one_win = musique_features("1p", "PLAT")
    five_wins = musique_features("1p1p1p1p1p", "PLAT")
    assert one_win["mus_win5"] < five_wins["mus_win5"] < 1.0
    assert math.isclose(one_win["mus_win5"], (1 + 2 * 0.1) / 3)


def test_window_is_five_most_recent_and_family_share():
    f = musique_features("1a1a1a1a1a9p9p9p", "ATTELE_VOLTE")
    assert math.isclose(f["mus_score5"], (5 + 2 * 0.3) / 7)
    assert f["mus_samefam5"] == 1.0
    assert musique_features("1p2p", "ATTELE_AUTOSTART")["mus_samefam5"] == 0.0
