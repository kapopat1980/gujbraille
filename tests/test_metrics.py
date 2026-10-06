import numpy as np
from gujbraille.metrics import align, cer, classification_metrics, levenshtein, mean_sd_ci, wer


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert cer("abc def", "abd def") == (1, 7)
    assert wer("a b c", "a x c") == (1, 3)


def test_align():
    pairs = align([1, 2, 3, 4], [1, 3, 4])
    assert pairs == [(0, 0), (2, 1), (3, 2)]


def test_macro_f1_is_mean_of_per_class():
    y = [1, 1, 1, 2, 2, 3]
    p = [1, 1, 2, 2, 2, 1]
    m = classification_metrics(y, p)
    assert abs(m["macro_f1"] - np.mean([c["f1"] for c in m["per_class"]])) < 1e-12
    assert abs(m["accuracy"] - 4 / 6) < 1e-12


def test_mean_sd_ci():
    r = mean_sd_ci([96.33, 96.51, 96.74, 96.87, 96.20])
    assert abs(r["mean"] - 96.53) < 1e-9
    assert abs(r["sd"] - 0.2775) < 1e-3


def test_mcnemar():
    from gujbraille.metrics import mcnemar
    y = [1] * 20
    a = [1] * 20
    b = [1] * 10 + [0] * 10
    r = mcnemar(y, a, b)
    assert r["a_right_b_wrong"] == 10 and r["a_wrong_b_right"] == 0
    assert r["p_value"] < 0.01
