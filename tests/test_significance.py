import numpy as np

from lab.analysis.significance import (benjamini_hochberg, bootstrap_ci, holm_correction,
                                       paired_bootstrap_diff, paired_permutation_test)


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(0.6, 0.2, 300)
    mean, lo, hi = bootstrap_ci(x)
    assert lo < mean < hi and hi - lo < 0.1


def test_identical_scores_give_p_one():
    a = np.random.default_rng(1).random(100)
    assert paired_permutation_test(a, a) == 1.0


def test_clear_difference_is_significant():
    rng = np.random.default_rng(2)
    b = rng.random(200)
    a = b + 0.1 + rng.normal(0, 0.02, 200)
    assert paired_permutation_test(a, b, n_perm=2000) < 0.01
    _, lo, hi = paired_bootstrap_diff(a, b)
    assert lo > 0


def test_noise_is_not_significant():
    rng = np.random.default_rng(3)
    a, b = rng.random(200), rng.random(200)
    assert paired_permutation_test(a, b, n_perm=2000) > 0.01


def test_corrections_are_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.03, 0.2])
    h, bh = holm_correction(p), benjamini_hochberg(p)
    assert np.all(h >= p) and np.all(bh >= p) and np.all(h <= 1) and np.all(bh <= h + 1e-12)