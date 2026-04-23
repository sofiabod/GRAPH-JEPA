import numpy as np
from src.eval.wilcoxon import paired_wilcoxon, bonferroni_correct


def test_wilcoxon_identical_returns_one():
    # identical arrays -> p-value == 1.0 (no difference)
    a = np.random.randn(50)
    p, _ = paired_wilcoxon(a, a)
    assert p == 1.0


def test_wilcoxon_large_difference():
    # clearly different arrays -> p < 0.01
    a = np.ones(50)
    b = np.zeros(50)
    p, _ = paired_wilcoxon(a, b)
    assert p < 0.01


def test_wilcoxon_returns_tuple():
    a = np.random.randn(30)
    b = a + 0.5
    result = paired_wilcoxon(a, b)
    assert len(result) == 2  # (p_value, statistic)


def test_bonferroni_correct():
    # bonferroni multiplies p-values by n_tests, clamped to 1.0
    pvals = [0.01, 0.05, 0.10]
    corrected = bonferroni_correct(pvals, n_tests=6)
    assert len(corrected) == 3
    assert abs(corrected[0] - min(0.01 * 6, 1.0)) < 1e-9
    assert abs(corrected[1] - min(0.05 * 6, 1.0)) < 1e-9
    assert abs(corrected[2] - min(0.10 * 6, 1.0)) < 1e-9  # 0.10 * 6 = 0.60


def test_bonferroni_clamps_to_one():
    pvals = [0.5]
    corrected = bonferroni_correct(pvals, n_tests=4)
    assert corrected[0] == 1.0  # 0.5 * 4 = 2.0, clamped to 1.0
