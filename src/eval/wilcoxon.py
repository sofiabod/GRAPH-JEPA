import numpy as np
from scipy.stats import wilcoxon
from typing import List, Tuple


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    # paired wilcoxon signed-rank test: is a > b?
    # returns (p_value, statistic)
    # if a == b exactly, scipy raises ValueError; handle it
    diff = a - b
    if np.all(diff == 0):
        return 1.0, 0.0
    result = wilcoxon(diff, alternative='greater')
    return float(result.pvalue), float(result.statistic)


def bonferroni_correct(pvals: List[float], n_tests: int) -> List[float]:
    # multiply each p-value by n_tests, clamp to 1.0
    return [min(p * n_tests, 1.0) for p in pvals]
