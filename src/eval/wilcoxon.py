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


def bootstrap_ci_on_delta(a: np.ndarray, b: np.ndarray,
                          n_resamples: int = 10000,
                          ci: float = 0.95,
                          seed: int = 0) -> dict:
    """percentile bootstrap CI on the mean of (a - b), where (a, b) are paired.

    resamples paired observations with replacement, recomputes mean delta,
    returns the empirical 2.5/97.5 percentiles (for ci=0.95). standard
    nonparametric way to put error bars on a paired comparison without
    assuming normality of the difference distribution.
    """
    diff = np.asarray(a) - np.asarray(b)
    n = diff.size
    if n == 0:
        return {"mean_delta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_resamples": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    return {
        "mean_delta": float(diff.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci,
        "n_resamples": n_resamples,
        "n_pairs": int(n),
    }
