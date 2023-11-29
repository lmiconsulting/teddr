import numpy as np
import statsmodels.api as sm


def gaussian_kernel(x, y, sigma):
    return np.exp(-np.square(x - y) / (2 * sigma * sigma))


def cohen_d(s1, s2):
    """Calculates Cohen's d

    Parameters
    ----------
    s1 : list
        First sample, from the reference distribution P.
    s2 : list
        Second sample, from the distribution Q
    """
    return (np.mean(s1) - np.mean(s2)) / (
        np.sqrt((np.std(s1) ** 2 + np.std(s2) ** 2) / 2)
    )


def midpoint_survival_sampler(survival_fn, n_samples=1):
    """Inverse transform sampling from a lifelines survival function.

    Parameters
    ----------
    survival_fn : lifelines.KaplanMeierFitter
        your survival function
    n_samples : int, optional
        number of samples to generate.
        default is 1.

    Returns
    -------
    samples : numpy.ndarray
        Array of generated samples.
    """
    cdf_df = survival_fn.cumulative_density_
    mid_cdf_values = cdf_df.mean(axis=1).values
    times = cdf_df.index.values
    u = np.random.rand(n_samples)
    samples = np.interp(u, mid_cdf_values, times)
    return samples


def double_interval_survival_sampler(
    survival_fn, n_samples=1, uncertainty_type="uniform"
):
    """Generate samples from the Turnbull estimator using inverse transform sampling
    with double sampling within CDF interval bounds.

    Parameters
    ----------
    survival_fn : lifelines.KaplanMeierFitter
        your survival function
    n_samples : int, optional
        number of samples to generate.
        default is 1.
    uncertainty_type : str
        either "uniform" or "gaussian"

    Returns
    -------
    samples : numpy.ndarray
        Array of generated samples.

    Notes
    -----
    Main idea:
    1. for a given random value u, identify the corresponding interval on the time axis
    using the average of the upper and lower CDF values.
    2. use a uniform distribution to sample within the identified interval, based on the
    difference between the upper and lower CDF values for that interval.
    """
    cdf_df = survival_fn.cumulative_density_
    upper_cdf_values = cdf_df["Turnbull Estimator_upper"].values
    lower_cdf_values = cdf_df["Turnbull Estimator_lower"].values
    mid_cdf_values = (upper_cdf_values + lower_cdf_values) / 2

    u = np.random.rand(n_samples)
    times = cdf_df.index.values
    mid_times = np.interp(u, mid_cdf_values, times)
    interval_indices = np.searchsorted(times, mid_times)

    if uncertainty_type == "uniform":
        lower_bounds = lower_cdf_values[interval_indices]
        upper_bounds = upper_cdf_values[interval_indices]
        time_diffs = np.append(np.diff(times), times[-1])
        samples = (
            mid_times
            + (u - (upper_bounds + lower_bounds) / 2) * time_diffs[interval_indices]
        )
    elif uncertainty_type == "gaussian":
        sigma = (upper_cdf_values - mid_cdf_values) / 1.96
        interval_indices[interval_indices == len(times)] -= 1
        sampled_stddevs = sigma[interval_indices]
        samples = mid_times + norm.rvs(scale=sampled_stddevs)
    else:
        raise ValueError(
            "Invalid uncertainty_type. Choose either 'uniform' or 'gaussian'."
        )
    return samples


def calculate_odds_ratio(s1, s2, with_ci=False, return_pvalue=False):
    """Calculate odds ratio between two samples.

    Parameters
    ----------
    s1 : list
        control samples
    s2 : list
        treatment samples
    with_ci : bool
        Whether to also return a 95% confidence interval about the odds ratio
    return_pvalue : bool
        Whether to also return the p value for the Beta coefficient
    """
    y = [0] * len(s1) + [1] * len(s2)  # 0 for control group, 1 for treatment group
    X = s1 + s2
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    odds_ratio = np.exp(result.params[1])
    p_value = result.pvalues[1]
    if return_pvalue:
        yield p_value
    if with_ci:
        conf_int = result.conf_int()
        conf_int_lower, conf_int_upper = conf_int[1, 0], conf_int[1, 1]
        odds_ratio_ci = (np.exp(conf_int_lower), np.exp(conf_int_upper))
        yield odds_ratio_ci
    yield odds_ratio


def mmd(s1, s2, unbiased=False):
    """Compute the Maximum Mean Discrepancy (MMD)

    Parameters
    ----------
    s1, s2 : array-like, shape (n_samples,)
        1-dimensional samples from distributions P and Q, respectively.
        They can be of different lengths.
    unbiased : bool, optional (default=False)
        If True, computes the unbiased estimator of MMD. Otherwise, computes
        the biased estimator. The unbiased estimator might be more accurate
        but is generally slower due to more terms to compute.

    References
    ----------
    .. [1] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Scholkopf, Alexander Smola
           A Kernel Two-Sample Test
           https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    """

    def rbf_kernel(x, y, sigma):
        """Gaussian RBF kernel"""
        return np.exp(-np.abs(x - y) ** 2 / (2 * sigma**2))

    s1 = np.array(s1)
    s2 = np.array(s2)
    m, n = len(s1), len(s2)

    combined_sample = np.hstack((s1, s2))
    pairwise_dists = np.abs(combined_sample[:, None] - combined_sample)
    sigma = np.median(pairwise_dists)

    # kernel Gram matrices
    KXX = rbf_kernel(s1[:, None], s1, sigma)
    KYY = rbf_kernel(s2[:, None], s2, sigma)
    KXY = rbf_kernel(s1[:, None], s2, sigma)

    if unbiased:
        np.fill_diagonal(KXX, 0)
        np.fill_diagonal(KYY, 0)
        MMD_u = (
            np.sum(KXX) / (m * (m - 1))
            + np.sum(KYY) / (n * (n - 1))
            - 2 * np.sum(KXY) / (m * n)
        ) ** 0.5
        return MMD_u
    else:
        MMD_b = (
            np.sum(KXX) / (m**2) + np.sum(KYY) / (n**2) - 2 * np.sum(KXY) / (m * n)
        ) ** 0.5
        return MMD_b

def pct_less(vals, ref):
    """Calculate the percentage of values in a list that are less than a reference value.

    Parameters
    ----------
    vals : list or array-like
        A list or array containing numeric values to compare against the reference.
    ref : float or int
        The reference value against which the values in `vals` are compared.

    Examples
    --------
    >>> pct_less([1, 2, 3, 4, 5], 3)
    0.4
    """
    return sum(v < ref for v in vals) / len(vals)

def holm_bonferroni(p_values, alpha=0.05):
    """Apply the Holm-Bonferroni method to adjust p-values for multiple comparisons.

    Parameters
    ----------
    p_values : list or array-like
        A list of p-values to adjust.
    alpha : float, optional
        The significance level to use, by default 0.05.

    Examples
    --------
    >>> holm_bonferroni([0.01, 0.04, 0.03, 0.05])
    2
    """
    sorted_p_values = sorted((p, i) for i, p in enumerate(p_values))
    m = len(p_values)
    adjusted_alpha = [alpha / (m - i) for i in range(m)]
    significant = [False] * m
    
    for (p_value, original_index), alpha in zip(sorted_p_values, adjusted_alpha):
        if p_value <= alpha:
            significant[original_index] = True
        else:
            break
    return significant.count(True)