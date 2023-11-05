import random
import numpy as np
import lifelines
import scipy

from . import stats


class TEDDR:
    """
    Parameters
    ----------
    locale_intervals : list of tuple
        List of (start, end) intervals for which the Kaplan-Meier estimator
        should be fit

    Examples
    --------
    >>> teddr = TEDDR(locale_intervals=[(0, 1), (1, 2)])
    >>> teddr.sample(10)
    """

    def __init__(self, locale_intervals):
        self.survival_fn = self._get_survival_fn(locale_intervals=locale_intervals)
        return None

    def _get_survival_fn(self, locale_intervals):
        """Fit the Turnbull estimator to the given locale intervals

        Parameters
        ----------
        locale_intervals : list of tuple
            List of (start, end) intervals

        Returns
        -------
        KaplanMeierFitter
            The fitted Kaplan-Meier estimator
        """
        kmf = lifelines.KaplanMeierFitter(label="Turnbull Estimator")
        kmf.fit_interval_censoring(
            [i[0] for i in locale_intervals],
            [i[1] for i in locale_intervals],
        )
        return kmf

    def sample(self, n_samples, cdf_uncertainty="gaussian"):
        """Two-stage sampling from the fitted estimator

        Parameters
        ----------
        n_samples : int
            Number of samples to be drawn
        cdf_uncertainty : str
            either "uniform" or "gaussian"

        Returns
        -------
        list of float
            Samples drawn from the estimator.
        """
        kmf_samples = stats.double_interval_survival_sampler(
            survival_fn=self.survival_fn, n_samples=n_samples
        )
        return kmf_samples

    def ks_distance(self, samples):
        """ """
        df = self.survival_fn.cumulative_density_
        samples_sorted = np.sort(samples)
        cdf_samples = np.arange(1, len(samples_sorted) + 1) / len(samples_sorted)

        def teddr1_cdf(val):
            if val in df.index:
                return df.loc[val, "Turnbull Estimator_upper"]
            else:
                return df.reindex([val], method="nearest").iloc[0, 0]

        cdf_teddr1_values = [teddr1_cdf(val) for val in samples_sorted]
        ks_statistic = np.max(np.abs(cdf_samples - np.array(cdf_teddr1_values)))
        return ks_statistic


class TEDDRTester:
    """Tests for comparing two TEDDR distributions.

    Parameters
    ----------
    locale_intervals1 : list of tuple
        Intervals for the first TEDDR
    locale_intervals2 : list of tuple
        Intervals for the second TEDDR
    cdf_uncertainty : str
        either "uniform" or "gaussian"

    Examples
    --------
    >>> tester = TEDDRTester(locale_intervals1=[(0, 1)], locale_intervals2=[(1, 2)])
    >>> U, p = tester.utest(10)
    >>> D, p = tester.kstest(10)
    """

    def __init__(
        self, locale_intervals1, locale_intervals2=None, cdf_uncertainty="uniform"
    ):
        self.teddr1 = TEDDR(locale_intervals=locale_intervals1)
        if locale_intervals2 is not None:
            self.teddr2 = TEDDR(locale_intervals=locale_intervals2)
            self.two_sample = True
        else:
            self.two_sample = False
        self.cdf_uncertainty = cdf_uncertainty
        return None

    def _bisample(self, n_samples):
        """Two-stage sampling from each estimator

        Parameters
        ----------
        n_samples : int
            Number of samples to be drawn from each
        """
        if not self.two_sample:
            raise NotImplementedError
        samples1 = self.teddr1.sample(
            n_samples=n_samples, cdf_uncertainty=self.cdf_uncertainty
        )
        samples2 = self.teddr2.sample(
            n_samples=n_samples, cdf_uncertainty=self.cdf_uncertainty
        )
        return list(samples1), list(samples2)

    def utest(self, n_samples=100, trials=100, return_samples=False):
        """Monte Carlo U-Test for Censored Transportation Events

        Parameters
        ----------
        n_samples : int
            Number of samples to be drawn from each TEDDR object
        trials : int
            The number of test trials to run
        return_samples : bool
            Whether you want your aggregated samples back

        Note
        ----
        This tests against the alternative hypothesis that the second
        distribution is stochastically greater than the first.
        """
        if not self.two_sample:
            raise NotImplementedError
        samples1 = []
        samples2 = []
        if trials == 1:
            samples1, samples2 = self._bisample(n_samples=n_samples)
            S, p = scipy.stats.mannwhitneyu(samples1, samples2, alternative="greater")
        elif trials > 1:
            ps = []
            for _ in range(trials):
                if return_samples:
                    U, p, s1, s2 = self.utest(
                        n_samples=n_samples, trials=1, return_samples=True
                    )
                    samples1 += s1
                    samples2 += s2
                else:
                    U, p = self.utest(
                        n_samples=n_samples, trials=1, return_samples=False
                    )
                ps.append(p)
            S, p = scipy.stats.combine_pvalues(ps, method="fisher")
        else:
            raise ValueError("Trial number must be positive.")
        if return_samples:
            return S, p, samples1, samples2
        return S, p
