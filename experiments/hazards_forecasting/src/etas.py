"""ETAS-type expert for the weekly-count forecaster: expected counts from background + Omori-Utsu aftershock sums.

lambda_k = mu * 7 + K * sum_i exp(ALPHA (m_i - MC)) * int_{week k} (t - t_i + C)^(-P) dt   over events i up to the origin.
ALPHA, P, C are fixed typical values (Ogata 1988; Utsu 1961) chosen before looking at any evaluation; mu and K are
refitted every 26 weeks on the last 10 years of *past* data with non-negative least squares, the only fitted numbers.
The forecaster hands the expert only the past series; the catalog is indexed by the number of weeks seen so far, so
nothing after the origin is ever read.
"""
import numpy as np
from scipy.optimize import nnls

ALPHA, P, C, MC = 1.8, 1.1, 0.05, 4.5          # exp-productivity per magnitude unit, Omori exponent, c in days, M_c
WEEKS_FIT, REFIT_EVERY = 520, 26


def _integral(a, b, p=P, c=C):
    """int_a^b (u + c)^-p du for 0 <= a < b (days)."""
    return ((a + c) ** (1 - p) - (b + c) ** (1 - p)) / (p - 1)


class ETASExpert:
    def __init__(self, t_days, mags, week_ends_days):
        self.t, self.m = np.asarray(t_days, float), np.asarray(mags, float)
        self.ends = np.asarray(week_ends_days, float)          # end (day number) of every week of the series
        self.w = np.exp(ALPHA * (self.m - MC))
        self._params, self._key = (0.0, 0.0), None

    def _S(self, T0, T1):
        """Aftershock sum for the window (T0, T1] using events up to T0 only."""
        k = np.searchsorted(self.t, T0, side="right")
        lo = np.searchsorted(self.t, T0 - 365.0)
        ti, wi = self.t[lo:k], self.w[lo:k]
        return float(np.sum(wi * _integral(T0 - ti, T1 - ti))) if len(ti) else 0.0

    def _fit_params(self, n, y):
        i0 = max(n - WEEKS_FIT, 1)
        rows, ys = [], []
        for j in range(i0, n):                                   # week j forecast from the end of week j-1
            rows.append([7.0, self._S(self.ends[j - 1], self.ends[j])])
            ys.append(np.expm1(y[j]))
        A, b = np.array(rows), np.array(ys)
        sc = np.array([1.0, max(A[:, 1].mean(), 1e-9)])
        x, _ = nnls(A / sc, b)
        self._params = tuple(x / sc)

    def fit(self, y):
        y = np.asarray(y, float)
        n = len(y)
        key = n // REFIT_EVERY
        if key != self._key and n > 60:
            self._fit_params(n, y)
            self._key = key
        self.n = n
        self.ref = float(np.mean(np.expm1(y[-52:])))
        return self

    def predict(self, h):
        mu, K = self._params
        T = self.ends[self.n - 1]
        out = []
        for k in range(1, int(h) + 1):
            lam = mu * 7.0 + K * self._S(T, T + 7.0 * k) - (K * self._S(T, T + 7.0 * (k - 1)) if k > 1 else 0.0)
            out.append(np.log1p(max(lam, 0.0)))
        return np.array(out)


class GatedETAS:
    """ETAS while an aftershock sequence is under way (last week's count >= RATIO x the mean of the previous 52 weeks, a
    fixed rule), exponential smoothing otherwise. Post-hoc design (added after seeing that the plain ETAS expert, trained
    on all weeks, is rarely right outside sequences); reported as such."""
    RATIO = 3.0

    def __init__(self, etas, alpha=0.1):
        self.etas, self.alpha = etas, alpha

    def fit(self, y):
        y = np.asarray(y, float)
        self.etas.fit(y)
        lev = y[0]
        for v in y[1:]:
            lev += self.alpha * (v - lev)
        self.level = lev
        base = float(np.mean(np.expm1(y[-53:-1]))) if len(y) > 53 else float(np.mean(np.expm1(y)))
        self.on = float(np.expm1(y[-1])) >= self.RATIO * max(base, 1.0)
        return self

    def predict(self, h):
        return self.etas.predict(h) if self.on else np.full(int(h), self.level)
