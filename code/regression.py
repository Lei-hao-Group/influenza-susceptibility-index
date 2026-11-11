"""Regression, cross-validation and variance-decomposition utilities."""
import itertools
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from isi import ISIBuilder


def season_dummies(season):
    """Season fixed effects: one dummy per season, first season as reference."""
    return pd.get_dummies(season, drop_first=True).astype(float).values


def r2_ols(y, *blocks):
    """In-sample R2 of an OLS regression of y on an intercept and the blocks."""
    X = np.column_stack([np.ones(len(y))] + [b for b in blocks if b is not None and np.size(b)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols_clustered(X, y, names, groups):
    """OLS with conventional and cluster-robust standard errors.

    The cluster-robust variance uses the CR1 small-sample correction,
    c = G/(G-1) * (n-1)/(n-k), and t tests on G-1 degrees of freedom, where G
    is the number of clusters (Cameron and Miller, 2015).
    """
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    n, k = A.shape
    bread = np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(bread * (resid ** 2).sum() / (n - k)))
    meat = np.zeros((k, k))
    labels = np.unique(groups)
    for g in labels:
        m = groups == g
        u = A[m].T @ resid[m]
        meat += np.outer(u, u)
    G = len(labels)
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    se_cl = np.sqrt(np.diag(bread @ meat @ bread) * c)
    return pd.DataFrame({"term": ["intercept"] + list(names), "coef": b, "se": se,
                         "p": 2 * stats.t.sf(np.abs(b / se), n - k),
                         "se_cluster": se_cl,
                         "p_cluster": 2 * stats.t.sf(np.abs(b / se_cl), G - 1)})


def partial_corr(x, y, z, method="spearman"):
    """Partial correlation of x and y given z (one or more control variables).

    With method="spearman" all variables are ranked first. The P value uses
    n - 2 - k degrees of freedom, k being the number of control variables.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(z, float)
    Z = Z[:, None] if Z.ndim == 1 else Z
    if method == "spearman":
        x, y, Z = rankdata(x), rankdata(y), np.apply_along_axis(rankdata, 0, Z)
    A = np.column_stack([Z, np.ones(len(x))])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = float(stats.pearsonr(rx, ry)[0])
    df = len(x) - 2 - Z.shape[1]
    t = abs(r) * np.sqrt(df / max(1 - r ** 2, 1e-300))
    return r, float(2 * stats.t.sf(t, df))


class ISIRegression(BaseEstimator, RegressorMixin):
    """kappa ~ season fixed effects + ISI (or dimension scores).

    The normalization ranges and CRITIC weights are learned from the data
    passed to ``fit`` only, so that in cross-validation the held-out data do
    not enter the construction of the index.

    X layout: the first len(framework) columns are the raw indicators in the
    order of the framework, the remaining columns are the season dummies.
    use = "isi" (the index) or "dimensions" (optionally a subset of them).
    """

    def __init__(self, framework=None, use="isi", dimensions=None):
        self.framework = framework
        self.use = use
        self.dimensions = dimensions

    def _design(self, X):
        n = len(self.framework)
        raw = pd.DataFrame(X[:, :n], columns=self.framework["variable"].tolist())
        _, D, index = self.builder_.transform(raw)
        if self.use == "isi":
            F = index.values[:, None]
        else:
            F = D[list(self.dimensions) if self.dimensions is not None else D.columns].values
        return np.column_stack([np.ones(len(X)), F, X[:, n:]])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n = len(self.framework)
        raw = pd.DataFrame(X[:, :n], columns=self.framework["variable"].tolist())
        self.builder_ = ISIBuilder(self.framework).fit(raw)
        self.coef_, *_ = np.linalg.lstsq(self._design(X), np.asarray(y, float), rcond=None)
        return self

    def predict(self, X):
        return self._design(np.asarray(X, dtype=float)) @ self.coef_


def cv_r2(make_model, X, y, folds=None, n_splits=5, repeats=20):
    """Five-fold cross-validated R2.

    folds=None : random splits, repeated ``repeats`` times (seeds 0, 1, ...);
                 the mean R2 over the repetitions is returned.
    folds=array: fold label of every observation (province-level
                 cross-validation: all seasons of a province share a label).
    In each split the predictions of all observations are pooled and one R2 is
    calculated against the total sum of squares of y.
    """
    scores = []
    if folds is not None:
        folds = np.asarray(folds)
        splits_list = [[(np.where(folds != f)[0], np.where(folds == f)[0]) for f in np.unique(folds)]]
    else:
        splits_list = [list(KFold(n_splits, shuffle=True, random_state=rep).split(X)) for rep in range(repeats)]
    for splits in splits_list:
        pred = np.zeros(len(y))
        for train, test in splits:
            model = make_model().fit(X[train], y[train])
            pred[test] = np.ravel(model.predict(X[test]))
        scores.append(r2_score(y, pred))
    return float(np.mean(scores))


def variance_shares(y, province, season):
    """Shares of the total variance of y due to differences between provinces,
    differences between seasons, and province-season-specific deviations."""
    y = pd.Series(np.asarray(y, float))
    ss = ((y - y.mean()) ** 2).sum()
    pm = y.groupby(np.asarray(province)).transform("mean")
    sm = y.groupby(np.asarray(season)).transform("mean")
    between_province = ((pm - y.mean()) ** 2).sum() / ss
    between_season = ((sm - y.mean()) ** 2).sum() / ss
    return {"between_provinces": between_province, "between_seasons": between_season,
            "province_season_specific": 1 - between_province - between_season}


def shapley_dimensions(D, y, S):
    """Exact Shapley decomposition of the R2 gained by adding the dimension
    scores D to the season-only model, over all orders of entry."""
    dims = list(D.columns)
    k = len(dims)
    cache = {}
    for size in range(k + 1):
        for subset in itertools.combinations(dims, size):
            cache[frozenset(subset)] = r2_ols(y, S, *[D[c].values for c in subset])
    base, full = cache[frozenset()], cache[frozenset(dims)]
    rows = []
    for j in dims:
        rest = [c for c in dims if c != j]
        value = sum(factorial(len(s)) * factorial(k - len(s) - 1) / factorial(k)
                    * (cache[frozenset(s) | {j}] - cache[frozenset(s)])
                    for size in range(k) for s in itertools.combinations(rest, size))
        rows.append({"dimension": j, "shapley_r2": value, "shapley_share": value / (full - base),
                     "r2_added_alone": cache[frozenset([j])] - base,
                     "r2_added_last": full - cache[frozenset(dims) - {j}]})
    return pd.DataFrame(rows), cache


def forward_selection(cache, dims):
    """Order in which the dimensions enter when each step adds the dimension
    that increases the in-sample R2 the most."""
    order = []
    while len(order) < len(dims):
        best = max([c for c in dims if c not in order], key=lambda c: cache[frozenset(order + [c])])
        order.append(best)
    return order


def weight_sweep(D, y, S, grid=None):
    """R2 of the ISI model when the weight of one dimension goes from 0 to 0.6
    and the other dimensions share the remaining weight equally. The slope is
    the central difference at equal weights (0.2)."""
    grid = np.round(np.arange(0, .601, .02), 2) if grid is None else grid
    k = D.shape[1]
    i0 = int(np.where(np.isclose(grid, 1 / k))[0][0])
    curves, slopes = {}, {}
    for c in D.columns:
        vals = []
        for w in grid:
            wt = pd.Series((1 - w) / (k - 1), index=D.columns)
            wt[c] = w
            vals.append(r2_ols(y, S, (D * wt).sum(axis=1).values))
        curves[c] = np.array(vals)
        slopes[c] = (vals[i0 + 1] - vals[i0 - 1]) / (grid[i0 + 1] - grid[i0 - 1])
    return grid, curves, slopes


def shapley_indicators(X, y, S, n_orders=1000, seed=20260914):
    """Monte Carlo Shapley decomposition of the R2 gained by adding all
    indicators to the season-only model, averaged over random orders of entry.
    Returns the mean contribution of each indicator and its Monte Carlo SE."""
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    B = np.column_stack([np.ones(len(y)), S])
    sst = ((y - y.mean()) ** 2).sum()

    def r2(cols):
        A = np.column_stack([B, Xs[:, cols]]) if cols else B
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        return 1 - ((y - A @ b) ** 2).sum() / sst

    rng = np.random.default_rng(seed)
    contrib = np.zeros((n_orders, X.shape[1]))
    base = r2([])
    for t in range(n_orders):
        cols, prev = [], base
        for j in rng.permutation(X.shape[1]):
            cols.append(j)
            now = r2(cols)
            contrib[t, j] = now - prev
            prev = now
    return contrib.mean(axis=0), contrib.std(axis=0, ddof=1) / np.sqrt(n_orders)


def season_only_model():
    return LinearRegression()
