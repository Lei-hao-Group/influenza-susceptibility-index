"""Construction of the influenza susceptibility index (ISI).

The index is built in three steps: normalization of each indicator, weighting
of the indicators within each dimension, and aggregation of the dimension
scores. The construction used in the paper is min-max normalization, CRITIC
weights within dimensions and the arithmetic mean of the five dimension scores.
The alternative choices used in the sensitivity analysis are also implemented.
"""
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# normalization: every function returns values where higher = more susceptible
# --------------------------------------------------------------------------
def norm_minmax(x, direction):
    lo, hi = x.min(), x.max()
    z = (x - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=x.index)
    return z if direction == "+" else 1 - z


def norm_zscore(x, direction):
    """z-score mapped to (0, 1) with the logistic function."""
    s = x.std(ddof=1)
    z = (x - x.mean()) / s if s > 0 else pd.Series(0.0, index=x.index)
    z = 1 / (1 + np.exp(-z))
    return z if direction == "+" else 1 - z


def norm_rank(x, direction):
    """Percentile rank among all observations."""
    z = x.rank(pct=True)
    return z if direction == "+" else 1 - z


NORMALIZATIONS = {"minmax": norm_minmax, "zscore": norm_zscore, "rank": norm_rank}


# --------------------------------------------------------------------------
# weights (each function takes the normalized indicators of one dimension)
# --------------------------------------------------------------------------
def critic_weights(Z):
    """CRITIC weights (Diakoulaki et al., 1995).

    C_j = sd_j * sum_k (1 - r_jk), w_j = C_j / sum(C). An indicator gets a
    larger weight when it varies more and is less correlated with the other
    indicators of the same dimension.
    """
    if Z.shape[1] == 1:
        return pd.Series(1.0, index=Z.columns)
    sd = Z.std(ddof=1)
    conflict = (1 - Z.corr().fillna(0)).sum(axis=1)
    c = sd * conflict
    return c / c.sum()


def entropy_weights(Z):
    p = Z.div(Z.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    k = 1 / np.log(len(Z))
    e = -k * (p * np.log(p.replace(0, np.nan))).fillna(0).sum(axis=0)
    d = 1 - e
    return d / d.sum()


def equal_weights(Z):
    return pd.Series(1 / Z.shape[1], index=Z.columns)


def cv_weights(Z):
    """Weights proportional to the coefficient of variation."""
    cv = (Z.std(ddof=1) / Z.mean().replace(0, np.nan)).fillna(0)
    return cv / cv.sum()


WEIGHTS = {"critic": critic_weights, "entropy": entropy_weights,
           "equal": equal_weights, "cv": cv_weights}


# --------------------------------------------------------------------------
# the construction used in the paper, usable inside cross-validation
# --------------------------------------------------------------------------
class ISIBuilder:
    """Learn normalization ranges and CRITIC weights on one data set and apply
    them to any other.

    framework : DataFrame with columns ``variable``, ``dimension`` and
                ``direction`` ("+" if higher values mean higher susceptibility,
                "-" otherwise), one row per indicator.
    """

    def __init__(self, framework):
        self.framework = framework.reset_index(drop=True)
        self.variables = self.framework["variable"].tolist()
        self.directions = self.framework.set_index("variable")["direction"]
        self.dimensions = sorted(self.framework["dimension"].unique())

    def _normalize(self, X):
        z = (X - self.lo_) / (self.hi_ - self.lo_).replace(0, np.nan)
        z = z.fillna(0.5)
        flip = self.directions[self.directions == "-"].index
        z[flip] = 1 - z[flip]
        return z

    def fit(self, X):
        X = X[self.variables].astype(float)
        self.lo_, self.hi_ = X.min(), X.max()
        Z = self._normalize(X)
        self.weights_ = {}
        for dim in self.dimensions:
            cols = self.framework.loc[self.framework["dimension"] == dim, "variable"].tolist()
            self.weights_[dim] = critic_weights(Z[cols])
        return self

    def transform(self, X):
        """Return (normalized indicators, dimension scores, ISI)."""
        Z = self._normalize(X[self.variables].astype(float))
        D = pd.DataFrame({dim: (Z[w.index] * w).sum(axis=1) for dim, w in self.weights_.items()})
        return Z, D, D.mean(axis=1)

    def weight_table(self):
        """CRITIC weight of each indicator and its weight in the ISI."""
        rows = []
        for dim in self.dimensions:
            for var, w in self.weights_[dim].items():
                rows.append({"dimension": dim, "variable": var, "direction": self.directions[var],
                             "critic_weight": w, "weight_in_isi": w / len(self.dimensions)})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# alternative constructions (sensitivity analysis)
# --------------------------------------------------------------------------
def build_index(X, framework, norm="minmax", within="critic", between="equal",
                aggregation="arithmetic", drop_dimensions=(), drop_variables=()):
    """Build the dimension scores and the index under one set of choices.

    between     : "equal", "critic", "entropy" or "count" (weights proportional
                  to the number of indicators in each dimension)
    aggregation : "arithmetic" or "geometric" (weighted means of the scores)
    """
    f = NORMALIZATIONS[norm]
    fw = framework[~framework["variable"].isin(drop_variables)]
    Z = pd.DataFrame({r.variable: f(X[r.variable].astype(float), r.direction) for r in fw.itertuples()})
    scores = {}
    for dim in sorted(fw["dimension"].unique()):
        if dim in drop_dimensions:
            continue
        cols = fw.loc[fw["dimension"] == dim, "variable"].tolist()
        scores[dim] = (Z[cols] * WEIGHTS[within](Z[cols])).sum(axis=1)
    D = pd.DataFrame(scores)
    if between == "equal":
        w = pd.Series(1 / D.shape[1], index=D.columns)
    elif between == "count":
        n = pd.Series({dim: (fw["dimension"] == dim).sum() for dim in D.columns}, dtype=float)
        w = n / n.sum()
    else:
        w = WEIGHTS[between](D)
    if aggregation == "arithmetic":
        index = (D * w).sum(axis=1)
    elif aggregation == "geometric":
        index = np.exp((np.log(D.clip(lower=1e-6)) * w).sum(axis=1))
    else:
        raise ValueError(aggregation)
    return D, index
