"""Build the ISI, relate it to kappa and run the sensitivity analyses.

Usage
-----
    python run_analysis.py --kappa fits/kappa_estimates.csv --out results

``province_season_indicators.csv`` has one row per province-season with the columns
``province``, ``season`` (starting year, e.g. 2010 for 2010/11) and
one column per indicator, named as in ``study_config.py``.
Kappa estimates are supplied separately through ``--kappa``.
Rows are used in the order of the file; the random cross-validation splits
depend on that order.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from isi import ISIBuilder, build_index
from study_config import INDICATOR_FRAMEWORK, PROVINCE_GROUPS
from regression import (ISIRegression, cv_r2, forward_selection, ols_clustered, partial_corr,
                        r2_ols, season_dummies, season_only_model, shapley_dimensions,
                        shapley_indicators, variance_shares, weight_sweep)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def load(args):
    panel = pd.read_csv(args.panel)
    if "kappa" not in panel.columns:
        if not args.kappa:
            raise SystemExit("--kappa is required: supply the collected SEIRS estimates")
        estimates = pd.read_csv(args.kappa)
        if "included" in estimates.columns:
            estimates = estimates.loc[estimates["included"].isin([True, 1, "True", "1"])]
        panel = panel.merge(estimates[["province", "season", "kappa"]],
                            on=["province", "season"], how="inner", sort=False, validate="one_to_one")
    framework = pd.read_csv(args.framework) if args.framework else pd.DataFrame(INDICATOR_FRAMEWORK)
    groups = (pd.read_csv(args.provinces) if args.provinces else pd.DataFrame(PROVINCE_GROUPS)).set_index("province")
    missing = set(framework["variable"]) - set(panel.columns)
    if missing:
        raise SystemExit("indicators missing from the panel: %s" % sorted(missing))
    panel["north"] = panel["province"].map(groups["region"].eq("north").astype(float))
    panel["cv_group"] = panel["province"].map(groups["cv_group"])
    return panel, framework


def province_means(panel, values, name):
    return pd.DataFrame({"province": panel["province"].values, name: np.asarray(values)}).groupby("province")[name].mean()


def main_analysis(panel, framework, out):
    variables = framework["variable"].tolist()
    y = panel["kappa"].values.astype(float)
    province = panel["province"].values
    north = panel["north"].values
    folds = panel["cv_group"].values
    S = season_dummies(panel["season"])
    X = np.column_stack([panel[variables].values.astype(float), S])
    summary = {}

    # ---- index -----------------------------------------------------------
    builder = ISIBuilder(framework).fit(panel)
    Z, D, isi = builder.transform(panel)
    dims = list(D.columns)
    scores = pd.concat([panel[["province", "season", "kappa"]], D, isi.rename("ISI")], axis=1)
    scores.to_csv(os.path.join(out, "isi_scores.csv"), index=False)
    weights = builder.weight_table()
    weights["rho_with_kappa"] = [stats.spearmanr(panel[v], y)[0] for v in weights["variable"]]

    # ---- kappa and the ISI ------------------------------------------------
    summary.update({"variance_" + k: v for k, v in variance_shares(y, province, panel["season"]).items()})
    rho, p_rho = stats.spearmanr(isi, y)
    rho_ns, p_rho_ns = partial_corr(isi, y, north)
    pm = pd.DataFrame({"kappa": province_means(panel, y, "kappa"), "ISI": province_means(panel, isi, "ISI"),
                       "north": province_means(panel, north, "north")})
    r_pm, p_pm = stats.pearsonr(pm["ISI"], pm["kappa"])
    r_pm_ns, p_pm_ns = partial_corr(pm["ISI"], pm["kappa"], pm["north"], method="pearson")
    slope_pm = np.polyfit(pm["ISI"], pm["kappa"], 1)[0]
    summary.update(dict(spearman_rho=rho, spearman_p=p_rho, partial_rho_north_south=rho_ns, partial_rho_p=p_rho_ns,
                        province_mean_pearson_r=r_pm, province_mean_p=p_pm, province_mean_r2=r_pm ** 2,
                        province_mean_slope=slope_pm, province_mean_partial_r=r_pm_ns, province_mean_partial_p=p_pm_ns))

    # ---- regression models -------------------------------------------------
    def isi_model(use="isi", dimensions=None):
        return lambda: ISIRegression(framework, use, dimensions)

    summary.update(dict(
        season_only_r2=r2_ols(y, S),
        season_only_cv_random=cv_r2(season_only_model, S, y),
        season_only_cv_province=cv_r2(season_only_model, S, y, folds=folds),
        isi_model_r2=r2_ols(y, S, isi.values),
        isi_model_cv_random=cv_r2(isi_model(), X, y),
        isi_model_cv_province=cv_r2(isi_model(), X, y, folds=folds),
        reference_province_season_r2=r2_ols(y, S, pd.get_dummies(province, drop_first=True).astype(float).values)))
    names = ["season_%d" % i for i in range(S.shape[1])]
    coef = pd.concat([
        ols_clustered(np.column_stack([isi.values, S]), y, ["ISI"] + names, province).assign(model="ISI model"),
        ols_clustered(np.column_stack([isi.values, north, S]), y, ["ISI", "north"] + names, province).assign(model="ISI model + north"),
        ols_clustered(np.column_stack([D.values, S]), y, dims + names, province).assign(model="dimension model")])
    coef.to_csv(os.path.join(out, "regression_coefficients.csv"), index=False)

    # ---- contributions of the dimensions ---------------------------------
    shap, cache = shapley_dimensions(D, y, S)
    shap["weight_in_isi"] = 1 / len(dims)
    shap["share_of_isi_variance"] = [np.cov(D[c], isi)[0, 1] / len(dims) / isi.var() for c in dims]
    grid, curves, slopes = weight_sweep(D, y, S)
    shap["r2_slope_at_equal_weight"] = [slopes[c] for c in dims]
    shap.to_csv(os.path.join(out, "dimension_contributions.csv"), index=False)
    pd.DataFrame({"weight": grid, **curves}).to_csv(os.path.join(out, "dimension_weight_sweep.csv"), index=False)

    order = forward_selection(cache, dims)
    steps = [dict(step="season only", r2=cache[frozenset()], cv_random=summary["season_only_cv_random"],
                  cv_province=summary["season_only_cv_province"])]
    for i in range(1, len(order) + 1):
        current = order[:i]
        steps.append(dict(step="+" + order[i - 1], r2=cache[frozenset(current)],
                          cv_random=cv_r2(isi_model("dimensions", current), X, y),
                          cv_province=cv_r2(isi_model("dimensions", current), X, y, folds=folds)))
    steps = pd.DataFrame(steps)
    steps["r2_gain"] = steps["r2"].diff()
    steps.to_csv(os.path.join(out, "forward_selection.csv"), index=False)

    # ---- contributions of the indicators ---------------------------------
    phi, se = shapley_indicators(panel[variables].values.astype(float), y, S)
    ind = weights.set_index("variable").loc[variables].reset_index()
    ind["share_of_isi_variance"] = [ind.loc[i, "weight_in_isi"] * np.cov(Z[v], isi)[0, 1] / isi.var() for i, v in enumerate(variables)]
    ind["shapley_r2"], ind["shapley_mc_se"], ind["shapley_share"] = phi, se, phi / phi.sum()
    ind.to_csv(os.path.join(out, "indicator_contributions.csv"), index=False)
    summary["weights_vs_shapley_rho"], summary["weights_vs_shapley_p"] = stats.spearmanr(ind["weight_in_isi"], ind["shapley_share"])

    # ---- composition of each province's ISI -----------------------------
    Dp = D.groupby(province).mean()
    Zp = Z.groupby(province).mean()
    dim_dev = (Dp - Dp.mean()) / len(dims)
    ind_dev = (Zp - Zp.mean()) * ind.set_index("variable")["weight_in_isi"]
    table = pm[["ISI", "kappa"]].join(Dp.add_prefix("score_")).join(dim_dev.add_prefix("contribution_"))
    table.sort_values("ISI", ascending=False).to_csv(os.path.join(out, "province_isi.csv"))
    ind_dev.rename_axis("province").to_csv(os.path.join(out, "province_indicator_contributions.csv"))

    pd.Series(summary, name="value").rename_axis("quantity").to_csv(os.path.join(out, "summary.csv"))
    return summary


def sensitivity_analysis(panel, framework, out):
    variables = framework["variable"].tolist()
    y = panel["kappa"].values.astype(float)
    north = panel["north"].values
    folds = panel["cv_group"].values
    S = season_dummies(panel["season"])
    X = np.column_stack([panel[variables].values.astype(float), S])

    # ---- 96 constructions: 3 normalizations x 4 x 4 weightings x 2 aggregations
    rows = []
    for norm, within, between, agg in itertools.product(("minmax", "zscore", "rank"), ("critic", "entropy", "equal", "cv"),
                                                        ("equal", "critic", "entropy", "count"), ("arithmetic", "geometric")):
        _, index = build_index(panel, framework, norm, within, between, agg)
        rho, p = stats.spearmanr(index, y)
        rows.append(dict(normalization=norm, within=within, between=between, aggregation=agg, rho=rho, p=p))
    pd.DataFrame(rows).to_csv(os.path.join(out, "sensitivity_constructions.csv"), index=False)

    # ---- leaving out one indicator or one dimension ----------------------
    base = stats.spearmanr(build_index(panel, framework)[1], y)[0]
    rows = []
    for v in variables:
        rho = stats.spearmanr(build_index(panel, framework, drop_variables=(v,))[1], y)[0]
        rows.append(dict(removed=v, kind="indicator", rho=rho, change=rho - base))
    for d in sorted(framework["dimension"].unique()):
        rho = stats.spearmanr(build_index(panel, framework, drop_dimensions=(d,))[1], y)[0]
        rows.append(dict(removed=d, kind="dimension", rho=rho, change=rho - base))
    pd.DataFrame(rows).to_csv(os.path.join(out, "sensitivity_leave_one_out.csv"), index=False)

    # ---- reversing the direction of indicators on which studies disagree --
    season_cv = cv_r2(season_only_model, S, y, folds=folds)
    rows = []
    for v in [None] + framework.loc[framework["contested"] == 1, "variable"].tolist():
        fw = framework.copy()
        if v is not None:
            fw.loc[fw["variable"] == v, "direction"] = fw.loc[fw["variable"] == v, "direction"].map({"+": "-", "-": "+"})
        index = ISIBuilder(fw).fit(panel).transform(panel)[2].values
        rho, p = stats.spearmanr(index, y)
        prho, pp = partial_corr(index, y, north)
        cv = cv_r2(lambda: ISIRegression(fw, "isi"), X, y, folds=folds)
        rows.append(dict(reversed=v or "none (as in the paper)", rho=rho, p=p, partial_rho=prho, partial_p=pp,
                         cv_province=cv, cv_gain_over_season_only=cv - season_cv))
    pd.DataFrame(rows).to_csv(os.path.join(out, "sensitivity_direction.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join(DATA, "province_season_indicators.csv"))
    ap.add_argument("--kappa", help="collected SEIRS estimates with province, season and kappa columns")
    ap.add_argument("--framework", help="optional CSV overriding the indicator definitions")
    ap.add_argument("--provinces", help="optional CSV overriding the province groups")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    panel_, framework_ = load(a)
    s = main_analysis(panel_, framework_, a.out)
    sensitivity_analysis(panel_, framework_, a.out)
    for k_, v_ in s.items():
        print("%-34s %.4f" % (k_, v_))
