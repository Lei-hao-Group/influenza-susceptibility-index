# Influenza susceptibility index (ISI)

Data and Python code for the manuscript "An influenza susceptibility index to
characterize spatial heterogeneity in influenza transmission in China".

The study estimates the influenza base transmission potential (kappa) of 30
Chinese provinces in nine influenza seasons (2010/11 to 2018/19) with a
specific humidity-driven SEIRS model. It then builds an influenza
susceptibility index from 28 province-level indicators in five dimensions and
relates the index to kappa.

## Contents

```
data/
  province_weekly_ili_plus.csv   weekly ILI+ by province
  province_daily_ili_plus.csv    daily ILI+ by province (input of the SEIRS model)
  indicator_framework.csv        the 28 indicators: dimension, direction, source
  province_groups.csv            north/south classification and cross-validation groups
code/
  seirs_mcmc.py                  SEIRS model and MCMC estimation of kappa
  isi.py                         construction of the index
  regression.py                  regression, cross-validation, Shapley decomposition
  run_analysis.py                main analysis and sensitivity analyses
```

## Data

ILI+ is the proportion of visits due to influenza-like illness multiplied by
the proportion of specimens positive for influenza. The surveillance data were
provided by the Chinese National Influenza Center.

- `province_weekly_ili_plus.csv`: `province`, `week_start` (the Sunday starting
  the week), `ili_plus`. 30 provinces, week 40 of 2010 to week 39 of 2019.
- `province_daily_ili_plus.csv`: `province`, `date`, `ili_plus`. The weekly
  series interpolated to days with a cubic spline, with negative values set to
  0. The SEIRS model is fitted to this series, and the weekly ILI+ in the
  figures of the paper is its sum over each 7 days.
- `indicator_framework.csv`: `variable` (column name used by the code),
  `indicator`, `dimension`, `dimension_name`, `direction` (`+` if higher values
  mean higher susceptibility), `contested` (1 for the four indicators whose
  direction is reversed in the sensitivity analysis) and `source`.
- `province_groups.csv`: `province`, `region` (north or south of the
  Qinling-Huaihe line) and `cv_group` (the five groups of six provinces used in
  province-level cross-validation).

The other two inputs come from public sources:

- Daily specific humidity by province, calculated from ERA5 2 m dewpoint
  temperature and surface pressure with Equation S1 of the paper. Columns:
  `province`, `date`, `specific_humidity` (kg/kg).
- The 28 indicators, from the China Statistical Yearbook (2011 to 2019
  editions), the 2010 and 2020 population censuses and ERA5. Table S1 of the
  paper gives the source of each indicator. The panel file has one row per
  province-season, with the columns `province`, `season` (starting year),
  `kappa` and one column per indicator, named as in `indicator_framework.csv`.

## Requirements

Python 3.12 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Usage

Estimate kappa for every province-season and collect the estimates:

```
cd code
python seirs_mcmc.py --humidity humidity.csv --out fits
python seirs_mcmc.py --collect fits
```

Each fit runs 50 walkers for 35,000 steps, and `--workers` sets the number of
fits run in parallel. To fit one province-season:

```
python seirs_mcmc.py --humidity humidity.csv --out fits --province Zhejiang --season 2012
```

`fits/kappa_estimates.csv` lists kappa (posterior mean), the convergence
diagnostics and R2 of every fit. Its `included` column marks the
province-seasons that meet the inclusion criteria: R-hat below 1.05 and
effective sample size above 400 for all parameters, and R2 of at least 0.40.

Build the index and run the regression and sensitivity analyses:

```
python run_analysis.py --panel panel.csv --out results
```

Output tables in `results/`:

- `summary.csv`: correlations of the ISI with kappa, variance shares, in-sample
  and cross-validated R2
- `isi_scores.csv`, `province_isi.csv`: the index and dimension scores by
  province-season and by province
- `regression_coefficients.csv`: coefficients with province-clustered standard
  errors
- `dimension_contributions.csv`, `forward_selection.csv`,
  `dimension_weight_sweep.csv`: contributions of the five dimensions
- `indicator_contributions.csv`, `province_indicator_contributions.csv`:
  weights and contributions of the 28 indicators
- `sensitivity_constructions.csv`, `sensitivity_leave_one_out.csv`,
  `sensitivity_direction.csv`: the 96 alternative constructions, leave-one-out
  results and direction reversals

## Reproducibility

Random seeds are fixed. MCMC seeds are derived from the province name and
season, and the random cross-validation splits follow the row order of the
panel file.

## License

The code is released under the MIT License. If you use the data, please cite
the paper.
