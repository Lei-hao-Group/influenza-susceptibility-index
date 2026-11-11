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
  province_weekly_ili_plus.csv    weekly ILI+ by province
  province_season_indicators.csv  28 indicators for each province-season
code/
  seirs_mcmc.py                  SEIRS model and MCMC estimation of kappa
  isi.py                         construction of the index
  regression.py                  regression, cross-validation, Shapley decomposition
  run_analysis.py                main analysis and sensitivity analyses
  study_config.py                indicator definitions and province groups
```

## Data

ILI+ is the proportion of visits due to influenza-like illness multiplied by
the proportion of specimens positive for influenza. The surveillance data were
provided by the Chinese National Influenza Center.

- `province_weekly_ili_plus.csv`: `province`, `week_start` (the Sunday starting
  the week), `ili_plus`. 30 provinces, week 40 of 2010 to week 39 of 2019.
- `province_season_indicators.csv`: all 28 indicator values for the 30 study
  provinces in each of the nine seasons (270 rows). `season` is the starting
  year, from 2010 (2010/11) to 2018 (2018/19). The 28 indicator columns retain
  the values and units used in the paper. The file has 30 columns: `province`,
  `season` and the 28 indicators. All 28 indicators are populated in every row.

The indicators come from the China Statistical Yearbook (2011 to 2019
editions), the 2010 and 2020 population censuses and ERA5. Table S1 of the
paper gives their sources. `code/study_config.py` defines each indicator's
dimension, direction and source, the four direction reversals used in the
sensitivity analysis, and the provinces' north/south and cross-validation
groups.

The SEIRS model takes daily ILI+ and daily specific humidity as inputs.
Supply the daily ILI+ file through `--ili`, with the columns `province`,
`date` and `ili_plus`. The daily ILI+ used in the paper was interpolated from
the weekly surveillance series with a cubic spline, with negative values
set to 0. Supply humidity through `--humidity`, with the columns `province`,
`date` and `specific_humidity` (kg/kg). Humidity is calculated from ERA5 2 m
dewpoint temperature and surface pressure with Equation S1 of the paper.

## Requirements

Python 3.12 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Usage

Estimate kappa for every province-season and collect the estimates:

```
cd code
python seirs_mcmc.py --ili daily_ili.csv --humidity humidity.csv --out fits
python seirs_mcmc.py --collect fits
```

Each fit runs 50 walkers for 35,000 steps, and `--workers` sets the number of
fits run in parallel. To fit one province-season:

```
python seirs_mcmc.py --ili daily_ili.csv --humidity humidity.csv --out fits --province Zhejiang --season 2012
```

`fits/kappa_estimates.csv` lists kappa (posterior mean), the convergence
diagnostics and R2 of every fit. Its `included` column marks the
province-seasons that meet the inclusion criteria: R-hat below 1.05 and
effective sample size above 400 for all parameters, and R2 of at least 0.40.

Build the index and run the regression and sensitivity analyses using
`data/province_season_indicators.csv` and the collected kappa estimates:

```
python run_analysis.py --kappa fits/kappa_estimates.csv --out results
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
