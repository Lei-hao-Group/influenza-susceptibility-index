"""Estimate the influenza base transmission potential (kappa) of each
province-season with a specific humidity-driven SEIRS model.

    dS/dt = -beta(t) S I + delta R
    dE/dt =  beta(t) S I - epsilon E
    dI/dt =  epsilon E - gamma I
    dR/dt =  gamma I - delta R
    beta(t) = kappa + sigma * exp(-omega * q(t))

q(t) is the daily mean specific humidity of the province. kappa, sigma, the
initial proportions infectious and immune and the standard deviation of the
observation error are estimated by MCMC (emcee); the latent, infectious and
immune periods, omega and the ILI+ scaling factor phi are fixed.

Usage
-----
    python seirs_mcmc.py --humidity humidity.csv --out fits            # all province-seasons
    python seirs_mcmc.py --humidity humidity.csv --out fits --province Zhejiang --season 2012
    python seirs_mcmc.py --collect fits                                # table of kappa estimates

``humidity.csv`` has the columns ``province``, ``date`` and
``specific_humidity`` (kg/kg, daily mean over the grid cells of the province).
"""
import argparse
import concurrent.futures
import multiprocessing
import os
import zlib

import arviz as az
import emcee
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ILI_FILE = os.path.join(HERE, "..", "data", "province_daily_ili_plus.csv")

SEASONS = list(range(2010, 2019))           # 2010/11 to 2018/19, by starting year
SEASON_START_WEEK, SEASON_END_WEEK = 40, 39

# fixed parameters
LATENT_DAYS = 1.0
INFECTIOUS_DAYS = 2.0
IMMUNE_DAYS = 365.0
OMEGA = 180.0
PHI = 1.84
SCALE = 7.0 / PHI                           # ILI+ is a weekly proportion: 7 days of new infections / phi

RK4_SUBSTEPS = 8                            # integration step of 1/8 day

# MCMC settings
N_WALKERS = 50
N_STEPS = 35000
BURN_IN = 12000
THIN = 5
PARAMS = ["kappa", "sigma", "log_I0", "f_R", "log_tau"]
BOUNDS = [(0.0, 3.0),                       # kappa
          (0.0, 4.0),                       # sigma
          (np.log(1e-9), np.log(0.02)),     # log initial proportion infectious
          (0.0, 0.70),                      # initial proportion immune, as a share of 1 - E(0) - I(0)
          (-16.0, 1.0)]                     # log SD of the observation error
N_DIM = len(PARAMS)

# inclusion criteria
MAX_RHAT, MIN_ESS, MIN_R2 = 1.05, 400, 0.40


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@njit(fastmath=True)
def _rhs(S, E, I, R, beta, d_lat, d_inf, d_imm):
    pop = S + E + I + R
    if pop < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    f_se = beta * (S / pop) * (I / pop)
    f_ei = E / d_lat
    f_ir = I / d_inf
    f_rs = (R / pop) / d_imm
    return -f_se + f_rs, f_se - f_ei, f_ei - f_ir, f_ir - f_rs


@njit(fastmath=True)
def _rk4(y0, n, hum, kappa, sigma, d_lat, d_inf, d_imm, sub):
    """Daily series of E from a fourth-order Runge-Kutta integration.

    hum holds exp(-omega * q) on a grid of half steps (see humidity_term)."""
    out = np.empty(n)
    S, E, I, R = y0[0], y0[1], y0[2], y0[3]
    out[0] = E
    dt = 1.0 / sub
    k = 0
    for i in range(1, n):
        for _ in range(sub):
            b1 = kappa + sigma * hum[k]
            b2 = kappa + sigma * hum[k + 1]
            b4 = kappa + sigma * hum[k + 2]
            a1, a2, a3, a4 = _rhs(S, E, I, R, b1, d_lat, d_inf, d_imm)
            c1, c2, c3, c4 = _rhs(S + dt / 2 * a1, E + dt / 2 * a2, I + dt / 2 * a3, R + dt / 2 * a4,
                                  b2, d_lat, d_inf, d_imm)
            d1, d2, d3, d4 = _rhs(S + dt / 2 * c1, E + dt / 2 * c2, I + dt / 2 * c3, R + dt / 2 * c4,
                                  b2, d_lat, d_inf, d_imm)
            e1, e2, e3, e4 = _rhs(S + dt * d1, E + dt * d2, I + dt * d3, R + dt * d4,
                                  b4, d_lat, d_inf, d_imm)
            S += dt / 6 * (a1 + 2 * c1 + 2 * d1 + e1)
            E += dt / 6 * (a2 + 2 * c2 + 2 * d2 + e2)
            I += dt / 6 * (a3 + 2 * c3 + 2 * d3 + e3)
            R += dt / 6 * (a4 + 2 * c4 + 2 * d4 + e4)
            k += 2
        out[i] = E
    return out


def humidity_term(t, q, omega=OMEGA, sub=RK4_SUBSTEPS):
    """exp(-omega * q(t)) on the half-step grid, q interpolated linearly between days."""
    half = np.arange(0, (len(t) - 1) * sub * 2 + 1) * (1.0 / (2 * sub)) + t[0]
    return np.exp(-omega * np.interp(half, t, q))


def initial_state(first_obs, I0, f_R):
    """E(0) is set so that the prediction on day 0 equals the first observation;
    the rest of the population is split into R(0) and S(0)."""
    E0 = max(first_obs, 1e-9) / SCALE * LATENT_DAYS
    rest = 1.0 - E0 - I0
    if rest <= 0:
        return None
    R0 = rest * f_R
    return np.array([rest - R0, E0, I0, R0])


def simulate(theta, first_obs, n, hum):
    """Predicted daily ILI+ for theta = (kappa, sigma, log_I0, f_R)."""
    kappa, sigma, log_I0, f_R = theta
    y0 = initial_state(first_obs, np.exp(log_I0), f_R)
    if y0 is None:
        return None
    E = _rk4(y0, n, hum, kappa, sigma, LATENT_DAYS, INFECTIOUS_DAYS, IMMUNE_DAYS, RK4_SUBSTEPS)
    pred = (E / LATENT_DAYS) * SCALE
    return pred if np.all(np.isfinite(pred)) else None


def r_squared(obs, pred):
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return np.nan if ss_tot < 1e-12 else 1 - np.sum((obs - pred) ** 2) / ss_tot


# --------------------------------------------------------------------------
# likelihood and priors
# --------------------------------------------------------------------------
def sse(theta, obs, n, hum):
    pred = simulate(theta, obs[0], n, hum)
    return 1e20 if pred is None else float(np.sum((obs - pred) ** 2))


def log_prior(theta, first_obs):
    for value, (lo, hi) in zip(theta, BOUNDS):
        if not lo < value < hi:
            return -np.inf
    E0 = max(first_obs, 1e-9) / SCALE * LATENT_DAYS
    return 0.0 if 1.0 - E0 - np.exp(theta[2]) > 0 else -np.inf


def log_likelihood(theta, obs, n, hum):
    pred = simulate(theta[:4], obs[0], n, hum)
    if pred is None:
        return -np.inf
    var = np.exp(theta[4]) ** 2 + 1e-16
    ll = -0.5 * np.sum((obs - pred) ** 2 / var + np.log(2 * np.pi * var))
    return ll if np.isfinite(ll) else -np.inf


def log_posterior(theta, obs, n, hum):
    lp = log_prior(theta, obs[0])
    return lp + log_likelihood(theta, obs, n, hum) if np.isfinite(lp) else -np.inf


def initial_walkers(center, args, rng, spread=0.02, max_tries=6000):
    """Walkers scattered around the least-squares solution, with standard
    deviations of 2% of the width of each prior interval."""
    width = np.array([hi - lo for lo, hi in BOUNDS])
    pos = np.empty((N_WALKERS, N_DIM))
    done = tries = 0
    while done < N_WALKERS and tries < max_tries:
        tries += 1
        cand = center + spread * width * rng.standard_normal(N_DIM)
        if np.isfinite(log_posterior(cand, *args)):
            pos[done] = cand
            done += 1
    for k in range(done, N_WALKERS):
        while True:
            cand = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])
            if np.isfinite(log_posterior(cand, *args)):
                pos[k] = cand
                break
    if len(np.unique(pos, axis=0)) < N_WALKERS:
        pos += 1e-6 * width * rng.standard_normal(pos.shape)
    return pos


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def interpolate_weekly(weekly):
    """Weekly ILI+ (dated to the Sunday starting each week) interpolated to a
    daily series with a cubic spline; negative values are set to 0.

    Used to prepare data/province_daily_ili_plus.csv."""
    s = weekly.set_index("week_start")["ili_plus"].sort_index()
    daily = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    return daily.interpolate(method="cubic").clip(lower=0).bfill().ffill()


def season_window(season):
    start = pd.to_datetime("%d-%d-0" % (season, SEASON_START_WEEK), format="%Y-%U-%w")
    end = pd.to_datetime("%d-%d-6" % (season + 1, SEASON_END_WEEK), format="%Y-%U-%w")
    return start, end


def load_season(ili_file, humidity_file, province, season):
    ili = pd.read_csv(ili_file, parse_dates=["date"])
    hum = pd.read_csv(humidity_file, parse_dates=["date"])
    y = ili[ili["province"] == province].set_index("date")["ili_plus"]
    q = hum[hum["province"] == province].set_index("date")["specific_humidity"]
    start, end = season_window(season)
    d = pd.concat([y.rename("y"), q.rename("q")], axis=1, join="inner").loc[start:end]
    if len(d) < 50:
        return None
    return d["y"].values.astype(float), d["q"].values.astype(float)


# --------------------------------------------------------------------------
# one province-season
# --------------------------------------------------------------------------
def fit_one(province, season, ili_file, humidity_file, out_root, n_steps=N_STEPS, burn_in=BURN_IN):
    out_dir = os.path.join(out_root, "%s_%d" % (province.replace(" ", "_"), season))
    if os.path.exists(os.path.join(out_dir, "fit_metrics.csv")):
        return province, season, "done before"
    data = load_season(ili_file, humidity_file, province, season)
    if data is None:
        return province, season, "no data"
    os.makedirs(out_dir, exist_ok=True)
    obs, q = data
    n = len(obs)
    hum = humidity_term(np.arange(n, dtype=float), q)
    args = (obs, n, hum)
    seed = zlib.crc32(("%s_%d" % (province, season)).encode("utf-8")) % (2 ** 31)
    rng = np.random.default_rng(seed)

    # starting values: differential evolution on the residual sum of squares
    de = differential_evolution(sse, BOUNDS[:4], args=args, strategy="best1bin", maxiter=200, popsize=25,
                                workers=1, polish=True, seed=seed)
    pred0 = simulate(de.x, obs[0], n, hum)
    if pred0 is None:
        pred0 = np.full_like(obs, float(np.mean(obs)))
    log_tau0 = float(np.log(np.sqrt(max(np.mean((obs - pred0) ** 2), 1e-14))))
    log_tau0 = float(np.clip(log_tau0, BOUNDS[4][0] + 1, BOUNDS[4][1] - 1))

    # posterior sampling: differential evolution and snooker moves
    sampler = emcee.EnsembleSampler(N_WALKERS, N_DIM, log_posterior, args=args,
                                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    sampler.run_mcmc(initial_walkers(np.append(de.x, log_tau0), args, rng), n_steps, progress=False)

    # convergence diagnostics, each walker treated as a chain
    idata = az.from_emcee(sampler, var_names=PARAMS).sel(draw=slice(burn_in, None))
    summary = az.summary(idata, round_to=6)
    summary.to_csv(os.path.join(out_dir, "posterior_summary.csv"))

    # fitted curve: daily median of curves simulated from 500 posterior draws
    flat = sampler.get_chain(discard=burn_in, thin=THIN, flat=True)
    draws = flat[rng.integers(0, len(flat), size=min(500, len(flat)))]
    curves = np.array([c for c in (simulate(d[:4], obs[0], n, hum) for d in draws) if c is not None])
    fitted = np.percentile(curves, 50, axis=0)
    pd.DataFrame({"day": np.arange(n), "observed": obs, "fitted": fitted,
                  "lower": np.percentile(curves, 2.5, axis=0), "upper": np.percentile(curves, 97.5, axis=0)}
                 ).to_csv(os.path.join(out_dir, "fitted_curve.csv"), index=False)
    pd.DataFrame([{"province": province, "season": season, "kappa": float(summary.loc["kappa", "mean"]),
                   "sigma": float(summary.loc["sigma", "mean"]), "r2": r_squared(obs, fitted),
                   "max_rhat": float(summary["r_hat"].max()), "min_ess": float(summary["ess_bulk"].min()),
                   "acceptance": float(np.mean(sampler.acceptance_fraction)), "n_steps": n_steps, "burn_in": burn_in}]
                 ).to_csv(os.path.join(out_dir, "fit_metrics.csv"), index=False)
    return province, season, "ok"


def collect(out_root):
    """Gather all fits and flag the province-seasons kept for further analysis."""
    rows = [pd.read_csv(os.path.join(out_root, d, "fit_metrics.csv")) for d in sorted(os.listdir(out_root))
            if os.path.exists(os.path.join(out_root, d, "fit_metrics.csv"))]
    table = pd.concat(rows, ignore_index=True)
    table["converged"] = (table["max_rhat"] < MAX_RHAT) & (table["min_ess"] > MIN_ESS)
    table["included"] = table["converged"] & (table["r2"] >= MIN_R2)
    table.to_csv(os.path.join(out_root, "kappa_estimates.csv"), index=False)
    print("%d fits, %d converged, %d included" % (len(table), table["converged"].sum(), table["included"].sum()))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ili", default=ILI_FILE)
    ap.add_argument("--humidity")
    ap.add_argument("--out", default="fits")
    ap.add_argument("--province")
    ap.add_argument("--season", type=int)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--burn", type=int, default=BURN_IN)
    ap.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    ap.add_argument("--collect", metavar="DIR")
    a = ap.parse_args()
    if a.collect:
        collect(a.collect)
    else:
        if not a.humidity:
            ap.error("--humidity is required")
        provinces = [a.province] if a.province else sorted(pd.read_csv(a.ili)["province"].unique())
        seasons = [a.season] if a.season else SEASONS
        os.makedirs(a.out, exist_ok=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs = [pool.submit(fit_one, p, s, a.ili, a.humidity, a.out, a.steps, a.burn) for s in seasons for p in provinces]
            for job in concurrent.futures.as_completed(jobs):
                print("%s %d: %s" % job.result(), flush=True)
        collect(a.out)
