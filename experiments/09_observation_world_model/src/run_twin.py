"""Stage 0: identical-twin test of learned assimilation against classical assimilation on Lorenz-96.

Question the source paper raises, reduced to the smallest setting that can answer it: can an analysis step learned
from data, given no observation-error statistics, land close to the optimal filter — including recovering *which*
instrument to trust — where a classical filter is only as good as the covariance it is handed?

Protocol
--------
* 40-variable Lorenz-96, RK4, dt = 0.05. Observations every step of 20 of the 40 variables (every other one).
* Two instrument types with errors differing by 6x: 10 "reliable" channels (sigma = 0.25) and 10 "noisy" ones
  (sigma = 1.5). The classical filters are told the right numbers, an average number, or the numbers swapped;
  the learned filter is told nothing.
* Every method starts from a zero state and assimilates CYCLES = 24 steps (the first SPINUP_CYCLES = 12 are long
  enough for all of them to forget the zero state and are never scored), then forecasts HORIZON = 6 steps with no
  observations at all. Evaluation cases are independent trajectories, so the bootstrap is over independent cases.
* The classical filters are handed the *true* dynamics and only have to get the statistics right; the learned
  filter has to learn both. The comparison therefore bounds the learned filter from below.

Run:  python src/run_twin.py --steps 4000            (full run, ~3 min on a laptop CPU)
      python src/run_twin.py --smoke                  (a change you want to check quickly)
      python src/run_twin.py --steps 4000 --no-recon  (the ablation reported in README.md)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from common import (
    CYCLES,
    HORIZON,
    N_EVAL_ICS,
    N_OBS,
    N_STATE,
    N_TRAIN_TRAJ,
    N_TUNING_ICS,
    N_VAL_WINDOWS,
    NOISY_INDEX,
    OBS_INDEX,
    RELIABLE_INDEX,
    RESULTS,
    SIGMA_OBS,
    SPINUP_CYCLES,
    TRAIN_TRAJ_LEN,
    bootstrap_ci,
    desroziers,
    diagnostic,
    hollingsworth_lonnberg,
    lorenz96_step,
    make_trajectories,
    observe,
    rmse,
)
from learned_filter import LearnedFilter
from pinneapple_analysis.state_estimation.kalman import EnsembleKalmanFilter, ExtendedKalmanFilter

RELIABLE_CH = np.isin(OBS_INDEX, RELIABLE_INDEX)
NOISY_CH = np.isin(OBS_INDEX, NOISY_INDEX)
H_OBS = lambda x: x[OBS_INDEX]


def r_covariance(sigma: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(sigma, dtype=float) ** 2)


def sigma_variants() -> dict[str, np.ndarray]:
    """Observation-error specifications handed to the classical filters."""
    scalar = np.full(N_OBS, float(np.sqrt(np.mean(SIGMA_OBS ** 2))))
    swapped = np.where(RELIABLE_CH, SIGMA_OBS[NOISY_CH][0], SIGMA_OBS[RELIABLE_CH][0])
    return {"true": SIGMA_OBS.copy(), "scalar": scalar, "swapped": swapped}


# ---------------------------------------------------------------------------------------------- classical filters
def run_ekf(truth: np.ndarray, rng: np.random.Generator, sigma: np.ndarray, q: float, p0: float = 10.0):
    """EKF cycle from a zero state. Returns analysis, prior, observations and the analysis gains per cycle."""
    ekf = ExtendedKalmanFilter(n_state=N_STATE, n_obs=N_OBS, f=lorenz96_step, h=H_OBS,
                               Q=np.eye(N_STATE) * q, R=r_covariance(sigma))
    ekf.initialize(np.zeros(N_STATE), np.eye(N_STATE) * p0)
    obs = observe(truth[:CYCLES], rng)
    prior, post, gains = [], [], []
    for t in range(CYCLES):
        x_pred, _ = ekf.predict()
        res = ekf.update(obs[t])
        prior.append(x_pred)
        post.append(res["x"].copy())
        gains.append(np.diag(res["K"][OBS_INDEX]))
    post = np.stack(post)
    fc = forecast(post[-1])
    return {"prior": np.stack(prior), "posterior": post, "forecast": fc, "observations": obs,
            "gain": np.stack(gains)}


def run_enkf(truth: np.ndarray, rng: np.random.Generator, sigma: np.ndarray, q: float, inflation: float,
             n_ens: int = 80):
    """Stochastic EnKF cycle from a zero state (``pinneapple_analysis`` EnsembleKalmanFilter)."""
    filt = EnsembleKalmanFilter(n_state=N_STATE, n_obs=N_OBS, f=lorenz96_step, h=H_OBS,
                                Q=np.eye(N_STATE) * q, R=r_covariance(sigma), n_ens=n_ens,
                                inflation=inflation, seed=0)
    filt.initialize(np.zeros(N_STATE), np.eye(N_STATE) * 10.0)
    obs = observe(truth[:CYCLES], rng)
    prior, post = [], []
    for t in range(CYCLES):
        filt.predict()
        prior.append(filt.mean.copy())
        post.append(filt.update(obs[t])["x"])
    post = np.stack(post)
    return {"prior": np.stack(prior), "posterior": post, "forecast": forecast(post[-1]),
            "observations": obs}


def forecast(state: np.ndarray, steps: int = HORIZON) -> np.ndarray:
    out, x = [], np.asarray(state, dtype=float)
    for _ in range(steps):
        x = lorenz96_step(x)
        out.append(x.copy())
    return np.stack(out)


def persistence(truth: np.ndarray, steps: int = HORIZON) -> np.ndarray:
    """Optimistic persistence: the forecast is issued from the true state at the last analysis time, so lead k is
    compared with truth[CYCLES + k - 1]. An upper bound on a no-model, not an implementable filter."""
    return np.repeat(truth[CYCLES - 1][None], steps, axis=0)


# --------------------------------------------------------------------------------------------------- tuning
def tune_classical(tuning: np.ndarray) -> dict[str, float]:
    """Tune Q (and the EnKF inflation) on cases independent of the evaluation set, with the true sigma.

    Scored on the post-spin-up cycles only, like every reported number.
    """
    def score(out, case):
        return rmse(out["posterior"][SPINUP_CYCLES:], case[SPINUP_CYCLES:CYCLES])

    best_q, best_rmse = 1e-3, np.inf
    for q in (1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 3e-1):
        errs = []
        for i, case in enumerate(tuning):
            errs.append(score(run_ekf(case, np.random.default_rng(i), SIGMA_OBS, q), case))
        if np.mean(errs) < best_rmse:
            best_q, best_rmse = q, float(np.mean(errs))
    best_inf, best_enkf = 1.02, np.inf
    for inflation in (1.0, 1.02, 1.05, 1.10, 1.20):
        errs = []
        for i, case in enumerate(tuning):
            errs.append(score(run_enkf(case, np.random.default_rng(i), SIGMA_OBS, best_q, inflation), case))
        if np.mean(errs) < best_enkf:
            best_inf, best_enkf = inflation, float(np.mean(errs))
    return {"q": best_q, "ekf_analysis_rmse": best_rmse, "inflation": best_inf,
            "enkf_analysis_rmse": best_enkf}


# ------------------------------------------------------------------------------------------------- evaluation
def summarise(name: str, cases: list[dict], clim_diagnostic: np.ndarray, forecast_only: bool = False) -> dict:
    """Aggregate one method over the independent evaluation cases, on the post-spin-up cycles.

    ``forecast_only`` is for the optimistic persistence bar, which has no analysis to diagnose.
    """
    def ci(values):
        mean, lo, hi = bootstrap_ci(values)
        return {"mean": mean, "lo": lo, "hi": hi}

    unobserved = np.setdiff1d(np.arange(N_STATE), OBS_INDEX)
    analysis_all, analysis_unobserved, forecast_leads = [], [], []
    analysis_rel, analysis_noi = [], []
    gain_rel, gain_noi, desz_rel, desz_noi = [], [], [], []
    for case in cases:
        truth = case["truth"]
        scored = truth[SPINUP_CYCLES:CYCLES]
        post, prior = case["posterior"], case["prior"]
        post = post.mean(1) if post.ndim == 3 else post          # ensemble axis, for the learned filter
        prior = prior.mean(1) if prior.ndim == 3 else prior
        post, prior = post[SPINUP_CYCLES:], prior[SPINUP_CYCLES:]
        obs = case["observations"][SPINUP_CYCLES:]
        fc = case["forecast"]
        fc = fc.mean(1) if fc.ndim == 3 else fc                  # ensemble axis, for the learned filter
        forecast_leads.append(rmse(fc, truth[CYCLES:], axis=tuple(range(1, fc.ndim))))
        if forecast_only:
            continue
        analysis_all.append(rmse(post, scored))
        analysis_unobserved.append(rmse(post[:, unobserved], scored[:, unobserved]))
        # The decomposition matters: a cell that copies its observations is visible here and invisible in the
        # state-wide RMSE.
        analysis_rel.append(rmse(post[:, RELIABLE_INDEX], scored[:, RELIABLE_INDEX]))
        analysis_noi.append(rmse(post[:, NOISY_INDEX], scored[:, NOISY_INDEX]))
        gain = hollingsworth_lonnberg(prior[:, OBS_INDEX], post[:, OBS_INDEX], obs)
        gain_rel.append(np.nanmean(gain[RELIABLE_CH]))
        gain_noi.append(np.nanmean(gain[NOISY_CH]))
        desz = desroziers(prior[:, OBS_INDEX], post[:, OBS_INDEX], obs)
        desz_rel.append(np.nanmean(desz[RELIABLE_CH]))
        desz_noi.append(np.nanmean(desz[NOISY_CH]))

    leads = np.stack(forecast_leads)                         # (cases, HORIZON)
    out = {"forecast_leads": [ci(leads[:, k]) for k in range(HORIZON)]}
    if forecast_only:
        return out
    out.update({
        "rmse_analysis": ci(analysis_all),
        "rmse_analysis_reliable": ci(analysis_rel),
        "rmse_analysis_noisy": ci(analysis_noi),
        "rmse_analysis_unobserved": ci(analysis_unobserved),
        "effective_gain_reliable": ci(gain_rel),
        "effective_gain_noisy": ci(gain_noi),
        "desroziers_reliable": ci(desz_rel),
        "desroziers_noisy": ci(desz_noi),
    })
    if "crps" in cases[0]:
        out["crps"] = ci([c["crps"][SPINUP_CYCLES:].mean() for c in cases])
        spread = np.mean([c["spread"][SPINUP_CYCLES:].mean() for c in cases])
        err = np.mean([rmse(c["posterior"][SPINUP_CYCLES:].mean(1), c["truth"][SPINUP_CYCLES:CYCLES])
                       for c in cases])
        out["spread_skill"] = float(spread / err) if err > 0 else float("nan")
        emu_unobs = [rmse(c["emulated_state"][SPINUP_CYCLES:][:, unobserved],
                          c["truth"][SPINUP_CYCLES:CYCLES][:, unobserved]) for c in cases]
        out["emulator_unobserved_rmse"] = ci(emu_unobs)
        out["emulator_diagnostic_rmse"] = ci([
            rmse(c["emulated_diagnostic"][SPINUP_CYCLES:], diagnostic(c["truth"][SPINUP_CYCLES:CYCLES]))
            for c in cases])
        out["diagnostic_climatology_rmse"] = ci([
            rmse(np.repeat(clim_diagnostic[None], CYCLES - SPINUP_CYCLES, axis=0),
                 diagnostic(c["truth"][SPINUP_CYCLES:CYCLES])) for c in cases])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="short training, fewer cases")
    ap.add_argument("--steps", type=int, default=None, help="training steps for the learned filter")
    ap.add_argument("--no-recon", action="store_true",
                    help="ablation: drop the reconstruction term on visible channels, which is what rewards "
                         "copying the observation; keep the held-out term and the rollout")
    args = ap.parse_args()

    steps = args.steps if args.steps else (300 if args.smoke else 3000)
    n_eval = 12 if args.smoke else N_EVAL_ICS
    t_start = time.time()

    print("generating truth trajectories", flush=True)
    train = make_trajectories(N_TRAIN_TRAJ, TRAIN_TRAJ_LEN, seed=11)
    val = make_trajectories(max(4, N_VAL_WINDOWS // 4), CYCLES + HORIZON, seed=12)
    tuning = make_trajectories(N_TUNING_ICS, CYCLES + HORIZON, seed=14)
    cases = make_trajectories(n_eval, CYCLES + HORIZON, seed=13)
    clim_diagnostic = diagnostic(train).mean(axis=(0, 1))

    print("tuning the classical filters (true sigma, independent cases)", flush=True)
    tuned = tune_classical(tuning)
    print(f"  Q = {tuned['q']}, EnKF inflation = {tuned['inflation']}, "
          f"EKF analysis {tuned['ekf_analysis_rmse']:.3f}, EnKF analysis {tuned['enkf_analysis_rmse']:.3f}",
          flush=True)

    sigmas = sigma_variants()
    method_specs = {
        "ekf_trueR": ("ekf", sigmas["true"]),
        "ekf_scalarR": ("ekf", sigmas["scalar"]),
        "ekf_swappedR": ("ekf", sigmas["swapped"]),
        "enkf_trueR": ("enkf", sigmas["true"]),
        "enkf_swappedR": ("enkf", sigmas["swapped"]),
    }

    print("running the classical filters", flush=True)
    collected: dict[str, list[dict]] = {name: [] for name in method_specs}
    collected["persistence"] = []
    for i, truth in enumerate(cases):
        for name, (kind, sigma) in method_specs.items():
            rng = np.random.default_rng(1000 + i)
            if kind == "ekf":
                out = run_ekf(truth, rng, sigma, tuned["q"])
            else:
                out = run_enkf(truth, rng, sigma, tuned["q"], tuned["inflation"])
            out["truth"] = truth
            collected[name].append(out)
        collected["persistence"].append({
            "truth": truth, "observations": observe(truth[:CYCLES], np.random.default_rng(1000 + i)),
            # Optimistic persistence: each analysis is the truth one step earlier, and the forecast is issued from
            # the true state at the forecast start. A no-model bar, not an implementable filter.
            "posterior": np.concatenate([truth[:1], truth[:CYCLES - 1]]),
            "prior": np.concatenate([truth[:1], truth[:CYCLES - 1]]),
            "forecast": persistence(truth),
        })

    print("training the learned filter", flush=True)
    learned = LearnedFilter(seed=0, w_recon=0.0 if args.no_recon else 1.0)
    learned.fit(train, val, steps=steps)
    print("running the learned filter", flush=True)
    learned_cases = []
    for i, truth in enumerate(cases):
        out = learned.run_case(truth, rng=np.random.default_rng(1000 + i), torch_seed=i)
        out["truth"] = truth
        learned_cases.append(out)
    collected["learned"] = learned_cases

    summary = {name: summarise(name, runs, clim_diagnostic, forecast_only=(name == "persistence"))
               for name, runs in collected.items()}
    summary["reference_optimal_gain"] = {
        "reliable": float(np.mean([c["gain"][:, RELIABLE_CH].mean() for c in collected["ekf_trueR"]])),
        "noisy": float(np.mean([c["gain"][:, NOISY_CH].mean() for c in collected["ekf_trueR"]])),
    }
    summary["true_sigma_obs"] = {"reliable": float(SIGMA_OBS[RELIABLE_CH][0]), "noisy": float(SIGMA_OBS[NOISY_CH][0])}

    payload = {
        "config": {"n_state": N_STATE, "n_obs": N_OBS, "sigma_reliable": float(SIGMA_OBS[RELIABLE_CH][0]),
                   "sigma_noisy": float(SIGMA_OBS[NOISY_CH][0]), "cycles": CYCLES, "horizon": HORIZON,
                   "dt": 0.05, "forcing": 8.0, "training_steps": steps, "eval_cases": n_eval,
                   "tuned": tuned},
        "methods": summary,
        "training_history": learned.history,
        "runtime_s": round(time.time() - t_start, 1),
    }
    tag = "smoke" if args.smoke else ("norecon" if args.no_recon else "full")
    out_path = RESULTS / f"twin_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}  ({payload['runtime_s']} s)")
    header = f"{'method':<14}{'analysis':>10}{'unobs':>10}{'4-step':>10}{'gain rel':>10}{'gain noi':>10}{'Dz rel':>9}"
    print("\n" + header + "\n" + "-" * len(header))
    for name, s in summary.items():
        if not isinstance(s, dict) or "rmse_analysis" not in s:
            continue
        print(f"{name:<14}{s['rmse_analysis']['mean']:>10.3f}{s['rmse_analysis_unobserved']['mean']:>10.3f}"
              f"{s['forecast_leads'][min(3, HORIZON - 1)]['mean']:>10.3f}"
              f"{s['effective_gain_reliable']['mean']:>10.3f}{s['effective_gain_noisy']['mean']:>10.3f}"
              f"{s['desroziers_reliable']['mean']:>9.3f}")
    pf = summary["persistence"]["forecast_leads"]
    print(f"persistence (optimistic, no model): 1-step {pf[0]['mean']:.3f}  "
          f"4-step {pf[min(3, HORIZON - 1)]['mean']:.3f}")
    print(f"\nreference optimal gain (true R): reliable {summary['reference_optimal_gain']['reliable']:.3f}, "
          f"noisy {summary['reference_optimal_gain']['noisy']:.3f}; true sigma_o^2: "
          f"{summary['true_sigma_obs']['reliable'] ** 2:.3f} / {summary['true_sigma_obs']['noisy'] ** 2:.3f}")


if __name__ == "__main__":
    main()
