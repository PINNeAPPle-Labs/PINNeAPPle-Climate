"""Shared configuration and helpers for the observation-to-observation world model experiments.

Stage 0 is an identical-twin study on Lorenz-96. It isolates the one mechanism the source paper turns on — an
analysis step *learned from data* instead of derived from a supplied error covariance — at a scale that runs in
minutes on a laptop CPU, before any satellite byte is downloaded.

What the twin does contain (the mechanism under test):

* a state inferred from observations only, with no reanalysis and no observation-error statistics as input;
* a learned prior and a learned update applied whenever observations arrive;
* two observation types with errors differing by a factor of six ("reliable" surface-like, "noisy"
  retrieval-like), so that recovering *which* instrument to trust is a measurable outcome;
* an emulator that decodes a nonlinear diagnostic that is never assimilated (the twin's stand-in for the paper's
  unseen precipitation / cloud-top-height / solar-irradiance targets);
* a generative ensemble, so spread-skill and CRPS are evaluable.

What it does **not** contain, and why (see README.md for the stage plan): latent compression (n = 40 is already
small), view geometry, patch locality, satellite data. Those arrive at Stage 1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("PINNEAPPLE_SRC", Path.home() / "Documents/GitHub/pinneapple-labs/PINNeAPPle"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------------------------ twin
N_STATE = 40
FORCING = 8.0
DT = 0.05
SPINUP = 2000

# Observation network: 20 of the 40 variables, split into two instrument types with very different error
# magnitudes. The classical filters are told these numbers (or given wrong ones, as the ablation requires); the
# learned filter is not told anything and has to infer them.
OBS_INDEX = np.arange(0, N_STATE, 2)
RELIABLE_INDEX = OBS_INDEX[0::2]
NOISY_INDEX = OBS_INDEX[1::2]
SIGMA_OBS = np.where(np.isin(OBS_INDEX, RELIABLE_INDEX), 0.25, 1.5)
N_OBS = len(OBS_INDEX)

# Cycle used everywhere: CYCLES assimilated hours (the first SPINUP_CYCLES of which are never scored, since every
# method starts from a zero state), then HORIZON hours with no observations at all.
CYCLES = 24
SPINUP_CYCLES = 12
HORIZON = 6
WINDOW = CYCLES + HORIZON

# The never-assimilated diagnostic: a thresholded, nonlinear field — the closest cheap analogue of a precipitation
# product. Assimilated variables are its inputs; the threshold makes it non-invertible from any single variable.
DIAGNOSTIC_THRESHOLD = 2.5

# Training/validation/evaluation sizes. Evaluation initial conditions are independent trajectories, so the
# bootstrap over them is a bootstrap over independent cases.
N_TRAIN_TRAJ, TRAIN_TRAJ_LEN = 8, 400
N_VAL_WINDOWS = 80
N_EVAL_ICS = 60
N_TUNING_ICS = 10


def lorenz96_tendency(x: np.ndarray, forcing: float = FORCING) -> np.ndarray:
    """dx_i/dt for Lorenz-96 (Lorenz 1996), cyclic in i. Same model as ``pinneapple_analysis.data_assimilation.models``."""
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + forcing


def lorenz96_step(x: np.ndarray, dt: float = DT, forcing: float = FORCING) -> np.ndarray:
    """One RK4 step, matching ``lorenz96_step`` in ``pinneapple_analysis.data_assimilation.models``."""
    k1 = lorenz96_tendency(x, forcing)
    k2 = lorenz96_tendency(x + 0.5 * dt * k1, forcing)
    k3 = lorenz96_tendency(x + 0.5 * dt * k2, forcing)
    k4 = lorenz96_tendency(x + dt * k3, forcing)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def trajectory(length: int, rng: np.random.Generator, spinup: int = SPINUP) -> np.ndarray:
    """A trajectory of the truth after ``spinup`` steps from a perturbation of the unstable equilibrium."""
    x = FORCING * np.ones(N_STATE)
    x[0] += 0.01
    for _ in range(spinup):
        x = lorenz96_step(x)
    out = np.empty((length, N_STATE), dtype=np.float64)
    out[0] = x
    for k in range(1, length):
        out[k] = lorenz96_step(out[k - 1])
    return out


def make_trajectories(n: int, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([trajectory(length, rng) for _ in range(n)])


def observe(truth: np.ndarray, rng: np.random.Generator, sigma: np.ndarray | None = None) -> np.ndarray:
    """Noisy observations of ``OBS_INDEX`` at every step. ``truth`` is (..., N_STATE) -> (..., N_OBS)."""
    sigma = SIGMA_OBS if sigma is None else sigma
    noise = rng.standard_normal(truth.shape[:-1] + (N_OBS,)) * sigma
    return truth[..., OBS_INDEX] + noise


def diagnostic(state: np.ndarray, threshold: float = DIAGNOSTIC_THRESHOLD) -> np.ndarray:
    """The field that is never assimilated: a thresholded (precipitation-like) nonlinear transform of the state."""
    return np.maximum(state - threshold, 0.0)


# ---------------------------------------------------------------------------------------------------- metrics
def rmse(a: np.ndarray, b: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


def bias(a: np.ndarray, b: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    return np.mean(np.asarray(a) - np.asarray(b), axis=axis)


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95 % percentile CI over independent cases (evaluation initial conditions)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def crps_fair(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair (unbiased) CRPS for a finite ensemble. ``ensemble`` is (M, ...), ``truth`` is (...)."""
    m = ensemble.shape[0]
    term1 = np.abs(ensemble - truth).mean(axis=0)
    if m < 2:
        return term1
    pair = np.abs(ensemble[:, None] - ensemble[None, :]).sum(axis=(0, 1)) / (2 * m * (m - 1))
    return term1 - pair


# ------------------------------------------------------------------------------------------------- diagnostics
def hollingsworth_lonnberg(background: np.ndarray, analysis: np.ndarray, observation: np.ndarray,
                           channels: np.ndarray | None = None) -> np.ndarray:
    """Effective gain per channel: slope of the regression of (A - B) on (O - B).

    With a correct filter the slope is the optimal gain for that channel; a filter that distrusts a good
    instrument (or trusts a bad one) shows up as a slope away from the truth-supplied reference.

    ``background``/``analysis``/``observation`` are (T, m) arrays of the observed channels.
    """
    d = np.asarray(observation) - np.asarray(background)
    inc = np.asarray(analysis) - np.asarray(background)
    if channels is not None:
        d, inc = d[:, channels], inc[:, channels]
    d, inc = d.reshape(-1, d.shape[-1]), inc.reshape(-1, inc.shape[-1])
    num = np.sum(d * inc, axis=0)
    den = np.sum(d * d, axis=0)
    return num / np.where(den > 0, den, np.nan)


def desroziers(background: np.ndarray, analysis: np.ndarray, observation: np.ndarray,
               channels: np.ndarray | None = None) -> np.ndarray:
    """Desroziers estimate of the observation-error variance per channel: mean((O - A)(O - B)).

    It converges to the true R when the gain is optimal, which is how the paper argues that its learned update
    implies instrument errors close to the known ones without being told them.
    """
    oa = np.asarray(observation) - np.asarray(analysis)
    ob = np.asarray(observation) - np.asarray(background)
    if channels is not None:
        oa, ob = oa[:, channels], ob[:, channels]
    return np.mean(oa * ob, axis=0)
