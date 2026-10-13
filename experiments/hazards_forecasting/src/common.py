"""Shared protocol for the hazard-forecasting experiments.

Anti-overfitting rules applied in every experiment (see README.md):

1. Strictly chronological. Every forecast is issued with data available at its origin; labels arrive only after their
   lead time (``run_delayed``), never earlier.
2. Learned experts are refitted on an expanding window of *matured* past cases; nothing is tuned on the evaluation
   period. Hyper-parameters are fixed in the code before the evaluation period is looked at.
3. The ensemble weights are the online ones of ``pinneapple_physics.ensemble`` / ``pinneapple_systems.time_series``.
4. A warm-up period (online learning on, metrics off) separates the start of the stream from the reported period.
5. Every result is a skill score against honest baselines (persistence / climatology) with a moving-block bootstrap
   confidence interval, so "no skill" is reported as such.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# PINNeAPPle: $PINNEAPPLE_SRC, else a sibling checkout ../PINNeAPPle next to this repository, else the installed package
SRC = Path(os.environ.get("PINNEAPPLE_SRC", ROOT.parents[2] / "PINNeAPPle"))
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


# ============================================================================================== statistics
def block_bootstrap(x: np.ndarray, stat=np.mean, block: int = 20, n: int = 2000, seed: int = 0):
    """Moving-block bootstrap (autocorrelated errors): returns (estimate, lo95, hi95). PINNeAPPle ``block_bootstrap``."""
    from pinneapple_analysis.uncertainty.resampling import block_bootstrap as _bb
    return _bb(x, stat=stat, block=block, n=n, seed=seed, level=0.95)


def skill(err_model: np.ndarray, err_ref: np.ndarray, block: int = 20, seed: int = 0):
    """Skill = 1 - E|model| / E|reference| with a paired block-bootstrap CI (positive = model better)."""
    from pinneapple_analysis.uncertainty.resampling import skill_score
    return skill_score(err_model, err_ref, block=block, n=2000, seed=seed, level=0.95)


# ============================================================================================== delayed labels
def run_delayed(ens, queries, references, available_at, keep=False):
    """Prequential run where the label of case k is known only at time ``available_at[k]``.

    Cases must be sorted by their issue time ``t_k`` (= the order given). Before predicting case k, every earlier
    case whose label has matured (``available_at[j] <= t_k``) is fed to ``ens.update``. A label is never used before
    it exists in reality, even with several overlapping events (storms) in flight.

    ``queries[k]["t"]`` is the issue time (a number comparable with ``available_at``).
    Returns per-case dicts with ensemble/expert predictions and the weights used.
    Thin wrapper over PINNeAPPle ``PhysicsEnsemble.run_delayed`` (labels still pending at the end are not learned).
    """
    run = ens.run_delayed(queries, references, available_at, issue_times=[q["t"] for q in queries],
                          keep_predictions=True, flush=False)
    return [{"prediction": p["prediction"], "lower": p["lower"], "upper": p["upper"], "active": p["active"],
             "experts": p["expert_predictions"], "weights": dict(zip(run.names, w.tolist()))}
            for p, w in zip(run.predictions, run.weights)]


def cluster_skill(err_model, err_ref, groups, n=2000, seed=0):
    """1 - E|model|/E|ref| with a bootstrap that resamples whole groups (storms, events): cases of one group are
    strongly dependent, so resampling single cases would give CIs that are too narrow."""
    from pinneapple_analysis.uncertainty.resampling import skill_score
    return skill_score(err_model, err_ref, groups=groups, n=n, seed=seed, level=0.95)
