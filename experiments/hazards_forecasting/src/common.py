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

SRC = Path(os.environ.get("PINNEAPPLE_SRC", Path.home() / "Documents/GitHub/pp-ensemble-exp"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


# ============================================================================================== statistics
def block_bootstrap(x: np.ndarray, stat=np.mean, block: int = 20, n: int = 2000, seed: int = 0):
    """Moving-block bootstrap (autocorrelated errors): returns (estimate, lo95, hi95)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    starts = rng.integers(0, max(len(x) - block, 1), size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, : len(x)]
    boots = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def skill(err_model: np.ndarray, err_ref: np.ndarray, block: int = 20, seed: int = 0):
    """Skill = 1 - E|model| / E|reference| with a paired block-bootstrap CI (positive = model better)."""
    em, er = np.asarray(err_model, float), np.asarray(err_ref, float)
    ok = np.isfinite(em) & np.isfinite(er)
    em, er = em[ok], er[ok]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(em) / block))
    starts = rng.integers(0, max(len(em) - block, 1), size=(2000, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(2000, -1)[:, : len(em)]
    boots = 1 - em[idx].mean(1) / er[idx].mean(1)
    return float(1 - em.mean() / er.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ============================================================================================== delayed labels
def run_delayed(ens, queries, references, available_at, keep=False):
    """Prequential run where the label of case k is known only at time ``available_at[k]``.

    Cases must be sorted by their issue time ``t_k`` (= the order given). Before predicting case k, every earlier
    case whose label has matured (``available_at[j] <= t_k``) is fed to ``ens.update``. A label is never used before
    it exists in reality, even with several overlapping events (storms) in flight.

    ``queries[k]["t"]`` is the issue time (a number comparable with ``available_at``).
    Returns per-case dicts with ensemble/expert predictions and the weights used.
    """
    import heapq

    pending, out = [], []
    for k, q in enumerate(queries):
        while pending and pending[0][0] <= q["t"]:
            _, j, last = heapq.heappop(pending)
            ens._last = last
            ens.update(queries[j], references[j])
        p = ens.predict(q)
        last = ens._last
        heapq.heappush(pending, (available_at[k], k, last))
        out.append({"prediction": p["prediction"], "lower": p["lower"], "upper": p["upper"], "active": p["active"],
                    "experts": p["expert_predictions"], "weights": p["weights"]})
    return out


def cluster_skill(err_model, err_ref, groups, n=2000, seed=0):
    """1 - E|model|/E|ref| with a bootstrap that resamples whole groups (storms, events): cases of one group are
    strongly dependent, so resampling single cases would give CIs that are too narrow."""
    em, er, g = np.asarray(err_model, float), np.asarray(err_ref, float), np.asarray(groups)
    ok = np.isfinite(em) & np.isfinite(er)
    em, er, g = em[ok], er[ok], g[ok]
    u, inv = np.unique(g, return_inverse=True)
    sm, sr, cnt = (np.bincount(inv, w, len(u)) for w in (em, er, np.ones_like(em)))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(u), size=(n, len(u)))
    boots = 1 - sm[pick].sum(1) / sr[pick].sum(1)
    return float(1 - em.mean() / er.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
