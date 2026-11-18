"""Figures for the Stage 0 twin, read from ``results/twin.json``.

Three panels, one claim each:

1. forecast error against lead time, all methods, with 95 % bootstrap bands on the two the text argues about;
2. the two assimilation diagnostics — effective gain and the Desroziers error estimate, per instrument type,
   against the values the true covariances imply;
3. the analysis error decomposed by channel type, which is where a cell that copies its observations shows up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paperkit"))   # shared figure style
from common import FIGS, RESULTS
from paperkit import CAT, GRID, INK2, MUTED, savefig, setup_mpl

#: (key, label, colour, linestyle, width). ``learned_ablation`` only exists when --ablation is passed.
METHODS = [
    ("learned", "learned filter, paper weights", CAT[0], "-", 2.4),
    ("learned_ablation", "learned filter, held-out objective", "#0d366b", "-", 2.0),
    ("ekf_trueR", "EKF, true per-channel R", CAT[1], "-", 1.5),
    ("ekf_scalarR", "EKF, one R for all", CAT[3], "-", 1.5),
    ("ekf_swappedR", "EKF, R swapped", CAT[6], "-", 1.5),
    ("enkf_trueR", "EnKF, true per-channel R", CAT[2], "--", 1.5),
    ("enkf_swappedR", "EnKF, R swapped", CAT[7], ":", 1.5),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None, help="results JSON (default results/twin_full.json)")
    ap.add_argument("--ablation", default=None, help="the --no-recon JSON, overlaid as one more series")
    args = ap.parse_args()
    path = Path(args.results) if args.results else RESULTS / "twin_full.json"
    payload = json.loads(path.read_text())
    methods = dict(payload["methods"])
    ablation_path = Path(args.ablation) if args.ablation else RESULTS / "twin_norecon.json"
    if ablation_path.exists():
        methods["learned_ablation"] = json.loads(ablation_path.read_text())["methods"]["learned"]

    series = [entry for entry in METHODS if entry[0] in methods]
    setup_mpl()
    _forecast_figure(methods, series)
    _diagnostics_figure(methods, payload, series)
    _decomposition_figure(methods, series)


def _forecast_figure(methods: dict, series: list) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, label, colour, style, width in series:
        leads = methods[key]["forecast_leads"]
        x = np.arange(1, len(leads) + 1)
        ax.plot(x, [lead["mean"] for lead in leads], color=colour, linestyle=style, linewidth=width, label=label)
        if key in ("learned", "ekf_trueR"):
            ax.fill_between(x, [lead["lo"] for lead in leads], [lead["hi"] for lead in leads],
                            color=colour, alpha=0.13, linewidth=0)
    persistence = [lead["mean"] for lead in methods["persistence"]["forecast_leads"]]
    ax.plot(np.arange(1, len(persistence) + 1), persistence, color=MUTED, linestyle="--", linewidth=1.4,
            label="persistence (optimistic)")
    ax.set_xlabel("forecast lead (Lorenz-96 steps of dt = 0.05)")
    ax.set_ylabel("RMSE of the state (40 variables)")
    ax.set_title("Observation-free forecast (shaded: 95 % bootstrap)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    savefig(fig, FIGS / "fig1_forecast_rmse.png")


def _diagnostics_figure(methods: dict, payload: dict, series: list) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    keys = [entry[0] for entry in series]
    labels = [entry[1] for entry in series]
    x = np.arange(len(keys))
    width = 0.38

    for offset, (field, name, colour) in enumerate([("effective_gain_reliable", "reliable channels (true $\\sigma$ = 0.25)",
                                                      CAT[0]),
                                                     ("effective_gain_noisy", "noisy channels (true $\\sigma$ = 1.5)",
                                                      CAT[1])]):
        values = [methods[key][field]["mean"] for key in keys]
        err = [[methods[key][field]["mean"] - methods[key][field]["lo"] for key in keys],
               [methods[key][field]["hi"] - methods[key][field]["mean"] for key in keys]]
        left.bar(x + (offset - 0.5) * width, values, width, color=colour, label=name, alpha=0.85)
        left.errorbar(x + (offset - 0.5) * width, values, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9,
                      capsize=2)
    ref = payload["methods"]["reference_optimal_gain"]
    left.axhline(ref["reliable"], color=CAT[0], linestyle="--", linewidth=1.0)
    left.axhline(ref["noisy"], color=CAT[1], linestyle="--", linewidth=1.0)
    left.text(-0.45, ref["reliable"] + 0.03, "optimal gain, reliable", color=CAT[0], fontsize=7, ha="left")
    left.text(-0.45, ref["noisy"] + 0.03, "optimal gain, noisy", color=CAT[1], fontsize=7, ha="left")
    left.set_xticks(x, labels, rotation=28, ha="right", fontsize=7)
    left.set_ylabel("effective gain, $(A-B)/(O-B)$")
    left.set_title("Does the filter weight each instrument as it should?", fontsize=9)

    true_sigma = payload["methods"]["true_sigma_obs"]
    for offset, (field, name, colour, truth) in enumerate([
            ("desroziers_reliable", "reliable", CAT[0], true_sigma["reliable"] ** 2),
            ("desroziers_noisy", "noisy", CAT[1], true_sigma["noisy"] ** 2)]):
        values = [max(methods[key][field]["mean"], 1e-4) for key in keys]
        right.bar(x + (offset - 0.5) * width, values, width, color=colour, alpha=0.85, label=f"{name} channels")
        right.axhline(truth, color=colour, linestyle="--", linewidth=1.0)
    right.set_yscale("log")
    right.set_xticks(x, labels, rotation=28, ha="right", fontsize=7)
    right.set_ylabel("Desroziers $\\hat\\sigma_o^2$ (log)")
    right.set_title("Implied observation error (dashed: the true variance)", fontsize=9)
    right.legend(frameon=False, fontsize=7, loc="lower right")

    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    savefig(fig, FIGS / "fig2_assimilation_diagnostics.png")


def _decomposition_figure(methods: dict, series: list) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    keys = [entry[0] for entry in series]
    labels = [entry[1] for entry in series]
    x = np.arange(len(keys))
    width = 0.26
    groups = [("rmse_analysis_reliable", "reliable channels", CAT[0]),
              ("rmse_analysis_noisy", "noisy channels", CAT[1]),
              ("rmse_analysis_unobserved", "never observed", CAT[2])]
    for offset, (field, name, colour) in enumerate(groups):
        values = [methods[key][field]["mean"] for key in keys]
        ax.bar(x + (offset - 1) * width, values, width, color=colour, alpha=0.85, label=name)
    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("analysis RMSE")
    ax.set_title("Where the analysis error sits: copying observations shows up as $\\sigma$ on their channels",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    savefig(fig, FIGS / "fig3_analysis_decomposition.png")


if __name__ == "__main__":
    main()
