"""One summary figure per hazard from results/*.json: skill of the adaptive ensemble with 95 % CIs."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGS, RESULTS


def errbar(ax, labels, vals, title, xlabel):
    y = range(len(labels))
    est = [v[0] for v in vals]
    ax.errorbar(est, y, xerr=[[e - v[1] for e, v in zip(est, vals)], [v[2] - e for e, v in zip(est, vals)]], fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(list(y), labels)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
eq = json.loads((RESULTS / "earthquakes.json").read_text())
errbar(ax[0, 0], list(eq), [eq[r]["h1"]["skill_vs_climatology"] for r in eq],
       "Earthquakes, weekly M>=4.5 count, h=1: skill vs climatology", "skill (1 - MAE/MAE_clim), 95 % block-bootstrap CI")
vo = json.loads((RESULTS / "volcanoes.json").read_text())
errbar(ax[0, 1], list(vo), [vo[r]["h1"]["skill_vs_climatology"] for r in vo],
       "Volcanoes, monthly eruption starts, h=1: skill vs climatology", "skill, 95 % block-bootstrap CI")
ts = json.loads((RESULTS / "tsunamis.json").read_text())["primary_shift_3"]
names = ["ensemble_combine", "ensemble_select", "abe_1979", "spreading_calibrated", "ridge", "hgb", "knn"]
errbar(ax[1, 0], names, [ts["skill_vs_median_history"][n] for n in names],
       "Tsunami run-up (log10 H): skill vs historical median", "skill, 95 % event-cluster bootstrap CI")
try:
    hu = json.loads((RESULTS / "hurricanes.json").read_text())
    key = "skill_vs_persistence"
    names = ["ensemble_combine", "ensemble_select", "kaplan_demaria", "ridge", "hgb", "mlp", "knn"]
    errbar(ax[1, 1], names, [hu["wind"][key][n] for n in names], "Hurricane intensity 12-48 h: skill vs persistence",
           "skill, 95 % storm-cluster bootstrap CI")
except Exception as e:  # noqa: BLE001
    ax[1, 1].text(0.1, 0.5, f"hurricane results not available\n{e}")
fig.suptitle("Adaptive ensembles on public hazard data (chronological, no look-ahead)")
fig.tight_layout()
fig.savefig(FIGS / "skill_summary.png", dpi=120)
print("saved")
