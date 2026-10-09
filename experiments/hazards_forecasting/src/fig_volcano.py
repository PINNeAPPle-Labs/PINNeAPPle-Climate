"""Volcano thermal activity: 1-week-ahead forecast vs VIIRS observation, with the satellite scenes (MODIS true colour +
VIIRS 375 m thermal-anomaly layer, NASA GIBS) for the most active, the worst-forecast and a quiet week."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS
from download_viirs_volcano import VOLC
from exp_volcano_viirs import EVAL_FROM, evaluate
from gibs import snapshot

LAYERS = "MODIS_Terra_CorrectedReflectance_TrueColor,VIIRS_SNPP_Thermal_Anomalies_375m_All"


def main(name):
    s, y, run, res = evaluate(name)
    lat, lon, hw = VOLC[name]
    tgt = s.index[run.origins + 1] if False else s.index[np.minimum(run.origins + 1, len(s) - 1)]
    m = (tgt.year >= EVAL_FROM) & (run.origins + 1 < len(y))
    obs, fc = y[run.origins + 1][m], run.forecast[m, 0]
    t = tgt[m]
    fig = plt.figure(figsize=(16, 8.5))
    ax = fig.add_subplot(2, 1, 1)
    ax.fill_between(t, np.maximum(run.lower[m, 0], 0), run.upper[m, 0], alpha=0.25, label="90 % interval")
    ax.plot(t, fc, label="adaptive ensemble, 1 week ahead")
    ax.plot(t, obs, "k-", lw=0.7, label="observed (VIIRS thermal pixels)")
    r = res["h1"]
    ax.set_title(f"{name.title()}: skill vs persistence {r['skill_vs_naive'][0]:+.2f} [{r['skill_vs_naive'][1]:+.2f},{r['skill_vs_naive'][2]:+.2f}], "
                 f"vs climatology {r['skill_vs_climatology'][0]:+.2f}; 90 % interval covers {r['coverage_90']:.0%}", fontsize=10)
    ax.set_ylabel("log(1 + thermal pixels / day)"); ax.legend(loc="upper left")
    picks = {"most active week": int(np.argmax(obs)), "largest forecast error": int(np.argmax(np.abs(obs - fc))),
             "quiet week": int(np.argmin(obs + 0.001 * np.arange(len(obs))))}
    for k, (lab, i) in enumerate(picks.items()):
        ax.axvline(t[i], color="r", ls=":", lw=0.8)
        a = fig.add_subplot(2, 3, 4 + k)
        day = (t[i] - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        a.imshow(snapshot((lat - hw, lon - hw, lat + hw, lon + hw), day, layers=LAYERS, width=700),
                 extent=(lon - hw, lon + hw, lat - hw, lat + hw))
        a.plot(lon, lat, "c+", ms=10)
        a.set_title(f"{lab}, scene {day}: observed {np.expm1(obs[i]):.1f}, forecast {np.expm1(fc[i]):.1f} px/day", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / f"volcano_{name}_vs_observed.png", dpi=110)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["kilauea"]:
        main(n)
