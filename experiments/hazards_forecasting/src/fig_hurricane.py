"""Hurricane intensity: 24 h and 48 h forecasts vs best track, with the MODIS scene at peak intensity (NASA GIBS)."""
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, RESULTS
from gibs import snapshot

STORMS = {"Irma 2017": "2017242N16333", "Dorian 2019": "2019236N10314", "Ian 2022": "2022266N12294", "Haiyan 2013": "2013306N07162"}


def main():
    d = pickle.load(open(RESULTS / "hurricanes_wind_predictions.pkl", "rb"))
    cs, runs = d["cases"], d["runs"]["combine"]
    fig, ax = plt.subplots(2, 4, figsize=(20, 9), gridspec_kw={"height_ratios": [1, 1.1]})
    for k, (name, sid) in enumerate(STORMS.items()):
        idx = [i for i, c in enumerate(cs) if c["sid"] == sid]
        t = pd.to_datetime([cs[i]["time"] for i in idx])
        obs = np.array([cs[i]["v0"] for i in idx])
        a = ax[0, k]
        a.plot(t, obs, "k-", lw=2, label="best track (satellite-based)")
        for lead, col in ((1, "C0"), (3, "C3")):                      # LEADS = 12, 24, 36, 48 h -> indices 1 and 3
            f = np.array([runs[i]["prediction"][lead] for i in idx])
            lo = np.array([runs[i]["lower"][lead] for i in idx]); hi = np.array([runs[i]["upper"][lead] for i in idx])
            tt = t + pd.Timedelta(hours=12 * (lead + 1))
            a.plot(tt, f, color=col, lw=1.3, label=f"ensemble {12 * (lead + 1)} h ahead" + (" (90 % interval, clipped to 0-200 kt)" if lead == 1 else ""))
            if lead == 1:
                a.fill_between(tt, np.maximum(lo, 0), np.minimum(hi, 200), color=col, alpha=0.15)   # winds cannot be negative
        pers = np.array([cs[i]["v0"] for i in idx])
        a.plot(t + pd.Timedelta(hours=24), pers, "C7--", lw=1, label="persistence 24 h")
        a.set_title(f"{name}: MAE 24 h {np.mean(np.abs(np.array([runs[i]['prediction'][1] for i in idx]) - np.array([cs[i]['vf'][1] for i in idx]))):.1f} kt "
                    f"(persistence {np.mean(np.abs(pers - np.array([cs[i]['vf'][1] for i in idx]))):.1f})", fontsize=10)
        a.set_ylabel("max wind (kt)"); a.set_ylim(0, 200); a.tick_params(axis="x", rotation=30, labelsize=8)
        if k == 0:
            a.legend(fontsize=8, loc="upper left")
        pk = int(np.argmax(obs))
        la, lo_ = np.array([cs[i]["lat"] for i in idx]), np.array([cs[i]["lon"] for i in idx])
        b = ax[1, k]
        half = 9
        box = (la[pk] - half, lo_[pk] - half, la[pk] + half, lo_[pk] + half)
        b.imshow(snapshot(box, t[pk].strftime("%Y-%m-%d"), width=700), extent=(box[1], box[3], box[0], box[2]))
        b.plot(lo_, la, "y-", lw=1.5)
        b.plot(lo_[pk], la[pk], "r*", ms=14)
        b.set_title(f"MODIS {t[pk]:%Y-%m-%d}: peak {obs[pk]:.0f} kt (red star), best track in yellow", fontsize=9)
        b.set_xlim(box[1], box[3]); b.set_ylim(box[0], box[2])
    fig.tight_layout()
    fig.savefig(FIGS / "hurricane_intensity_vs_observed.png", dpi=105)


if __name__ == "__main__":
    main()
