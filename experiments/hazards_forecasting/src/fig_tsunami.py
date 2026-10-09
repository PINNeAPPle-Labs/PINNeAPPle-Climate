"""Observed vs predicted tsunami run-up on satellite imagery (NASA GIBS Blue Marble relief + bathymetry)."""
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, RESULTS
from gibs import snapshot

EVENTS = {"Sumatra 2004 (Mw 9.1)": 178, "Chile 2010 (Mw 8.8)": 206, "Tohoku 2011 (Mw 9.1)": 212}
SHIFT = 3.0


def main():
    d = pickle.load(open(RESULTS / "tsunamis_v2_predictions.pkl", "rb"))
    fig, ax = plt.subplots(3, 3, figsize=(17, 15))
    for r, (name, i) in enumerate(EVENTS.items()):
        c, run = d["cases"][i], d["combine"][i]
        lat, lon = c["slat"], c["slon"]
        s, n = np.percentile(lat, [1, 99]); w, e = np.percentile(lon, [1, 99])
        pad = 0.15 * max(n - s, e - w, 10)
        bbox = (max(s - pad, -85), max(w - pad, -180), min(n + pad, 85), min(e + pad, 180))
        img = snapshot(bbox, "2011-03-12", layers="BlueMarble_ShadedRelief_Bathymetry", width=1100)
        obs, pred = c["y"] - SHIFT, run["prediction"] - SHIFT
        lo, hi = run["lower"] - SHIFT, run["upper"] - SHIFT
        vmin, vmax = -1.5, 1.5
        for k, (v, ttl) in enumerate(((obs, "observed run-up"), (pred, "ensemble forecast (made before the labels)"))):
            a = ax[r, k]
            a.imshow(img, extent=(bbox[1], bbox[3], bbox[0], bbox[2]), aspect="auto")
            sc = a.scatter(lon, lat, c=v, s=14, cmap="turbo", vmin=vmin, vmax=vmax, edgecolor="k", linewidth=0.2)
            a.plot(c["elon"], c["elat"], "*", color="w", mec="k", ms=16)
            a.set_title(f"{name.split(' (')[0]}: {ttl.split(' (')[0]}", fontsize=10)
            a.set_xlim(bbox[1], bbox[3]); a.set_ylim(bbox[0], bbox[2])
        plt.colorbar(sc, ax=ax[r, :2], shrink=0.8, label="log10 run-up height (m)")
        a = ax[r, 2]
        a.scatter(obs, pred, s=6, alpha=0.3)
        a.plot([-2, 2.5], [-2, 2.5], "k-", lw=0.8)
        cov = np.mean((obs >= lo) & (obs <= hi))
        a.set(xlabel="observed log10 H", ylabel="forecast log10 H", title=f"{len(obs)} sites; MAE {np.mean(np.abs(obs - pred)):.2f} dex; 90% interval covers {cov:.0%}")
        a.set_aspect("equal")
    fig.suptitle("Tsunami run-up (v2, bathymetry-aware adaptive ensemble) vs NOAA observations, over NASA Blue Marble satellite imagery", y=0.995)
    fig.savefig(FIGS / "tsunami_map_vs_observed.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
