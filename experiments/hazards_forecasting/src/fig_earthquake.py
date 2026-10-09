"""Japan, weekly M>=4.5 rate: forecast vs observed around the 2011 Tohoku earthquake, with MODIS imagery and epicentres."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS
from exp_earthquakes import EVAL_YEAR, H, REGIONS, experts, load, weekly
from gibs import snapshot
from pinneapple_systems.time_series import AdaptiveForecaster

BOX = REGIONS["japan"]                                    # lat0, lat1, lon0, lon1
WEEKS = ["2010-09-12", "2011-03-13", "2011-03-20"]       # quiet week, week of the Mw 9.1 mainshock, first aftershock week


def main():
    df = load()
    s = weekly(df, BOX)
    y = np.log1p(s.values.astype(float))
    run = AdaptiveForecaster(experts(), horizon=H).run(y, start=int(np.argmax(s.index.year >= EVAL_YEAR - 5)))
    orig = s.index[run.origins]                            # forecast at origin for the week after (h=1)
    tgt = orig + pd.Timedelta(weeks=1)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 1, 1)
    m = (tgt >= "2010-06-01") & (tgt <= "2011-12-31")
    obs = np.expm1(y[run.origins + 1][m]); fc = np.expm1(run.forecast[m, 0])
    ax.fill_between(tgt[m], np.expm1(np.maximum(run.lower[m, 0], 0)), np.expm1(run.upper[m, 0]), alpha=0.25, label="90 % interval")
    ax.plot(tgt[m], fc, label="adaptive ensemble, 1 week ahead")
    ax.plot(tgt[m], obs, "k.-", lw=0.8, label="observed (USGS)")
    ax.set_yscale("log"); ax.set_ylabel("M>=4.5 per week, Japan box"); ax.legend()
    ax.set_title("No forecast anticipates the mainshock (week of 2011-03-13); the ensemble under-predicts the first aftershock weeks and catches up after 2-3 weeks")
    lat0, lat1, lon0, lon1 = BOX
    for k, w in enumerate(WEEKS):
        a = fig.add_subplot(2, 3, 4 + k)
        w = pd.Timestamp(w, tz="UTC")
        img = snapshot((lat0, lon0, lat1, lon1), (w - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), width=800)
        a.imshow(img, extent=(lon0, lon1, lat0, lat1), aspect="auto")
        sel = df[(df.time > w - pd.Timedelta(days=7)) & (df.time <= w) & df.latitude.between(lat0, lat1) & df.longitude.between(lon0, lon1)]
        a.scatter(sel.longitude, sel.latitude, s=(sel.mag - 3.5) ** 3 * 4, facecolor="none", edgecolor="yellow", linewidth=0.8)
        i = int(np.argmin(np.abs((tgt - w).days)))
        a.set_title(f"week ending {w:%Y-%m-%d}: observed {len(sel)}, forecast {fc[np.argmin(np.abs((tgt[m] - w).days))]:.0f} "
                    f"[{np.expm1(max(run.lower[m, 0][np.argmin(np.abs((tgt[m] - w).days))], 0)):.0f}-{np.expm1(run.upper[m, 0][np.argmin(np.abs((tgt[m] - w).days))]):.0f}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "earthquake_japan_vs_observed.png", dpi=110)


if __name__ == "__main__":
    main()
