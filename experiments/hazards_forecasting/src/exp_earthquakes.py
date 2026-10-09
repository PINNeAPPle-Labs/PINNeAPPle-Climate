"""Earthquakes: weekly rate forecasting with the adaptive forecasting ensemble (USGS ComCat, M>=4.5, 1980-2025).

Target: log(1 + weekly count of M>=4.5) in a region, horizons 1..4 weeks. Earthquakes are not predictable in time or
place beyond clustering (aftershocks) and the long-run rate; the experiment is therefore also a *no-false-skill* test:
the ensemble must beat persistence/climatology only where clustering gives something, and must never claim more.
Experts: the default pool + climatology (expanding mean) + an Omori-type decay of the recent excess over the mean.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import DATA, RESULTS, block_bootstrap, skill
from etas import ETASExpert, GatedETAS
from pinneapple_systems.time_series import AdaptiveForecaster, default_experts

REGIONS = {"global": None, "japan": (30, 46, 128, 148), "chile_peru": (-45, -5, -80, -65),
           "indonesia": (-11, 6, 94, 141), "california_baja": (30, 42, -125, -114)}
H, START_YEAR, EVAL_YEAR = 4, 1980, 1995        # online learning from 1980, metrics from 1995 (warm-up of 15 years)


@dataclass
class Climatology:
    mean_: float = 0.0

    def fit(self, y):
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, h):
        return np.full(int(h), self.mean_)


@dataclass
class OmoriDecay:
    """Mean + recent excess decaying as the Omori-Utsu law, 1/(1 + h/c)^p (p=1, c=1 week); excess over the mean of the
    last 52 weeks. Fixed constants (typical aftershock values), not fitted."""
    c: float = 1.0
    p: float = 1.0
    base_: float = 0.0
    excess_: float = 0.0

    def fit(self, y):
        y = np.asarray(y, float)
        self.base_ = float(np.mean(y[-52:]))
        self.excess_ = float(y[-1] - self.base_)
        return self

    def predict(self, h):
        k = np.arange(1, int(h) + 1)
        return self.base_ + self.excess_ / (1 + k / self.c) ** self.p


def load():
    df = pd.concat([pd.read_csv(f, usecols=["time", "latitude", "longitude", "mag", "type"])
                    for f in sorted((DATA / "usgs").glob("comcat_*.csv"))])
    df = df[df["type"] == "earthquake"].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def weekly(df, box):
    if box:
        la0, la1, lo0, lo1 = box
        df = df[df.latitude.between(la0, la1) & df.longitude.between(lo0, lo1)]
    s = df.set_index("time").resample("W").size()
    s = s[s.index.year >= START_YEAR]
    return s


def experts(catalog=None, week_index=None, etas=True):
    ex = default_experts(season_length=1)
    ex["climatology"] = Climatology()
    ex["omori"] = OmoriDecay()
    if etas and catalog is not None:
        t0 = pd.Timestamp("1970-01-01", tz="UTC")
        ex["etas"] = ETASExpert((catalog.time - t0).dt.total_seconds().values / 86400, catalog.mag.values,
                                (week_index - t0).total_seconds().values / 86400)
        ex["etas_gated"] = GatedETAS(ex["etas"])
    return ex


def region_catalog(df, box):
    if box:
        la0, la1, lo0, lo1 = box
        df = df[df.latitude.between(la0, la1) & df.longitude.between(lo0, lo1)]
    return df.sort_values("time")


def main(etas=True):
    global ETAS
    ETAS = etas
    df = load()
    out = {}
    for name, box in REGIONS.items():
        s = weekly(df, box)
        y = np.log1p(s.values.astype(float))
        start = int(np.argmax(s.index.year >= EVAL_YEAR - 5))           # forecasts begin 5 years before the metrics
        run = AdaptiveForecaster(experts(region_catalog(df, box), s.index, etas=ETAS), horizon=H, max_history=None if ETAS else 1000).run(y, start=start)
        res = {"n_weeks": int(len(y)), "mean_weekly_count": float(s.mean())}
        eval_mask = s.index[run.origins].year >= EVAL_YEAR
        for h in range(1, H + 1):
            idx = run.origins + h
            ok = (idx < len(y)) & eval_mask
            truth = y[idx[ok]]
            ens = np.abs(run.forecast[ok, h - 1] - truth)
            ref = {n: np.abs(run.expert_forecasts[ok, i, h - 1] - truth) for i, n in enumerate(run.expert_names)}
            mae = {n: float(np.mean(e)) for n, e in ref.items()}
            best = min(mae, key=mae.get)
            lo, hi = run.lower[ok, h - 1], run.upper[ok, h - 1]
            res[f"h{h}"] = {
                "mae_ensemble": float(ens.mean()), "mae_climatology": mae["climatology"], "mae_naive": mae["naive"],
                "best_single_in_hindsight": best, "mae_best_single": mae[best],
                "skill_vs_climatology": skill(ens, ref["climatology"]),
                "skill_vs_naive": skill(ens, ref["naive"]),
                "skill_vs_best_single": skill(ens, ref[best]),
                "coverage_90": float(np.mean((truth >= lo) & (truth <= hi))),
            }
        out[name] = res
        r1 = res["h1"]
        print(f"{name:16s} h1 MAE ens {r1['mae_ensemble']:.3f} clim {r1['mae_climatology']:.3f} naive {r1['mae_naive']:.3f} "
              f"| skill vs clim {r1['skill_vs_climatology'][0]:+.3f} [{r1['skill_vs_climatology'][1]:+.3f},"
              f"{r1['skill_vs_climatology'][2]:+.3f}] | cov {r1['coverage_90']:.2f}")
    (RESULTS / ("earthquakes_v2.json" if etas else "earthquakes.json")).write_text(json.dumps(out, indent=1))
    return out


ETAS = True

if __name__ == "__main__":
    main()
