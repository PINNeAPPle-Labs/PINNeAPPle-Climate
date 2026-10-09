"""Tropical cyclones: 12-48 h intensity and track forecasts from satellite-era best tracks (IBTrACS v04r01, 1980-2025).

IBTrACS merges the agencies' best tracks, which in the satellite era are built from geostationary/polar-orbiting
imagery (Dvorak) and microwave/scatterometer fixes. A *case* is a 6-hourly position of a tropical storm (>= 25 kt) with
24 h of history and 48 h of future; the target is the intensity (kt) or the displacement (km) at +12/24/36/48 h.

Protocol (see common.py): stream from 1990, label of a case available 48 h after its origin (several storms in flight at
once), learned experts refitted every 1 January on cases matured before that date, metrics from 2005. No operational
guidance (NHC/JTWC forecasts, SST, shear) is used: only what the best track itself holds up to the origin.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, FIGS, RESULTS, cluster_skill, run_delayed
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable

LEADS = np.array([2, 4, 6, 8])                    # 6-hourly steps -> 12, 24, 36, 48 h
STREAM_FROM, EVAL_FROM, T0 = 1990, 2005, pd.Timestamp("1980-01-01")
BASINS = ["NA", "EP", "WP", "NI", "SI", "SP"]
KT_TO_KMH = 1.852


# ---------------------------------------------------------------------------------------------------- data
def load_cases():
    cols = ["SID", "SEASON", "BASIN", "ISO_TIME", "LAT", "LON", "USA_WIND", "DIST2LAND", "TRACK_TYPE"]
    d = pd.read_csv(DATA / "ibtracs" / "ibtracs.ALL.csv", skiprows=[1], usecols=cols, low_memory=False,
                    keep_default_na=False, na_values=[" ", ""])
    d = d[(d.TRACK_TYPE == "main")]
    d["time"] = pd.to_datetime(d.ISO_TIME)
    d = d[d.time.dt.hour % 6 == 0].drop_duplicates(["SID", "time"]).sort_values(["SID", "time"])
    for c in ("LAT", "LON", "USA_WIND", "DIST2LAND"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.USA_WIND.notna() & d.LAT.notna() & (d.SEASON <= 2025)]
    rows = []
    for sid, g in d.groupby("SID", sort=False):
        t = g.time.values.astype("datetime64[h]").astype(np.int64)
        if len(g) < 14:
            continue
        v, la, lo, dl = g.USA_WIND.values, g.LAT.values, g.LON.values, g.DIST2LAND.fillna(1000).values
        basin = g.BASIN.iloc[0]
        for i in range(4, len(g) - 8):
            if (t[i + 8] - t[i - 4]) != 12 * 6 or v[i] < 25:       # needs a gap-free 24 h past and 48 h future
                continue
            if not np.all(np.isfinite(v[[i - 4, i - 2, i - 1, i]])) or not np.all(np.isfinite(v[i + LEADS])):
                continue
            ts = pd.Timestamp(g.time.values[i])
            # displacement of the past 6/12/24 h (km): local east/north from lat/lon differences
            def disp(j):
                dy = (la[i] - la[j]) * 111.0
                dx = (((lo[i] - lo[j] + 180) % 360) - 180) * 111.0 * np.cos(np.radians(la[i]))
                return dx, dy
            (dx6, dy6), (dx12, dy12), (dx24, dy24) = disp(i - 1), disp(i - 2), disp(i - 4)
            fut = []
            for k in LEADS:
                dy = (la[i + k] - la[i]) * 111.0
                dx = (((lo[i + k] - lo[i] + 180) % 360) - 180) * 111.0 * np.cos(np.radians(la[i]))
                fut += [dx, dy]
            doy = ts.dayofyear / 365.25 * 2 * np.pi
            feat = [v[i], v[i] - v[i - 1], v[i] - v[i - 2], v[i] - v[i - 4], la[i], abs(la[i]), np.sin(np.radians(lo[i])),
                    np.cos(np.radians(lo[i])), dx6, dy6, dx12, dy12, dx24, dy24, np.log1p(dl[i]), float(dl[i] <= 0),
                    np.sin(doy), np.cos(doy)] + [float(basin == b) for b in BASINS]
            rows.append({"sid": sid, "t": (ts - T0).total_seconds() / 86400, "year": ts.year, "basin": basin,
                         "x": np.array(feat), "v0": v[i], "dl": dl[i], "vf": v[i + LEADS], "fut_xy": np.array(fut),
                         "lat": la[i], "lon": lo[i], "time": str(ts), "xy6": (dx6, dy6)})
    rows.sort(key=lambda r: r["t"])
    for i, r in enumerate(rows):
        r["i"] = i
    return rows


# ---------------------------------------------------------------------------------------------------- experts
class Learned:
    """A model refitted each 1 January on cases whose label matured before that date. ``kind`` picks the regressor;
    ``task`` the target ('wind': change of intensity, 'track': displacement from persistence of motion)."""

    def __init__(self, kind, task, train_pool):
        self.kind, self.task, self.pool = kind, task, train_pool
        self.year, self.model, self.caches = None, None, {}

    def _target(self, r):
        if self.task == "wind":
            return r["vf"] - r["v0"]
        return r["fut_xy"] - persistence_track(r)

    def _fit(self, year):
        cut = (pd.Timestamp(f"{year}-01-01") - T0).total_seconds() / 86400
        tr = [r for r in self.pool if r["t"] + 2 <= cut and r["year"] >= 1980]
        X, Y = np.array([r["x"] for r in tr]), np.array([self._target(r) for r in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=30.0).fit(Xs, Y)
        elif self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.multioutput import MultiOutputRegressor
            m = MultiOutputRegressor(HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                   min_samples_leaf=200, l2_regularization=10.0,
                                                                   random_state=0)).fit(Xs, Y)
        elif self.kind == "mlp":
            from sklearn.neural_network import MLPRegressor
            ys = Y.std(0) + 1e-9
            m = (MLPRegressor((32, 32), alpha=1e-1, max_iter=60, random_state=0).fit(Xs, Y / ys), ys)
        elif self.kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            m = KNeighborsRegressor(60, weights="distance").fit(Xs[:, :16], Y)
        self.model, self.mu, self.sd, self.year = m, mu, sd, year

    def __call__(self, q):
        y = int(q["year"])
        if y not in self.caches:                 # fit once per year, predict that year's cases in one batch
            self._fit(y)
            idx = [r for r in self.pool if r["year"] == y]
            xs = (np.array([r["x"] for r in idx]) - self.mu) / self.sd
            if self.kind == "mlp":
                m, ys = self.model
                d = m.predict(xs) * ys
            elif self.kind == "knn":
                d = self.model.predict(xs[:, :16])
            else:
                d = self.model.predict(xs)
            self.caches[y] = {r["i"]: d[k] for k, r in enumerate(idx)}
        return self.base(q) + self.caches[y][q["i"]]

    def base(self, q):
        return q["v0"] * np.ones(len(LEADS)) if self.task == "wind" else persistence_track(q)


def persistence_track(r):
    """Constant velocity from the last 6 h of motion (km at +12..+48 h), the CLIPER-style no-skill reference."""
    dx6, dy6 = r["xy6"]
    return np.concatenate([[dx6 * k, dy6 * k] for k in LEADS])


def kaplan_demaria(q, vb=26.7, R=0.9, alpha=0.095):
    """Persistence, except that over land the winds decay as V(t) = Vb + (R*V0 - Vb) exp(-alpha t)
    (Kaplan & DeMaria 1995; Atlantic landfalls, alpha in 1/h). Closed-form physical expert."""
    v0 = q["v0"] * np.ones(len(LEADS))
    if q["dl"] > 0:
        return v0
    return np.maximum(vb + (R * q["v0"] - vb) * np.exp(-alpha * 6 * LEADS), 15.0)


def build(task, pool):
    if task == "wind":
        ex = [from_callable("persistence", lambda q: q["v0"] * np.ones(len(LEADS)), cost=1),
              from_callable("kaplan_demaria", kaplan_demaria, cost=1)]
    else:
        ex = [from_callable("const_velocity", persistence_track, cost=1)]
    for kind in ("ridge", "hgb", "mlp", "knn"):
        ex.append(from_callable(kind, Learned(kind, task, pool), cost=5))
    return ex


def run_task(task, rows):
    stream = [r for r in rows if r["year"] >= STREAM_FROM]
    ref = [r["vf"] if task == "wind" else r["fut_xy"] for r in stream]
    avail = [r["t"] + 2.0 for r in stream]
    out = {}
    experts = build(task, rows)                       # shared: a model fitted for a year serves both modes
    for mode in ("select", "combine"):
        ens = PhysicsEnsemble(experts, mode=mode)
        res = run_delayed(ens, stream, ref, avail)
        out[mode] = res
    return stream, ref, out


def summarize(task, stream, ref, runs):
    ev = np.array([r["year"] >= EVAL_FROM for r in stream])
    sid = np.array([r["sid"] for r in stream])[ev]
    y = np.array(ref)[ev]
    names = list(runs["select"][0]["experts"].keys())
    base = "persistence" if task == "wind" else "const_velocity"

    def err(pred):                                   # per-case mean absolute error over the 4 leads (kt or km)
        return np.abs(pred - y).reshape(len(y), -1).mean(1) if task == "wind" else \
            np.linalg.norm((pred - y).reshape(len(y), -1, 2), axis=2).mean(1)

    preds = {n: np.array([r["experts"].get(n, np.full_like(y[0], np.nan)) for r, e in zip(runs["select"], ev) if e]) for n in names}
    preds["ensemble_select"] = np.array([r["prediction"] for r, e in zip(runs["select"], ev) if e])
    preds["ensemble_combine"] = np.array([r["prediction"] for r, e in zip(runs["combine"], ev) if e])
    E = {n: err(p) for n, p in preds.items()}
    unit = "kt" if task == "wind" else "km"
    res = {"unit": unit, "n_cases": int(len(y)), "n_storms": int(len(np.unique(sid))), "mae": {n: float(e.mean()) for n, e in E.items()},
           "skill_vs_" + base: {n: cluster_skill(E[n], E[base], sid) for n in E if n != base}}
    best = min((n for n in names), key=lambda n: E[n].mean())
    res["best_single_in_hindsight"] = best
    res["skill_ensemble_select_vs_best_single"] = cluster_skill(E["ensemble_select"], E[best], sid)
    res["skill_ensemble_combine_vs_best_single"] = cluster_skill(E["ensemble_combine"], E[best], sid)
    # per lead (select / combine / best single / baseline)
    lead_err = {}
    for n in ("ensemble_select", "ensemble_combine", best, base):
        d = np.abs(preds[n] - y) if task == "wind" else np.linalg.norm((preds[n] - y).reshape(len(y), -1, 2), axis=2)
        lead_err[n] = (d.mean(0)).tolist()
    res["mae_by_lead_12_24_36_48h"] = lead_err
    # coverage of the 90 % interval for the select ensemble
    lo = np.array([r["lower"] for r, e in zip(runs["select"], ev) if e]); hi = np.array([r["upper"] for r, e in zip(runs["select"], ev) if e])
    res["coverage_90"] = float(np.mean((y >= lo) & (y <= hi)))
    # which expert the ensemble leads with, over the evaluation period
    act = [r["active"] for r, e in zip(runs["select"], ev) if e]
    res["active_share"] = {n: float(np.mean([a == n for a in act])) for n in names}
    return res


def main(tasks=("wind", "track")):
    rows = load_cases()
    print("cases", len(rows), "storms", len({r["sid"] for r in rows}), flush=True)
    allres = {}
    for task in tasks:
        stream, ref, runs = run_task(task, rows)
        import pickle
        keep = [{k: r[k] for k in ("sid", "t", "year", "basin", "lat", "lon", "time", "v0", "vf", "fut_xy")} for r in stream]
        pickle.dump({"cases": keep, "runs": runs}, open(RESULTS / f"hurricanes_{task}_predictions.pkl", "wb"))
        allres[task] = summarize(task, stream, ref, runs)
        r = allres[task]
        print(task, {k: round(v, 2) for k, v in r["mae"].items()}, "best", r["best_single_in_hindsight"], "cov", round(r["coverage_90"], 2),
              flush=True)
        (RESULTS / "hurricanes.json").write_text(json.dumps(allres, indent=1))
    return allres


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("wind", "track"))
