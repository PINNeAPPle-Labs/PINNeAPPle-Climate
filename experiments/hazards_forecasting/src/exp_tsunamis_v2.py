"""Tsunamis v2: bathymetry-aware forecasts (ETOPO 2022 at 5 arc-minutes).

v1 predicted run-up from magnitude, distance and depth only and had no way to know the local amplification of a coast.
v2 adds, for every site, (a) the median ocean depth within 0.5 deg and the share of shelf cells (<200 m), (b) the depth
at the epicentre, (c) the shallow-water travel time along the great circle and its mean depth, and a physical expert
that applies Green's law with the real depths:

    log10 H = 0.5 (Mw - 8) - 0.5 log10 sin(theta) + 0.25 log10(h_source / h_site) + c

(source displacement ~ rupture length ~ 10^(0.5 Mw), great-circle spreading 1/sqrt(sin theta), shoaling (h0/h)^(1/4);
Lamb 1932, Green 1838), with c the median of past residuals, the only fitted number. Post-hoc: added after the v1
evaluation showed that run-up scatter within an event was not explained by distance. Same stream and evaluation period.
"""
import json
import pickle

import numpy as np
import pandas as pd

import exp_tsunamis as v1
from common import DATA, RESULTS, cluster_skill, run_delayed
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable

G = 9.81
_B = np.load(DATA / "bathy" / "etopo_5min.npz")
Z, LAT, LON = _B["z"], _B["lat"], _B["lon"]
STEP = 5 / 60


def cell(lat, lon):
    i = np.clip(np.round((np.asarray(lat) - LAT[0]) / STEP).astype(int), 0, len(LAT) - 1)
    j = np.clip(np.round((((np.asarray(lon) + 180) % 360) - 180 - LON[0]) / STEP).astype(int), 0, len(LON) - 1)
    return i, j


def depth_at(lat, lon):
    i, j = cell(lat, lon)
    return -Z[i, j]


def site_stats(lat, lon, r=6):
    """Median ocean depth and shelf share (<200 m) in a (2r+1)^2 window (r = 6 cells ~ 0.5 deg) around each site."""
    i, j = cell(lat, lon)
    med, shelf = np.empty(len(i)), np.empty(len(i))
    for k in range(len(i)):
        w = -Z[max(i[k] - r, 0): i[k] + r + 1, np.arange(j[k] - r, j[k] + r + 1) % Z.shape[1]]
        oc = w[w > 0]
        med[k] = np.median(oc) if oc.size else 10.0
        shelf[k] = np.mean(oc < 200) if oc.size else 1.0
    return np.maximum(med, 10.0), shelf


def path_stats(elat, elon, lat, lon, n=48):
    """Travel time (h) and mean depth along the great circle from the epicentre to each site (ocean samples only)."""
    la0, lo0 = np.radians(elat), np.radians(elon)
    la1, lo1 = np.radians(np.asarray(lat, float))[:, None], np.radians(np.asarray(lon, float))[:, None]
    d = np.arccos(np.clip(np.sin(la0) * np.sin(la1[:, 0]) + np.cos(la0) * np.cos(la1[:, 0]) * np.cos(lo1[:, 0] - lo0), -1, 1))
    f = np.linspace(0, 1, n)[None, :]
    a, b = np.sin((1 - f) * d[:, None]) / np.maximum(np.sin(d[:, None]), 1e-9), np.sin(f * d[:, None]) / np.maximum(np.sin(d[:, None]), 1e-9)
    x = a * np.cos(la0) * np.cos(lo0) + b * np.cos(la1) * np.cos(lo1)
    y = a * np.cos(la0) * np.sin(lo0) + b * np.cos(la1) * np.sin(lo1)
    z = a * np.sin(la0) + b * np.sin(la1)
    plat, plon = np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))
    h = np.maximum(depth_at(plat, plon), 20.0)
    ds = (d * 6371e3 / (n - 1))[:, None]
    return (ds / np.sqrt(G * h)).sum(1) / 3600.0, h.mean(1)


def add_features(c):
    hs, sh = site_stats(c["slat"], c["slon"])
    tt, hp = path_stats(c["elat"], c["elon"], c["slat"], c["slon"])
    hsrc = max(float(depth_at(c["elat"], c["elon"])), 100.0)
    c["h_site"], c["shelf"], c["tt"], c["h_path"], c["h_src"] = hs, sh, tt, hp, hsrc
    return c


def feats_b(q):
    base = v1.feats(q)
    return np.column_stack([base, np.log10(q["h_site"]), q["shelf"], np.full(len(q["dist"]), np.log10(q["h_src"])),
                            np.log10(q["tt"] + 0.1), np.log10(q["h_path"])])


class LearnedB(v1.Learned):
    def __init__(self, kind, pool):
        super().__init__(kind, pool)

    def _fit(self, year):
        cut = (pd.Timestamp(f"{year}-01-01") - v1.T0).days
        tr = [c for c in self.pool if c["t"] + 2 <= cut]
        self.year = year
        self.med = float(np.median(np.concatenate([c["y"] for c in tr]))) if tr else v1.SHIFT
        if len(tr) < 30:
            self.model = None
            return
        X = np.vstack([feats_b(c) for c in tr]); y = np.concatenate([c["y"] for c in tr])
        w = np.concatenate([np.full(len(c["y"]), 1.0 / len(c["y"])) for c in tr]); w = w * len(w) / w.sum()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=10.0).fit(Xs, y, sample_weight=w)
        elif self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                                       l2_regularization=10.0, random_state=0).fit(Xs, y, sample_weight=w)
        elif self.kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            self.cols = [0, 1, 2, 7, 8]                 # Mw, log distance, depth, log site depth, shelf share
            self.model = KNeighborsRegressor(25, weights="distance").fit(Xs[:, self.cols], y)

    def __call__(self, q):
        if self.year != q["year"]:
            self._fit(q["year"])
        if self.model is None:
            return np.full(len(q["dist"]), self.med)
        Xs = (feats_b(q) - self.mu) / self.sd
        return self.model.predict(Xs[:, self.cols] if self.kind == "knn" else Xs)


class Hybrid(LearnedB):
    """Physics + learned correction: Green's-law shape (extrapolates in magnitude and distance) plus a strongly regularised
    model of its residual. Post-hoc (added after seeing that learned models saturate for the largest events)."""
    def _fit(self, year):
        cut = (pd.Timestamp(f"{year}-01-01") - v1.T0).days
        tr = [c for c in self.pool if c["t"] + 2 <= cut]
        self.year = year
        self.med = 0.0
        if len(tr) < 30:
            self.model = None
            return
        X = np.vstack([feats_b(c) for c in tr]); y = np.concatenate([c["y"] - GreenBathy.shape(c) for c in tr])
        w = np.concatenate([np.full(len(c["y"]), 1.0 / len(c["y"])) for c in tr]); w = w * len(w) / w.sum()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=100.0).fit(Xs, y, sample_weight=w)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=80,
                                                       l2_regularization=10.0, random_state=0).fit(Xs, y, sample_weight=w)

    def __call__(self, q):
        if self.year != q["year"]:
            self._fit(q["year"])
        base = GreenBathy.shape(q)
        if self.model is None:
            return base
        return base + self.model.predict((feats_b(q) - self.mu) / self.sd)


class GreenBathy:
    def __init__(self, pool):
        self.pool, self.year, self.c = pool, None, 0.0

    @staticmethod
    def shape(q):
        th = np.clip(q["dist"] / 6371.0, 1e-3, np.pi - 1e-3)
        return (0.5 * (q["mw"] - 8) - 0.5 * np.log10(np.sin(th)) + 0.25 * np.log10(q["h_src"] / q["h_site"]) + v1.SHIFT)

    def __call__(self, q):
        if self.year != q["year"]:
            cut = (pd.Timestamp(f"{q['year']}-01-01") - v1.T0).days
            r = [np.median(c["y"] - self.shape(c)) for c in self.pool if c["t"] + 2 <= cut]
            self.c, self.year = (float(np.median(r)) if len(r) >= 5 else 0.0), q["year"]
        return self.shape(q) + self.c


def main(shift=3.0):
    v1.SHIFT = shift
    cases = [add_features(c) for c in v1.load()]
    experts = [from_callable("abe_1979", v1.abe, cost=1), from_callable("spreading_calibrated", v1.Spreading(cases), cost=1),
               from_callable("green_bathymetry", GreenBathy(cases), cost=1), from_callable("median_history", v1.Learned("median", cases), cost=1)]
    experts += [from_callable(k, v1.Learned(k, cases), cost=3) for k in ("ridge", "hgb", "knn")]
    experts += [from_callable(k + "_bathy", LearnedB(k, cases), cost=3) for k in ("ridge", "hgb", "knn")]
    experts += [from_callable("hybrid_" + k, Hybrid(k, cases), cost=3) for k in ("ridge", "hgb")]
    runs = {m: run_delayed(PhysicsEnsemble(experts, mode=m), cases, [c["y"] for c in cases], [c["t"] + 2.0 for c in cases])
            for m in ("select", "combine")}
    ev = [i for i, c in enumerate(cases) if c["year"] >= v1.EVAL_FROM]
    names = list(runs["select"][0]["experts"].keys())
    yy = np.concatenate([cases[i]["y"] for i in ev])
    gid = np.concatenate([np.full(len(cases[i]["y"]), cases[i]["id"]) for i in ev])
    P = {n: np.concatenate([runs["select"][i]["experts"][n] for i in ev]) for n in names}
    P["ensemble_select"] = np.concatenate([runs["select"][i]["prediction"] for i in ev])
    P["ensemble_combine"] = np.concatenate([runs["combine"][i]["prediction"] for i in ev])
    E = {n: np.abs(p - yy) for n, p in P.items()}
    best = min(names, key=lambda n: E[n].mean())
    lo = np.concatenate([runs["combine"][i]["lower"] for i in ev]); hi = np.concatenate([runs["combine"][i]["upper"] for i in ev])
    out = {"n_events_eval": len(ev), "n_sites_eval": int(len(yy)), "typical_factor": {n: float(10 ** e.mean()) for n, e in E.items()},
           "best_single_in_hindsight": best, "skill_vs_median_history": {n: cluster_skill(E[n], E["median_history"], gid) for n in E if n != "median_history"},
           "skill_vs_abe_1979": {n: cluster_skill(E[n], E["abe_1979"], gid) for n in E if n != "abe_1979"},
           "skill_combine_vs_best_single": cluster_skill(E["ensemble_combine"], E[best], gid),
           "skill_select_vs_best_single": cluster_skill(E["ensemble_select"], E[best], gid),
           "coverage_90_combine": float(np.mean((yy >= lo) & (yy <= hi))),
           "correlation_within_event": None}
    # share of the within-event variance of the observation that each model explains (R2 after removing event means)
    def within_r2(p):
        yc, pc = yy.copy(), p.copy()
        for g in np.unique(gid):
            m = gid == g
            yc[m] -= yc[m].mean(); pc[m] -= pc[m].mean()
        return float(1 - np.sum((yc - pc) ** 2) / np.sum(yc ** 2))
    out["within_event_r2"] = {n: within_r2(p) for n, p in P.items()}
    json.dump(out, open(RESULTS / "tsunamis_v2.json", "w"), indent=1)
    pickle.dump({"cases": [{k: v for k, v in c.items()} for c in cases], "select": runs["select"], "combine": runs["combine"], "shift": shift},
                open(RESULTS / "tsunamis_v2_predictions.pkl", "wb"))
    print({k: round(v, 2) for k, v in out["typical_factor"].items()}, "best", best)
    print("combine vs best", [round(a, 2) for a in out["skill_combine_vs_best_single"]], "cov", round(out["coverage_90_combine"], 2))
    print("within-event R2", {k: round(v, 2) for k, v in out["within_event_r2"].items()})
    v1p = json.load(open(RESULTS / "tsunamis.json"))["primary_shift_3"]
    print("v1 factors", {k: round(v, 2) for k, v in v1p["typical_factor"].items()})


if __name__ == "__main__":
    main()
