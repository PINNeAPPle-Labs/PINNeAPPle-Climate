"""Tsunamis: forecast of the run-up height at each coastal site from the source earthquake (NOAA NCEI/WDS database).

A *case* is one earthquake-generated tsunami; the field to predict is log10(run-up height) at the observed sites, from
what is known minutes after the earthquake: moment magnitude, depth, epicentre and each site's location and distance.
Experts: Abe (1979) closed form, a spherical-spreading law with an online-calibrated constant, ridge, boosted trees,
analog (kNN), and the historical median. Stream: events from 1960, label available 2 days after the event, learned
experts refitted each 1 January on earlier events; metrics from 1990. Cluster bootstrap over events.
"""
import json

import numpy as np
import pandas as pd

from common import DATA, RESULTS, cluster_skill, run_delayed
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable

STREAM_FROM, EVAL_FROM, SHIFT = 1960, 1990, 3.0       # target = log10(H / 1 m) + SHIFT (relative L2 needs a non-zero field)
T0 = pd.Timestamp("1900-01-01")


def load():
    ru = pd.DataFrame(json.load(open(DATA / "tsunami" / "runups.json")))
    ev = pd.DataFrame(json.load(open(DATA / "tsunami" / "events.json"))).set_index("id")
    ru = ru[(ru.year >= STREAM_FROM) & ru.runupHt.notna() & (ru.runupHt > 0) & ru.sourceEqMagnitude.notna() &
            ru.distFromSource.notna() & ru.latitude.notna() & ru.tsunamiEventId.isin(ev.index) & (ru.doubtful != "y")]
    cases = []
    for eid, g in ru.groupby("tsunamiEventId"):
        if len(g) < 3:
            continue
        e = ev.loc[eid]
        if pd.isna(e.month) or pd.isna(e.day) or pd.isna(e.latitude):
            continue
        t = pd.Timestamp(year=int(e.year), month=max(int(e.month), 1), day=max(int(e.day), 1))
        cases.append({"id": int(eid), "t": (t - T0).days, "year": int(e.year), "mw": float(g.sourceEqMagnitude.iloc[0]),
                      "depth": float(e.eqDepth) if pd.notna(e.eqDepth) else np.nan, "elat": float(e.latitude), "elon": float(e.longitude),
                      "dist": np.maximum(g.distFromSource.values.astype(float), 5.0), "slat": g.latitude.values, "slon": g.longitude.values,
                      "y": np.log10(g.runupHt.values.astype(float)) + SHIFT})
    cases.sort(key=lambda c: c["t"])
    return cases


def feats(q):
    d = q["dist"]
    return np.column_stack([np.full(len(d), q["mw"]), np.log10(d), np.full(len(d), np.nan_to_num(q["depth"], nan=30.0)),
                           np.abs(q["slat"]), np.sin(np.radians(q["slon"])), np.cos(np.radians(q["slon"])),
                           q["mw"] * np.log10(d)])


class Learned:
    def __init__(self, kind, pool):
        self.kind, self.pool, self.year, self.model, self.med = kind, pool, None, None, None

    def _fit(self, year):
        cut = (pd.Timestamp(f"{year}-01-01") - T0).days
        tr = [c for c in self.pool if c["t"] + 2 <= cut]
        self.year = year
        self.med = float(np.median(np.concatenate([c["y"] for c in tr]))) if tr else SHIFT
        if len(tr) < 30:
            self.model = None
            return
        X = np.vstack([feats(c) for c in tr]); y = np.concatenate([c["y"] for c in tr])
        w = np.concatenate([np.full(len(c["y"]), 1.0 / len(c["y"])) for c in tr])      # each event counts once
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=10.0).fit(Xs, y, sample_weight=w * len(w) / w.sum())
        elif self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                                       l2_regularization=10.0, random_state=0).fit(Xs, y, sample_weight=w * len(w) / w.sum())
        elif self.kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor(25, weights="distance").fit(Xs[:, :3], y)

    def __call__(self, q):
        if self.year != q["year"]:
            self._fit(q["year"])
        if self.kind == "median" or self.model is None:
            return np.full(len(q["dist"]), self.med)
        Xs = (feats(q) - self.mu) / self.sd
        return self.model.predict(Xs[:, :3] if self.kind == "knn" else Xs)


class Spreading:
    """H = c * 10^(0.5 (Mw - 8)) / sqrt(sin(theta)): source displacement scales with the rupture length (10^(0.5 Mw)),
    amplitude falls as the square root of the great-circle spreading. ``c`` is the median past log-ratio, updated only from
    matured events (online calibration, the single fitted constant)."""
    def __init__(self, pool):
        self.pool, self.year, self.c = pool, None, 0.0

    @staticmethod
    def shape(q):
        th = np.clip(q["dist"] / 6371.0, 1e-3, np.pi - 1e-3)
        return 0.5 * (q["mw"] - 8) - 0.5 * np.log10(np.sin(th)) + SHIFT

    def __call__(self, q):
        if self.year != q["year"]:
            cut = (pd.Timestamp(f"{q['year']}-01-01") - T0).days
            r = [np.median(c["y"] - self.shape(c)) for c in self.pool if c["t"] + 2 <= cut]
            self.c, self.year = (float(np.median(r)) if len(r) >= 5 else 0.0), q["year"]
        return self.shape(q) + self.c


def abe(q):
    """Abe (1979): Mt = log10 H + log10 Delta + 5.55 (H in m, Delta in km), with Mt = Mw."""
    return np.log10(10 ** (q["mw"] - 5.55) / q["dist"]) + SHIFT


def run(shift):
    global SHIFT
    SHIFT = shift
    cases = load()
    print("events", len(cases), "sites", sum(len(c["y"]) for c in cases))
    experts = [from_callable("abe_1979", abe, cost=1), from_callable("spreading_calibrated", Spreading(cases), cost=1),
               from_callable("median_history", Learned("median", cases), cost=1)]
    experts += [from_callable(k, Learned(k, cases), cost=3) for k in ("ridge", "hgb", "knn")]
    out = {}
    runs = {}
    for mode in ("select", "combine"):
        runs[mode] = run_delayed(PhysicsEnsemble(experts, mode=mode), cases, [c["y"] for c in cases], [c["t"] + 2.0 for c in cases])
    ev = [i for i, c in enumerate(cases) if c["year"] >= EVAL_FROM]
    names = list(runs["select"][0]["experts"].keys())
    y = [cases[i]["y"] for i in ev]
    gid = np.concatenate([np.full(len(v), cases[i]["id"]) for i, v in zip(ev, y)])
    yy = np.concatenate(y)
    P = {n: np.concatenate([runs["select"][i]["experts"][n] for i in ev]) for n in names}
    P["ensemble_select"] = np.concatenate([runs["select"][i]["prediction"] for i in ev])
    P["ensemble_combine"] = np.concatenate([runs["combine"][i]["prediction"] for i in ev])
    E = {n: np.abs(p - yy) for n, p in P.items()}
    best = min(names, key=lambda n: E[n].mean())
    lo = np.concatenate([runs["select"][i]["lower"] for i in ev]); hi = np.concatenate([runs["select"][i]["upper"] for i in ev])
    out = {"n_events_eval": len(ev), "n_sites_eval": int(len(yy)), "mae_log10": {n: float(e.mean()) for n, e in E.items()},
           "typical_factor": {n: float(10 ** e.mean()) for n, e in E.items()}, "best_single_in_hindsight": best,
           "skill_vs_median_history": {n: cluster_skill(E[n], E["median_history"], gid) for n in E if n != "median_history"},
           "skill_vs_abe_1979": {n: cluster_skill(E[n], E["abe_1979"], gid) for n in E if n != "abe_1979"},
           "skill_select_vs_best_single": cluster_skill(E["ensemble_select"], E[best], gid),
           "skill_combine_vs_best_single": cluster_skill(E["ensemble_combine"], E[best], gid),
           "coverage_90": float(np.mean((yy >= lo) & (yy <= hi))),
           "active_share": {n: float(np.mean([runs["select"][i]["active"] == n for i in ev])) for n in names}}
    if shift == 3.0:
        import pickle
        pickle.dump({"cases": cases, "select": runs["select"], "combine": runs["combine"], "shift": shift},
                    open(RESULTS / "tsunamis_predictions.pkl", "wb"))
    return out


def main():
    primary = run(3.0)                                  # fixed before looking at any result
    sens = {str(s): run(s) for s in (1.5, 2.0, 4.0)}    # sensitivity to the only arbitrary choice: reported, not selected
    allres = {"primary_shift_3": primary, "sensitivity": sens}
    (RESULTS / "tsunamis.json").write_text(json.dumps(allres, indent=1))
    for k, v in {"3.0": primary, **sens}.items():
        f = v["typical_factor"]
        print(f"shift {k}: factor ens_select {f['ensemble_select']:.2f} ens_combine {f['ensemble_combine']:.2f} "
              f"best {v['best_single_in_hindsight']} {f[v['best_single_in_hindsight']]:.2f} abe {f['abe_1979']:.2f} "
              f"median {f['median_history']:.2f} | combine vs best {v['skill_combine_vs_best_single'][0]:+.2f} cov {v['coverage_90']:.2f}")


if __name__ == "__main__":
    main()
