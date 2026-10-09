"""Volcanoes: monthly eruption-start rate (Smithsonian GVP Holocene eruptions) with the adaptive forecasting ensemble.

Target: monthly count of eruptions that start in a region (only eruptions with a known start month), 1979-2025
(satellite era, when remote observation made reporting of remote volcanoes more complete), horizons 1..6 months.
Honest scope: the GVP is a catalog of reported eruptions, not a satellite time series; the MODVOLC thermal-alert
endpoint (MODIS) returned no data when probed, so thermal series are future work. Compared with the stationary
Poisson rate (climatology) and persistence, with block-bootstrap CIs.
"""
import json

import numpy as np
import pandas as pd

from common import DATA, RESULTS, skill
from exp_earthquakes import Climatology
from pinneapple_systems.time_series import AdaptiveForecaster, default_experts

REGIONS = {"global": None, "indonesia": (-11, 6, 94, 141), "kamchatka_kuril_japan": (30, 60, 130, 175),
           "central_south_america": (-56, 22, -110, -60)}
H, START_YEAR, EVAL_YEAR = 6, 1979, 1995


def load():
    f = json.loads((DATA / "volcano" / "gvp_eruptions.json").read_text())["features"]
    rows = [{"lon": x["geometry"]["coordinates"][0], "lat": x["geometry"]["coordinates"][1], **x["properties"]} for x in f]
    d = pd.DataFrame(rows)
    d = d[(d.StartDateYear >= START_YEAR) & (d.StartDateMonth.fillna(0) > 0) & (d.Activity_Type == "Confirmed Eruption")]
    d["t"] = pd.to_datetime(dict(year=d.StartDateYear.astype(int), month=d.StartDateMonth.astype(int), day=1))
    return d.drop_duplicates(["Eruption_Number"])


def monthly(d, box):
    if box:
        la0, la1, lo0, lo1 = box
        d = d[d.lat.between(la0, la1) & d.lon.between(lo0, lo1)]
    idx = pd.date_range(f"{START_YEAR}-01-01", "2025-12-01", freq="MS")
    return d.groupby("t").size().reindex(idx, fill_value=0)


def main():
    d, out = load(), {}
    for name, box in REGIONS.items():
        s = monthly(d, box)
        y = s.values.astype(float)
        ex = default_experts(season_length=12)
        ex["climatology"] = Climatology()
        start = int(np.argmax(s.index.year >= EVAL_YEAR - 5))
        run = AdaptiveForecaster(ex, horizon=H).run(y, start=start)
        em = s.index[run.origins].year >= EVAL_YEAR
        res = {"n_months": len(y), "mean_monthly_starts": float(y.mean())}
        for h in (1, 3, 6):
            idx = run.origins + h
            ok = (idx < len(y)) & em
            truth = y[idx[ok]]
            ens = np.abs(run.forecast[ok, h - 1] - truth)
            ref = {n: np.abs(run.expert_forecasts[ok, i, h - 1] - truth) for i, n in enumerate(run.expert_names)}
            mae = {n: float(e.mean()) for n, e in ref.items()}
            best = min(mae, key=mae.get)
            res[f"h{h}"] = {"mae_ensemble": float(ens.mean()), "mae_climatology": mae["climatology"],
                            "mae_naive": mae["naive"], "best_single_in_hindsight": best, "mae_best_single": mae[best],
                            "skill_vs_climatology": skill(ens, ref["climatology"]), "skill_vs_naive": skill(ens, ref["naive"]),
                            "skill_vs_best_single": skill(ens, ref[best]),
                            "coverage_90": float(np.mean((truth >= run.lower[ok, h - 1]) & (truth <= run.upper[ok, h - 1])))}
        out[name] = res
        r = res["h1"]
        print(f"{name:24s} mean {res['mean_monthly_starts']:.2f}/month h1 MAE ens {r['mae_ensemble']:.3f} clim {r['mae_climatology']:.3f} "
              f"best {r['best_single_in_hindsight']} {r['mae_best_single']:.3f} | skill vs clim {r['skill_vs_climatology'][0]:+.3f} "
              f"[{r['skill_vs_climatology'][1]:+.3f},{r['skill_vs_climatology'][2]:+.3f}] cov {r['coverage_90']:.2f}")
    (RESULTS / "volcanoes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
