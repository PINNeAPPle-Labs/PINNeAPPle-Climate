"""Volcanoes from satellite: weekly VIIRS thermal-anomaly activity (NASA GIBS, Suomi NPP 375 m, 2012-2025).

Series: log(1 + mean daily number of thermal-anomaly pixels in a 0.4 deg box around the volcano) per week. Forecast 1-4
weeks ahead with the adaptive ensemble (default pool + climatology + a seasonal-free Omori-like decay for post-eruption
cooling). Online from 2012; forecasts from 2015, metrics from 2016. Skill vs persistence and vs climatology with
block-bootstrap CIs. Cloud cover and swath gaps make a day without detection ambiguous (no eruption vs not seen); the
series therefore underestimates activity and is noisy, which is part of the test.
"""
import json

import numpy as np
import pandas as pd

from common import DATA, RESULTS, skill
from exp_earthquakes import Climatology, OmoriDecay
from pinneapple_systems.time_series import AdaptiveForecaster, default_experts

H, FORECAST_FROM, EVAL_FROM = 4, 2015, 2016


def weekly_series(name):
    d = pd.read_csv(DATA / "volcano" / f"viirs_{name}.csv", parse_dates=["date"]).set_index("date")
    d = d[d.pixels >= 0]                                   # failed requests are dropped, not treated as zeros
    return d.pixels.resample("W").mean().dropna()


def experts():
    ex = default_experts(season_length=1)
    ex["climatology"] = Climatology()
    ex["omori"] = OmoriDecay()
    return ex


def evaluate(name):
    s = weekly_series(name)
    y = np.log1p(s.values)
    run = AdaptiveForecaster(experts(), horizon=H).run(y, start=int(np.argmax(s.index.year >= FORECAST_FROM)))
    em = s.index[run.origins].year >= EVAL_FROM
    res = {"n_weeks": len(y), "frac_weeks_active": float(np.mean(s > 0))}
    for h in range(1, H + 1):
        idx = run.origins + h
        ok = (idx < len(y)) & em
        truth = y[idx[ok]]
        ens = np.abs(run.forecast[ok, h - 1] - truth)
        ref = {n: np.abs(run.expert_forecasts[ok, i, h - 1] - truth) for i, n in enumerate(run.expert_names)}
        mae = {n: float(e.mean()) for n, e in ref.items()}
        best = min(mae, key=mae.get)
        res[f"h{h}"] = {"mae_ensemble": float(ens.mean()), "mae_naive": mae["naive"], "mae_climatology": mae["climatology"],
                        "best_single_in_hindsight": best, "mae_best_single": mae[best],
                        "skill_vs_naive": skill(ens, ref["naive"]), "skill_vs_climatology": skill(ens, ref["climatology"]),
                        "skill_vs_best_single": skill(ens, ref[best]),
                        "coverage_90": float(np.mean((truth >= run.lower[ok, h - 1]) & (truth <= run.upper[ok, h - 1])))}
    return s, y, run, res


def main(names):
    out = {}
    for n in names:
        _, _, _, res = evaluate(n)
        out[n] = res
        r = res["h1"]
        print(f"{n:12s} active weeks {res['frac_weeks_active']:.0%} h1 MAE ens {r['mae_ensemble']:.3f} naive {r['mae_naive']:.3f} clim {r['mae_climatology']:.3f}"
              f" | skill vs naive {r['skill_vs_naive'][0]:+.2f} [{r['skill_vs_naive'][1]:+.2f},{r['skill_vs_naive'][2]:+.2f}] vs clim {r['skill_vs_climatology'][0]:+.2f}"
              f" | cov {r['coverage_90']:.2f}")
    (RESULTS / "volcano_viirs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or ["kilauea"])
