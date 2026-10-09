# Hazard forecasting with adaptive ensembles (earthquakes, tsunamis, volcanoes, tropical cyclones)

Experiments with the adaptive forecasting ensembles of PINNeAPPle (`pinneapple_systems.time_series.AdaptiveForecaster`,
`pinneapple_physics.ensemble.PhysicsEnsemble`; Fixed-Share weights + AdaHedge + adaptive conformal intervals) on public
data. Goal: forecasts that do not overfit, and honest statements when there is no skill.

The ensemble module is not on PINNeAPPle `main` yet; the experiments were run against branch
`claude/clever-archimedes-sajlck` (set `PINNEAPPLE_SRC` to a checkout of it).

## Protocol (src/common.py)

1. Strictly chronological. A label is used only after it exists (`run_delayed`: cases in flight at the same time, label
   available at issue time + lead time).
2. Learned experts are refitted on an expanding window of matured past cases; no hyper-parameter is tuned on the
   evaluation period; a warm-up period separates the start of the stream from the reported metrics.
3. Every result is a skill score against persistence / climatology, with a block (or event-cluster) bootstrap 95 % CI.
   "No skill" is reported as such. The best single model in hindsight is shown as an upper bar, not a competitor.

## Data (public, no login)

| Hazard | Data | Where it is satellite |
|---|---|---|
| Volcanoes | NASA GIBS VIIRS Suomi-NPP 375 m thermal anomalies, daily, 2012-2025 (5 volcanoes) | the target series itself |
| Tropical cyclones | IBTrACS v04r01 best tracks (built from satellite imagery fixes), 1980-2025 | the track/intensity data; MODIS scenes for the figures |
| Earthquakes | USGS ComCat M>=4.5, 1980-2025 | MODIS scenes as context only |
| Tsunamis | NOAA NCEI/WDS run-ups and events | Blue Marble imagery as context only |
| Tsunami bathymetry (v2) | NOAA ETOPO 2022, 5 arc-min, via OPeNDAP | derived partly from satellite altimetry (context for the physics expert) |
| Volcano eruption catalog | Smithsonian GVP | none (reported eruptions) |

Earthquake and tsunami targets come from seismic-network and field catalogs; the satellite imagery in their figures is
background, not input. Satellite products for them (InSAR, altimetry) need an Earthdata login and are future work.
The MODVOLC alert endpoint returned no data when probed.

## Results (evaluation period, chronological, no look-ahead)

**Earthquakes** (weekly log-count of M>=4.5, h = 1 week, evaluated 1995-2025; v2 with an ETAS expert, `results/earthquakes_v2.json`;
v1 is kept in `results/earthquakes.json`). Skill vs climatology: global +0.45 [+0.40,+0.49], Japan +0.23, Chile-Peru +0.21,
Indonesia +0.21, California-Baja +0.21 (all CIs above 0), from aftershock clustering and catalog non-stationarity. Against the
best single expert in hindsight the ensemble ties (CIs contain 0) except California-Baja (+0.08 [+0.05,+0.11]). 90 % intervals
cover 90 %. v2 added an ETAS expert (`src/etas.py`: Omori-Utsu aftershock sums with fixed alpha, p, c; only background rate and
productivity are fitted, online, from the past) and a gated version active only while a sequence is under way (post-hoc). A
first version of the ETAS expert had a bug (the forecaster hands experts only the last 1000 weeks, so a series index derived from
the history length froze at week 1000); fixed with `max_history=None`. `figures/earthquake_japan_vs_observed.png`: nothing
anticipates the Tohoku mainshock; the ETAS expert predicts the aftershock level one week later (295 vs 570 observed) but the ensemble
gives it only ~14 % weight at that moment (weights come from past errors and no earlier Japan sequence was this large), so the
ensemble forecast is still low (25 vs 570) for that first week and follows from the second week on.

**Volcanoes, catalog** (GVP, monthly eruption starts): no skill against the Poisson rate (skill -0.1 % to -2.4 %, CIs
contain 0); the ensemble collapses to climatology at no cost, coverage 0.89-0.91.

**Volcanoes, VIIRS thermal series** (weekly, h = 1; metrics 2016-2025):

| Volcano | skill vs persistence | skill vs climatology | 90 % coverage |
|---|---|---|---|
| Kilauea | -0.01 [-0.07,+0.04] | +0.79 | 0.89 |
| Etna | +0.03 [-0.00,+0.06] | +0.38 | 0.90 |
| Nyiragongo | +0.13 [+0.07,+0.17] | +0.48 | 0.89 |
| Erta Ale | +0.02 [-0.03,+0.07] | +0.62 | 0.90 |
| Sakurajima | +0.07 [+0.01,+0.13] | +0.21 | 0.90 |

Weekly persistence is already strong; the ensemble matches it and beats it clearly only at Nyiragongo (skill rises with lead, +0.13 at 1 week to +0.21 at 4) and Sakurajima. The figures now show the cumulative error advantage over persistence (where the gains come from, e.g. Etna after 2018) and `figures/volcano_skill_by_lead.png` the skill by lead for all five volcanoes.
Limitations: a cloud-covered scene looks like "no detection" (e.g. Nyiragongo, 2020-01-30), so the series underestimates
activity; the count is in drawn pixels of the GIBS layer, not radiative power. A first Nyiragongo series with a 0.2 deg
box was contaminated by vegetation fires ~15 km from the crater and was replaced by a 0.04 deg box (the old files are not
published). Boxes: Kilauea 0.20, Etna 0.10, Nyiragongo 0.04, Erta Ale and Sakurajima 0.05 deg half-width.

**Tsunamis** (run-up at coastal sites, 212 events / 18,236 sites from 1990, typical multiplicative error factor), v1:
historical median 5.3, Abe (1979) closed form 3.0, calibrated spreading law 3.8, ridge 3.5, boosted trees 2.9, kNN 2.47
(best in hindsight), **ensemble (combine) 2.76**, ensemble (select) 3.04. The ensemble beats the median by +0.39
[+0.27,+0.52] but not the best single model in hindsight (-0.12 [-0.36,+0.06]). Robust to the one arbitrary choice (offset
of the log field, 1.5-4). Weakness: near the source the forecast saturates (about 4 m while observed run-ups reach tens of
metres); Sumatra 2004 intervals cover only 54 % (`figures/tsunami_map_vs_observed.png`). Catalog sites are those where a
run-up was measured (selection bias).

*v2 attempts to fix this (post-hoc, `src/exp_tsunamis_v2.py`, `results/tsunamis_v2.json`), all negative or marginal:* ETOPO 2022 bathymetry
at 5 arc-minutes gives each site the local ocean depth, shelf share, the shallow-water travel time and mean depth along the great circle;
a Green's-law expert with the real depths (3.81); the same features in ridge/trees/kNN (kNN improves 2.47 -> 2.35); and a physics +
learned-residual hybrid (3.54 / 2.93, no better than the pure learned models). The ensemble does not improve (combine 2.89, select 2.90,
slightly worse than v1 because more weak experts dilute it) and does not beat the best single model (-0.24 [-0.41,+0.07]); 90 % coverage
is 0.91. About half of the within-event variance of log run-up is explained (R2 0.45-0.48 for the best models), the rest is local
coastal amplification that magnitude, distance and 9 km bathymetry do not carry. The saturation for Sumatra 2004 and Tohoku 2011 remains.
Resolving it needs source-resolved simulation (fault geometry, nested high-resolution bathymetry), which is future work. The figure shows v2.

**Tropical cyclones, intensity** (IBTrACS best tracks, 1990-2025 stream, metrics from 2005: 30,714 cases from 1,631 storms, 12-48 h
lead, MAE averaged over the four leads). Persistence 16.4 kt; Kaplan-DeMaria over-land decay 16.4 (no gain: storms rarely are over
land at the origin); ridge 13.6; boosted trees 12.3; kNN 12.6; ensemble (combine) 13.2, ensemble (select) 13.3. Skill vs persistence
(storm-cluster bootstrap): trees +0.25 [+0.24,+0.26], kNN +0.23, ridge +0.17, **ensemble (combine) +0.20 [+0.18,+0.21]**. The ensemble is
clearly worse than the best single model in hindsight (-0.07 [-0.08,-0.06]); the leader (trees, 32 % of the time, kNN 30 %) only
emerges slowly because labels arrive 48 h after the forecast, and with ~8 storms alive at once the weights see far fewer independent
outcomes than cases. 90 % intervals cover 89 %, but are wide (the interval comes from expert spread and a conformal factor, not a
per-case model of uncertainty) and the ensemble misses rapid intensification (Ian 2022: 24 kt MAE at 24 h vs 27 for persistence).
`figures/hurricane_intensity_vs_observed.png`: Irma, Dorian, Ian and Haiyan against the best track, with the MODIS scene at peak.
An MLP expert was planned, but with the installed scikit-learn it raised on every call (positional `loss` argument), so the
ensemble ran with five experts (persistence, Kaplan-DeMaria, ridge, trees, kNN); the code now excludes it explicitly.
Track forecasts (`src/exp_hurricanes.py track`) were still running when this was published and will be added later.

## Reproduce

```
python src/download_data.py                  # IBTrACS, USGS, NOAA tsunami, GVP (public)
python src/download_viirs_volcano.py         # ~1 h, GIBS WMS; the CSVs are already in data/volcano/
python src/download_bathymetry.py           # ETOPO 2022 at 5 arc-min via OPeNDAP (tsunami v2)
python src/exp_earthquakes.py; python src/exp_volcanoes.py; python src/exp_volcano_viirs.py kilauea etna nyiragongo erta_ale sakurajima
python src/exp_tsunamis.py; python src/exp_tsunamis_v2.py
python src/exp_hurricanes.py wind track      # hours; the prediction pickles are not in the repo
python src/fig_hurricane.py
python src/fig_earthquake.py; python src/fig_tsunami.py; python src/fig_volcano.py kilauea etna nyiragongo erta_ale sakurajima; python src/fig_volcano_summary.py
```

Sources: IBTrACS (Knapp et al. 2010), USGS ComCat, NOAA NCEI/WDS Global Historical Tsunami Database, Smithsonian GVP,
NASA GIBS (VIIRS thermal anomalies, MODIS corrected reflectance, Blue Marble), Abe (1979), Kaplan & DeMaria (1995),
Herbster & Warmuth (1998), de Rooij et al. (2014), Gibbs & Candes (2021).
