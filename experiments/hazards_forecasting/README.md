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
| Volcano eruption catalog | Smithsonian GVP | none (reported eruptions) |

Earthquake and tsunami targets come from seismic-network and field catalogs; the satellite imagery in their figures is
background, not input. Satellite products for them (InSAR, altimetry) need an Earthdata login and are future work.
The MODVOLC alert endpoint returned no data when probed.

## Results (evaluation period, chronological, no look-ahead)

**Earthquakes** (weekly log-count of M>=4.5, h = 1 week, evaluated 1995-2025). Skill vs climatology: global +0.28
[+0.22,+0.34], Japan +0.20, Chile-Peru +0.12, Indonesia +0.14, California-Baja +0.16 (all CIs above 0), from aftershock
clustering and catalog non-stationarity. Against the best single expert in hindsight the ensemble is equal (CIs contain
0) except California-Baja (+0.07 [+0.03,+0.10]). 90 % intervals cover 90 %. `figures/earthquake_japan_vs_observed.png`:
no model anticipates the Tohoku mainshock, and the ensemble under-predicts the first aftershock weeks (14 vs 741
observed), catching up after 2-3 weeks.

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

Weekly persistence is already strong; the ensemble matches it and beats it clearly only at Nyiragongo and Sakurajima.
Limitations: a cloud-covered scene looks like "no detection" (e.g. Nyiragongo, 2020-01-30), so the series underestimates
activity; the count is in drawn pixels of the GIBS layer, not radiative power. A first Nyiragongo series with a 0.2 deg
box was contaminated by vegetation fires ~15 km from the crater and was replaced by a 0.04 deg box (the old files are not
published). Boxes: Kilauea 0.20, Etna 0.10, Nyiragongo 0.04, Erta Ale and Sakurajima 0.05 deg half-width.

**Tsunamis** (run-up at coastal sites, 212 events / 18,236 sites from 1990, typical multiplicative error factor):
historical median 5.3, Abe (1979) closed form 3.0, calibrated spreading law 3.8, ridge 3.5, boosted trees 2.9, kNN 2.47
(best in hindsight), **ensemble (combine) 2.76**, ensemble (select) 3.04. The ensemble beats the median by +0.39
[+0.27,+0.52] but not the best single model in hindsight (-0.12 [-0.36,+0.06]). Robust to the one arbitrary choice (offset
of the log field, 1.5-4). 90 % interval coverage 0.88. Weakness: near the source the forecast saturates (about 4 m while
observed run-ups reach tens of metres); Sumatra 2004 intervals cover only 54 %
(`figures/tsunami_map_vs_observed.png`). Catalog sites are those where a run-up was measured (selection bias).

**Tropical cyclones**: experiment implemented (`src/exp_hurricanes.py`: 71,233 cases from 3,696 storms, intensity and
track at 12-48 h, six experts including a Kaplan-DeMaria decay model) but still running when this was published; results
and figures will be added in a follow-up commit.

## Reproduce

```
python src/download_data.py                  # IBTrACS, USGS, NOAA tsunami, GVP (public)
python src/download_viirs_volcano.py         # ~1 h, GIBS WMS; the CSVs are already in data/volcano/
python src/exp_earthquakes.py; python src/exp_volcanoes.py; python src/exp_volcano_viirs.py kilauea etna nyiragongo erta_ale sakurajima
python src/exp_tsunamis.py
python src/fig_earthquake.py; python src/fig_tsunami.py; python src/fig_volcano.py kilauea etna nyiragongo erta_ale sakurajima
```

Sources: IBTrACS (Knapp et al. 2010), USGS ComCat, NOAA NCEI/WDS Global Historical Tsunami Database, Smithsonian GVP,
NASA GIBS (VIIRS thermal anomalies, MODIS corrected reflectance, Blue Marble), Abe (1979), Kaplan & DeMaria (1995),
Herbster & Warmuth (1998), de Rooij et al. (2014), Gibbs & Candes (2021).
