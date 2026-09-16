# Physics AI for Climate & Earth Systems: A Reference Landscape

*A seed document for an open-source reference repository on physics-informed / scientific machine learning applied to weather, climate, and natural-disaster prediction.*

> **Status note:** This document reflects the public state of the field as of September 2026. AI-driven weather and climate modeling is moving faster than almost any other applied-ML domain — expect model names, benchmark leaderboards, and operational status to shift within months. Wherever a specific claim could not be independently verified in this research pass, it is flagged as such rather than stated as fact.

## Table of Contents

1. [Why This Field, Why Now](#why-this-field-why-now)
2. [Academic & Research Lab Landscape](#academic--research-lab-landscape)
3. [Model Architectures](#model-architectures)
4. [Commercial Products & Startups](#commercial-products--startups)
5. [Numerical Methods & Baseline Solvers](#numerical-methods--baseline-solvers)
6. [Geometry & Grid Resources](#geometry--grid-resources)
7. [Datasets & Benchmarks](#datasets--benchmarks)
8. [Standards & Validation](#standards--validation)
9. [Open Challenges & Gaps](#open-challenges--gaps)
10. [Roadmap: Building the Reference Hub](#roadmap-building-the-reference-hub)
11. [Sources](#sources)

---

## Why This Field, Why Now

Between 2022 and 2026, machine learning went from a curiosity in numerical weather prediction (NWP) to an operational component at the world's leading forecast centers. ECMWF now runs its **Artificial Intelligence Forecasting System (AIFS)** alongside its physics-based Integrated Forecasting System (IFS) in production, and Google DeepMind's **WeatherNext** models have been used to forecast real hurricanes (including Hurricane Melissa's rapid intensification during the 2025 Atlantic season) in near-real time. At the same time, hybrid physics-ML general circulation models like **NeuralGCM** are starting to blur the line between "weather model" and "climate model," and diffusion-based ensemble systems like **GenCast** are reshaping how forecasters think about probabilistic prediction and computational cost.

This document catalogs the landscape — labs, papers, companies, datasets, numerical baselines, grids, standards bodies, and open problems — as a starting point for a public reference repository.

---

## Academic & Research Lab Landscape

### Operational forecast centers with active ML programs
- **ECMWF (European Centre for Medium-Range Weather Forecasts)** — the single most influential institution in this space. Produces **ERA5** reanalysis (the de facto standard ML training set), runs the physics-based **IFS**, and as of February 25, 2025 operationally runs **AIFS**, its own machine-learned forecasting system, in parallel with IFS. An ensemble version of AIFS followed in July 2025, and **AIFS v2** has since been documented in *Geoscientific Model Development*. ECMWF also co-developed **NeuralGCM** with Google Research.
- **NOAA** (including NCEP, GFDL, and the Physical Sciences Laboratory) — operates the **Global Forecast System (GFS)**, now built on the FV3 dynamical core, and is a partner institution for hybrid ML-physics research (e.g., Ai2's collaboration with GFDL).
- **NASA GISS (Goddard Institute for Space Studies)** — develops **ModelE**, NASA's coupled atmosphere-ocean climate model. Recent work (Elsaesser et al. 2025, *JAMES*) used a neural-network surrogate of ModelE to build a machine-learning-calibrated physics ensemble (CPE) spanning 45 physics parameters — voted GISS's top internal publication of 2025. Related work applies ML emulation to atmospheric composition/aerosol schemes ("Smart-NINT").
- **NCAR (National Center for Atmospheric Research)** — NSF-sponsored, develops the **Community Earth System Model (CESM)** and the **MPAS** (Model for Prediction Across Scales) unstructured-mesh model; active in ML-based Earth system model research and hosts community verification/benchmarking discussions (e.g., re-broadcasting the NeurIPS Climate Change AI workshop).
- **University of Reading (UK)** — a leading academic center for data assimilation and ML-weather research, closely tied to ECMWF (many Reading graduates and faculty collaborate directly with ECMWF's research department).

### Industrial and non-profit AI labs
- **Google DeepMind / Google Research** — the most prolific publisher in this space: **GraphCast** (Science, 2023), **GenCast** (Nature, 2024/2025), and **NeuralGCM** (Nature, 2024), plus the productized **WeatherNext** model family (WeatherNext 2, WeatherNext 3) and the **WeatherBench2** benchmark.
- **Microsoft Research** — **ClimaX** (ICML 2023, first general-purpose weather/climate foundation model) and **Aurora** (Nature, May 2025, "a foundation model for the Earth system"), plus the **Planetary Computer** data platform.
- **NVIDIA** — **FourCastNet** (2022) and the broader **Earth-2** platform, built on NVIDIA's **PhysicsNeMo** (formerly Modulus) framework for physics-ML.
- **Huawei Cloud** — **Pangu-Weather** (Nature, 2023), the first AI model to be shown, in a peer-reviewed paper, to beat a leading NWP system (ECMWF's IFS) on deterministic accuracy at several lead times.
- **Allen Institute for AI (Ai2)** — a dedicated Climate Modeling team building ML surrogates for climate model components; flagship result is **ACE (Ai2 Climate Emulator)**, a ~200M-parameter autoregressive emulator of a 100 km-resolution global atmospheric model, developed with NOAA GFDL and Lawrence Livermore National Laboratory (LLNL) as partners. Related collaboration with UC San Diego produced **Spherical DYffusion**, a generative model claimed to simulate ~100 years of climate in ~25 hours (~25x faster than the reference model).
- **UC Berkeley / Lawrence Berkeley National Laboratory (LBNL)** — Berkeley AI Research (BAIR) runs a Climate Initiative reading group; LBNL's climate science and "ML for Science" (ml4sci.lbl.gov) groups work on extreme-weather detection in exascale climate simulations and physically-consistent ML methods.

### Key venues
- **NeurIPS "Tackling Climate Change with Machine Learning" workshop**, organized by the nonprofit **Climate Change AI (CCAI)** — the field's flagship interdisciplinary venue, running annually since 2019; the 2025 edition (San Diego) featured keynotes from Microsoft Research and UC San Diego, and the 2026 edition is scheduled at NeurIPS in Sydney.
- **AMS (American Meteorological Society)** annual meeting and its journal ***Artificial Intelligence for the Earth Systems (AIES)*** — the primary meteorological-society venue bridging operational forecasters and ML researchers.
- Domain journals increasingly carrying this work: *Nature*, *Science*, *Science Advances*, *Journal of Advances in Modeling Earth Systems (JAMES)*, *Geoscientific Model Development (GMD)*, and NeurIPS/ICML proceedings (Datasets & Benchmarks track).

---

## Model Architectures

### Graph neural networks — GraphCast (DeepMind, 2023)
GraphCast encodes the atmosphere as a multi-mesh graph and uses message-passing GNNs to autoregressively predict the next state at 0.25° resolution across 37 pressure levels, using 6-hour steps out to 10 days. In evaluation on ERA5-derived ground truth, GraphCast outperformed ECMWF's HRES on more than 90% of 1,380+ verification targets, while running in under a minute on a single TPU/GPU node versus hours on a supercomputer for physics-based NWP. Published in *Science* (Lam et al., "Learning skillful medium-range global weather forecasting," 2023); code and weights open-sourced by DeepMind.

### Fourier Neural Operators — FourCastNet (NVIDIA, 2022)
FourCastNet combines the **Adaptive Fourier Neural Operator (AFNO)** — a token-mixing operation that learns in the Fourier domain, well suited to PDE-governed systems — with a Vision Transformer backbone. Trained on ERA5 at 0.25° resolution, it generates a week-long global forecast in under 2 seconds on a GPU (vs. hours for IFS), matching IFS accuracy on large-scale fields and exceeding it on high-frequency variables like precipitation. Two arXiv releases (2202.11214, 2208.05419) document the model and its scaling to production-class throughput; it underlies NVIDIA's Earth-2 platform and PhysicsNeMo examples.

### Vision-transformer weather models — Pangu-Weather (Huawei, 2023) and Aurora (Microsoft, 2025)
- **Pangu-Weather** introduces a **3D Earth-Specific Transformer (3DEST)** to natively handle the non-uniform 3D structure of atmospheric data (pressure levels are not evenly spaced physically), trained on ERA5 (1979–2021) with hierarchical temporal aggregation (1h/3h/6h/24h sub-models) to minimize autoregressive rollout error. It produces a 24-hour global forecast in 1.4 seconds on a single V100 GPU and was the first ML model published in *Nature* to demonstrably outperform a leading operational NWP system on standard accuracy metrics.
- **Aurora** is a large-scale foundation model pretrained on over one million hours of heterogeneous geophysical data, with specialized fine-tuned "heads" for medium-resolution weather, high-resolution weather, air quality, tropical cyclone tracks, and ocean waves. It reports outperforming several operational baselines across these domains at roughly 5,000x lower compute cost than conventional NWP, and was published in *Nature* in May 2025 ("A foundation model for the Earth system").

### Diffusion models for probabilistic/ensemble forecasting — GenCast (DeepMind, 2024/2025)
GenCast reframes ensemble forecasting as conditional generative modeling: a diffusion model produces an ensemble of stochastic 15-day forecasts (0.25°, 12-hour steps, 80+ variables), generating each member in roughly a minute on a TPU. It outperformed ECMWF's operational ensemble (ENS, 50 members) on 97.4% of 1,320 evaluation targets and showed particular strength on extreme events, tropical cyclone tracks, and wind-power-relevant variables. Published in *Nature* ("Probabilistic weather forecasting with machine learning," 2024/2025). NVIDIA's **CorrDiff** (part of Earth-2) and Fudan's **FuXi-ENS** are other notable diffusion/generative approaches to ensemble generation, aimed at sidestepping the linear cost-per-member of classical ensembles.

### Hybrid physics-ML general circulation models — NeuralGCM (Google Research + ECMWF, 2024)
NeuralGCM couples a **differentiable dynamical core** (solving the physical equations of atmospheric motion, thereby structurally preserving mass/momentum/energy conservation) with learned neural-network parameterizations for sub-grid processes (convection, cloud formation, radiation) that are typically hand-tuned in traditional GCMs. It produces skillful 1–15 day weather forecasts and stable multi-decade climate simulations, running ~1,200 simulated years per day on a single TPU — enabling large-ensemble climate experiments that are computationally prohibitive with conventional GCMs. A 2025 *Science Advances* follow-up extended NeuralGCM training to satellite-observed precipitation. This hybrid category — differentiable/structurally-constrained solver plus learned components — is widely seen as the architecture most likely to satisfy both skill and physical-consistency requirements simultaneously.

### Foundation models — ClimaX (Microsoft, 2023)
ClimaX extends a Transformer with custom variable-encoding and aggregation blocks so a single pretrained backbone (self-supervised on CMIP6 climate-model output) can be fine-tuned across heterogeneous tasks: weather forecasting at multiple lead times/resolutions, regional downscaling, and climate projection — including variables and grids unseen during pretraining. It's the earliest widely-cited attempt at a general-purpose "foundation model" for the weather/climate domain, predating Aurora.

### Downscaling and extreme-event modeling
Downscaling — translating coarse global model output to high-resolution regional fields — is increasingly handled by generative models: NVIDIA's CorrDiff (diffusion-based, 12.5x resolution increase, claimed ~1,000x speedup and ~3,000x energy reduction vs. numerical downscaling) is the leading industrial example. Academic work spans CNNs, GANs, and diffusion super-resolution for regional climate; a 2024 review in *AIES* ("Enhancing Regional Climate Downscaling through Advances in Machine Learning") surveys the space. For extreme-event detection/attribution, CNN-based classifiers are used to detect atmospheric rivers, tropical cyclones, and heatwaves in large simulation archives, and a distinct research line — "ML-based extreme event attribution" (Science Advances, 2024) — uses ML to estimate how much climate change altered the probability of a specific observed event, though the validity of ML methods under the required counterfactual distribution shift remains actively debated.

---

## Commercial Products & Startups

| Company | Focus | Notes |
|---|---|---|
| **NVIDIA Earth-2** | Platform for AI-accelerated weather/climate simulation & visualization | Built on FourCastNet + CorrDiff (generative downscaling, 2km-class detail) + PhysicsNeMo; early adopters include The Weather Company and Taiwan's Central Weather Administration (typhoon prediction) |
| **Google DeepMind (WeatherNext)** | Productized GraphCast/GenCast lineage | WeatherNext 2 (0.25°, fine-tuned on ECMWF HRES) and WeatherNext 3 (ingests live geostationary satellite data, hourly re-initialization, up to 0.05°/5km, 15-day probabilistic forecasts with 64 ensemble members); used operationally during the 2025 Atlantic hurricane season, correctly flagging Hurricane Melissa's rapid intensification |
| **Microsoft (Aurora, Planetary Computer)** | Foundation model + geospatial data platform | Aurora spans weather, air quality, ocean waves, cyclones; Planetary Computer provides curated Earth observation datasets/APIs for downstream ML |
| **Tomorrow.io** (formerly ClimaCell) | "Weather intelligence & resilience platform" | Building its own radar/weather satellite constellation ("SpaceX of weather"); ~$425M raised, ~$1B valuation (Feb 2026); serves 250+ enterprise/government customers |
| **Jupiter Intelligence** | Physical climate-risk analytics for insurers, corporates, governments | Competitor to Tomorrow.io in the climate-risk-analytics segment |
| **Atmo AI** | AI weather forecasting for national meteorological agencies and enterprises | *(Noted from general industry knowledge; specific current claims/partnerships not independently verified in this research pass — flag for follow-up.)* |
| **Spire Global** | Space-based weather data (GNSS radio occultation + other sensors) | Observes atmosphere from ocean surface to ~120km altitude; sells proprietary weather data/model outputs (e.g., ranked #1 in wind-forecast accuracy in a cited case study) |
| **WindBorne Systems** | Long-duration smart weather balloon constellation + proprietary ML model "WeatherMesh" | Stanford-spinout (founded 2019); claims to beat GraphCast by ~11% on 500hPa geopotential height in internal benchmarks; customers include the U.S. National Weather Service, U.S. Air Force, and U.S. Navy; raised a $37M Series B (Khosla Ventures, Galvanize) at a ~$250M valuation |
| **Salient Predictions** | Subseasonal-to-seasonal (S2S) forecasting, 2–52 weeks out | Founded 2019; technology traces to Woods Hole Oceanographic Institution/MIT research; customers include AB InBev, Syngenta, Zurich Insurance, BASF; reports outperforming NOAA GEFS/ECMWF/climatology benchmarks at S2S lead times |
| **One Concern** | AI-driven disaster resilience/risk platform | Stanford-founded (2015); product suite includes Seismic Concern (earthquake), Flood Concern (physics-informed flood risk mapping), and Fire Concern (wildfire spread + economic impact); used by cities including San Francisco and Los Angeles |
| **Understory** | Ground-based hyperlocal weather sensor network + risk analytics | *(Referenced in the disaster-tech/climate-risk startup space; not independently re-verified in this pass — flag for follow-up.)* |

---

## Numerical Methods & Baseline Solvers

Understanding ML weather/climate models requires understanding what they are compared against — and increasingly, hybridized with.

### Operational NWP systems
- **ECMWF IFS (Integrated Forecasting System)** — the global "gold standard" deterministic and ensemble NWP system; historically built on a spectral-transform dynamical core, run at ~9km resolution operationally, with **4D-Var** as its signature data-assimilation method. ERA5 is produced by running IFS in reanalysis mode. IFS is the baseline every major ML weather model (GraphCast, Pangu-Weather, FourCastNet, GenCast) is benchmarked against.
- **NOAA GFS (Global Forecast System)** — the U.S. operational global model, now built on the **FV3 (Finite-Volume Cubed-Sphere)** dynamical core developed at GFDL, replacing GFS's older spectral core as part of NOAA's Unified Forecast System (UFS) initiative.
- **WRF (Weather Research and Forecasting Model)** — the leading community mesoscale/regional NWP model (NCAR/NOAA/university consortium), with ARW and NMM dynamical-core variants; widely used both as an operational regional forecasting tool and as a synthetic data generator for training/validating ML downscaling models.

### Climate GCMs (General Circulation Models)
- **CESM (Community Earth System Model)**, developed at NCAR — a leading community coupled atmosphere-ocean-land-ice model, CMIP-contributing, and a common target for ML parameterization research (e.g., ClimSim's CRM-in-the-loop design mirrors CESM-style host models).
- **E3SM (Energy Exascale Earth System Model)**, DOE-funded — designed for exascale HPC, used for high-resolution climate projection and DOE climate-energy applications.
- **GFDL models (AM4, CM4, SPEAR, ESM4)** — NOAA GFDL's family of atmosphere/climate/seasonal-prediction models, all built on the FV3 dynamical core; GFDL is also Ai2's primary partner for ML-emulator research (ACE).

### Dynamical cores and numerical methods
Two broad families dominate: **spectral-transform methods** (historically ECMWF's approach, efficient for smooth global fields but with scaling limitations at very high resolution) and **finite-volume methods on structured grids** (e.g., GFDL's FV3), which scale better on modern massively-parallel hardware. This tension between spectral and finite-volume/finite-difference approaches directly motivated the grid choices discussed below, and is echoed in ML architectures: GraphCast's graph structure and NeuralGCM's differentiable dynamical core are direct descendants of finite-volume-style spatial discretization thinking, while FourCastNet's use of Fourier-domain operators is a spiritual successor to spectral methods.

### Data assimilation and its ML-hybrid successors
- **4D-Var** — ECMWF's flagship variational method, minimizing a cost function that balances a background forecast against a time window of observations, subject to the model's dynamical constraints.
- **Ensemble Kalman Filter (EnKF)** — an alternative (and in various hybrid forms, complementary) approach using ensemble-estimated, flow-dependent background-error covariances; NOAA's GDAS uses a hybrid 4D-EnVar approach combining both philosophies.
- **ML-hybrid data assimilation** is an active research frontier: approaches integrate learned components (e.g., learned observation operators, learned error-covariance models, or fully learned state-estimation networks) within the classical 4D-Var/EnKF mathematical framework so that the ML component remains constrained by known dynamics and observational likelihood. ECMWF researchers (e.g., Bonavita, Laloyaux) have published extensively on both the promise and current limitations of this direction. Operational ML models like AIFS still rely on conventional (often IFS-derived) analyses as initial conditions, though "end-to-end learned assimilation" is a stated long-term research goal across multiple groups. WeatherNext 3's move to ingest live satellite observations directly and reinitialize hourly is a step in this direction — using observations as model input rather than through an explicit separate assimilation step.

---

## Geometry & Grid Resources

Global atmospheric and oceanic models must discretize a sphere, and grid choice has first-order implications for both numerical stability and ML architecture design.

- **Regular latitude-longitude grid** — simplest and most common for ML training/inference (it is a natural fit for convolutional and transformer architectures designed for image-like tensors), but suffers a coordinate singularity at the poles, where grid cells shrink drastically, forcing tiny time steps or special filtering in numerical solvers. Most ML weather models (GraphCast at 0.25° lat-lon, FourCastNet, Pangu-Weather) consume ERA5 pre-regridded to a regular lat-lon grid for exactly this convenience, despite its numerical downsides for classical solvers.
- **Reduced Gaussian grid** — ECMWF's native operational grid (used internally by IFS and in ERA5's original production), which reduces the number of grid points near the poles to roughly equalize physical cell area, avoiding some lat-lon pathologies while remaining quasi-regular.
- **Cubed-sphere grid** — projects a cube onto the sphere, giving quasi-uniform cell areas and much better scalability on parallel hardware than lat-lon grids (cited speedups of ~20x for the equivalent grid density). This is the grid used by GFDL's FV3 dynamical core (and hence NOAA's GFS/UFS and GFDL's climate models), and by NeuralGCM's differentiable dynamical core. Its drawback is 8 cube-corner singularities (vs. 2 poles) where cell topology is irregular.
- **Icosahedral grid** — tiles the sphere with (mostly) hexagonal cells derived from subdividing an icosahedron, offering near-uniform cell area and no directional bias. Used operationally by Germany's DWD/MPI-M **ICON** model and NCAR's **MPAS**. Well suited to local adaptive mesh refinement but is inherently unstructured, complicating some ML and HPC tooling built for regular grids.
- **ERA5 grid standard** — publicly distributed at ~31km (0.25°) regular lat-lon resolution with 137 vertical levels from the surface to 80km, hourly from 1940–present; this has become the de facto common grid/resolution standard for ML weather model training, benchmarking (WeatherBench2), and cross-model comparison, even though it is a regridded/interpolated product of ECMWF's native reduced Gaussian grid.

---

## Datasets & Benchmarks

- **ERA5** (ECMWF/Copernicus Climate Change Service) — *the* standard ML training dataset for weather models: hourly global reanalysis from 1940 to near-present, 0.25° resolution, 137 vertical levels, produced via 4D-Var data assimilation of historical observations into the IFS model. Available via the Copernicus Climate Data Store and as a public AWS Open Data registry dataset. Nearly every major model in this document (GraphCast, FourCastNet, Pangu-Weather, ClimaX, Aurora's pretraining mix) trains on ERA5.
- **WeatherBench / WeatherBench2** (Google Research, Rasp et al.) — the standard open benchmark for global medium-range (1–14 day) ML weather forecasting. WeatherBench2 provides an open evaluation framework, standardized training/ground-truth/baseline data, and a continuously updated public leaderboard; its evaluation protocol deliberately mirrors operational verification practice at centers like ECMWF, making it a bridge between ML benchmarking culture and meteorological verification culture.
- **ClimSim** (NeurIPS 2023 Datasets & Benchmarks, award-winning) — the largest dataset built specifically for hybrid physics-ML climate emulation: 5.7 billion input-output pairs derived from cloud-resolving model (CRM) simulations nested inside a coarser host climate model, targeting ML parameterization of sub-grid processes like convection and cloud formation. A follow-up, **ClimSim-Online** (JMLR 2025), provides a live-coupled training/evaluation framework, and the dataset has since anchored a $50,000 Kaggle competition ("LEAP") for hybrid physics-ML climate emulation.
- **NOAA/NASA satellite datasets** — includes NOAA's GOES geostationary satellite archives, NASA's MODIS/VIIRS Earth observation products, and NASA's broader Earthdata catalog; increasingly used both as ML training inputs and, as in WeatherNext 3, as direct model inputs for observation-driven reinitialization.
- **ClimateSet** — a large-scale, ML-ready dataset built from CMIP6 climate model output, intended to lower the barrier to training ML climate emulators without requiring direct access to raw CMIP archives.
- **Synthetic/scenario generators** — CMIP6 itself functions as a multi-model "synthetic scenario" ensemble (different climate models run under shared socioeconomic/emissions pathways — the SSPs); ML groups increasingly also generate synthetic perturbed-parameter ensembles (e.g., NASA GISS's ModelE calibrated physics ensemble) using neural-network surrogates to cheaply explore parameter space that would be computationally prohibitive to sample directly with the full physical model.

---

## Standards & Validation

- **WMO (World Meteorological Organization)** sets the overarching international framework for forecast verification through its **WMO Integrated Processing and Prediction System (WIPPS)** and the **Joint Working Group on Forecast Verification Research (JWGFVR)**, which issues recommended verification standards used by national meteorological services worldwide.
- **AI-specific verification adaptation** — WMO has explicitly begun adapting these standards for ML forecasts, including hosting a dedicated **"Workshop: Verification of AI models in Operational Centers"**, aimed at establishing metrics and best practices that let both AI and traditional NWP outputs be trusted and compared fairly, and at working toward optimal hybrid AI/NWP operational workflows.
- **WeatherBench2's protocol** was explicitly designed to "stay close to" this operational/WMO verification tradition, making it something of a de facto bridge standard between the ML research community and operational meteorology.
- **IPCC relevance** — the IPCC's assessment reports (AR6 and beyond) are built on multi-model ensembles under the **CMIP** (Coupled Model Intercomparison Project) framework, combined with structured uncertainty characterization across model uncertainty, internal variability, and emissions-scenario uncertainty. ML is beginning to intersect this process in two ways: (1) as a **post-hoc uncertainty-reduction tool** — e.g., using observed-pattern "emergent constraints" learned via ML to narrow the spread of CMIP6-based warming projections (with some studies reporting 47–70% uncertainty reduction versus raw multi-model spread or IPCC AR6 baseline estimates), and (2) as a potential **future generation mechanism** for the ensembles themselves (ML-emulated GCM ensembles), which raises open methodological questions about how IPCC-style structured uncertainty assessment should treat ML-generated scenarios — an area with no settled community consensus yet.

---

## Open Challenges & Gaps

1. **Physical consistency and conservation-law violations.** Pure data-driven ML forecast models are typically trained against pixel-wise error metrics (e.g., RMSE) with no built-in guarantee of mass, energy, or momentum conservation, or even basic physical plausibility (e.g., negative precipitation has been documented). ECMWF's own research (Bonavita, *Geophysical Research Letters*, 2024, "On Some Limitations of Current Machine Learning Weather Prediction Models") catalogs these gaps directly. Mitigations include physics-informed loss penalties, architecturally-enforced conservation (as in NeuralGCM's differentiable dynamical core), and hybrid structural approaches generally — but no current large-scale ML weather model fully matches the physical consistency guarantees of a classical NWP model.
2. **Extreme-event tail prediction.** Extreme events are by construction rare in training data, biasing models toward common-case skill and leaving tail behavior poorly characterized; some evidence points to a deeper "spectral bias" in neural networks toward smooth/low-frequency solutions, independent of data imbalance alone. This is operationally critical since extreme events (hurricanes, heatwaves, extreme precipitation) are exactly where forecast value is highest.
3. **Long-range climate projection reliability.** ML models trained on historical climate statistics face a fundamental distribution-shift problem when projecting forward under warming scenarios that differ structurally from the training period — the "climate-invariant ML" research line (e.g., Beucler et al.) exists specifically to address poor extrapolation of learned parameterizations to untrained climate regimes. Pure ML climate emulators remain far less battle-tested than physics-based GCMs for century-scale projection, which is why most credible near-term efforts (NeuralGCM, Ai2's ACE) are explicitly hybrid rather than end-to-end learned.
4. **Uncertainty quantification (UQ) at scale.** Diffusion-based ensemble models (GenCast, FuXi-ENS) and conformal-prediction-based approaches are promising cheaper alternatives to classical ensemble methods, but rigorous, WMO-comparable UQ for ML forecasts — especially for rare/extreme scenarios and for multi-decadal climate projection — is still an active, unsettled research area.
5. **Computational cost of ensemble ML forecasting at scale.** While individual ML forecasts are dramatically cheaper than a single NWP run, generating large, high-resolution, high-frequency ensembles (as WeatherNext 3's 64-member hourly-updated system attempts) still represents substantial aggregate compute and energy cost — an underexplored area for the "physics AI is inherently cheap" narrative once taken to full operational, high-resolution, high-cadence scale.
6. **Validity of ML-based extreme event attribution.** Using ML for climate-change attribution of specific extreme events requires extrapolating to counterfactual ("what if there were no climate change") conditions that represent a significant distribution shift from training data — a methodological question flagged explicitly in recent literature (arXiv:2511.19039, "Validity in machine learning for extreme event attribution") as unresolved.
7. **Governance and evaluation standardization.** As more centers (ECMWF, NOAA, DWD, private companies) field competing ML models, the field still lacks fully settled, universally-adopted standards for fair comparison across resolution, ensemble size, training data overlap with test periods, and compute budget — WeatherBench2 and WMO's AI verification workshop are early steps, not a finished framework.

---

## Roadmap: Building the Reference Hub

A concrete, phased plan for an open-source repository to become the leading public reference for "Physics AI for Climate & Earth Systems" over 1–3 years.

### Phase 1 (Months 0–3): Foundation and curation
- Launch as a structured, well-tagged "awesome-list"-style catalog: papers (with links to arXiv/journal/DOI), open-source code repos, pretrained weights, and datasets, organized by the taxonomy in this document (architecture family, application domain, institution).
- Publish clear **model cards** for each major model (GraphCast, Pangu-Weather, FourCastNet, Aurora, GenCast, NeuralGCM, ClimaX, AIFS) covering training data, resolution, benchmark scores (WeatherBench2-aligned where possible), compute cost, and known limitations — directly addressing the "physical consistency" and "fair comparison" gaps above.
- Establish contribution guidelines and a lightweight governance model (maintainers, review process) from day one so the repo can scale via community PRs rather than sole-maintainer bottleneck.

### Phase 2 (Months 3–9): Reproducibility and benchmarking infrastructure
- Build (or integrate with) a reproducible benchmark harness on top of **WeatherBench2** and **ClimSim**, so new models/papers can be evaluated on a common, versioned leaderboard rather than self-reported numbers.
- Publish hands-on tutorials/notebooks: e.g., "train a minimal GNN weather model on ERA5," "run inference with open GraphCast/Aurora weights," "evaluate a downscaling model against WMO-style verification metrics."
- Start a "numerical methods primer" section explaining IFS/GFS/WRF, spectral vs. finite-volume vs. graph/operator-learning approaches, and data assimilation (4D-Var/EnKF) at a level accessible to ML practitioners without a meteorology background — and vice versa for domain scientists new to ML.

### Phase 3 (Months 9–18): Community and ecosystem integration
- Cross-list with existing community infrastructure rather than duplicating it: Climate Change AI's wiki/paper database, NVIDIA PhysicsNeMo examples, Hugging Face model hub tags for Earth-science models, and WMO's evolving AI-verification guidance.
- Host or co-host a recurring open community call / working group modeled on Climate Change AI's structure, focused specifically on the climate/earth-systems physics-AI subdomain (as distinct from CCAI's broader mitigation/adaptation/policy scope).
- Track and summarize operational deployments (ECMWF AIFS updates, WeatherNext version releases, NOAA UFS ML integration efforts) as a living "state of operational AI weather" page — this is currently scattered across press releases and technical blogs with no single canonical tracker.

### Phase 4 (Months 18–36): Establish as the reference
- Publish a citable, versioned "state of the field" survey/report derived from the repo content (targeting a Climate Change AI workshop or AIES submission) to establish academic credibility and a canonical citation.
- Sponsor or co-sponsor an open benchmark challenge (in the spirit of the ClimSim/LEAP Kaggle competition) to drive new contributions and community engagement.
- Pursue formal recognition/partnership signals — e.g., inclusion in WMO's AI-verification community discussions, cross-links from ECMWF/NOAA/Ai2 documentation, and adoption by university courses as a teaching reference.
- Establish long-term sustainability: a lightweight non-profit/fiscal-sponsorship structure or affiliation with an existing body (e.g., Climate Change AI, NumFOCUS-style scientific open-source umbrella) to ensure the repo survives beyond any single maintainer's bandwidth.

---

## Sources

**Landmark model papers**
- [Learning skillful medium-range global weather forecasting (GraphCast) — Science](https://www.science.org/doi/10.1126/science.adi2336)
- [GraphCast — Google DeepMind blog](https://deepmind.google/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/)
- [FourCastNet — arXiv:2202.11214](https://arxiv.org/abs/2202.11214)
- [FourCastNet — arXiv:2208.05419](https://arxiv.org/abs/2208.05419)
- [Pangu-Weather — Huawei Nature publication announcement](https://www.huawei.com/en/news/2023/7/pangu-ai-model-nature-publish)
- [Aurora: A foundation model for the Earth system — Nature](https://www.nature.com/articles/s41586-025-09005-y)
- [Aurora — Microsoft Research](https://www.microsoft.com/en-us/research/blog/introducing-aurora-the-first-large-scale-foundation-model-of-the-atmosphere/)
- [GenCast: Diffusion-based ensemble forecasting — arXiv:2312.15796](https://arxiv.org/abs/2312.15796)
- [Probabilistic weather forecasting with machine learning (GenCast) — Nature](https://www.nature.com/articles/s41586-024-08252-9)
- [Neural general circulation models for weather and climate (NeuralGCM) — Nature](https://www.nature.com/articles/s41586-024-07744-y)
- [NeuralGCM — Google Research blog](https://research.google/blog/fast-accurate-climate-modeling-with-neuralgcm/)
- [Neural general circulation models for modeling precipitation — Science Advances](https://www.science.org/doi/10.1126/sciadv.adv6891)
- [ClimaX: A foundation model for weather and climate — arXiv:2301.10343](https://arxiv.org/abs/2301.10343)
- [ClimaX — GitHub](https://github.com/microsoft/ClimaX)

**Operational systems**
- [ECMWF's AI forecasts become operational (AIFS)](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ai-forecasts-become-operational)
- [AIFS Single 1.1.0 update — GMD](https://gmd.copernicus.org/articles/19/4703/2026/)
- [WeatherNext — Google for Developers](https://developers.google.com/weathernext)
- [WeatherNext 3 — Google DeepMind](https://deepmind.google/science/weathernext/)
- [NVIDIA Announces Earth Climate Digital Twin — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-announces-earth-climate-digital-twin)
- [FV3: Finite-Volume Cubed-Sphere Dynamical Core — GFDL](https://www.gfdl.noaa.gov/fv3/)

**Labs and institutional research**
- [Climate modeling — Ai2](https://allenai.org/climate-modeling)
- [ACE: A fast, skillful learned global atmospheric model — Climate Change AI](https://www.climatechange.ai/papers/neurips2023/14)
- [Using Machine Learning to Generate a GISS ModelE Calibrated Physics Ensemble — JAMES](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024MS004713)
- [BAIR Climate Initiative Group — UC Berkeley](https://ai-climate.berkeley.edu/reading-group.html)

**Datasets and benchmarks**
- [ECMWF Reanalysis v5 (ERA5) dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- [ERA5 atmospheric reanalysis — Climate Data Guide (NCAR)](https://climatedataguide.ucar.edu/climate-data/era5-atmospheric-reanalysis)
- [WeatherBench 2 — arXiv:2308.15560](https://arxiv.org/abs/2308.15560)
- [WeatherBench 2 — Google Research blog](https://research.google/blog/weatherbench-2-a-benchmark-for-the-next-generation-of-data-driven-weather-models/)
- [ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation — NeurIPS 2023](https://neurips.cc/virtual/2023/poster/73569)
- [ClimSim-Online — JMLR](https://jmlr.org/papers/v26/24-1014.html)

**Standards and validation**
- [Workshop: Verification of AI models in Operational Centers — WMO](https://community.wmo.int/events/workshop-verification-of-ai-models-operational-centers)
- [Joint Working Group on Forecast Verification Research — WMO](https://community.wmo.int/site/knowledge-hub/governance/research-board/scientific-steering-committee-ssc/joint-working-group-forecast-verification-research)

**Open challenges**
- [On Some Limitations of Current Machine Learning Weather Prediction Models — GRL](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107377)
- [Physics-informed machine learning: case studies for weather and climate modelling — Phil. Trans. R. Soc. A](https://royalsocietypublishing.org/rsta/article/379/2194/20200093/41210/Physics-informed-machine-learning-case-studies-for)
- [Machine learning–based extreme event attribution — Science Advances](https://www.science.org/doi/10.1126/sciadv.adl3242)
- [Validity in machine learning for extreme event attribution — arXiv:2511.19039](https://arxiv.org/pdf/2511.19039)
- [Machine learning reduces uncertainty in Earth System Models — CMCC](https://www.cmcc.it/article/machine-learning-reduces-uncertainty-in-earth-system-models)

**Commercial/startup coverage**
- [Tomorrow.io raises $175M — SiliconANGLE](https://siliconangle.com/2026/02/03/tomorrow-io-raises-175m-deploy-ai-native-weather-satellite-constellation/)
- [WindBorne breaks world record for weather forecast accuracy (WeatherMesh)](https://windbornesystems.com/blog/windborne-breaks-world-record-for-most-accurate-global-weather-forecasts)
- [AI makes weather prediction better — TechCrunch on WindBorne](https://techcrunch.com/2026/08/05/ai-makes-weather-prediction-better-can-windborne-make-it-lucrative/)
- [Spire Global — weather & climate data from space](https://spire.com/weather-climate/)
- [Salient Predictions — subseasonal-to-seasonal forecasting](https://www.salientpredictions.com/blog/subseasonal-to-seasonal-weather-forecasting-with-sam-levang-from-salient-predictions)
- [One Concern expands AI disaster platform with predictive flood solution](https://www.prnewswire.com/news-releases/one-concern-expands-ai-disaster-platform-with-launch-of-predictive-flood-solution-300751064.html)

**Venues**
- [Climate Change AI — Events](https://www.climatechange.ai/events)
- [Tackling Climate Change with Machine Learning — NeurIPS 2026](https://www.climatechange.ai/events/neurips2026)
