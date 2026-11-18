## 09 — Observation-to-observation world model, Stage 0: the Lorenz-96 twin

Stage 0 of [issue #19](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Climate/issues/19). Before any satellite byte
is downloaded, this asks the question the campaign turns on in the smallest setting that can answer it:

> Can an analysis step **learned from data**, given no observation-error statistics, land close to the optimal
> filter — including recovering *which* instrument to trust — where a classical filter is only as good as the
> covariance it is handed?

The source paper is WeatherDreamer ([arXiv:2611.18393](https://arxiv.org/abs/2611.18393)): a world model that
learns the data-assimilation cycle end to end from 18 observing systems without a reanalysis product. It
reproduces the three pieces that make that possible — learned prior, learned update on arrival of observations,
learned observation emulator — and the diagnostics the paper uses to argue its learned update recovers the
instrument errors without being told them.

### What is reproduced, and what is not

| Paper | Here | Why |
|---|---|---|
| HEALPix Nside=128, global | Lorenz-96, 40 variables, cyclic | the twin is the standard data-assimilation test bed; no geometry to build |
| 18 observing systems, 4 years | 2 instrument types × 10 channels, 8 training trajectories | the instrument *contrast* is what is under test, not the count |
| latent state `h ∈ R^512`, `z ∈ R^64` per node | no compression (n = 40) | the mechanism under test is the learned update, not the compression |
| GRU with message-passing linear maps | residual MLP | Lorenz-96 is densely coupled, so global mixing is the right inductive bias |
| 18-way message passing → learned gain | one MLP producing a full `W(z_f, innovation, mask)` gain | the paper's cell collapsed to its effect |
| FiLM-injected AR(1) noise | AR(1) noise added to the state before the update | same generative ensemble, cheaper plumbing |
| LIIF decoders, arbitrary query coordinates | emulator head evaluated on the never-assimilated channels | the "unseen variable" test without the geometry |
| view geometry as excess air mass | absent | needs a satellite; this is Stage 3 of the campaign |
| analysis + 12 h forecast, 8 members, 1.5 min | 24 assimilated steps + 6 observation-free steps, 4 members | same loop, scaled |

### Protocol

* 40-variable Lorenz-96, RK4, `dt = 0.05`, `F = 8`.
* Observations every step of 20 of the 40 variables, split into two instrument types whose errors differ by 6×:
  10 **reliable** channels with `σ = 0.25` and 10 **noisy** ones with `σ = 1.5`. The observable part of the
  state is known exactly; what has to be inferred is how much to believe it.
* Every method starts from a **zero state** and assimilates 24 steps. The first 12 are never scored: they are the
  spin-up that lets any of them forget the zero state (the paper zero-initialises for the same reason).
* After the last analysis, 6 steps with **no observations at all**.
* 60 evaluation cases, each an **independent trajectory**, so the 95 % bootstrap is over independent cases.
* The classical filters get the **true dynamics** and only have to get the statistics right; the learned filter
  has to learn both. The comparison bounds the learned filter from below.
* `Q` and the EnKF inflation are tuned on 10 separate cases, with the true `σ`, and then frozen. Nothing is tuned
  on the evaluation cases.

### Results

Full run: 4 000 training steps, 60 cases, 148 s on one laptop CPU. `results/twin_full.json`, `results/twin_full.log`.

**Analysis and forecast error** (RMSE of the 40-variable state, mean and 95 % bootstrap CI):

| method | analysis | never-observed channels | forecast, lead 4 |
|---|---|---|---|
| learned filter, paper loss weights | 0.883 [0.870, 0.894] | 0.635 [0.624, 0.647] | 0.259 [0.248, 0.271] |
| learned filter, held-out objective only | **0.349 [0.341, 0.357]** | **0.402 [0.391, 0.412]** | **0.176** |
| EKF, true per-channel R | 0.835 [0.696, 0.979] | 1.027 [0.841, 1.220] | 0.528 [0.482, 0.577] |
| EKF, one R for all channels | 0.708 [0.642, 0.778] | 0.819 [0.723, 0.920] | 0.611 [0.572, 0.652] |
| EKF, R swapped between the two types | 2.074 [1.902, 2.255] | 2.598 [2.358, 2.856] | 1.672 [1.530, 1.817] |
| EnKF (80 members), true per-channel R | 1.267 [1.140, 1.407] | 1.639 [1.465, 1.832] | 0.881 [0.803, 0.976] |
| EnKF, R swapped | 3.343 [3.182, 3.513] | 4.422 [4.185, 4.668] | 3.321 [3.144, 3.507] |
| persistence (optimistic, not a filter) | — | — | 3.062 |

**The two assimilation diagnostics**, per instrument type. The optimal gains implied by the true covariances are
0.79 (reliable) and 0.27 (noisy); the true observation-error variances are 0.0625 and 2.25.

| method | gain, reliable | gain, noisy | Desroziers `σ̂²_o`, reliable | Desroziers `σ̂²_o`, noisy | CRPS | spread / skill |
|---|---|---|---|---|---|---|
| learned, paper weights | 0.977 | 0.992 | 0.003 | 0.022 | 0.518 | 0.20 |
| learned, held-out objective | 0.304 | 0.042 | 0.082 | **2.289 [2.226, 2.352]** | **0.173** | 0.64 |
| EKF, true R | 0.780 | 0.226 | 0.038 | — | — | — |
| EKF, one R | 0.410 | 0.357 | 0.224 | — | — | — |
| EKF, R swapped | 0.412 | 0.786 | 2.397 | — | — | — |
| EnKF, true R | 0.844 | 0.358 | 0.036 | — | — | — |

**Emulator** (learned filter, held-out objective): the never-assimilated half of the state is decoded at RMSE
0.325, and a thresholded nonlinear diagnostic that is never assimilated at all (the twin's stand-in for a
precipitation product) at RMSE 0.191 against 1.882 for climatology — a factor of 10, from a latent built only
from the other 20 channels.

Figures: [figures/fig1_forecast_rmse.png](figures/fig1_forecast_rmse.png),
[figures/fig2_assimilation_diagnostics.png](figures/fig2_assimilation_diagnostics.png),
[figures/fig3_analysis_decomposition.png](figures/fig3_analysis_decomposition.png).

### What the numbers say

1. **Which instrument to trust is the whole problem, and it is worth a factor of 2.5.** Give the EKF the right
   numbers swapped between the two types — a plausible operational bookkeeping error, same total error budget —
   and its analysis goes from 0.835 to 2.074 and its 4-step forecast from 0.528 to 1.672. Nothing else in this
   experiment costs that much.

2. **The learned filter with the paper's own loss weights does *not* reproduce the paper's calibration claim.**
   Its effective gain is 0.98 on the reliable channels *and* 0.99 on the noisy ones: it copies its observations.
   The consequence is visible exactly where it should be — its analysis error on the noisy channels is 1.499,
   i.e. `σ = 1.5`, the observation error itself, against 0.773 for the EKF that was told the truth — and in the
   Desroziers estimate, which comes out at 0.003 where the truth is 0.0625. The paper reports the opposite
   (learned error magnitudes converging on the known instrument errors). At this scale the reconstruction term
   `L_recon` on *visible* channels is minimised by copying them, and that optimum wins.

3. **The ablation identifies the ingredient.** Drop the reconstruction term on visible channels, keep the term
   that reproduces *withheld* observations and the rollout CRPS (the paper's block-mask weighting pushed to its
   limit), and the same architecture becomes a filter: gain 0.30 on the reliable channels against 0.04 on the
   noisy ones — it separates the two types by 7× where the optimum is 2.9× — and the Desroziers estimate lands at
   **2.289 [2.226, 2.352] for a true variance of 2.25**, a 2 % miss, without the number ever being supplied. The
   analysis error drops to 0.349 and beats every classical variant, including the one given the true covariances.

4. **The learned prior is worth more than the filter is.** Both learned variants win the forecast and the
   never-observed channels; the held-out-objective one wins them outright (0.176 against 0.528 at lead 4) while
   its gain is *smaller* than optimal, i.e. it over-smooths. A gain that over-smooths cannot beat a correct-gain
   filter by better filtering: it wins because its dynamics, trained as a conditional-mean predictor over eight
   trajectories, is more accurate one step ahead than a chaotic model propagated with a tuned, isotropic `Q`.
   Any claim that a learned filter beats a classical one must separate those two effects, which is what Stage 2
   is for.

5. **Its forecast error falls with lead time, and that is a defect, not a win.** From 0.401 at lead 1 to 0.259 at
   lead 4: the learned prior contracts towards the attractor faster than Lorenz-96 actually diverges, so it
   denoises the analysis rather than propagating instability. The EKF and the EnKF all grow with lead, as they
   must. Reported as it is; Stage 1 has to check that the learned dynamics reproduces error growth before any of
   its forecasts are believed.

Also worth recording, because it is the classical side of the same coin: the EKF given **one** `R` for all
channels beats the EKF given the **true** per-channel `R` on the state-wide analysis (0.708 against 0.835,
non-overlapping CIs) while being clearly worse on the reliable channels (0.345 against 0.206) and better on the
never-observed ones (0.819 against 1.027). A wrong `R` is compensating for a wrong `B`: the background
covariance is a tuned isotropic `Q`, and it is the term a classical system cannot learn. That is the gap the
learned filter exploits, and it is the gap the paper claims to close — not the observation errors alone.

### Deviations and caveats

* Lorenz-96 is not the atmosphere: no geometry, no sparsity pattern changing in time, no biases, one scale. The
  paper's observation geometry — patches, blocks, a limb — is what makes filtering *necessary*; here it is
  possible to be a poor filter and still forecast well, which is exactly what result 2 shows.
* The learned filter sees 4 ensemble members at evaluation and 2 during training against the paper's 8; its
  spread is under-dispersive (spread/skill 0.20–0.64 against 1.0 for a calibrated ensemble), which the paper
  reports it does not suffer from.
* Tuning effort is asymmetric in the classical filters' favour on the dynamics (they are handed the true
  equations) and symmetric on the statistics (all of them, including the learned one, get one tuned scalar knob
  or none).
* One seed, one architecture, 4 000 steps. Nothing here is a claim about the paper's numbers at its own scale.

### What this changes for Stage 1

1. **The loss, not the architecture, decides whether the cell filters.** The objective must reward predicting what
   was *withheld*; reconstruction of what was given teaches copying. Stage 1 should keep both terms separated and
   report the effective gain from the start, so the failure mode is visible in the first experiment rather than
   the third.
2. **Observation geometry is the missing forcing.** With 20 of 40 variables observed every hour at `σ ≤ 1.5`,
   being a poor filter is survivable. The ABI/ISD observation network is sparse, irregular and has gaps, which is
   the condition under which the paper's mechanism is not optional.
3. **Report the decomposition, always.** The state-wide RMSE hid a 6× error on the noisy channels behind a good
   score. Per-channel-type and never-observed numbers go in the protocol, not the appendix.
4. **Check error growth before believing a forecast score.** A contracting prior looks skilful at short leads.

### Files

| File | What |
|---|---|
| `src/common.py` | twin configuration, Lorenz-96, observation network, metrics and the two DA diagnostics |
| `src/learned_filter.py` | prior, learned gain, emulator, AR(1) ensemble, the loss and the training loop |
| `src/run_twin.py` | classical baselines through `pinneapple_analysis`, tuning, evaluation, `results/*.json` |
| `src/make_figures.py` | the three figures, from the JSON only — no number is typed by hand |
| `results/twin_full.json` | paper loss weights (default) |
| `results/twin_norecon.json` | the `--no-recon` ablation |

### How to run

```bash
cd experiments/09_observation_world_model/src
python run_twin.py --steps 4000            # full run, ~150 s on a laptop CPU
python run_twin.py --steps 4000 --no-recon # the ablation above
python make_figures.py                     # figures/, reads results/*.json
python run_twin.py --smoke                 # a few seconds, for a change you want to check
```

Needs `torch`, `numpy`, `matplotlib` and a PINNeAPPle checkout on `PINNEAPPLE_SRC`
(default `~/Documents/GitHub/pinneapple-labs/PINNeAPPle`); the classical filters are
`pinneapple_analysis.state_estimation.kalman.ExtendedKalmanFilter` and `EnsembleKalmanFilter`, used as they are.
