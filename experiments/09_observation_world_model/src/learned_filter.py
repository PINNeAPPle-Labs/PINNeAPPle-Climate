"""The learned filter: prior, analysis cell and emulator, trained end to end on observations.

The architecture follows the source paper where a 40-variable state allows it:

* **prior** — a residual MLP over the state, i.e. the learned dynamics, standing in for the paper's GRU with
  message-passing linear maps. No locality here: Lorenz-96 is densely coupled, so global mixing is the right
  inductive bias and the deviation is one of geometry, not of mechanism;
* **analysis cell** — a *learned gain*, the paper's bipartite message passing collapsed to its effect: the
  increment is ``W(z_f, innovation, mask) @ innovation`` with ``W`` produced per state and per observation.
  Nothing about the observation-error covariance is an input; the update sees only the prior state, the
  innovation and which channels are present. The mask and the zero-innovation convention for absent channels are
  the paper's learned "no-observation" path;
* **emulator** — a decoder from the analysis state, evaluated on the channels that are never assimilated (the
  twin's stand-in for the paper's unseen precipitation / cloud-top-height / irradiance targets);
* **ensemble** — AR(1) noise injected into the state before the update, the cheap analogue of the paper's FiLM
  noise injection. Gives CRPS and spread-skill with no covariance supplied anywhere.

Training objective, same structure and weights as the paper (``L = 1.0 L_recon + 0.5 L_prior + 1.0 L_rollout``),
with RMS forms for reconstruction and prior so the three terms have comparable magnitude: ``L_recon`` masked
observation error, ``L_prior`` one-step prediction error against the stop-gradient analysis, ``L_rollout`` fair
CRPS over ``ROLLOUT`` observation-free steps. The emulator is a small extra head on the same loss.
"""
from __future__ import annotations

import numpy as np
import torch
from common import CYCLES, HORIZON, N_OBS, N_STATE, OBS_INDEX, crps_fair, diagnostic, observe
from torch import nn

torch.set_default_dtype(torch.float64)

ROLLOUT = 3                 # observation-free steps in the training rollout loss (0.15 time units)
MEMBERS_TRAIN = 2           # the paper's M = 2 during training
MEMBERS_EVAL = 4
AR1_RHO = 0.9


def crps_fair_torch(ensemble: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Fair CRPS, differentiable: the training-time twin of ``common.crps_fair``. (M, ...) against (...)."""
    m = ensemble.shape[0]
    term1 = (ensemble - truth).abs().mean(dim=0)
    if m < 2:
        return term1
    pair = (ensemble[:, None] - ensemble[None, :]).abs().sum(dim=(0, 1)) / (2 * m * (m - 1))
    return term1 - pair


class NeuralFilter(nn.Module):
    """Learned prior + learned analysis cell + emulator."""

    def __init__(self, n_state: int = N_STATE, n_obs: int = N_OBS, hidden: int = 128) -> None:
        super().__init__()
        self.n_state, self.n_obs = n_state, n_obs
        self.register_buffer("obs_index", torch.as_tensor(OBS_INDEX, dtype=torch.long))

        self.prior = nn.Sequential(nn.Linear(n_state, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, n_state))
        self.gain = nn.Sequential(nn.Linear(n_state + 3 * n_obs, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, n_state * n_obs))
        self.emulator = nn.Sequential(nn.Linear(n_state, hidden), nn.SiLU(),
                                      nn.Linear(hidden, n_state))
        self.log_sigma_noise = nn.Parameter(torch.tensor(-2.0))

        # Start from the prior-only solution with no spread: the update and the ensemble grow out of a
        # well-defined deterministic baseline instead of starting from garbage.
        for head in (self.prior, self.gain):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """One-step prior: residual, so the model starts as persistence and learns the increment."""
        return z + self.prior(z)

    def update(self, zf: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Analysis step. ``zf`` (..., n); ``y``/``mask`` broadcast over leading dims."""
        while y.dim() < zf.dim():
            y, mask = y.unsqueeze(-2), mask.unsqueeze(-2)
        z_obs = zf.index_select(-1, self.obs_index)
        innovation = (y - z_obs) * mask                     # absent channels carry no information
        features = torch.cat([zf, innovation, mask.expand_as(innovation), y.expand_as(innovation)], dim=-1)
        gain = self.gain(features).view(*zf.shape[:-1], self.n_state, self.n_obs)
        return zf + (gain @ innovation.unsqueeze(-1)).squeeze(-1)

    def emulated_state(self, z: torch.Tensor) -> torch.Tensor:
        return self.emulator(z)

    def noise(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        return torch.exp(self.log_sigma_noise) * torch.randn(shape, generator=generator)


def random_mask(batch: int, n_obs: int, generator: torch.Generator, p_drop: float = 0.30,
                p_all: float = 0.10, p_block: float = 0.20) -> torch.Tensor:
    """Observation availability, following the paper's masking recipe: per-channel dropout, whole-network
    dropout (forecast mode) and one contiguous block drawn per window."""
    mask = (torch.rand(batch, n_obs, generator=generator) > p_drop).to(torch.float64)
    mask[torch.rand(batch, generator=generator) < p_all] = 0.0
    half = n_obs // 2
    for b in torch.nonzero(torch.rand(batch, generator=generator) < p_block).flatten().tolist():
        start = int(torch.randint(0, n_obs - half + 1, (1,), generator=generator).item())
        mask[b, start:start + half] = 0.0
    return mask


class LearnedFilter:
    """Training driver and inference API for :class:`NeuralFilter`."""

    def __init__(self, seed: int = 0, hidden: int = 128, lr: float = 2e-3, w_recon: float = 1.0,
                 w_hidden: float = 2.0, w_prior: float = 0.5, w_rollout: float = 2.0,
                 w_emulator: float = 0.2) -> None:
        torch.manual_seed(seed)
        self.model = NeuralFilter(hidden=hidden)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.w_recon, self.w_hidden = w_recon, w_hidden
        self.w_prior, self.w_rollout = w_prior, w_rollout
        self.w_emulator = w_emulator
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.history: list[dict[str, float]] = []

    # ----------------------------------------------------------------------------------------------- data
    def sample_batch(self, trajectories: np.ndarray, batch: int, rng: np.random.Generator):
        """Random windows of (obs, mask, truth) from long truth trajectories."""
        n_traj, traj_len = trajectories.shape[:2]
        need = CYCLES + ROLLOUT + 1
        starts = rng.integers(0, traj_len - need, size=batch)
        which = rng.integers(0, n_traj, size=batch)
        truth = np.stack([trajectories[w, s:s + need] for w, s in zip(which, starts)])
        obs = observe(truth, rng)
        mask = random_mask(batch, N_OBS, self.generator).numpy()
        return (torch.as_tensor(obs[:, :CYCLES]),
                torch.as_tensor(mask[:, None, :].repeat(CYCLES, axis=1)),
                torch.as_tensor(truth))

    # ------------------------------------------------------------------------------------------- training
    def sequence_losses(self, obs: torch.Tensor, mask: torch.Tensor, truth: torch.Tensor,
                        anchors: torch.Tensor, generator: torch.Generator) -> dict[str, torch.Tensor]:
        """Roll the filter through a window and return the losses.

        ``obs``/``mask`` are (B, L, m), ``truth`` is (B, L + ROLLOUT, n) and ``anchors`` indexes the steps from
        which the observation-free rollout is scored.
        """
        batch, length, _ = obs.shape
        z = torch.zeros(batch, MEMBERS_TRAIN, self.model.n_state)
        eps = torch.zeros_like(z)
        decay = float(np.sqrt(1.0 - AR1_RHO ** 2))
        recon, recon_hidden, prior_terms, states = [], [], [], []
        for t in range(length):
            eps = AR1_RHO * eps + decay * self.model.noise(z.shape, generator)
            za = self.model.update(self.model.dynamics(z) + eps, obs[:, t], mask[:, t])
            z_obs = za.index_select(-1, self.model.obs_index)
            present = mask[:, t].unsqueeze(1).expand_as(z_obs)
            err = (z_obs - obs[:, t].unsqueeze(1)) ** 2
            recon.append(torch.sqrt((err * present).sum() / present.sum().clamp(min=1.0)))
            # The paper's block-mask term: the same cell, formed without these channels, must reproduce the
            # observation that was withheld. Without it the reconstruction loss is minimised by copying the
            # observation and the analysis never learns to trust its own prior.
            withheld = 1.0 - present
            recon_hidden.append(torch.sqrt((err * withheld).sum() / withheld.sum().clamp(min=1.0)))
            states.append(za)
            z = za

        # L_prior: the dynamics must predict the next analysis. Targets are stop-gradient, so this trains the
        # prior and not the analysis cell (the paper's z-hat against stop-gradient z).
        for t in range(length - 1):
            pred = self.model.dynamics(states[t].detach())
            prior_terms.append(torch.sqrt(((pred - states[t + 1].detach()) ** 2).mean()))

        # L_rollout: fair CRPS over observation-free steps, backpropagated through the rollout.
        rollout, emulator = [], []
        for a in anchors.tolist():
            z_roll = states[a]
            for step in range(ROLLOUT):
                if step > 0:
                    z_roll = z_roll + self.model.noise(z_roll.shape, generator)
                z_roll = self.model.dynamics(z_roll)
                rollout.append(crps_fair_torch(z_roll.permute(1, 0, 2), truth[:, a + 1 + step]).mean())
            emulator.append(torch.sqrt(((self.model.emulated_state(states[a].mean(1))
                                         - truth[:, a]) ** 2).mean()))

        return {"recon": torch.stack(recon).mean(), "hidden": torch.stack(recon_hidden).mean(),
                "prior": torch.stack(prior_terms).mean(),
                "rollout": torch.stack(rollout).mean(), "emulator": torch.stack(emulator).mean(),
                "sigma_noise": torch.exp(self.model.log_sigma_noise).detach()}

    def fit(self, trajectories: np.ndarray, val_trajectories: np.ndarray, steps: int = 2500, batch: int = 32,
            log_every: int = 250) -> None:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=steps)
        rng = np.random.default_rng(0)
        anchors = torch.tensor([8, 16, 22], dtype=torch.long)
        for step in range(steps):
            obs, mask, truth = self.sample_batch(trajectories, batch, rng)
            losses = self.sequence_losses(obs, mask, truth, anchors, self.generator)
            loss = (self.w_recon * losses["recon"] + self.w_hidden * losses["hidden"]
                    + self.w_prior * losses["prior"]
                    + self.w_rollout * losses["rollout"] + self.w_emulator * losses["emulator"])
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            sched.step()
            if step % log_every == 0 or step == steps - 1:
                record = {k: float(v) for k, v in losses.items()}
                record["step"] = step
                record["loss"] = float(loss)
                record.update({f"val_{k}": v for k, v in self.validate(val_trajectories).items()})
                self.history.append(record)
                print(f"step {step:5d}  loss {record['loss']:.3f}  recon {record['recon']:.3f}  "
                      f"hidden {record['hidden']:.3f}  prior {record['prior']:.3f}  "
                      f"rollout {record['rollout']:.3f}  "
                      f"sigma_n {record['sigma_noise']:.3f}  val analysis {record['val_rmse_analysis']:.3f}  "
                      f"val forecast {record['val_rmse_forecast']:.3f}", flush=True)

    @torch.no_grad()
    def validate(self, trajectories: np.ndarray, n_cases: int = 20) -> dict[str, float]:
        analysis, forecast = [], []
        for case in range(min(n_cases, trajectories.shape[0])):
            out = self.run_case(trajectories[case], rng=np.random.default_rng(case), torch_seed=case)
            analysis.append(out["rmse_analysis"])
            forecast.append(out["rmse_forecast"])
        return {"rmse_analysis": float(np.mean(analysis)), "rmse_forecast": float(np.mean(forecast))}

    # ------------------------------------------------------------------------------------------ inference
    @torch.no_grad()
    def run_case(self, truth: np.ndarray, rng: np.random.Generator, torch_seed: int = 0,
                 members: int = MEMBERS_EVAL, noise: bool = True) -> dict[str, np.ndarray]:
        """One case: CYCLES assimilated steps from a zero state, then HORIZON observation-free steps.

        The state starts at zero and the instrument errors are whatever the data contains — the paper's spin-up,
        and the reason no background covariance is needed.
        """
        generator = torch.Generator().manual_seed(int(torch_seed))
        model = self.model
        obs = observe(truth[:CYCLES], rng)
        decay = float(np.sqrt(1.0 - AR1_RHO ** 2))
        z = torch.zeros(members, model.n_state)
        eps = torch.zeros_like(z)
        prior, post = [], []
        for t in range(CYCLES):
            if noise:
                eps = AR1_RHO * eps + decay * model.noise(z.shape, generator)
            zf = model.dynamics(z) + eps
            za = model.update(zf, torch.as_tensor(obs[t]).expand(members, -1), torch.ones(members, N_OBS))
            prior.append(zf.numpy())
            post.append(za.numpy())
            z = za
        prior, post = np.stack(prior), np.stack(post)          # (CYCLES, M, n)

        fc, z = [], torch.as_tensor(post[-1])
        for _ in range(HORIZON):
            z = model.dynamics(z) + (model.noise(z.shape, generator) if noise else 0.0)
            fc.append(z.numpy())
        fc = np.stack(fc)                                       # (HORIZON, M, n)

        mean_post, mean_fc = post.mean(1), fc.mean(1)
        return {
            "prior": prior, "posterior": post, "forecast": fc, "observations": obs,
            "rmse_analysis": float(np.sqrt(np.mean((mean_post - truth[:CYCLES]) ** 2))),
            "rmse_forecast": float(np.sqrt(np.mean((mean_fc - truth[CYCLES:]) ** 2))),
            "spread": post.std(1),        # per-cycle ensemble spread over members
            "crps": crps_fair(post.swapaxes(0, 1), truth[:CYCLES]),
            "emulated_state": model.emulated_state(torch.as_tensor(mean_post)).numpy(),
            "emulated_diagnostic": diagnostic(model.emulated_state(torch.as_tensor(mean_post)).numpy()),
        }
