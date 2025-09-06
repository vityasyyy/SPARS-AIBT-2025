import random
import numpy as np
from typing import Tuple
import torch as T
from torch.func import vmap
import logging
from HPCv3.Simulator.MachineMonitor import Monitor
from torchviz import make_dot
logger = logging.getLogger("runner")


class Reward:
    """
    reward_per_node = α * (-energy_waste_per_node) + β * (-waiting_time_metric)
    where waiting_time_metric is a scalar (mean wait of finished jobs) broadcast to all nodes.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 # mean wait per finished job (True) or sum (False)
                 use_mean_wait: bool = True,
                 device: str = "cuda",
                 require_grad: bool = True):   # keep True to match your current pipeline
        self.alpha = alpha
        self.beta = beta
        self.use_mean_wait = use_mean_wait
        self.device = T.device(device)
        self.require_grad = require_grad

    @staticmethod
    def _get_first_key(d: dict, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _compute_waiting_time_scalar(self, monitor) -> float:
        """
        Compute a scalar waiting-time metric from monitor logs:
        wait_j = max(0, start_time_j - submit_time_j) for each job with both times available.
        Returns 0.0 if no valid pairs found.
        """
        # Build lookup: submit time per job_id (or unique job dict key)
        submit_by_id = {}
        for job in monitor.jobs_submission_log:
            # Try several likely key names to be robust
            jid = self._get_first_key(job, ["job_id", "id", "jid"])
            submit = self._get_first_key(
                job, ["submit_time", "arrival_time", "arrival", "queued_time", "time"])
            if jid is not None and submit is not None:
                submit_by_id[jid] = float(submit)

        waits = []
        for job in monitor.jobs_execution_log:
            jid = self._get_first_key(job, ["job_id", "id", "jid"])
            # start_time should be part of the job dict when it started; try common aliases
            start = self._get_first_key(
                job, ["start_time", "dispatch_time", "begin_time"])
            submit = submit_by_id.get(jid, None)
            if submit is not None and start is not None:
                wait = float(start) - float(submit)
                if wait > 0:
                    waits.append(wait)

        if not waits:
            return 0.0
        if self.use_mean_wait:
            return float(sum(waits) / len(waits))
        else:
            return float(sum(waits))

    def calculate_reward(self, monitor):
        # 1) Energy-waste (per-node); keep your original sign convention (negative is bad)
        wasted_energy = [-entry['energy_waste'] for entry in monitor.energy]
        wasted_energy = T.tensor(
            wasted_energy, dtype=T.float32, device=self.device, requires_grad=self.require_grad
        )  # shape: [num_nodes]

        # 2) Waiting-time (scalar) → broadcast to all nodes, negative sign to penalize large waits
        wait_scalar = self._compute_waiting_time_scalar(
            monitor)   # python float
        waiting_vec = T.full(
            (len(monitor.energy),), -wait_scalar,
            dtype=T.float32, device=self.device, requires_grad=self.require_grad
        )

        # 3) Weighted combination (per-node vector)
        reward = self.alpha * wasted_energy + self.beta * waiting_vec
        return reward


def discounted_returns(rewards, gamma, time_dim=-1):
    """
    rewards: tensor with time along `time_dim`
    gamma: float or 0-D tensor (can require_grad)
    returns: discounted-to-go along `time_dim` (same shape as rewards)
    """
    assert T.is_tensor(rewards), "rewards must be a tensor"
    dtype, device = rewards.dtype, rewards.device

    if time_dim != -1:
        rewards = rewards.transpose(time_dim, -1)  # shape: [..., seq_len]

    seq_len = rewards.size(-1)

    # gamma as tensor
    if T.is_tensor(gamma):
        gamma = gamma.to(device=device, dtype=dtype)
    else:
        gamma = T.tensor(gamma, device=device, dtype=dtype)

    weighted = rewards.view(-1, 1) * gamma
    flipped = T.flip(weighted, dims=[-1])
    csum = T.cumsum(flipped, dim=-1)
    disc = T.flip(csum, dims=[-1]) / gamma

    # Put time axis back
    if time_dim != -1:
        disc = disc.transpose(-1, time_dim)

    return disc


def learn(agent, agent_opt, critic, critic_opt, done, saved_experiences, next_observation,
          gamma: float = 0.99, entropy_coef: float = 0.0, eps: float = 1e-12):
    """
    Batched A2C-style update.
    Expects saved_experiences as lists of Tensors with matching shapes per step:
      memory_features[t] : [N, D]
      memory_masks[t]    : [N] or [N,1]  (1=valid, 0=invalid)  (used for logprob reduction)
      memory_actions[t]  : one of:
         • [N]   (class index per node, e.g., 0/1)
         • [N,1] (class index per node)
         • [N,2] (one-hot per node for 2 classes)
      memory_rewards[t]  : scalar or tensor reducible to scalar (mean)
    next_observation = (next_features, next_masks) with next_features: [N,D]
    Agent forward: probs[B,N,2], entropy[B]; Critic forward: values[B]
    """
    memory_actions, memory_features, memory_masks, memory_rewards = saved_experiences
    next_features, _next_masks = next_observation

    device = next(agent.parameters()).device
    Tlen = len(memory_rewards)

    # ---------- stack along time ----------
    feats = T.cat(memory_features, dim=0)       # [T,N,D]
    masks = T.cat(memory_masks, dim=0)         # [T,N,2]
    if masks.dim() == 3 and masks.size(-1) == 1:
        masks = masks.squeeze(-1)                                  # [T,N]

    # rewards -> [T] (reduce any extra dims to a scalar per step)
    rews = T.stack([r.to(device).float().view(-1).mean() if isinstance(r, T.Tensor)
                    else T.tensor(float(r), device=device)
                    for r in memory_rewards])                      # [T]

    # actions stacked (kept flexible)
    acts = T.cat(memory_actions, dim=0)                     # [T, ...]
    if acts.dim() == 3 and acts.size(-1) == 1:
        # [T,N] if was [T,N,1]
        acts = acts.squeeze(-1)

    # ---------- bootstrap ----------
    with T.no_grad():
        nf = next_features.to(device).float()
        if nf.dim() == 2:  # [N,D]
            nf_b = nf.unsqueeze(0)                                 # [1,N,D]
        else:
            nf_b = nf
        bootstrap = T.zeros((), device=device) if done else critic(
            nf_b).view(-1).mean()

    # ---------- discounted returns [T] ----------
    returns = T.empty_like(rews)
    R = bootstrap
    for t in range(Tlen - 1, -1, -1):
        R = rews[t] + gamma * R
        returns[t] = R

    # ---------- single critic forward ----------
    values = critic(feats).float().view(Tlen)                      # [T]

    # ---------- single agent forward ----------
    # probs: [T,N,2], ent: [T]
    probs, ent = agent(feats, masks)
    # numerical safety
    probs = probs.clamp_min(eps)
    log_probs = probs.log()                                        # [T,N,2]

    # ---- compute selected log-prob per step (handle several action encodings) ----
    # per-node selected probability:
    if acts.dim() == 3 and acts.size(-1) == 2:
        # one-hot per node: [T,N,2]
        sel_p_node = (probs * acts).sum(dim=-1)                    # [T,N]
    elif acts.dim() == 2:
        # class index per node: [T,N]
        sel_p_node = probs.gather(
            dim=-1, index=acts.long().unsqueeze(-1)).squeeze(-1)  # [T,N]
    elif acts.dim() == 1:
        # single action per step: [T]
        # reduce probs across nodes first (masked mean), then pick class
        msum = masks.sum(dim=1).clamp_min(1.0)                     # [T]
        probs_step = (probs * masks.unsqueeze(-1)).sum(dim=1) / \
            msum.unsqueeze(-1)      # [T,2]
        sel_p_node = probs_step.gather(
            dim=-1, index=acts.long().unsqueeze(-1)).squeeze(-1)  # [T]
    else:
        raise ValueError(f"Unsupported actions shape {tuple(acts.shape)}")

    # reduce per-node to per-step (masked mean) if still node-wise
    if sel_p_node.dim() == 2:
        # masked mean over nodes
        msum = masks.sum(dim=(1, 2)).clamp_min(1.0)                     # [T]
        sel_p = (sel_p_node).sum(dim=1) / msum             # [T]
    else:
        sel_p = sel_p_node                                         # [T]

    logp = (sel_p + eps).log()                                     # [T]

    # ---------- losses ----------
    advantage = (returns - values).detach()                        # [T]
    actor_loss = -(logp * advantage).mean()
    if entropy_coef:
        # ent is already per-step [T] from Agent; no masking applied here
        actor_loss = actor_loss - entropy_coef * ent.float().mean()

    critic_loss = T.nn.functional.mse_loss(values, returns)

    # ---------- optimize ----------
    agent_opt.zero_grad(set_to_none=True)
    dot = make_dot(actor_loss, params=dict(agent.named_parameters()))
    dot.render("agent_loss_graph", format="pdf")

    actor_loss.backward()
    # T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=...)
    agent_opt.step()

    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    # T.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=...)
    critic_opt.step()

    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "returns_mean": float(returns.mean().detach().cpu()),
        "adv_mean": float(advantage.mean().detach().cpu()),
        "entropy": float(ent.float().mean().detach().cpu()),
    }


def get_feasible_mask(states):
    # fm[:, 0] = dibiarkan, dummy
    # fm[:, 1] = boleh matikan/tidak
    # fm[:, 2] = boleh hidupkan/tidak
    feasible_mask = np.ones((len(states), 3), dtype=np.float32)
    is_switching_off = np.asarray(
        [host['state'] == 'switching_off' for host in states])
    is_switching_on = np.asarray(
        [host['state'] == 'switching_on' for host in states])
    is_switching = np.logical_or(is_switching_off, is_switching_on)
    is_idle = np.asarray(
        [host['state'] == 'active' and host['job_id'] is None for host in states])
    is_sleeping = np.asarray(
        [host['state'] == 'sleeping' for host in states])
    is_allocated = np.asarray(
        [host['state'] == 'active' and host['job_id'] is None for host in states])

    # can it be switched off
    is_really_idle = np.logical_and(is_idle, np.logical_not(is_allocated))
    feasible_mask[:, 1] = np.logical_and(
        np.logical_not(is_switching), is_really_idle)

    # can it be switched on
    feasible_mask[:, 2] = np.logical_and(
        np.logical_not(is_switching), is_sleeping)
    # return cuma 2 action, update 15-09-2022
    return feasible_mask[:, 1:]


def feature_extraction(simulator) -> Tuple[np.ndarray, np.ndarray]:
    # === GLOBAL FEATURES ===
    num_sim_features = 5
    simulator_features = np.zeros((num_sim_features,), dtype=np.float32)

    job_num = len(simulator.jobs_manager.waiting_queue)
    simulator_features[0] = job_num

    arrival_rate = len(simulator.Monitor.jobs_submission_log) / (
        simulator.current_time - simulator.start_time
        + 1e-8)

    simulator_features[1] = arrival_rate

    mean_runtime_jobs_in_queue = sum(
        job["walltime"] for job in simulator.jobs_manager.waiting_queue) / (len(simulator.jobs_manager.waiting_queue) + 1e-8)

    simulator_features[2] = mean_runtime_jobs_in_queue

    total_energy_waste = sum(e["energy_waste"]
                             for e in simulator.Monitor.energy)

    simulator_features[3] = total_energy_waste

    mean_requested_walltime_jobs_in_queue = mean_runtime_jobs_in_queue
    simulator_features[4] = mean_requested_walltime_jobs_in_queue

    # expand simulator features for concatenation
    simulator_features = simulator_features[np.newaxis, ...]

# === NODE FEATURES ===
    num_node_features = 6
    hosts = list(simulator.PlatformControl.get_state())
    num_hosts = len(hosts)

    # Generate random values per node per feature
    node_features = np.random.uniform(0.0, 10.0, size=(
        num_hosts, num_node_features)).astype(np.float32)

    # Broadcast simulator features to match node_features rows
    simulator_features_broadcast = np.broadcast_to(
        simulator_features, (num_hosts, simulator_features.shape[1])
    )

    # Concatenate along features axis
    features = np.concatenate(
        (simulator_features_broadcast, node_features), axis=1)

    return features
