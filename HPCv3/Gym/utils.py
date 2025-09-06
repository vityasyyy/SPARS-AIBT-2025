import random
import numpy as np
from typing import Tuple
import torch as T
from torch.func import vmap
import logging
from HPCv3.Simulator.MachineMonitor import Monitor
from torchviz import make_dot
logger = logging.getLogger("runner")


class Reward():
    def __init__(self):
        self.alpha = 0.5
        self.beta = 0.5

    def calculate_reward(monitor: Monitor):
        wasted_energy = [-energy['energy_waste']
                         for energy in monitor.energy]
        wasted_energy = T.tensor(
            wasted_energy, dtype=T.float32, requires_grad=True, device='cuda')
        # logger.info(f"Wasted energy: {wasted_energy}")
        return wasted_energy


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
          gamma: float = 0.99, entropy_coef: float = 0.0):
    """
    Single-pass (reverse) loop, no batching (TorchScript-safe).
    - No Graphviz in hot path
    - Per-timestep returns, values, advantages
    - Avoids Python-scalar conversions
    """
    memory_actions, memory_features, memory_masks, memory_rewards = saved_experiences
    next_features, _next_masks = next_observation

    device = next(agent.parameters()).device
    Tlen = len(memory_rewards)

    # Helper: make any tensor -> scalar tensor on device (mean if it has >1 element)
    def _scalar(t):
        if isinstance(t, T.Tensor):
            return t.to(device).float().view(-1).mean()
        return T.tensor(float(t), device=device)

    # --- Bootstrap from next state (single critic call) ---
    with T.no_grad():
        bootstrap = T.zeros((), device=device) if done else _scalar(
            critic(next_features))

    # --- Allocate per-timestep arrays on device ---
    rewards = T.empty(Tlen, device=device)
    returns = T.empty(Tlen, device=device)
    values = T.empty(Tlen, device=device)
    logprobs = T.empty(Tlen, device=device)
    entropies = T.empty(Tlen, device=device)

    have_entropy = False

    # --- Single reverse pass: compute returns, values, logprobs ---
    R = bootstrap
    for t in range(Tlen - 1, -1, -1):
        # reward_t as scalar tensor
        rewards[t] = _scalar(memory_rewards[t])

        # critic value (scalar tensor)
        v_t = critic(memory_features[t]).float().view(-1).mean()
        values[t] = v_t

        # agent outputs
        out = agent(memory_features[t].to(device), memory_masks[t].to(device))
        if isinstance(out, (tuple, list)):
            lp_t = out[0]
            ent_t = out[1]
        else:
            lp_t, ent_t = out, None

        logprobs[t] = _scalar(lp_t)
        if ent_t is not None:
            entropies[t] = _scalar(ent_t)
            have_entropy = True

        # discounted return
        R = rewards[t] + gamma * R
        returns[t] = R

    # --- Losses ---
    advantage = (returns - values).detach()       # [T]
    actor_loss = -(logprobs * advantage).mean()
    if entropy_coef and have_entropy:
        actor_loss = actor_loss - entropy_coef * entropies.mean()

    critic_loss = T.nn.functional.mse_loss(values, returns)

    # --- Optimize actor ---
    agent_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    # T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=...)
    agent_opt.step()

    # --- Optimize critic ---
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    # T.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=...)
    critic_opt.step()

    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "returns_mean": float(returns.mean().detach().cpu()),
        "adv_mean": float(advantage.mean().detach().cpu()),
        "entropy": float(entropies.mean().detach().cpu()) if have_entropy else None,
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
    )
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
