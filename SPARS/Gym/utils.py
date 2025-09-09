from torch import nn
import numpy as np
import torch as T
import logging
from SPARS.Simulator.MachineMonitor import Monitor
logger = logging.getLogger("runner")


class Reward:
    """
    reward_per_node = α * (-energy_waste_per_node) + β * (-waiting_time_metric)
    where waiting_time_metric is a scalar (mean wait of finished jobs) broadcast to all nodes.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.9,
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

    def calculate_reward(self, monitor, waiting_queue, current_time):
        """
        Returns a scalar reward (torch tensor) on self.device.
        Higher energy waste or waiting time => more negative reward.
        """
        # 1) Totals (as floats)
        total_waste = sum(float(e.get('energy_waste', 0.0))
                          for e in monitor.energy)
        total_wait = sum(
            max(0.0, float(current_time) - float(j.get('subtime', 0.0)))
            for j in waiting_queue
        )

        # 2) To tensors on the correct device
        waste_t = T.tensor(total_waste, dtype=T.float32, device=self.device)
        wait_t = T.tensor(total_wait,  dtype=T.float32, device=self.device)

        # 3) Weighted penalty -> reward (scalar)
        penalty = self.alpha * waste_t + self.beta * wait_t
        reward = -penalty  # penalize waste & wait

        # (optional) logging as Python floats
        logger.info(
            f"total_waste={waste_t.item():.4f}, total_wait={wait_t.item():.4f}, reward={reward.item():.4f}")
        return reward  # 0-D tensor on self.device


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


def learn(model, model_opt, done, saved_experiences, next_observation,
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
    memory_features = T.stack(memory_features, dim=0)
    memory_actions = T.stack(memory_actions, dim=0)
    memory_rewads = T.stack(memory_rewards, dim=0)
    next_features, _next_masks = next_observation
    device = model.device

    rews = T.stack([r.to(device).float().view(-1).mean() if isinstance(r, T.Tensor)
                    else T.tensor(float(r), device=device)
                    for r in memory_rewards])

    device = next(model.parameters()).device
    Tlen = len(memory_rewards)

    logits, values = model(memory_features)
    next_logits, next_values = model(next_features)
    loc = logits.mean()

    # use std only when we have >1 element, else fallback to 1.0
    if logits.numel() > 1:
        std = logits.float().std(unbiased=False)   # avoid NaN
    else:
        std = T.tensor(1.0, device=logits.device, dtype=logits.dtype)

    scale = std.clamp_min(1e-6)  # avoid 0 or NaN
    dist = T.distributions.Normal(loc=loc, scale=scale)
    log_probs = dist.log_prob(memory_actions)
    entropy = dist.entropy().mean()

    bootstrap = T.zeros(
        (), device=device) if done else next_values.view(-1).mean()
    returns = T.empty_like(rews)
    R = bootstrap
    for t in range(Tlen - 1, -1, -1):
        R = rews[t] + gamma * R
        returns[t] = R

    advantages = gamma * returns - values

    # losses
    policy_loss = -(log_probs * advantages).mean()
    value_loss = (returns - values).pow(2).mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    # backprop
    model_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    model_opt.step()


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


def feature_extraction(simulator) -> np.ndarray:
    """
    Returns a 1D global feature vector of shape [11].
    The timestep axis is *implicit* (you append this per step to history_features).
    """

    # === GLOBAL (simulator-level) FEATURES ===
    # 1) #jobs in queue
    # 2) job arrival rate
    # 3) mean runtime of jobs in queue
    # 4) total energy waste
    # 5) mean requested walltime in queue (== runtime per your note)
    # 6) avg #required nodes by jobs in queue
    tq = simulator.jobs_manager.waiting_queue
    tnow = simulator.current_time
    t0 = simulator.start_time
    dt = max(tnow - t0, 1e-8)

    job_num = float(len(tq))
    arrival_rate = float(len(simulator.Monitor.jobs_submission_log)) / dt
    mean_runtime_q = (sum(job.get("runtime", 0.0) for job in tq) /
                      max(len(tq), 1e-8))
    total_waste = float(sum(e.get("energy_waste", 0.0)
                        for e in simulator.Monitor.energy))
    mean_req_wt_q = mean_runtime_q
    avg_req_nodes = (sum(job.get("res", 0.0) for job in tq) /
                     max(len(tq), 1e-8))

    sim_feats = np.array([
        job_num,
        arrival_rate,
        float(mean_runtime_q),
        total_waste,
        float(mean_req_wt_q),
        float(avg_req_nodes),
    ], dtype=np.float32)  # [6]

    # === AGGREGATED NODE FEATURES ===
    # 1) #computing nodes
    # 2) #idle nodes
    # 3) #sleeping nodes
    # 4) avg switching-on time
    # 5) avg switching-off time
    state = list(simulator.PlatformControl.get_state())
    computing_nodes = [n["id"] for n in state if n.get(
        "state") == "active" and n.get("job_id") is not None]
    idle_nodes = [n["id"] for n in state if n.get(
        "state") == "active" and n.get("job_id") is None]
    sleeping_nodes = [n["id"] for n in state if n.get("state") == "sleeping"]

    transitions_info = getattr(getattr(
        simulator.PlatformControl, "machines", object()), "machines_transition", [])
    sleeping_set, idle_set = set(sleeping_nodes), set(idle_nodes)
    switch_on_times, switch_off_times = [], []
    for node_info in transitions_info:
        nid = node_info.get("node_id")
        for tr in node_info.get("transitions", []):
            frm = tr.get("from")
            to = tr.get("to")
            tt = float(tr.get("transition_time", 0.0))
            if frm == "switching_on" and to == "active" and nid in sleeping_set:
                switch_on_times.append(tt)
            if frm == "switching_off" and to == "sleeping" and nid in idle_set:
                switch_off_times.append(tt)

    avg_switch_on = (sum(switch_on_times) /
                     max(len(switch_on_times),  1)) if switch_on_times else 0.0
    avg_switch_off = (sum(switch_off_times) /
                      max(len(switch_off_times), 1)) if switch_off_times else 0.0

    node_feats = np.array([
        float(len(computing_nodes)),
        float(len(idle_nodes)),
        float(len(sleeping_nodes)),
        float(avg_switch_on),
        float(avg_switch_off),
    ], dtype=np.float32)  # [5]

    # === FINAL 1D FEATURE VECTOR ===
    features = np.concatenate(
        [sim_feats, node_feats], axis=0).astype(np.float32)  # [11]
    return features
