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

    # Move time axis to the end for convenience
    if time_dim != -1:
        rewards = rewards.transpose(time_dim, -1)  # shape: [..., seq_len]

    seq_len = rewards.size(-1)

    # gamma as tensor
    if T.is_tensor(gamma):
        gamma = gamma.to(device=device, dtype=dtype)
    else:
        gamma = T.tensor(gamma, device=device, dtype=dtype)

    # steps = T.arange(seq_len, device=device, dtype=dtype)  # [seq_len]
    print(rewards.shape)
    print(gamma.shape)
    # print(steps.shape)
    # factors = gamma ** steps                                    # [seq_len]
    # factors = factors.view(*([1] * (rewards.dim() - 1)),
    #                        seq_len)  # [..., seq_len]

    weighted = rewards.view(-1, 1) * gamma
    flipped = T.flip(weighted, dims=[-1])
    csum = T.cumsum(flipped, dim=-1)
    disc = T.flip(csum, dims=[-1]) / gamma

    # Put time axis back
    if time_dim != -1:
        disc = disc.transpose(-1, time_dim)

    return disc


def learn(agent, agent_opt, critic, critic_opt, done, advantage, saved_experiences):
    saved_logprobs, saved_states, saved_masks, saved_rewards, next_state = saved_experiences
    # prepare returns
    if done:
        R = 0
    else:
        next_state = next_state.to(agent.device)
        R = critic(next_state).detach().item()
    returns = [0 for _ in range(len(saved_logprobs))]
    critic_vals = [0. for _ in range(len(saved_logprobs))]
    new_logprobs = []
    for i in range(len(returns)):
        actions, entropy = (agent(saved_states[i], saved_masks[i]))
        new_logprobs.append(actions)

    for i in range(len(returns)):
        R = saved_rewards[-i] + 0.9*R
        returns[-i] = R
        critic_vals[-i] = critic(saved_states[-i]).squeeze(0)

    new_logprobs = T.stack(new_logprobs)
    critic_vals = T.stack(critic_vals)
    # update actor
    new_logprobs = new_logprobs.view(1, 1, -1, 1)
    agent_loss = -(new_logprobs*advantage).sum()
    agent_opt.zero_grad(set_to_none=True)

    # assume agent_loss is your scalar tensor
    dot = make_dot(agent_loss, params=dict(agent.named_parameters()))

    # render to file (PDF/PNG/SVG)
    dot.render("agent_loss_graph", format="pdf")

    agent_loss.backward()
    # T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm="inf")
    agent_opt.step()
    returns = T.stack(returns, dim=0)
    returns = returns.view(1, 1, -1, 1)

    # update critic
    critic_loss = (returns-critic_vals)**2
    critic_loss = critic_loss.mean()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    # T.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.grad_norm)
    critic_opt.step()


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
    node_features = np.zeros(
        (len(hosts), num_node_features), dtype=np.float32)

    node_features[:, 0] = 0  # host_on_off
    node_features[:, 1] = 0  # host_active_idle
    node_features[:, 2] = 0  # current_idle_time
    node_features[:, 3] = 0  # remaining_runtime_percent
    node_features[:, 4] = 0  # normalized_wasted_energy
    node_features[:, 5] = 0  # normalized_switching_time

    # broadcast simulator features to match node_features rows
    simulator_features = np.broadcast_to(
        simulator_features, (node_features.shape[0],
                             simulator_features.shape[1])
    )

    # concatenate along features axis
    features = np.concatenate((simulator_features, node_features), axis=1)

    return features
