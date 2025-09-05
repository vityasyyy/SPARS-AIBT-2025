import numpy as np
from typing import Tuple
import torch as T
from torch.func import vmap


class Reward():
    def calculate_reward(state):
        reward = -state[0]['power']
        return reward


def learn(agent, agent_opt, critic, critic_opt, done, saved_experiences, next_state, discount_factor=0.99):
    saved_features, masks, saved_actions, saved_rewards = saved_experiences
    saved_features = T.tensor(
        saved_features, requires_grad=True).to(agent.device)

    # prepare returns
    if done:
        R = 0
    else:
        next_state = next_state.to(agent.device)
        R = critic(next_state).detach().item()
    saved_logprobs = T.tensor([np.log(prob)
                              for prob in saved_actions], requires_grad=True).to(agent.device)

    rewards = T.tensor(saved_rewards, dtype=T.float32,
                       device=agent.device, requires_grad=True)

    # Compute discounted returns
    R = 0.
    returns = T.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        R = rewards[i] + discount_factor * R
        returns[i] = R

    # Compute critic values in one batch (if saved_features is a tensor)

    critic_values_list = [critic(T.tensor(f, dtype=T.float32, device=agent.device)).squeeze(-1)
                          for f in saved_features]


# Convert list to tensor
    critic_values = T.stack(critic_values_list)

    # returns = T.tensor(returns, dtype=T.float32).to(agent.device)
    advantage = (returns - critic_values).detach()
    print(advantage.shape)
    print(saved_logprobs.shape)
    # update actor
    saved_logprobs = saved_logprobs.view(-1, 1, 1, 1)  # [T,1,1,1]
    agent_loss = -(saved_logprobs*advantage).sum()
    agent_opt.zero_grad(set_to_none=True)
    agent_loss.backward()
    T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()

    # update critic
    critic_loss = (returns-critic_values)**2
    critic_loss = critic_loss.mean()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    T.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1)
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
        job["walltime"] for job in simulator.jobs_manager.waiting_queue) / len(simulator.jobs_manager.waiting_queue)

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
