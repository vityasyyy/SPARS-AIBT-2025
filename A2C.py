import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
import heapq
from problem_generator.problem_generator import ProblemGenerator
import json
import torch.nn.functional as F

TOTAL_NODES = 100


class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.actor(state)


class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.critic(state)


class A2CAgent:
    def __init__(self, state_size, lr=0.1):
        self.actor = ActorNetwork(state_size)
        self.critic = CriticNetwork(state_size)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_prob = self.actor(state_tensor)
        alpha = action_prob.item() * 5 + 1
        beta = (1 - action_prob.item()) * 5 + 1
        dist = D.Beta(alpha, beta)
        action = dist.sample().item()

        return action

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)

        # Actor selects action
        action_probs = actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()

        # Take action and observe next state and reward
        next_state, reward, done, _, _ = env.step(action.item())

        # Critic estimates value function
        value = critic(state_tensor)
        next_value = critic(torch.FloatTensor(next_state))

        # Calculate TD target and Advantage
        td_target = reward + gamma * next_value * (1 - done)
        advantage = td_target - value

        # Critic update with MSE loss
        critic_loss = F.mse_loss(value, td_target.detach())
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        # Actor update
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * advantage.detach()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()


def calculate_wasted_energy(idle_nodes, idle_time):
    rate_waste_energy = 1
    return idle_nodes * idle_time * rate_waste_energy


def push_schedule_queue(schedule_queue, out_schedule_queue, event):
    heapq.heappush(schedule_queue, event)
    out_schedule_queue += [event]


class Trainer:
    def __init__(self, agent, generator, alpha, beta, num_episodes=100):
        self.agent = agent
        self.generator = generator
        self.alpha = alpha
        self.beta = beta
        self.num_episodes = num_episodes
        self.episode_rewards = []

    def train(self):
        optimizer_actor = optim.AdamW(actor.parameters())
        jobs = []
        count_jobs = 0
        new_jobs = self.generator.generate()
        jobs.extend(new_jobs)
        count_jobs += len(new_jobs)
        jobs.sort(key=lambda job: job['arrival'])

        for episode in range(self.num_episodes):
            env_state = {'total_nodes': TOTAL_NODES, 'active_nodes': 0,
                         'inactive_nodes': 0, 'idle_nodes': TOTAL_NODES}
            out_schedule_queue = []

            total_reward = 0
            total_waiting_time = 0
            schedule_queue = []
            total_energy_consumption = 0
            heapq.heapify(schedule_queue)
            job_started = 0
            jobs_queue = []
            heapq.heapify(jobs_queue)
            actions = []
            total_waiting_time_transient = 0

            for job in jobs:
                job['type'] = 'arrival'
                push_schedule_queue(
                    schedule_queue, out_schedule_queue, (job['arrival'], job))

            while schedule_queue:
                event = heapq.heappop(schedule_queue)
                event_time, job_data = event

                if job_data['type'] == 'arrival':
                    required_nodes = job_data['nodes']
                    available_nodes = env_state['idle_nodes']

                    if available_nodes >= required_nodes:
                        execution_event = {
                            'nodes': job_data['nodes'],
                            'actual_execution_time': job_data['actual_execution_time'],
                            'type': 'execution_start',
                            'arrival': job_data['arrival']
                        }
                        push_schedule_queue(
                            schedule_queue, out_schedule_queue, (event_time, execution_event))

                    else:
                        heapq.heappush(jobs_queue, event)
                elif job_data['type'] == 'execution_start':
                    job_started += 1
                    total_waiting_time += event_time - job_data['arrival']
                    env_state['idle_nodes'] -= job_data['nodes']
                    env_state['active_nodes'] += job_data['nodes']
                    execution_event = {
                        'nodes': job_data['nodes'],
                        'actual_execution_time': job_data['actual_execution_time'],
                        'type': 'execution_finished'
                    }
                    push_schedule_queue(
                        schedule_queue, out_schedule_queue,  (event_time + job_data['actual_execution_time'], execution_event))

                elif job_data['type'] == 'execution_finished':
                    rate_consumption = 1
                    total_energy_consumption += job_data['actual_execution_time'] * \
                        rate_consumption * job_data['nodes']

                    env_state['idle_nodes'] += job_data['nodes']
                    env_state['active_nodes'] -= job_data['nodes']

                    temp_idle = env_state['idle_nodes']
                    while True:
                        if len(jobs_queue) != 0 and (jobs_queue[0][1]['nodes'] <= env_state['idle_nodes']):
                            event_time, job_data = heapq.heappop(jobs_queue)
                            start_time = event_time
                            execution_event = {
                                'nodes': job_data['nodes'],
                                'actual_execution_time': job_data['actual_execution_time'],
                                'type': 'execution_start',
                                'arrival': job_data['arrival']
                            }
                            push_schedule_queue(
                                schedule_queue, out_schedule_queue,  (start_time + job_data['actual_execution_time'], execution_event))
                            temp_idle -= job_data['nodes']
                        else:
                            break

                state = [env_state['active_nodes'], env_state['idle_nodes'],
                         env_state['inactive_nodes'], float(len(jobs_queue))]
                action = self.agent.choose_action(state)
                actions.append(action)
                nodes_to_on = int(action * TOTAL_NODES)

                env_state['idle_nodes'] = max(
                    0, nodes_to_on - env_state['active_nodes'])

                env_state['inactive_nodes'] = TOTAL_NODES - \
                    env_state['idle_nodes'] - env_state['active_nodes']

                wasted_energy = calculate_wasted_energy(
                    env_state['idle_nodes'], event_time)
                total_waiting_time_transient = 0
                for jobs_ in jobs_queue:
                    total_waiting_time_transient += max(
                        event_time - jobs_[0], 0)

                reward = -self.alpha * wasted_energy - self.beta * total_waiting_time_transient
                total_reward += reward

                next_state = [env_state['active_nodes'], env_state['idle_nodes'],
                              env_state['inactive_nodes'], float(len(jobs_queue))]
                done = int(not jobs)
                self.agent.train(state, action, reward, next_state, done)

            self.episode_rewards.append(total_reward)
            print(
                f"Episode {episode+1}, Alpha: {self.alpha}, Beta: {self.beta}, Total Reward: {total_reward:.2f}")

            with open(f"my_data_{self.alpha}_{self.beta}_{episode}.json", "w") as f:
                out = [a for a in out_schedule_queue]
                out = sorted(out, key=lambda x: x[0])
                json.dump(out, f, indent=4)
            with open(f"actions_{self.alpha}_{self.beta}_{episode}.json", "w") as f:
                json.dump(actions, f, indent=4)
        return np.mean(self.episode_rewards)


def run_grid_search():
    state_size = 4
    generator = ProblemGenerator(num_jobs=100)

    best_reward = float('-inf')
    best_alpha, best_beta = None, None

    for alpha in [1]:
        for beta in [1]:
            print(f"Training with alpha={alpha}, beta={beta}...")
            agent = A2CAgent(state_size)
            trainer = Trainer(agent, generator,
                              alpha, beta, num_episodes=100)
            avg_reward = trainer.train()

            if avg_reward > best_reward:
                best_reward = avg_reward

    print(
        f"Best parameters: Alpha={best_alpha}, Beta={best_beta}, Reward={best_reward:.2f}")


def main():
    run_grid_search()


if __name__ == "__main__":
    main()
