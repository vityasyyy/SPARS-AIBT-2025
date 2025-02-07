import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import heapq
import json
from problem_generator.problem_generator import ProblemGenerator


TOTAL_NODES = 100


class A2CAgent:
    def __init__(self, state_size=4, action_size=1, gamma=0.99, lr=1e-4):
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size

        self.actor = ActorNetwork(state_size, action_size)
        self.critic = CriticNetwork(state_size)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        # Calling forward action to get action probability based on current state
        action_prob = self.actor(state)
        action = action_prob.detach().numpy()[0][0]
        return action

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0)
        reward = torch.FloatTensor([reward]).unsqueeze(0)

        value = self.critic(state)
        next_value = self.critic(next_state)

        target_value = reward + self.gamma * next_value * (1 - done)
        advantage = target_value - value

        log_prob = torch.log(self.actor(state) + 1e-5)
        actor_loss = -log_prob * advantage.detach()

        critic_loss = F.mse_loss(value, target_value.detach())

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        # Function to return probability action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Make sure the output is float in range 0 to 1
        action = torch.sigmoid(self.output_layer(x))
        return action


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        # Function to get value
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.output_layer(x)
        return value


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

        jobs = []
        count_jobs = 0
        new_jobs = self.generator.generate()
        jobs.extend(new_jobs)
        count_jobs += len(new_jobs)
        jobs.sort(key=lambda job: job['arrival'])

        for episode in range(self.num_episodes):

            # This part is to setup environment variables
            env_state = {'total_nodes': TOTAL_NODES, 'active_nodes': 0,
                         'inactive_nodes': 0, 'idle_nodes': TOTAL_NODES}
            out_schedule_queue = []
            # heapq.heapify(out_schedule_queue)

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
            prev_event_time = 0
            for job in jobs:
                job['type'] = 'arrival'
                push_schedule_queue(
                    schedule_queue, out_schedule_queue, (job['arrival'], job))
            while schedule_queue:  # Ini step, step adalah event based
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
                actions.append(float(action))
                nodes_to_on = int(action * TOTAL_NODES)

                env_state['idle_nodes'] = max(0, nodes_to_on -
                                              env_state['active_nodes'])

                env_state['inactive_nodes'] = TOTAL_NODES - \
                    env_state['idle_nodes'] - env_state['active_nodes']

                wasted_energy = calculate_wasted_energy(
                    env_state['idle_nodes'], (event_time - prev_event_time))

                prev_event_time = event_time

                total_waiting_time_transient = 0

                for jobs_ in jobs_queue:
                    total_waiting_time_transient += max(
                        event_time - jobs_[0], 0)

                reward = -self.alpha * wasted_energy - self.beta * total_waiting_time_transient
                total_reward += reward

                next_state = [env_state['active_nodes'], env_state['idle_nodes'],
                              env_state['inactive_nodes'], float(len(jobs_queue))]
                done = int(not schedule_queue)
                self.agent.train(state, action, reward, next_state, done)

            self.episode_rewards.append(total_reward)
            print(
                f"Episode {episode+1}, Alpha: {self.alpha}, Beta: {self.beta}, Total Reward: {total_reward:.2f}")

            with open(f"schedules/my_data_{self.alpha}_{self.beta}_{episode}.json", "w") as f:
                out = [a for a in out_schedule_queue]
                out = sorted(out, key=lambda x: x[0])
                json.dump(out, f, indent=4)
            with open(f"actions/actions_{self.alpha}_{self.beta}_{episode}.json", "w") as f:
                json.dump(actions, f, indent=4)
        return np.mean(self.episode_rewards)


def run_grid_search():
    state_size = 4
    generator = ProblemGenerator(num_jobs=1000)

    best_reward = float('-inf')
    best_alpha, best_beta = None, None

    # for alpha in range(1, 101):
    for alpha in (0.1, 0.2):
        for beta in (0.001, 0.002):
            print(f"Training with alpha={alpha}, beta={beta}...")
            agent = A2CAgent(state_size)
            trainer = Trainer(agent, generator,
                              alpha, beta, num_episodes=50)
            avg_reward = trainer.train()

            if avg_reward > best_reward:
                best_reward = avg_reward

    print(
        f"Best parameters: Alpha={best_alpha}, Beta={best_beta}, Reward={best_reward:.2f}")


def main():
    run_grid_search()


if __name__ == "__main__":
    main()
