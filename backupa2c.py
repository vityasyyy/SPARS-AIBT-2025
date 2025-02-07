import torch.nn.functional as F
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from problem_generator.scheduler import Scheduler
from problem_generator.problem_generator import ProblemGenerator
import matplotlib.pyplot as plt
import heapq

TOTAL_NODES = 100


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_prob = self.actor(state)
        state_value = self.critic(state)
        return action_prob, state_value


class A2CAgent:
    def __init__(self, state_size, lr=0.0001):
        self.model = ActorCriticNetwork(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_prob, _ = self.model(state_tensor)
        return action_prob.item()

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.FloatTensor([action])

        action_prob, state_value = self.model(state_tensor)
        _, next_state_value = self.model(next_state_tensor)

        delta = reward + (1 - done) * \
            next_state_value.item() - state_value.item()
        actor_loss = -torch.log(action_prob.squeeze(0)) * delta
        critic_loss = delta ** 2
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def calculate_wasted_energy(inactive_nodes, interarrival_time):
    return inactive_nodes * interarrival_time


class Trainer:
    def __init__(self, agent, generator, scheduler, num_episodes=100):
        self.agent = agent
        self.generator = generator
        self.scheduler = scheduler
        self.num_episodes = num_episodes
        self.episode_rewards = []

    def train(self):
        for episode in range(self.num_episodes):
            env_state = {'total_nodes': TOTAL_NODES, 'active_nodes': 0,
                         'inactive_nodes': TOTAL_NODES, 'idle_nodes': 0}
            count_jobs = 0
            total_reward = 0
            schedule_result = []
            jobs = []
            total_waiting_time = 0
            completed_jobs = 0

            while count_jobs < 1000 or jobs:
                if count_jobs < 1000:
                    new_jobs = self.generator.generate()
                    jobs.extend(new_jobs)
                    count_jobs += len(new_jobs)
                    jobs.sort(key=lambda job: job['arrival'])

                if jobs:
                    job = jobs.pop(0)
                    required_nodes = job['nodes']
                    allocated_nodes = [0.0] * env_state['total_nodes']
                    heapq.heapify(allocated_nodes)
                    start_time = max(job['arrival'], heapq.nlargest(
                        required_nodes, allocated_nodes)[-1])

                    for _ in range(required_nodes):
                        heapq.heappop(allocated_nodes)
                    for _ in range(required_nodes):
                        heapq.heappush(allocated_nodes,
                                       start_time + job['requested_execution_time'])

                    waiting_time = start_time - job['arrival']
                    total_waiting_time += waiting_time
                    completed_jobs += 1
                    avg_waiting_time = total_waiting_time / \
                        completed_jobs if completed_jobs > 0 else 0

                    schedule_result.append({
                        **job,
                        'start': start_time,
                        'actual_finish': start_time + job['actual_execution_time'],
                        'expected_finish': start_time + job['requested_execution_time'],
                        'waiting': waiting_time
                    })

                    state = [env_state['active_nodes'], env_state['idle_nodes'],
                             env_state['inactive_nodes'], avg_waiting_time]
                    action = self.agent.choose_action(state)
                    nodes_to_activate = int(action * TOTAL_NODES)

                    env_state['inactive_nodes'] = max(
                        0, env_state['inactive_nodes'] - nodes_to_activate)
                    env_state['active_nodes'] = min(
                        TOTAL_NODES, env_state['active_nodes'] + nodes_to_activate)
                    env_state['idle_nodes'] = TOTAL_NODES - \
                        env_state['active_nodes'] - env_state['inactive_nodes']

                    interarrival_time = np.random.uniform(0, 1) if jobs else 0
                    wasted_energy = calculate_wasted_energy(
                        env_state['inactive_nodes'], interarrival_time)
                    reward = -wasted_energy-(avg_waiting_time*100)

                    next_state = [env_state['active_nodes'], env_state['idle_nodes'],
                                  env_state['inactive_nodes'], avg_waiting_time]
                    done = int(not jobs)
                    self.agent.train(state, action, reward, next_state, done)

                    total_reward += reward
                    # print(
                    #     f"Activated {nodes_to_activate} nodes, Active: {env_state['active_nodes']}, Idle: {env_state['idle_nodes']}, Inactive: {env_state['inactive_nodes']}")

            self.episode_rewards.append(total_reward)
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

    def plot_results(self):
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()


def main():
    state_size = 4
    agent = A2CAgent(state_size)
    generator = ProblemGenerator()
    scheduler = Scheduler()
    trainer = Trainer(agent, generator, scheduler)
    trainer.train()
    trainer.plot_results()
    torch.save(agent.model.state_dict(), 'hpc_scheduler_a2c.pth')


if __name__ == "__main__":
    main()


v2

# hpc_scheduler_a2c_train.py


TOTAL_NODES = 100


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_prob = self.actor(state)
        state_value = self.critic(state)
        return action_prob, state_value


class A2CAgent:
    def __init__(self, state_size, lr=0.0001):
        self.model = ActorCriticNetwork(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_prob, _ = self.model(state_tensor)
        return action_prob.item()

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.FloatTensor([action])

        action_prob, state_value = self.model(state_tensor)
        _, next_state_value = self.model(next_state_tensor)

        delta = reward + (1 - done) * \
            next_state_value.item() - state_value.item()
        actor_loss = -torch.log(action_prob.squeeze(0)) * delta
        critic_loss = delta ** 2
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
            env_state = {'total_nodes': TOTAL_NODES, 'active_nodes': 0,
                         'inactive_nodes': 0, 'idle_nodes': TOTAL_NODES}
            out_schedule_queue = []
            # heapq.heapify(out_schedule_queue)

            total_reward = 0

            total_waiting_time = 0
            schedule_queue = []
            total_energy_consumption = 0
            heapq.heapify(schedule_queue)
            avg_waiting_time = 0
            job_started = 0
            jobs_queue = []
            heapq.heapify(jobs_queue)
            actions = []
            avg_waiting_time_transient = 0
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
                    avg_waiting_time = total_waiting_time / job_started
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

                env_state['idle_nodes'] = max(0, nodes_to_on -
                                              env_state['active_nodes'])

                env_state['inactive_nodes'] = TOTAL_NODES - \
                    env_state['idle_nodes'] - env_state['active_nodes']

                wasted_energy = calculate_wasted_energy(
                    env_state['idle_nodes'], event_time)
                avg_waiting_time_transient = 0
                for jobs in jobs_queue:
                    avg_waiting_time_transient += max(event_time - jobs[0], 0)
                avg_waiting_time_transient /= len(jobs_queue)
                reward = -self.alpha * wasted_energy - self.beta * avg_waiting_time_transient
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
    generator = ProblemGenerator(num_jobs=1000)

    best_reward = float('-inf')
    best_alpha, best_beta = None, None

    # for alpha in range(1, 101):
    for alpha in [1]:
        for beta in [0.001]:
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


utama

TOTAL_NODES = 100


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_prob = self.actor(state)
        state_value = self.critic(state)
        return action_prob, state_value


class A2CAgent:
    def __init__(self, state_size, lr=0.0001):
        self.model = ActorCriticNetwork(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_prob, _ = self.model(state_tensor)
        return action_prob.item()

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.FloatTensor([action])

        action_prob, state_value = self.model(state_tensor)
        _, next_state_value = self.model(next_state_tensor)

        delta = reward + (1 - done) * \
            next_state_value.item() - state_value.item()
        actor_loss = -torch.log(action_prob.squeeze(0)) * delta
        critic_loss = delta ** 2
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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

                env_state['idle_nodes'] = max(0, nodes_to_on -
                                              env_state['active_nodes'])

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
    generator = ProblemGenerator(num_jobs=1000)

    best_reward = float('-inf')
    best_alpha, best_beta = None, None

    for alpha in [1]:
        for beta in [1]:
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
