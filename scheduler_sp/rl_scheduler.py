from scheduler_sp.env import SPSimulator
import torch as T
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

def reward_function(before, after):
    weight_energy_waste = 1
    num_idle_res_after_action = len(after.simulator.available_resources)
    energy_waste = num_idle_res_after_action * weight_energy_waste

    waiting_time = sum(after.simulator.current_time - job['subtime'] 
                      for job in after.simulator.jobs_monitor.waiting_queue)
    
    alpha = 0.1
    beta = 0.9
    return -alpha * energy_waste - beta * waiting_time

class RLScheduler:
    def __init__(self, simulator: SPSimulator, actor, critic, lr=0.01):
        self.simulator = simulator
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.is_copied_instance = False
        self.count_step = 0


    def get_simulator_state(self):
        return [
            self.simulator.sim_monitor.total_waiting_time,
            sum(self.simulator.sim_monitor.energy_consumption),
            sum(self.simulator.sim_monitor.energy_waste)
        ]

    def apply_action(self, actions, nodes):
        switch_off = []
        switch_on = []
        for index, node in enumerate(nodes):
            action_value = actions[index].item() if actions.numel() > 1 else actions.item()

            if action_value == 0 and node not in self.simulator.inactive_resources:
                switch_off.append(node)
            elif action_value == 1 and node not in self.simulator.available_resources:
                switch_on.append(node)
        
        if switch_off:
            print(f'switch off: {switch_off}')
            self.simulator.switch_off(switch_off)
        if switch_on:
            print(f'switch on: {switch_on}')
            self.simulator.switch_on(switch_on)
        
        if len(switch_off) == 0 and len(switch_on) == 0:
            print('no action taken')

    def fcfs_schedule(self):
        for job in self.simulator.jobs_monitor.waiting_queue[:]:
            available = [h for h in self.simulator.get_not_allocated_resources() 
                        if h not in self.simulator.inactive_resources]
            
            if job['res'] <= len(available):
                reserved_node, need_activation_node = self.simulator.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
            else:
                break

    def backfill_schedule(self):
        if len(self.simulator.jobs_monitor.waiting_queue) >= 2:
            p_job = self.simulator.jobs_monitor.waiting_queue[0]
            backfilling_queue = self.simulator.jobs_monitor.waiting_queue[1:]
            
            not_reserved_resources = sorted(set(self.simulator.available_resources) - 
                                          set(self.simulator.reserved_resources))
            next_releases = [{'release_time': 0, 'node': nrs} for nrs in not_reserved_resources]
            
            for job in self.simulator.jobs_monitor.active_jobs:
                next_releases.extend({'release_time': job['finish_time'], 'node': node} 
                                  for node in job['allocated_resources'])
            
            next_releases.sort(key=lambda x: (x['release_time'], x['node']))
            
            if len(next_releases) >= p_job['res']:
                last_host = next_releases[p_job['res'] - 1]
                p_start_t = last_host['release_time']
                reservation = [r['node'] for r in next_releases 
                              if r['release_time'] <= p_start_t][-p_job['res']:]

                for job in backfilling_queue:
                    available = [h for h in self.simulator.get_not_allocated_resources() 
                                if h not in self.simulator.inactive_resources]
                    not_reserved = [h for h in available if h not in reservation]

                    if job['res'] <= len(not_reserved):
                        self.simulator.prioritize_and_start(job)
                    elif job['walltime'] and (job['walltime'] + self.simulator.current_time <= p_start_t):
                        self.simulator.prioritize_and_start(job)

    def easy_schedule(self):
        self.fcfs_schedule()
        self.backfill_schedule()
        
    def schedule(self):
        """Main entry point called by simulator after each event"""
        self.count_step += 1
        
        # 1. Take RL action if possible
        if not self.is_copied_instance:
            print(f'~~~ STEP {self.count_step} ~~~')
            print(f'aval res: {self.simulator.available_resources}')
            print(f'inac res: {self.simulator.inactive_resources}')
            self._rl_decision()
        else:
        # 2. Perform standard scheduling
            self.easy_schedule()
        

    def _rl_decision(self):
        nodes = self.simulator.get_not_allocated_resources()
        if not nodes:
            return

        # Capture pre-state
        pre_state = copy.deepcopy(self.simulator)
        state_before = T.tensor(self.get_simulator_state(), 
                              dtype=T.float32).unsqueeze(0).unsqueeze(0)
        value_before = self.critic(state_before)

        # Generate actions
        node_features = np.array([
            self.get_global_features() + self.get_node_features(node) 
            for node in nodes
        ], dtype=np.float32)
        
        action_probs = self.actor(T.tensor(node_features).unsqueeze(0))
        print(f'action probs: {action_probs}')
        
        actions = T.bernoulli(action_probs).float()  # Stochastic sampling

        # Apply actions
        self.apply_action(actions.squeeze(0), nodes)
        
        self.easy_schedule()
        
        # Create post-state copy for training
        post_copy = copy.deepcopy(self)
        post_copy.is_copied_instance = True
        post_copy.simulator.proceed()  # Process next event

        state_after = post_copy.get_simulator_state()
        print(f'state after: {state_after}')
        state_after = T.tensor(state_after, dtype=T.float32).unsqueeze(0).unsqueeze(0)
        value_after = self.critic(state_after)
        # Calculate reward and update networks
        reward = reward_function(pre_state, post_copy)
        self._update_networks(value_before, value_after, action_probs.squeeze(0), reward)

    def _update_networks(self, value_before, value_after, action_probs, reward):
        print(f'value before :{value_after}')
        print(f'value after :{value_before}')
        advantage = reward + value_after - value_before

        # Actor update
        self.actor_optimizer.zero_grad()
        entropy = -T.mean(action_probs * T.log(action_probs + 1e-6)) - T.mean((1 - action_probs) * T.log(1 - action_probs + 1e-6))
        loss_actor = -T.mean(advantage.detach() * T.log(action_probs + 1e-6)) - 0.1 * entropy
        loss_actor.backward()
        self.actor_optimizer.step()

        # Critic update
        self.critic_optimizer.zero_grad()
        loss_critic = self.loss_fn(value_before, reward + value_after.detach())
        loss_critic.backward()
        self.critic_optimizer.step()

    def get_global_features(self):
        return [
            len(self.simulator.jobs_monitor.waiting_queue),
            sum(self.simulator.current_time - j['subtime'] 
               for j in self.simulator.jobs_monitor.waiting_queue),
            sum(j['res'] for j in self.simulator.jobs_monitor.waiting_queue),
            len([n for n in self.simulator.available_resources 
                if n not in self.simulator.reserved_resources]),
            len(self.simulator.inactive_resources)
        ]

    def get_node_features(self, node_index, simulator=None):
        if simulator is None:
            simulator = self.simulator

        transition_cost = simulator.machines[node_index]['wattage_per_state']
        transition_time = 0
        if simulator.sim_monitor.nodes_action[node_index]['state'] == 'idle':
            transition_cost = transition_cost[4]
            transition_time = simulator.transition_time[0]
        elif simulator.sim_monitor.nodes_action[node_index]['state'] == 'sleeping':
            transition_cost = transition_cost[3]
            transition_time = simulator.transition_time[1]

        return [transition_cost, transition_time]