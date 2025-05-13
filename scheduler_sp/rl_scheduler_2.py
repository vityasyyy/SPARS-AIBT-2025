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
    def __init__(self, simulator: SPSimulator, node_manager, lr=0.0001):
        self.simulator = simulator
        self.node_manager = node_manager
        self.agent_optimizer = optim.Adam(self.node_manager.parameters(), lr=lr)
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
        for idx, node in enumerate(nodes):
            action = actions[idx].item()
            
            if self.simulator.sim_monitor.nodes_action[node]['state'] == 'computing':
                continue

            if action == 0 and node not in self.simulator.inactive_resources:
                switch_off.append(node)
            elif action == 1 and node not in self.simulator.available_resources:
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
            if job['res'] <= len(self.simulator.get_not_allocated_resources()):
                reserved_node, need_activation_node = self.simulator.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
            else:
                break

    def backfill_schedule(self):
        if len(self.simulator.jobs_monitor.waiting_queue) >= 2:
            p_job = self.simulator.jobs_monitor.waiting_queue[0]
            
            backfilling_queue = self.simulator.jobs_monitor.waiting_queue[1:]

            not_reserved_resources = self.simulator.resources_agenda
            next_releases = self.simulator.resources_agenda
            
            next_releases = sorted(
                next_releases, 
                key=lambda x: (x['release_time'], x['node'])
            )
            if len(next_releases) < p_job['res']:
                return
            
            
            last_host = next_releases[p_job['res'] - 1]
            p_start_t = last_host['release_time']
            
            candidates = [r['node'] for r in next_releases if r['release_time'] <= p_start_t]
            reservation = candidates[-p_job['res']:]

            not_reserved_resources = [r for r in not_reserved_resources if r not in reservation]

            
            for job in backfilling_queue:
                available = self.simulator.get_not_allocated_resources()
                not_reserved = [h for h in available if h not in reservation]
                if job['res'] <= len(not_reserved):
                    reserved_node, need_activation_node = self.simulator.prioritize_lowest_node(job['res'])
                    self.simulator.execution_start(job, reserved_node, need_activation_node)
                elif job['walltime'] and job['walltime'] + self.simulator.current_time <= p_start_t and job['res'] <= len(available):
                    reserved_node, need_activation_node = self.simulator.prioritize_lowest_node(job['res'])
                    self.simulator.execution_start(job, reserved_node, need_activation_node)
    
    def easy_schedule(self):
        self.fcfs_schedule()
        self.backfill_schedule()
        
    def schedule(self):
        self.count_step += 1
        
        if not self.is_copied_instance:
            print(f'~~~ STEP {self.count_step} ~~~')
            print(f'curr time: {self.simulator.current_time}')
            print(f'aval res: {self.simulator.available_resources}')
            print(f'inac res: {self.simulator.inactive_resources}')
            print(f'event: {self.simulator.event}')
            self._rl_decision()
        else:
            self.easy_schedule()
        
    def _rl_decision(self):
        nodes = self.simulator.get_not_allocated_resources()
        if not nodes:
            return

        pre_state = copy.deepcopy(self.simulator)
        state_before = T.tensor(self.get_simulator_state(), dtype=T.float32).unsqueeze(0).unsqueeze(0)

 
        node_features = [self._get_node_features(index) for index, node in enumerate(nodes)]


        node_features_tensor = T.tensor(node_features, dtype=T.float32)

        action_probs = self.node_manager(node_features_tensor)

        print(f'Action probabilities: {action_probs}')
        actions = (action_probs > 0.5).float() 

        for idx, node in enumerate(nodes):
            if self.simulator.sim_monitor.nodes_action[node]['state'] == 'computing':
                actions[0, idx] = 1.0

        self.apply_action(actions.squeeze(0), nodes)
        self.easy_schedule()
        
        post_copy = copy.deepcopy(self)
        post_copy.is_copied_instance = True
        n = 5
        for i in range(n):     
            if post_copy.simulator.events or post_copy.simulator.jobs_monitor.waiting_queue:
                post_copy.simulator.proceed() 
            else:
                break

        state_after = post_copy.get_simulator_state()
        state_after = T.tensor(state_after, dtype=T.float32).unsqueeze(0).unsqueeze(0)

        reward = reward_function(pre_state, post_copy)
        self._update_networks(action_probs.squeeze(0), reward)

    def _update_networks(self, action_probs, reward):
        print(f'Reward: {reward}')
        
        loss = -T.mean(reward * action_probs)
        print(f'Loss: {loss.item()}')

        
        self.agent_optimizer.zero_grad()
        loss.backward()

        self.agent_optimizer.step()
   
        
    def _get_global_features(self):

        return [
            self.simulator.current_time,
            len(self.simulator.jobs_monitor.waiting_queue),
            self.simulator.jobs_monitor.waiting_queue[0]['res'] 
            if len(self.simulator.jobs_monitor.waiting_queue) > 0 
            else 0
        ]

    def _get_node_unique_features(self, node_index):
        state = self.simulator.sim_monitor.nodes_action[node_index]['state']
        state_mapping = {
            'sleeping': 0,
            'turning_off': 1,
            'idle': 2,
            'turning_on': 3,
            'computing': 4
        }
        
        state_number = state_mapping.get(state, -1)  
        transition_cost = 0
        if state_number == 0:
            transition_cost = 190
        elif state_number == 2:
            transition_cost = 9
        
        return [state_number, 190, transition_cost, self.simulator.sim_monitor.nodes_action[node_index]['time']]

    def _get_node_features(self, node_index):
        return [self._get_global_features() + self._get_node_unique_features(node_index)]