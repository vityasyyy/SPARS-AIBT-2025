import itertools
import time
from HPCv2_Simulator.Simulator import SPSimulator
import torch as T
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

def reward_function(before, after):
    before_energy_waste = 0
    for node_energy_waste in before.simulator.sim_monitor.energy_waste:
        before_energy_waste += node_energy_waste
        
    after_energy_waste = 0
    for node_energy_waste in after.simulator.sim_monitor.energy_waste:
        after_energy_waste += node_energy_waste
        
    energy_waste = after_energy_waste - before_energy_waste
    
    # weight_energy_waste = 1
    # num_idle_res_after_action = len(after.simulator.node_manager.available_resources)
    # energy_waste = num_idle_res_after_action * weight_energy_waste

    before_waiting_time = sum(before.simulator.current_time - job['subtime'] 
                      for job in before.simulator.jobs_manager.waiting_queue)
    after_waiting_time = sum(after.simulator.current_time - job['subtime'] 
                      for job in after.simulator.jobs_manager.waiting_queue)
    
    waiting_time = after_waiting_time - before_waiting_time
    
    timespan =  after.simulator.current_time - before.simulator.current_time
    
    max_waiting_time = len(before.simulator.jobs_manager.waiting_queue) * timespan
    
    i = len(before.simulator.jobs_manager.waiting_queue)
    
    while i < len(after.simulator.jobs_manager.waiting_queue):
        max_waiting_time += after.simulator.current_time - after.simulator.jobs_manager.waiting_queue[i]['subtime'] 
        i+=1
        
    alpha = 0.5
    beta = 0.5
    
    reward = 0
    waiting_queue = [job for job in after.simulator.jobs_manager.waiting_queue if job not in before.simulator.jobs_manager.waiting_queue]
    waiting_time = len(waiting_queue) * timespan
    # if max_waiting_time == 0:
    #     reward = -alpha * energy_waste
    # else:
    #     reward = -alpha * energy_waste - beta * (waiting_time / max_waiting_time)
    reward = -alpha * energy_waste - beta * waiting_time

    if reward > 0:
        print('here')
    return reward

class RLScheduler:
    def __init__(self, simulator: SPSimulator, node_manager, lr=0.0001):
        self.simulator = simulator
        self.node_manager = node_manager
        self.agent_optimizer = optim.Adam(self.node_manager.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.is_copied_instance = False
        self.count_step = 0
        self.action = {'switch_off': None, 'switch_on': None}
        self.this_step_allocated = []

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

            if action == 0 and node not in self.simulator.node_manager.inactive_resources:
                switch_off.append(node)
            elif action == 1 and node not in self.simulator.node_manager.available_resources:
                switch_on.append(node)
        
        if switch_off:
            print(f'switch off: {switch_off}')
            e = {'type': 'switch_off', 'node': switch_off}
            timestamp = self.simulator.current_time
            self.simulator.jobs_manager.push_event(timestamp, e)
            self.action['switch_off'] = switch_off
        else:
            self.action['switch_off'] = None
            
            
        if switch_on:
            print(f'switch on: {switch_on}')
            e = {'type': 'switch_on', 'node': switch_on}
            timestamp = self.simulator.current_time
            self.simulator.jobs_manager.push_event(timestamp, e)
            self.action['switch_on'] = switch_on
        else:
            self.action['switch_on'] = None
        
        if len(switch_off) == 0 and len(switch_on) == 0:
            print('no action taken')

    def fcfs_schedule(self):

        
        for job in self.simulator.jobs_manager.waiting_queue[:]:
            available_resources, inactive_resources = self.simulator.node_manager.get_not_allocated_resources()
            if self.action['switch_off'] is not None:
                available_resources = list(set(available_resources) - set(self.action['switch_off']))
            if self.action['switch_on'] is not None:
                available_resources = list(set(available_resources) | set(self.action['switch_on']))
            available_resources = [resource for resource in available_resources if resource not in self.this_step_allocated]
            
            if job['res'] <= len(available_resources):
                if job['res'] <= len(available_resources):
                    allocated_nodes = available_resources[:job['res']]
                    self.this_step_allocated += allocated_nodes
                    if self.action['switch_on'] is not None:
                        switch_on_set = set(self.action['switch_on'])
                        need_activation_nodes = [node for node in allocated_nodes if node in switch_on_set]
                        allocated_nodes = [node for node in allocated_nodes if node not in switch_on_set]
                    else:
                        need_activation_nodes = []
                        
                    self.simulator.execution_start(job, allocated_nodes, need_activation_nodes)
            else:
                break

    def backfill_schedule(self):
        p_job = self.simulator.jobs_manager.waiting_queue[0]
            
        backfilling_queue = self.simulator.jobs_manager.waiting_queue[1:]
        
        not_reserved_resources = self.simulator.node_manager.resources_agenda
        next_releases = self.simulator.node_manager.resources_agenda
        
        next_releases = sorted(
            next_releases, 
            key=lambda x: (x['release_time'], x['node'])
        )
        
        if self.action['switch_off'] is not None:
            next_releases = [next_release for next_release in next_releases if next_release['node'] not in self.action['switch_off']]
                
        if self.action['switch_on'] is not None:
            for next_release in next_releases:
                if next_release['node'] in self.action['switch_on'] and next_release['node'] not in self.this_step_allocated:
                    next_release['release_time'] += 5

        if len(next_releases) < p_job['res']:
            return
        
        last_host = next_releases[p_job['res'] - 1]
        p_start_t = last_host['release_time']
                
        candidates = [r['node'] for r in next_releases if r['release_time'] <= p_start_t]
        head_job_reservation = candidates[-p_job['res']:]
        
        not_reserved_resources = [r for r in not_reserved_resources if r not in head_job_reservation]
        
        for job in backfilling_queue:
            available_resources, inactive_resources = self.simulator.node_manager.get_not_allocated_resources()
            if self.action['switch_off'] is not None:
                available_resources = list(set(available_resources) - set(self.action['switch_off']))
            if self.action['switch_on'] is not None:
                available_resources = list(set(available_resources) | set(self.action['switch_on']))
            
            available_resources = [resource for resource in available_resources if resource not in self.this_step_allocated]
            not_allocated_resources = available_resources
            
            not_reserved = [h for h in not_allocated_resources if h not in head_job_reservation]
            
            available_resources_not_reserved = [r for r in available_resources if r in not_reserved]


            if job['res'] <= len(not_reserved):
                if job['res'] <= len(available_resources_not_reserved):
                    allocated_nodes = available_resources_not_reserved[:job['res']]
                    self.this_step_allocated += allocated_nodes
                    if self.action['switch_on'] is not None:
                        switch_on_set = set(self.action['switch_on'])
                        need_activation_nodes = [node for node in allocated_nodes if node in switch_on_set]
                        allocated_nodes = [node for node in allocated_nodes if node not in switch_on_set]
                    else:
                        need_activation_nodes = []
                    
                    self.simulator.execution_start(job, allocated_nodes, need_activation_nodes)

            elif job['walltime'] and job['walltime'] + self.simulator.current_time <= p_start_t and job['res'] <= len(not_allocated_resources):
                if job['res'] <= len(available_resources):
                    allocate_nodes = available_resources[:job['res']]
                    if self.action['switch_on'] is not None:
                        switch_on_set = set(self.action['switch_on'])
                        need_activation_nodes = [node for node in allocated_nodes if node in switch_on_set]
                        allocated_nodes = [node for node in allocated_nodes if node not in switch_on_set]
                        if job['walltime'] + self.simulator.current_time + 5 <= p_start_t:
                            self.this_step_allocated += allocated_nodes + need_activation_nodes
                            self.simulator.execution_start(job, allocate_nodes, need_activation_nodes)
                    else:
                        need_activation_nodes = []
                        self.this_step_allocated += allocated_nodes
                        self.simulator.execution_start(job, allocate_nodes, [])

    
    def easy_schedule(self):
        self.fcfs_schedule()
        if len(self.simulator.jobs_manager.waiting_queue) >= 2:
            self.backfill_schedule()
        
    def schedule(self):
        self.count_step += 1
        
        if not self.is_copied_instance:
            
            print(f'~~~ STEP {self.count_step} ~~~')
            print(f'curr time: {self.simulator.current_time}')
            print(f'aval res: {self.simulator.node_manager.available_resources}')
            print(f'inac res: {self.simulator.node_manager.inactive_resources}')
            print(f'switch off res: {self.simulator.node_manager.on_off_resources}')
            print(f'switch on res: {self.simulator.node_manager.off_on_resources}')
            # print(f'reserved res: {self.simulator.node_manager.reserved_resources}')
            # print(f'agenda res: {self.simulator.node_manager.resources_agenda}')
            # print(f'event: {self.simulator.event}')
            
            
            inactive_resources = self.simulator.node_manager.inactive_resources
            available_resources = self.simulator.node_manager.available_resources
            on_off_resources = self.simulator.node_manager.on_off_resources
            off_on_resources = self.simulator.node_manager.off_on_resources
            reserved_resources = self.simulator.node_manager.reserved_resources
            active_jobs = self.simulator.jobs_manager.active_jobs
            computing_resources = []
            for job in active_jobs:
                computing_resources.extend(job['allocated_resources'])
                
            all_resources = (
                inactive_resources +
                available_resources +
                on_off_resources +
                off_on_resources +
                reserved_resources +
                computing_resources
            )

            # Cek duplikat
            unique_resources = set(all_resources)

            if len(unique_resources) != len(all_resources):
                # Ada duplikat!
                from collections import Counter
                duplicates = [item for item, count in Counter(all_resources).items() if count > 1]
                raise ValueError(f"Duplicate resource IDs detected: {duplicates}")
            self._rl_decision()
        else:
            self.easy_schedule()
        
    def _rl_decision(self):
        nodes = self.simulator.node_manager.get_not_allocated_resources()
        nodes = list(itertools.chain.from_iterable(nodes))
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
        n = 3
        for i in range(n):     
            if post_copy.simulator.jobs_manager.events or post_copy.simulator.jobs_manager.waiting_queue:
                post_copy.simulator.proceed() 
            else:
                break

        state_after = post_copy.get_simulator_state()
        state_after = T.tensor(state_after, dtype=T.float32).unsqueeze(0).unsqueeze(0)

        reward = reward_function(self, post_copy)
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
            len(self.simulator.jobs_manager.waiting_queue),
            self.simulator.jobs_manager.waiting_queue[0]['res'] 
            if len(self.simulator.jobs_manager.waiting_queue) > 0 
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