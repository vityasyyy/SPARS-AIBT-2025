import copy
import torch as T
import torch.nn as nn
import torch.optim as optim

from scheduler_batsim.backfilling import EASYScheduler
from .easy_scheduler import FCFSScheduler

def reward_function(before, after):
    weight_energy_waste = 1
    num_idle_res_after_action = len(after.simulator.available_resources)
    energy_waste = num_idle_res_after_action * weight_energy_waste

    waiting_time = sum(after.simulator.current_time - job['subtime'] 
                      for job in after.simulator.jobs_monitor.waiting_queue)
    
    alpha = 0.1
    beta = 0.9
    return -alpha * energy_waste - beta * waiting_time

class RLScheduler(EASYScheduler):
    def __init__(self, node_manager, lr=0.0001):
        self.is_copied_instance = False
        self.count_step = 0
        
    def schedule(self):
        self.count_step += 1
        
        if not self.is_copied_instance:
            print(f'~~~ STEP {self.count_step} ~~~')
            print(f'curr time: {self.simulator.current_time}')
            print(f'aval res: {self.simulator.available_resources}')
            print(f'inac res: {self.simulator.inactive_resources}')
            print(f'event: {self.simulator.event}')
            self.rl_decision()
        else:
            super().schedule()
            
        
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
        