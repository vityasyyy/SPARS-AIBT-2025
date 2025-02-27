import json
import heapq
from sched import scheduler
import pandas as pd
import copy
import numpy as np
import torch.optim as optim  # Import the optim module


class MyDict:
    def __init__(self, _dict: dict):
        if isinstance(_dict, MyDict):
            self._dict = copy.deepcopy(_dict._dict)
        else:
            self._dict = _dict
        
    def __lt__(self, other):
        priority_type = {'turn_on', 'turn_off', 'switch_on', 'switch_off'}

        if self._dict['type'] in priority_type:
            return True
        elif other._dict['type'] in priority_type:
            return False
        elif self._dict['type'] not in priority_type and other._dict['type'] in priority_type:
            return True
       
        
        if 'current_time' in self._dict and 'current_time' in other._dict:
            if self._dict['current_time'] < other._dict['current_time']:
                return True
            elif self._dict['current_time'] > other._dict['current_time']:
                return False
        
        if self._dict['type'] != 'execution_finished' and self._dict['type'] != 'execution_start':
            return self._dict['id'] < other._dict['id']
        else:
            if self._dict['type'] == 'execution_finished' and other._dict['type'] == 'execution_start':
                return True
            elif self._dict['type'] == 'execution_start' and other._dict['type'] == 'execution_finished':
                return False
            elif self._dict['type'] == 'pre_switch_on_check':
                return False
            elif other._dict['type'] == 'pre_switch_on_check':
                return True
            else:
                return self._dict['id'] < other._dict['id']
                
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value
        
    def has_key(self, key):
        return key in self._dict
    
class SPSimulator:
    def __init__(self, model, platform_path="platforms/spsim/platform.json", workload_path="workloads/simple_data_100.json"):        
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
            
        self.nb_res = self.platform_info['nb_res']
        self.machines = self.platform_info['machines']
        self.profiles = self.workload_info['profiles']
        self.transition_time = [self.platform_info['switch_off_time'], self.platform_info['switch_on_time']]
        self.model = model
        if self.model is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        
        self.jobs = []
        for job in self.workload_info['jobs']:
            job['type'] = 'arrival'
            
            heapq.heappush(self.jobs, (float(job['subtime']), MyDict(job)))
        
        self.sim_monitor = {
            "energy_consumption": [0] * len(self.machines),
            "nodes_action": [{'state': 'idle', 'time': 0} for _ in range(self.nb_res)],
            "total_idle_time": [0] * len(self.machines),
            'avg_waiting_time': 0,
            'waiting_event_count': 0,
            'finish_time': 0,
            'nb_res': pd.DataFrame([{'time': 0, 
                                'sleeping': 0, 
                                'sleeping_nodes': [], 
                                'switching_on': 0, 
                                'switching_on_nodes':[], 
                                'switching_off': 0,
                                'switching_off_nodes': [],  
                                'idle': 16,
                                'idle_nodes': list(range(self.nb_res)),
                                'computing': 0, 
                                'computing_nodes': [],
                                'unavailable': 0}]),
            'nodes': [[{'type': 'idle', 'starting_time': 0, 'finish_time': 0}] for _ in range(16)]

        }
        
        #EDIT HERE
        self.current_time = 0
        self.event = None
        self.available_resources = list(range(self.nb_res))
        self.reserved_resources = []
        self.inactive_resources = []
        self.on_off_resources = []
        self.off_on_resources = []
        self.schedule_queue = []
        self.waiting_queue = []
        self.waiting_queue_ney = []
        self.executed_jobs = []
        self.monitor_jobs=[]
        self.active_jobs = []
        self.reserved_count = 0
        self.step_count = 0
        
        self.arrival_count = 0
        self.total_req_res = 0
    
    def update_nb_res(self, current_time, event, _type, nodes):
        mask = self.sim_monitor['nb_res']['time'] == current_time
        
        for node_index in nodes:
            node_history = self.sim_monitor['nodes'][node_index]
            if node_history[len(node_history)-1]['type'] != _type:
                node_history[len(node_history)-1]['finish_time'] = current_time
                if _type == 'release':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'allocate':
                    node_history.append({'type': 'computing', 'starting_time': current_time, 'finish_time': event['walltime'] + current_time})
                elif _type == 'switch_off':
                    node_history.append({'type': 'switching_off', 'starting_time': current_time, 'finish_time': 1 + current_time})
                elif _type == 'switch_on':
                    node_history.append({'type': 'switching_on', 'starting_time': current_time, 'finish_time': 1 + current_time})
                elif _type == 'turn_off':
                    node_history.append({'type': 'sleeping', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'turn_on':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
                
        if mask.sum() == 0:
            last_row = self.sim_monitor['nb_res'].iloc[-1].copy()
            last_row['time'] = current_time
            self.sim_monitor['nb_res'] = pd.concat([self.sim_monitor['nb_res'], last_row.to_frame().T], ignore_index=True)
            mask = self.sim_monitor['nb_res']['time'] == current_time
        
        row_idx = self.sim_monitor['nb_res'].index[mask].tolist()[0]
        nodes_len = len(nodes)
        
        if _type == 'release':
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'computing'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = _nodes
            
            self.sim_monitor['nb_res'].at[row_idx, 'computing_nodes'] = [
                item for item in self.sim_monitor['nb_res'].at[row_idx, 'computing_nodes']
                if item.get('job_id') != event['id']
            ]
        
        elif _type == 'allocate':
            self.sim_monitor['nb_res'].at[row_idx, 'computing'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] -= nodes_len
            
            self.sim_monitor['nb_res'].at[row_idx, 'computing_nodes'] += [{'job_id': event['id'], 'nodes': nodes,'starting_time': current_time, 'finish_time': current_time+event['walltime']}]
            
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'])
            
        elif _type == 'switch_off':
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] = _nodes
            
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'])
            
        elif _type == 'turn_off':
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] = _nodes
               
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'])
            
        elif _type == 'switch_on':
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] = _nodes
            
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'])

        elif _type == 'turn_on':
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = _nodes 
            
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'])
            
    def check_backfilling(self, current_time, event, temp_available_resources, active_jobs, next_job, backfilled_node_count):
        temp_aval_res = temp_available_resources
        
        estimated_finish_time = current_time + event['walltime']
        last_job_active_job_finish_time_that_required_to_be_released = np.inf
        active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])

        for active_job in active_jobs:
            temp_available_resources += active_job['res']
            if temp_available_resources >= next_job['res']:
                last_job_active_job_finish_time_that_required_to_be_released = active_job['finish_time']
                break
            
        for active_job in active_jobs:
            if active_job['finish_time'] <= last_job_active_job_finish_time_that_required_to_be_released:
                temp_aval_res += active_job['res']

        return estimated_finish_time < last_job_active_job_finish_time_that_required_to_be_released or temp_aval_res >= event['res'] + next_job['res']
    
    def update_node_action(self, allocated, event, event_type, target_state):
        for node in allocated:
            self.sim_monitor['nodes_action'][node]['state'] = target_state
            self.sim_monitor['nodes_action'][node]['time'] = self.current_time

        self.update_nb_res(self.current_time, event, event_type, allocated)
    
    def update_energy_consumption(self):
        temp_index = 0
        for node_action in self.sim_monitor['nodes_action']:
            if node_action['state'] == 'sleeping':
                rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][0]
            elif node_action['state'] == 'idle':
                rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][1]
            elif node_action['state'] == 'computing':
                rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][2]
            elif node_action['state'] == 'switching_on':
                rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][3]
            elif node_action['state'] == 'switching_off':
                rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][4]
            
            duration = self.current_time - node_action['time']
            self.sim_monitor['energy_consumption'][temp_index] += (duration * rate_energy_consumption)
            
            self.sim_monitor['nodes_action'][temp_index]['time'] = self.current_time
        
            temp_index +=1
            
    def find_grouped_resources(self, resources, count):
        resources = sorted(resources)
        for i in range(len(resources) - count + 1):
            if resources[i + count - 1] - resources[i] == count - 1:
                return resources[i:i + count]
        return resources[:count]
    
    def print_energy_consumption(self):
        index = 0
        sum = 0
        for node_energy_consumption in self.sim_monitor['energy_consumption']:
            print(f'Energy consumption of node {index}: ', node_energy_consumption)
            sum+=node_energy_consumption
            index += 1
        print(f'Total energy consumption: {sum}')    
    
    def start_simulator(self, timeout = None):
        if timeout is not None:
            heapq.heappush(self.schedule_queue, (timeout, MyDict({'node': copy.deepcopy(self.available_resources), 'type': 'switch_off'})))
    
    
    def switch_off(self, node):
        valid_switch_off = [item for item in node if item in self.available_resources]
        
        if len(valid_switch_off) == 0:
            return
        
        self.available_resources = [item for item in self.available_resources if item not in valid_switch_off]
        self.on_off_resources.extend(valid_switch_off)
        self.on_off_resources = sorted(self.on_off_resources)
        self.update_node_action(valid_switch_off, self.event, 'switch_off', 'switching_off')
        
        heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[0], MyDict({'node': copy.deepcopy(valid_switch_off), 'type': 'turn_off' })))
    
    def turn_off(self, node):
        self.inactive_resources.extend(node)
        self.inactive_resources = sorted(self.inactive_resources)
        
        self.on_off_resources = [item for item in self.on_off_resources if item not in node]
        self.on_off_resources = sorted(self.on_off_resources)
        self.update_node_action(node, self.event, 'turn_off', 'sleeping')
    
    def switch_on(self, node):
        self.inactive_resources = [item for item in self.inactive_resources if item not in node]
        self.off_on_resources.extend(node)
        self.off_on_resources = sorted(self.off_on_resources)
        self.update_node_action(node, self.event, 'switch_on', 'switching_on')
        
        heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict({'node': copy.deepcopy(node), 'type': 'turn_on' })))
    
    def turn_on(self, node):
        self.available_resources.extend(node)
        self.available_resources = sorted(self.available_resources)
        self.off_on_resources = [item for item in self.off_on_resources if item not in node]
        self.off_on_resources = sorted(self.off_on_resources)
        self.update_node_action(node, self.event, 'turn_on', 'idle')
        
    def prioritize_lowest_node(self, num_needed_res):
        reserved_node = []
        need_activation_node = []
        
        for node_index in range(self.nb_res):
            if (node_index in self.available_resources or node_index in self.inactive_resources) and node_index not in self.reserved_resources:
                reserved_node.append(node_index)
        
                if node_index in self.inactive_resources:
                    need_activation_node.append(node_index)
                if len(reserved_node) == num_needed_res:
                    return reserved_node, need_activation_node
            
        return reserved_node, need_activation_node
        
    def proceed(self, timeout = None):
        if self.schedule_queue:
            self.current_time, self.event = heapq.heappop(self.schedule_queue)
        else:
            self.event = self.waiting_queue.pop(0)

        self.update_energy_consumption()
      
        if self.event['type'] == 'switch_off':
            self.switch_off(self.event['node'])
            
        elif self.event['type'] == 'turn_off':
            self.turn_off(self.event['node'])
            
        elif self.event['type'] == 'switch_on':
            self.switch_on(self.event['node'])
            
        elif self.event['type'] == 'turn_on':
            self.turn_on(self.event['node'])
            
        elif self.event['type'] == 'arrival':
            if len(self.available_resources) + len(self.inactive_resources) >= self.event['res']:
                if self.waiting_queue:
                    self.active_jobs = sorted(self.active_jobs, key=lambda x: x['finish_time'])
                    backfilled_node_count = 0
                    if self.check_backfilling(self.current_time, self.event, len(self.available_resources) + len(self.inactive_resources), self.active_jobs, self.waiting_queue[0], backfilled_node_count):
                        
                        reserved_node, need_activation_node = self.prioritize_lowest_node(self.event['res'])
                        
                        self.event['type'] = 'execution_start'
                        self.event['reserved_nodes'] = reserved_node
                        self.reserved_resources += reserved_node
                        self.reserved_resources = sorted(self.reserved_resources)
                        
                        self.waiting_queue_ney.append(self.event)
                        self.waiting_queue_ney = sorted(self.waiting_queue_ney)
                        
                        if len(need_activation_node) > 0:
                            heapq.heappush(self.schedule_queue, (self.current_time, MyDict({'type': 'switch_on', 'node': need_activation_node})))
                            heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict(self.event)))
                        else:
                            heapq.heappush(self.schedule_queue, (self.current_time, MyDict(self.event)))
                    else:
                        
                        self.event['current_time'] = self.current_time
                        self.waiting_queue.append(self.event)
                else:        
                    reserved_node, need_activation_node = self.prioritize_lowest_node(self.event['res'])
                    
                    self.event['type'] = 'execution_start'
                    self.event['reserved_nodes'] = reserved_node
                    self.reserved_resources += reserved_node
                    self.reserved_resources = sorted(self.reserved_resources)
                    
                    self.waiting_queue_ney.append(self.event)
                    self.waiting_queue_ney = sorted(self.waiting_queue_ney)
                    
                    if len(need_activation_node) > 0:
                        heapq.heappush(self.schedule_queue, (self.current_time, MyDict({'type': 'switch_on', 'node': need_activation_node})))
                        heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict(self.event)))
                    else:
                        heapq.heappush(self.schedule_queue, (self.current_time, MyDict(self.event)))
                        
            else:
                self.event['current_time'] = self.current_time
                self.waiting_queue.append(self.event)
                
        elif self.event['type'] == 'execution_start': 
            new_waiting_queue_ney = []
            for d in self.waiting_queue_ney:
                if d['id'] != self.event['id']:
                    new_waiting_queue_ney.append(d)
            self.waiting_queue_ney = new_waiting_queue_ney
            
            allocated = self.event['reserved_nodes']
            self.reserved_resources = [node for node in self.reserved_resources if node not in allocated]
            self.available_resources = [node for node in self.available_resources if node not in allocated]
            self.update_node_action(allocated, self.event, 'allocate', 'computing')

            finish_time = self.current_time + self.event['walltime']
            finish_event = {
                'id': self.event['id'],
                'res': self.event['res'],
                'walltime': self.event['walltime'],
                'type': 'execution_finished',
                'subtime': self.event['subtime'],
                'profile': self.event['profile'],
                'allocated_resources': allocated
            }
            
            if finish_event['subtime'] != self.current_time:
                self.sim_monitor['avg_waiting_time'] += (self.current_time - finish_event['subtime'])
                self.sim_monitor['waiting_event_count'] += 1
                
            heapq.heappush(self.schedule_queue, (finish_time, MyDict(finish_event)))
            
            finish_event['finish_time'] = finish_time
            self.active_jobs.append(finish_event)
            
            self.monitor_jobs.append({
                'job_id': self.event['id'],
                'workload_name': 'w0',
                'profile': self.event['profile'],
                'submission_time': self.event['subtime'],
                'requested_number_of_resources': self.event['res'],
                'requested_time': self.event['walltime'],
                'success': 0,
                'final_state': 'COMPLETED_WALLTIME_REACHED',
                'starting_time': self.current_time,
                'execution_time': self.event['walltime'],
                'finish_time': finish_time,
                'waiting_time': self.current_time - self.event['subtime'],
                'turnaround_time': finish_time - self.event['subtime'],
                'stretch': (finish_time - self.event['subtime']) / self.event['walltime'],
                'allocated_resources': allocated,
                'consumed_energy': -1
            })
        
        elif self.event['type'] == 'execution_finished':
            allocated = self.event['allocated_resources']
            self.available_resources.extend(allocated)
            self.available_resources.sort() 
            
            self.active_jobs = [active_job for active_job in self.active_jobs if active_job['id'] != self.event['id']]
            
            self.update_node_action(allocated, self.event, 'release', 'idle')
            
            skipbf = False
            for aj in self.active_jobs:
                if aj['finish_time'] == self.current_time:
                    skipbf=True
                    
            if skipbf == True:
                return
            
            reserved_count = 0
            for job in self.waiting_queue_ney:
                reserved_count += job['res']
                
            temp_available_resource = len(self.available_resources) + len(self.inactive_resources) + len(self.off_on_resources) - reserved_count

            self.waiting_queue = sorted(self.waiting_queue)

            switch_on_nodes = []
            
            for _ in range(len(self.waiting_queue)):
                job = self.waiting_queue[0]
                if temp_available_resource >= job['res']:
                    next_job = self.waiting_queue.pop(0)
                    next_job['type'] = 'execution_start'
                    temp_available_resource -= next_job['res']
                    
                    reserved_node, need_activation_node = self.prioritize_lowest_node(next_job['res'])
                    next_job['reserved_nodes'] = reserved_node
                    self.reserved_resources += reserved_node
                    self.reserved_resources = sorted(self.reserved_resources)
                    switch_on_nodes += need_activation_node
                    self.waiting_queue_ney.append(next_job)
                    self.waiting_queue_ney = sorted(self.waiting_queue_ney)
                    if len(need_activation_node) > 0:
                        heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict(next_job)))
                    else:
                        heapq.heappush(self.schedule_queue, (self.current_time, MyDict(next_job)))
                else:
                    break

            self.active_jobs = sorted(self.active_jobs, key=lambda x: x['finish_time'])
            backfilled_node_count = 0
            while True:
                is_pushed = False
                for k in range(0, len(self.waiting_queue)):
                    job = self.waiting_queue[k]
                    if temp_available_resource >= job['res']:
                        if not self.check_backfilling(self.current_time, job, temp_available_resource, self.active_jobs, self.waiting_queue[0], backfilled_node_count):
                            continue
                        backfilled_node_count += job['res']
                        next_job = self.waiting_queue.pop(k)
                        next_job['type'] = 'execution_start'
                        temp_available_resource -= next_job['res']
                        
                        reserved_node, need_activation_node = self.prioritize_lowest_node(next_job['res'])
                        next_job['reserved_nodes'] = reserved_node
                        switch_on_nodes += need_activation_node
                        self.reserved_resources += reserved_node
                        self.reserved_resources = sorted(self.reserved_resources)
                        self.waiting_queue_ney.append(next_job)
                        self.waiting_queue_ney = sorted(self.waiting_queue_ney)
                        
                        if len(need_activation_node) > 0:
                            heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict(next_job)))
                        else:
                            heapq.heappush(self.schedule_queue, (self.current_time, MyDict(next_job)))
                            
                        is_pushed = True
                        break
                if is_pushed == False:
                    break
            
            switch_on_nodes = sorted(switch_on_nodes)
            
            if len(switch_on_nodes) > 0:
                heapq.heappush(self.schedule_queue, (self.current_time, MyDict({'type': 'switch_on', 'node': copy.deepcopy(switch_on_nodes)})))    

        mask = self.sim_monitor['nb_res']['time'] == self.current_time
        has_idle = (self.sim_monitor['nb_res'].loc[mask, 'idle'] > 0).any()
        if has_idle:
            heapq.heappush(self.schedule_queue, (self.current_time + timeout, MyDict({'type':'switch_off', 'node': copy.deepcopy(self.available_resources)})))
        
        if len(self.waiting_queue) == 0 and len(self.schedule_queue) == 0:
            for x in self.sim_monitor['nodes']:
                if x[len(x)-1]['finish_time'] != self.current_time:
                    x[len(x)-1]['finish_time'] = self.current_time