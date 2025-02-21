import json
import heapq
import pandas as pd
import copy
import numpy as np

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
    def __init__(self, platform_path="platforms/spsim/platform.json", workload_path="workloads/simple_data_100.json"):        
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
            
        self.nb_res = self.platform_info['nb_res']
        self.machines = self.platform_info['machines']
        self.profiles = self.workload_info['profiles']
        self.transition_time = [self.platform_info['switch_off_time'], self.platform_info['switch_on_time']]
        
        self.jobs = []
        for job in self.workload_info['jobs']:
            job['type'] = 'arrival'
            
            heapq.heappush(self.jobs, (float(job['subtime']), MyDict(job)))
        
        self.sim_monitor = {
            "energy_consumption": [0] * len(self.machines),
            "start_idle": [0] * len(self.machines),
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
            'nodes': [
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}]
                    ]
        }
    
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
    
    def find_grouped_resources(self, resources, count):
        resources = sorted(resources)
        for i in range(len(resources) - count + 1):
            if resources[i + count - 1] - resources[i] == count - 1:
                return resources[i:i + count]
        return resources[:count]