import json
import pandas as pd
import copy
import numpy as np
from collections import defaultdict

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
            elif self._dict['type'] == 'execution_finished' and other._dict['type'] == 'execution_finished':
                return str(self._dict['id']) < str(other._dict['id'])
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
    
    def __str__(self):
       return f'{self["id"]}'
    
class JobsMonitor:
    def __init__(self, workload_info):
        self.num_jobs = len(workload_info['jobs'])
        self.num_jobs_finished = 0
        self.waiting_queue = []
        self.waiting_queue_ney = []
        self.executed_jobs = []
        self.monitor_jobs = []
        self.active_jobs = []
        self.reserved_count = 0
        self.arrival_count = 0
        self.total_req_res = 0
   
class SimMonitor:
    def __init__(self, nb_res, machines):
        self.energy_consumption = [0] * len(machines)
        self.nodes_action = [{'state': 'idle', 'time': 0} for _ in range(nb_res)]
        self.idle_time = [0] * len(machines)
        self.total_waiting_time = 0
        self.finish_time = 0
        self.nb_res = pd.DataFrame([{
            'time': 0, 'sleeping': 0, 'sleeping_nodes': [], 'switching_on': 0, 'switching_on_nodes': [],
            'switching_off': 0, 'switching_off_nodes': [], 'idle': nb_res, 'idle_nodes': list(range(nb_res)),
            'computing': 0, 'computing_nodes': [], 'unavailable': 0
        }])
        self.nodes = [[{'type': 'idle', 'starting_time': 0, 'finish_time': 0}] for _ in range(nb_res)]
        self.node_state_monitor = [{'idle': 0, 'switching_off': 0, 'switching_on': 0, 'computing': 0, 'sleeping': 0} for _ in range(nb_res)]
        self.energy_waste = [0] * len(machines)
    
    def update_energy_waste(self):
        for node_index, node in enumerate(self.node_state_monitor):
            self.energy_waste[node_index] = (node["idle"] * 190) + (node["switching_off"] * 9) + (node["switching_on"] * 190)
    
    
class SPSimulator:
    def __init__(self, scheduler, platform_path, workload_path, timeout=None):        
        self.scheduler = scheduler
        self.timeout = timeout
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)

        self.nb_res = self.platform_info['nb_res']
        self.machines = self.platform_info['machines']
        self.profiles = self.workload_info['profiles']
        self.transition_time = [self.platform_info['switch_off_time'], self.platform_info['switch_on_time']]

        self.is_finish = False
        self.sim_monitor = SimMonitor(self.nb_res, self.machines)
        self.jobs_monitor = JobsMonitor(self.workload_info)
        
        self.resources_agenda = [{'release_time': 0, 'node': i} for i in range(16)]
        self.current_time = 0
        self.last_event_time = 0
        self.event = None
        self.available_resources = list(range(self.nb_res))
        self.reserved_resources = []
        self.inactive_resources = []
        self.on_off_resources = []
        self.off_on_resources = []
        
        self.events = []
        
        for job in self.workload_info['jobs']:
            job['type'] = 'arrival'
            timestamp = job['subtime']

            self.push_event(timestamp, job)

        self.step_count = 0
        
        self.arrival_count = 0
        self.total_req_res = 0
    
    def update_nb_res(self, current_time, event, _type, nodes):
        mask = self.sim_monitor.nb_res['time'] == current_time
        
        for node_index in nodes:
            node_history = self.sim_monitor.nodes[node_index]
            if node_history[len(node_history)-1]['type'] != _type:
                node_history[len(node_history)-1]['finish_time'] = current_time
                if _type == 'release':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'allocate':
                    node_history.append({'type': 'computing', 'starting_time': current_time, 'finish_time': event['walltime'] + current_time})
                elif _type == 'switch_off':
                    node_history.append({'type': 'switching_off', 'starting_time': current_time, 'finish_time': self.transition_time[0] + current_time})
                elif _type == 'switch_on':
                    node_history.append({'type': 'switching_on', 'starting_time': current_time, 'finish_time': self.transition_time[1] + current_time})
                elif _type == 'turn_off':
                    node_history.append({'type': 'sleeping', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'turn_on':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
                
        if mask.sum() == 0:
            last_row = self.sim_monitor.nb_res.iloc[-1].copy()
            last_row['time'] = current_time
            self.sim_monitor.nb_res = pd.concat([self.sim_monitor.nb_res, last_row.to_frame().T], ignore_index=True)
            mask = self.sim_monitor.nb_res['time'] == current_time
        
        row_idx = self.sim_monitor.nb_res.index[mask].tolist()[0]
        nodes_len = len(nodes)
        
        if _type == 'release':
            self.sim_monitor.nb_res.at[row_idx, 'idle'] += nodes_len
            self.sim_monitor.nb_res.at[row_idx, 'computing'] -= nodes_len
            
            _nodes = self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] = _nodes
            
            self.sim_monitor.nb_res.at[row_idx, 'computing_nodes'] = [
                item for item in self.sim_monitor.nb_res.at[row_idx, 'computing_nodes']
                if item.get('job_id') != event['id']
            ]
        
        elif _type == 'allocate':
            self.sim_monitor.nb_res.at[row_idx, 'computing'] += nodes_len
            self.sim_monitor.nb_res.at[row_idx, 'idle'] -= nodes_len
            
            self.sim_monitor.nb_res.at[row_idx, 'computing_nodes'] += [{'job_id': event['id'], 'nodes': nodes,'starting_time': current_time, 'finish_time': current_time+event['walltime']}]
            
            self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'])
            
        elif _type == 'switch_off':
            self.sim_monitor.nb_res.at[row_idx, 'switching_off'] += nodes_len
            self.sim_monitor.nb_res.at[row_idx, 'idle'] -= nodes_len
            
            _nodes = self.sim_monitor.nb_res.at[row_idx, 'switching_off_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.nb_res.at[row_idx, 'switching_off_nodes'] = _nodes
            
            self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'])
            
        elif _type == 'turn_off':
            self.sim_monitor.nb_res.at[row_idx, 'sleeping'] += nodes_len
            self.sim_monitor.nb_res.at[row_idx, 'switching_off'] -= nodes_len
            
            _nodes = self.sim_monitor.nb_res.at[row_idx, 'sleeping_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.nb_res.at[row_idx, 'sleeping_nodes'] = _nodes
               
            self.sim_monitor.nb_res.at[row_idx, 'switching_off_nodes'] = [item for item in self.sim_monitor.nb_res.at[row_idx, 'switching_off_nodes'] if item not in nodes]
            self.sim_monitor.nb_res.at[row_idx, 'switching_off_nodes'] = sorted(self.sim_monitor.nb_res.at[row_idx, 'switching_off_nodes'])
            
        elif _type == 'switch_on':
            self.sim_monitor.nb_res.at[row_idx, 'switching_on'] += nodes_len
            self.sim_monitor.nb_res.at[row_idx, 'sleeping'] -= nodes_len
            
            _nodes = self.sim_monitor.nb_res.at[row_idx, 'switching_on_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.nb_res.at[row_idx, 'switching_on_nodes'] = _nodes
            
            self.sim_monitor.nb_res.at[row_idx, 'sleeping_nodes'] = [item for item in self.sim_monitor.nb_res.at[row_idx, 'sleeping_nodes'] if item not in nodes]
            self.sim_monitor.nb_res.at[row_idx, 'sleeping_nodes'] = sorted(self.sim_monitor.nb_res.at[row_idx, 'sleeping_nodes'])

        elif _type == 'turn_on':
            self.sim_monitor.nb_res.at[row_idx, 'idle'] += nodes_len
            self.sim_monitor.nb_res.at[row_idx, 'switching_on'] -= nodes_len
            
            _nodes = self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.nb_res.at[row_idx, 'idle_nodes'] = _nodes 
            
            self.sim_monitor.nb_res.at[row_idx, 'switching_on_nodes'] = [item for item in self.sim_monitor.nb_res.at[row_idx, 'switching_on_nodes'] if item not in nodes]
            self.sim_monitor.nb_res.at[row_idx, 'switching_on_nodes'] = sorted(self.sim_monitor.nb_res.at[row_idx, 'switching_on_nodes'])
            
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
            self.sim_monitor.nodes_action[node]['state'] = target_state
            self.sim_monitor.nodes_action[node]['time'] = self.current_time

        self.update_nb_res(self.current_time, event, event_type, allocated)
    
    def update_energy_consumption(self):
        for index, node_action in enumerate(self.sim_monitor.nodes_action):
            state = node_action['state']
            
            state_mapping = {
                'sleeping': 0,
                'idle': 1,
                'computing': 2,
                'switching_on': 3,
                'switching_off': 4
            }
            
            rate_energy_consumption = self.machines[index]['wattage_per_state'][state_mapping[state]]

            last_time = max(node_action['time'], self.last_event_time)
            duration = self.current_time - last_time  

            self.sim_monitor.energy_consumption[index] += duration * rate_energy_consumption
        
        
    def update_node_state_monitor(self):
        for index, node_action in enumerate(self.sim_monitor.nodes_action):
            state = node_action['state']
            last_time = max(node_action['time'], self.last_event_time) 
            duration = self.current_time - last_time  

            self.sim_monitor.node_state_monitor[index][state] += duration
            
    def update_idle_time(self):
        for index, node_action in enumerate(self.sim_monitor.nodes_action):
            if node_action['state'] == 'idle':
                last_time = max(node_action['time'], self.last_event_time) 
        
                self.sim_monitor.idle_time[index] += self.current_time - last_time
                 
    def update_total_waiting_time(self):
        num_job_in_queue = len(self.jobs_monitor.waiting_queue) + len(self.jobs_monitor.waiting_queue_ney)
        self.sim_monitor.total_waiting_time += num_job_in_queue * (self.current_time - self.last_event_time )
        
    def find_grouped_resources(self, resources, count):
        resources = sorted(resources)
        for i in range(len(resources) - count + 1):
            if resources[i + count - 1] - resources[i] == count - 1:
                return resources[i:i + count]
        return resources[:count]
    
    def print_energy_consumption(self):
        index = 0
        sum = 0
        for node_energy_consumption in self.sim_monitor.energy_consumption:
            print(f'Energy consumption of node {index}: ', node_energy_consumption)
            sum+=node_energy_consumption
            index += 1
        print(f'Total energy consumption: {sum}')    
    
    def start_simulator(self, timeout = None):
        if timeout is not None:
            e = {'type': 'switch_off', 'node': copy.deepcopy(self.available_resources)}
            ts = self.current_time + self.timeout
            self.push_event(ts, e)
    
    def switch_off(self, node):
        valid_switch_off = [item for item in node if item in self.available_resources]
        
        if len(valid_switch_off) == 0:
            return
        
        self.available_resources = [item for item in self.available_resources if item not in valid_switch_off]
        self.on_off_resources.extend(valid_switch_off)
        self.on_off_resources = sorted(self.on_off_resources)
        self.update_node_action(valid_switch_off, self.event, 'switch_off', 'switching_off')
        
        # for i in range(16):
        #     if i in node:
        #         self.resources_agenda[i]['release_time'] = self.current_time + self.transition_time[0]
                
        ts = self.current_time + self.transition_time[0]
        e = {'type': 'turn_off', 'node': copy.deepcopy(valid_switch_off)}
        self.push_event(ts, e)
    
    def turn_off(self, node):
        self.inactive_resources.extend(node)
        self.inactive_resources = sorted(self.inactive_resources)
                
        self.on_off_resources = [item for item in self.on_off_resources if item not in node]
        self.on_off_resources = sorted(self.on_off_resources)
        
        # for i in range(16):
        #     if i in node:
        #         self.resources_agenda[i]['release_time'] = 0
                
        self.update_node_action(node, self.event, 'turn_off', 'sleeping')
    
    def switch_on(self, node, update_resources_agenda = True):
        self.inactive_resources = [item for item in self.inactive_resources if item not in node]
        self.off_on_resources.extend(node)
        self.off_on_resources = sorted(self.off_on_resources)
        self.update_node_action(node, self.event, 'switch_on', 'switching_on')
        
        # if update_resources_agenda:
        #     for i in range(16):
        #         if i in node:
        #             self.resources_agenda[i]['release_time'] = self.current_time + self.transition_time[1]
        
        ts = self.current_time + self.transition_time[1]
        e = {'type': 'turn_on', 'node': copy.deepcopy(node)}
        self.push_event(ts, e)
    
    def turn_on(self, node):
        self.available_resources.extend(node)
        self.available_resources = sorted(self.available_resources)
        self.off_on_resources = [item for item in self.off_on_resources if item not in node]
        self.off_on_resources = sorted(self.off_on_resources)
        
        # for i in range(16):
        #     if i in node:
        #         self.resources_agenda[i]['release_time'] = 0
                
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
    
    def execution_start(self, job, reserved_node, need_activation_node):
        if len(reserved_node) + len(need_activation_node) < job['res']:
            print('not enough res')
            return
        
        self.reserved_resources += reserved_node
        self.reserved_resources = sorted(self.reserved_resources)
        job['reserved_nodes'] = reserved_node
        job['type'] = 'execution_start'

        for index, _job in enumerate(self.jobs_monitor.waiting_queue): 
            if _job['id'] == job['id']:
                self.jobs_monitor.waiting_queue.pop(index)
                break
        
        self.jobs_monitor.waiting_queue_ney.append(job)
        
        if job['id'] == 21:
            print('here')
        if len(need_activation_node) > 0:
            for i in range(16):
                if i in need_activation_node or i in reserved_node:
                    self.resources_agenda[i]['release_time'] = self.current_time + self.transition_time[1] + job['walltime']
            self.switch_on(need_activation_node, update_resources_agenda=False)
            ts = self.current_time + self.transition_time[1]
            self.push_event(ts, job)

        else:
            for i in range(16):
                if i in need_activation_node or i in reserved_node:
                    self.resources_agenda[i]['release_time'] = self.current_time + job['walltime']
            
            self.push_event(self.current_time, job)
            
    def get_not_allocated_resources(self):
        not_allocated_resources = self.available_resources + self.inactive_resources
        not_allocated_resources = [resource for resource in not_allocated_resources if resource not in self.reserved_resources]
        not_allocated_resources = sorted(not_allocated_resources)
        return not_allocated_resources
    
    def push_event(self, timestamp, event):
        found = None
        for x in self.events:
            if x['timestamp'] == timestamp:
                found = x
                break
        if found:
            found['events'].append(event)
        else:
            self.events.append({'timestamp':timestamp, 'events':[event]})
        
        self.events.sort(key=lambda x: x['timestamp'])
    
    def proceed(self):
        self.current_time, events = self.events.pop(0).values()
        self.update_energy_consumption()
        self.update_node_state_monitor()
        self.update_idle_time()
        self.update_total_waiting_time()
        self.sim_monitor.update_energy_waste()
        self.last_event_time = self.current_time
        for event in events:
            self.event = event
            if self.event['type'] == 'switch_off':
                self.switch_off(self.event['node'])

            elif self.event['type'] == 'turn_off':
                self.turn_off(self.event['node'])
                
            elif self.event['type'] == 'switch_on':
                self.switch_on(self.event['node'])
                
            elif self.event['type'] == 'turn_on':
                self.turn_on(self.event['node'])
                
            elif self.event['type'] == 'arrival':
                self.jobs_monitor.waiting_queue.append(self.event)
                    
            elif self.event['type'] == 'execution_start':
                new_waiting_queue_ney = []
                for d in self.jobs_monitor.waiting_queue_ney:
                    if d['id'] != self.event['id']:
                        new_waiting_queue_ney.append(d)
                        
                self.jobs_monitor.waiting_queue_ney = new_waiting_queue_ney
        
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
                
                self.push_event(finish_time, finish_event)    
                
                finish_event['finish_time'] = finish_time
                self.jobs_monitor.active_jobs.append(finish_event)
                
                self.jobs_monitor.monitor_jobs.append({
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
                self.jobs_monitor.num_jobs_finished += 1
                
                if self.jobs_monitor.num_jobs_finished == self.jobs_monitor.num_jobs:
                    self.sim_monitor.finish_time = self.current_time
                allocated = self.event['allocated_resources']
                for i in range(16):
                    if i in allocated:
                        self.resources_agenda[i]['release_time'] = 0
                        
                self.available_resources.extend(allocated)
                self.available_resources.sort()
                
                self.update_node_action(allocated, self.event, 'release', 'idle')
                
                self.jobs_monitor.active_jobs = [active_job for active_job in self.jobs_monitor.active_jobs if active_job['id'] != self.event['id']]
        
        mask = self.sim_monitor.nb_res['time'] == self.current_time
        has_idle = (self.sim_monitor.nb_res.loc[mask, 'idle'] > 0).any()
        
        self.scheduler.schedule()
        
        if has_idle and self.timeout is not None:
            e = {'type': 'switch_off', 'node': copy.deepcopy(self.available_resources)}
            ts = self.current_time + self.timeout
            self.push_event(ts, e)

        
        if len(self.jobs_monitor.waiting_queue) == 0 and len(self.events) == 0:
            for x in self.sim_monitor.nodes:
                if x[len(x)-1]['finish_time'] != self.current_time:
                    x[len(x)-1]['finish_time'] = self.current_time
            
            self.on_finish()
      
    def on_finish(self):
        self.print_energy_consumption()

        node_state_durations = []
        for node_list in self.sim_monitor.nodes:
            state_durations = defaultdict(float) 
            for entry in node_list:
                state = entry['type']
                duration = entry['finish_time'] - entry['starting_time']
                state_durations[state] += duration
            node_state_durations.append(dict(state_durations))
        
        for i, times in enumerate(node_state_durations):
            print(f"Node {i}: {times}")
            
        
        