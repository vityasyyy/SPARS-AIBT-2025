
import copy
from re import I
from HPCv3.RJMS.Algo.BaseAlgorithm import BaseAlgorithm
class SmartFCFS(BaseAlgorithm):

    def schedule(self):
        super().prep_schedule()
        self.smart_fcfs_switch_on()
        self.fcfs()
        self.add_call_me_later()
        self.switch_off_nodes()    
        
        return self.events
    
    def smart_fcfs_switch_on(self):
        self.fcfs_predicted = []
        self.future_free_nodes = []
        self.future_free_nodes_need_activation = []
        
        for agenda in self.resources_agenda:
            node_index = agenda['node_id']
            release_time = agenda['release_time']
            current_time = self.current_time

            # First condition: node is ready to be freed now
            if release_time <= current_time:
                if node_index not in self.ResourceManager.reserved_nodes:
                    switching_on = self.ResourceManager.get_switching_on_nodes()
                    switching_off = self.ResourceManager.get_switching_off_nodes()
                        
                    if node_index in self.inactive:
                        self.future_free_nodes_need_activation.append(node_index)
                        self.future_free_nodes.append(node_index)
                    elif (node_index in self.available):
                        self.future_free_nodes.append(node_index)
                    elif (node_index in switching_on and release_time == current_time):
                        self.future_free_nodes.append(node_index)
                    elif (node_index not in switching_off):
                        self.future_free_nodes.append(node_index)

        self.future_free_nodes.sort(key=lambda node: node in self.future_free_nodes_need_activation)
        need_activation_node = []
        
        waiting_queue = [job for job in self.JobsManager.waiting_queue if job['job_id'] not in self.scheduled]
        for job in waiting_queue:
            if job['res'] <= len(self.future_free_nodes):
                
                allocated_resource = self.future_free_nodes[:job['res']]
                self.future_free_nodes = self.future_free_nodes[job['res']:]
                for node in allocated_resource:
                    if node in self.future_free_nodes_need_activation:
                        need_activation_node.append(node)
                        self.future_free_nodes_need_activation.remove(node)
                self.fcfs_predicted.append({'id': job['job_id'], 'walltime': job['walltime'], 'resources': allocated_resource})
            else:
                break
            
        if len(need_activation_node) > 0:
            super().push_event(self.current_time, {'type': 'switch_on', 'node': need_activation_node})
                
    def add_call_me_later(self):
        if len(self.available) > 0 and self.timeout is not None:
            e = {'type': 'call_me_later'}
            timestamp = self.current_time + self.timeout
            super().push_event(timestamp, e)
        
    def switch_off_nodes(self):
        switch_off_nodes = []
        if self.current_time == 30:
            print('here')
        for node_index, node in enumerate(self.ResourceManager.nodes):
            if node['state'] == 'active' and node['job_id'] == None and node['duration'] >= self.timeout:
                switch_off_nodes.append(node_index)
        
        waiting_queue = [job for job in self.JobsManager.waiting_queue if job['job_id'] not in self.scheduled]
        valid_switch_off_nodes = []
        next_releases = copy.deepcopy(self.resources_agenda)
        if len(waiting_queue) > 0:
            for job in waiting_queue:
                
                if job['res'] <= len(next_releases):
                    last_host = next_releases[job['res'] - 1]
                    job_prediction_start = last_host['release_time']
                    job_candidates = [r['node_id'] for r in next_releases if r['release_time'] <= job_prediction_start]
                    job_reservation = job_candidates[-job['res']:]
                    
                    next_releases = [nr for nr in next_releases if nr['node_id'] not in job_reservation]

                    # Pre-index machines by id for quick lookup
                    machine_map = {m['id']: m for m in self.ResourceManager.machines}

                    for node in switch_off_nodes:
                        machine = machine_map.get(node)

                        # Extract transition times safely
                        switch_off_time = next(
                            (t['transition_time']
                            for t in machine['states']['switching_off']['transitions']
                            if t['state'] == 'sleeping'),
                            None
                        )

                        switch_on_time = next(
                            (t['transition_time']
                            for t in machine['states']['switching_on']['transitions']
                            if t['state'] == 'active'),
                            None
                        )

                        # If we don't have both times, skip
                        if switch_off_time is None or switch_on_time is None:
                            raise RuntimeError(f"Node {node} Missing switch_off_time or switch_on_time")

                        # Decision check
                        if job_prediction_start > self.current_time + switch_off_time + switch_on_time:
                            valid_switch_off_nodes.append(node)
                        
                        if len(valid_switch_off_nodes) == 0:
                            print('debug')
                    switch_off_nodes = valid_switch_off_nodes
                    valid_switch_off_nodes = []
                else:
                    break
                
        if len(switch_off_nodes) > 0:
            e = {'type': 'switch_off', 'nodes': switch_off_nodes}
            timestamp = self.current_time
            super().push_event(timestamp, e)
            
    def fcfs(self):
        waiting_queue = [job for job in self.JobsManager.waiting_queue if job['job_id'] not in self.scheduled]
        for job in waiting_queue:
            if job['res'] <= len(self.available) + len(self.inactive):
                if job['res'] <= len(self.available):
                    allocated_nodes = self.available[:job['res']]
                    event = {
                        'job_id': job['job_id'],
                        'walltime': job['walltime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': allocated_nodes
                    }
                    self.available = self.available[job['res']:]
                    self.allocated.extend(allocated_nodes)
                    compute_demand = job['walltime'] * job['res']
                    compute_power = sum(self.compute_speeds[i] for i in allocated_nodes)
                    finish_time = self.current_time + (compute_demand / compute_power)
                    for node in self.resources_agenda:
                        if node['node_id'] in allocated_nodes:
                            node['release_time'] = finish_time
                    self.scheduled.append(job['job_id'])
                    super().push_event(self.current_time, event)
            else:
                break