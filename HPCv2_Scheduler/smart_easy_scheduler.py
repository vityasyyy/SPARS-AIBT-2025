from scheduler_batsim import fcfs
from .smart_fcfs_scheduler import SmartFCFSScheduler

class SmartEasyScheduler(SmartFCFSScheduler):
    def schedule(self):
        super().smart_fcfs_switch_on()
        super().fcfs()
        if len(self.simulator.jobs_manager.waiting_queue) >= 2:
            self.smart_easy_switch_on()
            self.backfill()
        
        super().add_call_me_later()
        super().switch_off_nodes()
        if self.simulator.current_time == 248:  # Debugging line
            print('here')
            
    def switch_off_nodes(self):

        switch_off_nodes = []
        for node_index, node in enumerate(self.simulator.sim_monitor.nodes_action):
            if node['state'] == 'idle' and self.simulator.current_time - node['time'] >= self.timeout:
                switch_off_nodes.append(node_index)

        if len(self.simulator.jobs_manager.waiting_queue) > 0:
            next_releases = self.simulator.node_manager.resources_agenda
            next_releases = sorted(
                next_releases, 
                key=lambda x: (x['release_time'], x['node'])
            )
            
            fcfs_count = 0
            for job in self.simulator.jobs_manager.waiting_queue[:]:
                
                if job['res'] < len(next_releases):
                    last_host = next_releases[job['res'] - 1]
                    job_prediction_start = last_host['release_time']
                    job_candidates = [r['node'] for r in next_releases if r['release_time'] <= job_prediction_start]
                    job_reservation = job_candidates[-job['res']:]
                    
                    next_releases = [nr for nr in next_releases if nr['node'] not in job_reservation]
                            
                    if job_prediction_start < self.simulator.current_time + self.simulator.node_manager.transition_time[0] + self.simulator.node_manager.transition_time[1]:
                        switch_off_nodes = [node for node in switch_off_nodes if node not in job_reservation]
                    fcfs_count += 1
                else:
                    break
            
            backfilling_queue = self.simulator.jobs_manager.waiting_queue[fcfs_count:]
            head_job = backfilling_queue[0]
            backfilling_queue = backfilling_queue[1:]
            
            
            for job in backfilling_queue:
                if job['res'] < len(next_releases):
                    last_host = next_releases[job['res'] - 1]
                    job_prediction_start = last_host['release_time']
                    job_candidates = [r['node'] for r in next_releases if r['release_time'] <= job_prediction_start]
                    job_reservation = job_candidates[-job['res']:]
                    
                    next_releases = [nr for nr in next_releases if nr['node'] not in job_reservation]
                            
                    if job_prediction_start < self.simulator.current_time + self.simulator.node_manager.transition_time[0] + self.simulator.node_manager.transition_time[1]:
                        switch_off_nodes = [node for node in switch_off_nodes if node not in job_reservation]
                else:
                    break
                
        if len(switch_off_nodes) > 0:
            e = {'type': 'switch_off', 'node': switch_off_nodes}
            timestamp = self.simulator.current_time
            self.simulator.jobs_manager.push_event(timestamp, e)
        
    
    def smart_easy_switch_on(self):
        if self.simulator.current_time == 1741:  # Debugging line
            print('here')
        p_job = self.simulator.jobs_manager.waiting_queue[0]
            
        backfilling_queue = self.simulator.jobs_manager.waiting_queue[1:]
        
        next_releases = self.simulator.node_manager.resources_agenda
        
        next_releases = sorted(
            next_releases, 
            key=lambda x: (x['release_time'], x['node'])
        )
        
        if len(next_releases) < p_job['res']:
            return
        
        last_host = next_releases[p_job['res'] - 1]
        head_job_prediction_start = last_host['release_time']
                
        filtered = [r for r in next_releases if r['release_time'] <= head_job_prediction_start]
        filtered_sorted = sorted(filtered, key=lambda r: (r['release_time'], -r['node']))
        head_job_candidates = [r['node'] for r in filtered_sorted]
        head_job_reservation = head_job_candidates[-p_job['res']:]
        
        need_activation_node = []
        for job in backfilling_queue:
            available = self.future_free_nodes
            not_reserved = [h for h in available if h not in head_job_reservation]
            activation_delay = self.simulator.node_manager.transition_time[1]
            
            if job['res'] <= len(not_reserved):
                not_allocated_resources = self.future_free_nodes[:job['res']]
                self.future_free_nodes = self.future_free_nodes[job['res']:]
                
                for node in not_allocated_resources:
                    if node in self.future_free_nodes_need_activation:
                        need_activation_node.append(node)
                        self.future_free_nodes_need_activation.remove(node)
            
            elif job['walltime'] and job['walltime'] + self.simulator.current_time + activation_delay <= head_job_prediction_start and job['res'] <= len(available):
                not_allocated_resources = self.future_free_nodes[:job['res']]
                self.future_free_nodes = self.future_free_nodes[job['res']:]
                for node in not_allocated_resources:
                    if node in self.future_free_nodes_need_activation:
                        need_activation_node.append(node)
                        self.future_free_nodes_need_activation.remove(node)
        
        if len(need_activation_node) > 0:
            self.simulator.jobs_manager.push_event(self.simulator.current_time, {'type': 'switch_on', 'node': need_activation_node})
    
    def backfill(self):
        p_job = self.simulator.jobs_manager.waiting_queue[0]
            
        backfilling_queue = self.simulator.jobs_manager.waiting_queue[1:]
        
        not_reserved_resources = self.simulator.node_manager.resources_agenda
        next_releases = self.simulator.node_manager.resources_agenda
        
        next_releases = sorted(
            next_releases, 
            key=lambda x: (x['release_time'], x['node'])
        )
        
        if len(next_releases) < p_job['res']:
            return
        
        last_host = next_releases[p_job['res'] - 1]
        head_job_prediction_start = last_host['release_time']
                
        head_job_candidates = [r['node'] for r in next_releases if r['release_time'] <= head_job_prediction_start]
        head_job_reservation = head_job_candidates[-p_job['res']:]
        
        not_reserved_resources = [r for r in not_reserved_resources if r not in head_job_reservation]
        
        
        for job in backfilling_queue:
            available_resources, inactive_resources = self.simulator.node_manager.get_not_allocated_resources()
            not_allocated_resources = available_resources + inactive_resources
            not_reserved = [h for h in not_allocated_resources if h not in head_job_reservation]
            
            available_resources_not_reserved = [r for r in available_resources if r in not_reserved]
            inactive_resources_not_reserved = [r for r in inactive_resources if r in not_reserved]
            
            activation_delay = 0
            for reserved_node in head_job_reservation:
                if reserved_node in inactive_resources:
                    activation_delay = 5

            if job['res'] <= len(not_reserved):
                if job['res'] <= len(available_resources_not_reserved):
                    allocate_nodes = available_resources_not_reserved[:job['res']]
                    self.simulator.execution_start(job, allocate_nodes, [])
                    
                    e = {'type': 'call_me_later'}
                    timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                    self.simulator.jobs_manager.push_event(timestamp, e)
                else:
                    count_available_nodes = len(available_resources_not_reserved)
                    allocate_nodes = available_resources_not_reserved
                    need_activation_count = job['res'] - count_available_nodes
                    need_activation_node = inactive_resources_not_reserved[:need_activation_count]
                    self.simulator.execution_start(job, allocate_nodes, need_activation_node)
                    
                    e = {'type': 'call_me_later'}
                    timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                    self.simulator.jobs_manager.push_event(timestamp, e)

            elif job['walltime'] and job['walltime'] + self.simulator.current_time + activation_delay <= head_job_prediction_start and job['res'] <= len(not_allocated_resources):
                if job['res'] <= len(available_resources):
                    allocate_nodes = available_resources[:job['res']]
                    self.simulator.execution_start(job, allocate_nodes, [])
                    
                    e = {'type': 'call_me_later'}
                    timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                    self.simulator.jobs_manager.push_event(timestamp, e)
                else:
                    count_available_nodes = len(available_resources)
                    allocate_nodes = available_resources
                    need_activation_count = job['res'] - count_available_nodes
                    need_activation_node = inactive_resources[:need_activation_count]
                    self.simulator.execution_start(job, allocate_nodes, need_activation_node)
                    
                    e = {'type': 'call_me_later'}
                    timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                    self.simulator.jobs_manager.push_event(timestamp, e)