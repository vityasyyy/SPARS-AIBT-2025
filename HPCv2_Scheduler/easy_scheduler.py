from .fcfs_scheduler import FCFSScheduler

class EasyScheduler(FCFSScheduler):
    def schedule(self):
        super().fcfs()
        if len(self.simulator.jobs_manager.waiting_queue) >= 2:
            self.backfill()
        
        if self.simulator.current_time == 110:
            print('here')
            
        if len(self.simulator.node_manager.available_resources) > 0 and self.timeout is not None:
            e = {'type': 'call_me_later'}
            timestamp = self.simulator.current_time + self.timeout
            self.simulator.jobs_manager.push_event(timestamp, e)
        
        switch_off_nodes = []
        for node_index, node in enumerate(self.simulator.sim_monitor.nodes_action):
            if node['state'] == 'idle' and self.simulator.current_time - node['time'] >= self.timeout:
                switch_off_nodes.append(node_index)

        reserved_indices = {reserved_node['node_index'] for reserved_node in self.simulator.node_manager.reserved_resources}
        switch_off_nodes = [node for node in switch_off_nodes if node not in reserved_indices]
        if len(switch_off_nodes) > 0:
            e = {'type': 'switch_off', 'node': switch_off_nodes}
            timestamp = self.simulator.current_time
            self.simulator.jobs_manager.push_event(timestamp, e)
            
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
        p_start_t = last_host['release_time']
                
        candidates = [r['node'] for r in next_releases if r['release_time'] <= p_start_t]
        head_job_reservation = candidates[-p_job['res']:]
        
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
                else:
                    count_available_nodes = len(available_resources_not_reserved)
                    allocate_nodes = available_resources_not_reserved
                    need_activation_count = job['res'] - count_available_nodes
                    need_activation_node = inactive_resources_not_reserved[:need_activation_count]
                    self.simulator.execution_start(job, allocate_nodes, need_activation_node)

            elif job['walltime'] and job['walltime'] + self.simulator.current_time + activation_delay <= p_start_t and job['res'] <= len(not_allocated_resources):
                if job['res'] <= len(available_resources):
                    allocate_nodes = available_resources[:job['res']]
                    self.simulator.execution_start(job, allocate_nodes, [])
                else:
                    count_available_nodes = len(available_resources)
                    allocate_nodes = available_resources
                    need_activation_count = job['res'] - count_available_nodes
                    need_activation_node = inactive_resources[:need_activation_count]
                    self.simulator.execution_start(job, allocate_nodes, need_activation_node)