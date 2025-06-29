from .smart_fcfs_scheduler import SmartFCFSScheduler

class SmartEasyScheduler(SmartFCFSScheduler):
    def schedule(self):
        super().smart_fcfs_switch_on()
        super().fcfs()
        if len(self.simulator.jobs_manager.waiting_queue) >= 2:
            self.smart_easy_switch_on()
            self.backfill()
        
        super().add_call_me_later()
            
    def smart_easy_switch_on(self):
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
        reservation = candidates[-p_job['res']:]
        
        not_reserved_resources = [r for r in not_reserved_resources if r not in reservation]
        
        for agenda in self.resources_agenda:
            node_index = agenda['node']
            if agenda['release_time'] <= self.simulator.current_time:
                if node_index in self.simulator.node_manager.inactive_resources:
                    self.future_free_nodes_need_activation.append(node_index)
                    self.future_free_nodes.append(node_index)
                    
            if agenda['release_time'] == self.simulator.current_time + self.simulator.node_manager.transition_time[1]:
                if any(node_index not in resource_list for resource_list in (
                    self.simulator.node_manager.inactive_resources,
                    self.simulator.node_manager.available_resources,
                    self.simulator.node_manager.off_on_resources,
                    self.simulator.node_manager.on_off_resources,
                    self.simulator.node_manager.reserved_resources
                )):
                    self.future_free_nodes.append(node_index)
        
        need_activation_node = []
        for job in backfilling_queue:
            available = self.future_free_nodes
            not_reserved = [h for h in available if h not in reservation]
            activation_delay = self.simulator.node_manager.transition_time[1]
            
            if job['res'] <= len(not_reserved):
                not_allocated_resources = self.future_free_nodes[:job['res']]
                self.future_free_nodes = self.future_free_nodes[job['res']:]
                
                for node in not_allocated_resources:
                    if node in self.future_free_nodes_need_activation:
                        need_activation_node.append(node)
                        self.future_free_nodes_need_activation.remove(node)
            
            elif job['walltime'] and job['walltime'] + self.simulator.current_time + activation_delay <= p_start_t and job['res'] <= len(available):
                not_allocated_resources = future_free_nodes[:job['res']]
                future_free_nodes = future_free_nodes[job['res']:]
                for node in not_allocated_resources:
                    if node in self.future_free_nodes_need_activation:
                        need_activation_node.append(node)
                        self.future_free_nodes_need_activation.remove(node)

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
        reservation = candidates[-p_job['res']:]
        
        not_reserved_resources = [r for r in not_reserved_resources if r not in reservation]
        
        # smarty
        
                
        #stupidity
        
        for job in backfilling_queue:
            available = self.simulator.node_manager.get_not_allocated_resources()
            not_reserved = [h for h in available if h not in reservation]
            reserved_node, need_activation_node = self.simulator.node_manager.prioritize_lowest_node(job['res'])
            activation_delay = 5 

            if job['res'] <= len(not_reserved):
                reserved_node, need_activation_node = self.simulator.node_manager.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
                e = {'type': 'call_me_later'}
                timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                self.simulator.jobs_manager.push_event(timestamp, e)
            elif job['walltime'] and job['walltime'] + self.simulator.current_time + activation_delay <= p_start_t and job['res'] <= len(available):
                reserved_node, need_activation_node = self.simulator.node_manager.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
                e = {'type': 'call_me_later'}
                timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                self.simulator.jobs_manager.push_event(timestamp, e)
                