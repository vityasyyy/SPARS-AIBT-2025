class SmartFCFSScheduler:
    def __init__(self, simulator, timeout):
        self.simulator = simulator
        self.timeout = timeout

    def smart_fcfs_switch_on(self):
        self.resources_agenda = self.simulator.node_manager.resources_agenda
        self.future_free_nodes = []
        self.future_free_nodes_need_activation = []
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
        for job in self.simulator.jobs_manager.waiting_queue[:]:
            if job['res'] <= len(self.future_free_nodes):
                not_allocated_resources = self.future_free_nodes[:job['res']]
                self.future_free_nodes = self.future_free_nodes[job['res']:]
                
                for node in not_allocated_resources:
                    if node in self.future_free_nodes_need_activation:
                        need_activation_node.append(node)
                        self.future_free_nodes_need_activation.remove(node)
            else:
                break
            
        if len(need_activation_node) > 0:
            self.simulator.jobs_manager.push_event(self.simulator.current_time, {'type': 'switch_on', 'node': need_activation_node})
                
    def add_call_me_later(self):
        if len(self.simulator.node_manager.available_resources) > 0 and self.timeout is not None:
            e = {'type': 'call_me_later'}
            timestamp = self.simulator.current_time + self.timeout
            self.simulator.jobs_manager.push_event(timestamp, e)
        
        switch_off_nodes = []
        for node_index, node in enumerate(self.simulator.sim_monitor.nodes_action):
            if node['state'] == 'idle' and self.simulator.current_time - node['time'] >= self.timeout:
                switch_off_nodes.append(node_index)

        if len(switch_off_nodes) > 0:
            e = {'type': 'switch_off', 'node': switch_off_nodes}
            timestamp = self.simulator.current_time
            self.simulator.jobs_manager.push_event(timestamp, e)
            
            
    def schedule(self):
        self.smart_fcfs_switch_on()
        self.fcfs()
        self.add_call_me_later()
            
    def fcfs(self):
        
        for job in self.simulator.jobs_manager.waiting_queue[:]:
            not_allocated_resources = self.simulator.node_manager.get_not_allocated_resources()
 
            if job['res'] <= len(not_allocated_resources):
                reserved_node, need_activation_node = self.simulator.node_manager.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
                e = {'type': 'call_me_later'}
                timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                self.simulator.jobs_manager.push_event(timestamp, e)
            else:
                break