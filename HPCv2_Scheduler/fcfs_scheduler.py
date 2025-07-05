class FCFSScheduler:
    def __init__(self, simulator, timeout=None):
        self.simulator = simulator
        self.timeout = timeout
    def schedule(self):
        self.fcfs()
        
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
       
    def fcfs(self):
        if self.simulator.current_time == 815:
            print('here')
            
        for job in self.simulator.jobs_manager.waiting_queue[:]:
            available_resources, inactive_resources = self.simulator.node_manager.get_not_allocated_resources()

            if job['res'] <= len(available_resources) + len(inactive_resources):
                if job['res'] <= len(available_resources):
                    allocate_nodes = available_resources[:job['res']]
                    self.simulator.execution_start(job, allocate_nodes, [])
                else:
                    count_available_nodes = len(available_resources)
                    allocate_nodes = available_resources
                    need_activation_count = job['res'] - count_available_nodes
                    need_activation_node = inactive_resources[:need_activation_count]
                    self.simulator.execution_start(job, allocate_nodes, need_activation_node)
            else:
                break