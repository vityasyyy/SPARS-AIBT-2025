class SmartFCFSScheduler:
    def __init__(self, simulator, timeout):
        self.simulator = simulator
        self.timeout = timeout
        self.fcfs_predicted = []

    def smart_fcfs_switch_on(self):
        self.fcfs_predicted = []
        if self.simulator.current_time == 201:  # Debugging line
            print('here')
        self.resources_agenda = self.simulator.node_manager.resources_agenda
        self.future_free_nodes = []
        self.future_free_nodes_need_activation = []
        for agenda in self.resources_agenda:
            node_index = agenda['node']
            release_time = agenda['release_time']
            current_time = self.simulator.current_time
            transition_time = self.simulator.node_manager.transition_time[1]

            # First condition: node is ready to be freed now
            if release_time <= current_time + transition_time:
                if node_index not in self.simulator.node_manager.reserved_resources:
                        
                    if node_index in self.simulator.node_manager.inactive_resources:
                        self.future_free_nodes_need_activation.append(node_index)
                        self.future_free_nodes.append(node_index)
                    elif (node_index in self.simulator.node_manager.available_resources):
                        self.future_free_nodes.append(node_index)
                    elif (node_index in self.simulator.node_manager.off_on_resources and release_time == current_time + transition_time):
                        self.future_free_nodes.append(node_index)
                    elif (node_index not in self.simulator.node_manager.on_off_resources
                          ):
                        self.future_free_nodes.append(node_index)

        self.future_free_nodes.sort(key=lambda node: node in self.future_free_nodes_need_activation)
        need_activation_node = []
        for job in self.simulator.jobs_manager.waiting_queue[:]:
            if job['res'] <= len(self.future_free_nodes):
                
                allocated_resource = self.future_free_nodes[:job['res']]
                self.future_free_nodes = self.future_free_nodes[job['res']:]
                for node in allocated_resource:
                    if node in self.future_free_nodes_need_activation:
                        need_activation_node.append(node)
                        self.future_free_nodes_need_activation.remove(node)
                self.fcfs_predicted.append({'id': job['id'], 'walltime': job['walltime'], 'resources': allocated_resource})
            else:
                break
            
        if len(need_activation_node) > 0:
            
            if self.simulator.current_time == 201:  # Debugging line
                print('here')
            self.simulator.jobs_manager.push_event(self.simulator.current_time, {'type': 'switch_on', 'node': need_activation_node})
                
    def add_call_me_later(self):
        if len(self.simulator.node_manager.available_resources) > 0 and self.timeout is not None:
            e = {'type': 'call_me_later'}
            timestamp = self.simulator.current_time + self.timeout
            self.simulator.jobs_manager.push_event(timestamp, e)
        
    def switch_off_nodes(self):
        if self.simulator.current_time == 2036:  # Debugging line
            print('here')
        switch_off_nodes = []
        for node_index, node in enumerate(self.simulator.sim_monitor.nodes_action):
            if node['state'] == 'idle' and self.simulator.current_time - node['time'] >= self.timeout:
                switch_off_nodes.append(node_index)

        if len(self.simulator.jobs_manager.waiting_queue) > 0:
            next_releases = self.simulator.node_manager.resources_agenda
            for agenda in self.resources_agenda:
                node = agenda['node']
                if node in self.simulator.node_manager.inactive_resources:
                    agenda['release_time'] = self.simulator.current_time + self.simulator.node_manager.transition_time[1]
                    
            next_releases = sorted(
                next_releases, 
                key=lambda x: (x['release_time'], x['node'])
            )
            
            for job in self.simulator.jobs_manager.waiting_queue[:]:
                
                if job['res'] <= len(next_releases):
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
            if self.simulator.current_time == 1644 and 5 in switch_off_nodes:  # Debugging line
                print('here')
            e = {'type': 'switch_off', 'node': switch_off_nodes}
            timestamp = self.simulator.current_time
            self.simulator.jobs_manager.push_event(timestamp, e)
            
    def schedule(self):
        self.smart_fcfs_switch_on()
        self.fcfs()
        self.add_call_me_later()
        self.switch_off_nodes()

            
    def fcfs(self):

        for job in self.simulator.jobs_manager.waiting_queue[:]:
            available_resources, inactive_resources = self.simulator.node_manager.get_not_allocated_resources()
 
            if job['res'] <= len(available_resources) + len(inactive_resources):
                if job['res'] <= len(available_resources):
                    allocate_nodes = available_resources[:job['res']]
                    job_id = job['id']
                    match = next((entry for entry in self.fcfs_predicted if entry['id'] == job_id), None)
                    if match:
                        self.fcfs_predicted.remove(match)

                    self.simulator.execution_start(job, allocate_nodes, [])
                    
                    e = {'type': 'call_me_later'}
                    timestamp = self.simulator.current_time + job['walltime'] - self.simulator.node_manager.transition_time[1]
                    self.simulator.jobs_manager.push_event(timestamp, e)
            
            else:
                break