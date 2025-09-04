import copy


class BaseAlgorithm():
    def __init__(self, state, waiting_queue, start_time, timeout=None):
        self.state = state
        self.waiting_queue = waiting_queue
        self.events = []
        self.current_time = start_time
        self.timeout = timeout

        self.available = []
        self.inactive = []
        self.compute_speeds = []
        self.allocated = []
        self.scheduled = []
        self.call_me_later = []
        self.timeout_list = []

    def push_event(self, timestamp, event):
        found = next(
            (x for x in self.events if x['timestamp'] == timestamp), None)
        if found:
            found['events'].append(event)
        else:
            self.events.append({'timestamp': timestamp, 'events': [event]})
        self.events.sort(key=lambda x: x['timestamp'])

    def set_time(self, current_time):
        self.current_time = current_time

    def timeout_policy(self):
        add_event = False
        for node in self.state:
            if node['job_id'] == None and node['state'] == 'active' and node['id'] not in [t['node_id'] for t in self.timeout_list]:
                self.timeout_list.append(
                    {'node_id': node['id'], 'time': self.current_time + self.timeout})
                add_event = True

        if add_event:
            self.push_event(self.current_time + self.timeout,
                            {'type': 'call_me_later'})

        switch_off = []
        for node in self.state:
            for timeout_info in self.timeout_list:
                if node['id'] == timeout_info['node_id']:

                    if self.current_time >= timeout_info['time'] and node['id'] not in self.allocated and node['job_id'] is None and node['state'] == 'active':
                        switch_off.append(node['id'])
                    elif self.current_time >= timeout_info['time'] and node['job_id'] is not None and node['state'] == 'active':
                        self.timeout_list.remove(timeout_info)

        if len(switch_off) > 0:
            self.push_event(self.current_time, {
                            'type': 'switch_off', 'nodes': switch_off})

    def prep_schedule(self, new_state, waiting_queue, scheduled_queue):
        self.state = new_state
        self.waiting_queue = waiting_queue
        self.scheduled_queue = scheduled_queue

        for node in self.state:
            if node['state'] == 'sleeping':
                node['release_time'] += node['transition_time']

        self.events = []
        self.compute_speeds = [node['compute_speed']
                               for node in self.state]

        """ 
        GET ALL AVAILABLE NODES (INCLUDES THE RESERVED NODES)
        THEN CHECK IF A SCHEDULED JOB CAN BE EXECUTED
        """
        self.available = [
            node for node in self.state
            if node['state'] == 'active' and node['job_id'] is None
        ]

        self.inactive = [
            node for node in self.state
            if node['state'] == 'sleeping'
        ]

        self.allocated = []
        self.scheduled = []

        if len(self.scheduled_queue) > 0:
            for scheduled_job in self.scheduled_queue:
                executable = True
                for reserved_nodes in scheduled_job['nodes']:
                    if reserved_nodes not in self.available:
                        executable = False

                if executable:
                    allocated_nodes = scheduled_job['nodes']
                    self.available = [
                        node for node in self.available if node not in scheduled_job['nodes']]
                    self.allocated.extend(scheduled_job['nodes'])
                    compute_demand = scheduled_job['walltime'] * \
                        scheduled_job['res']
                    compute_power = sum(
                        self.compute_speeds[i] for i in allocated_nodes)
                    finish_time = self.current_time + \
                        (compute_demand / compute_power)
                    for node in self.state:
                        if node['id'] in allocated_nodes:
                            node['release_time'] = finish_time
                    self.scheduled_queue.remove(scheduled_job)
                    event = {
                        'job_id': scheduled_job['job_id'],
                        'subtime': scheduled_job['subtime'],
                        'walltime': scheduled_job['walltime'],
                        'res': scheduled_job['res'],
                        'type': 'execution_start',
                        'nodes': allocated_nodes
                    }
                    self.push_event(self.current_time, event)
        """ 
        CONTINUE EXECUTE SCHEDULING LOGIC WITHOUT CONSIDERING THE RESERVED NODES
        """
        reserved_nodes = [
            node for node in self.state
            if node['reserved']
        ]

        self.available = [
            node for node in self.available if node not in reserved_nodes
        ]
