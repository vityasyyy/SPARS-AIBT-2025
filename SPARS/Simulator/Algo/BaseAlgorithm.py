import copy


class BaseAlgorithm():
    def __init__(self, state, waiting_queue, start_time, jobs_manager, timeout=None):
        self.state = state
        self.waiting_queue = waiting_queue
        self.events = []
        self.current_time = start_time
        self.timeout = timeout

        self.jobs_manager = jobs_manager

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

    def remove_from_timeout_list(self, node_ids):
        ids = set(node_ids)
        self.timeout_list[:] = [ti for ti in self.timeout_list
                                if ti.get('node_id') not in ids]

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
                    allocated_ids = [node['id']
                                     for node in self.allocated]
                    if self.current_time >= timeout_info['time'] and node['id'] not in allocated_ids and node['job_id'] is None and node['state'] == 'active':
                        switch_off.append(node['id'])
                        self.timeout_list.remove(timeout_info)
                    elif self.current_time >= timeout_info['time'] and node['job_id'] is not None and node['state'] == 'active':
                        self.timeout_list.remove(timeout_info)

        if len(switch_off) > 0:
            self.push_event(self.current_time, {
                            'type': 'switch_off', 'nodes': switch_off})

    def prep_schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        if self.current_time == 450:
            print('x')
        self.state = new_state
        self.waiting_queue = waiting_queue
        self.scheduled_queue = scheduled_queue
        self.resources_agenda = copy.deepcopy(resources_agenda)
        self.resources_agenda = sorted(
            self.resources_agenda,
            key=lambda x: (x['release_time'], x['id'])
        )

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

        self.reserved_ids = []
        for job in scheduled_queue:
            self.reserved_ids.extend(job['nodes'])

        # if len(self.scheduled_queue) > 0:
        #     for scheduled_job in self.scheduled_queue:
        #         executable = True
        #         for reserved_nodes in scheduled_job['nodes']:
        #             if reserved_nodes not in self.available:
        #                 executable = False

        #         if executable:
        #             allocated_nodes = scheduled_job['nodes']
        #             self.available = [
        #                 node for node in self.available if node not in scheduled_job['nodes']]
        #             self.allocated.extend(scheduled_job['nodes'])

        #             self.scheduled_queue.remove(scheduled_job)
        #             event = {
        #                 'job_id': scheduled_job['job_id'],
        #                 'subtime': scheduled_job['subtime'],
        #                 'runtime': scheduled_job['runtime'],
        #                 'res': scheduled_job['res'],
        #                 'type': 'execution_start',
        #                 'nodes': allocated_nodes
        #             }
        #             self.push_event(self.current_time, event)
        """ 
        CONTINUE EXECUTE SCHEDULING LOGIC WITHOUT CONSIDERING THE RESERVED NODES
        """

        self.available = [
            node for node in self.available if node['id'] not in self.reserved_ids
        ]
