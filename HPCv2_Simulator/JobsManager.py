class JobsManager:
    def __init__(self, workload_info):
        self.num_jobs = len(workload_info['jobs'])
        self.num_jobs_finished = 0
        self.waiting_queue = []
        self.waiting_queue_ney = []
        self.executed_jobs = []
        self.monitor_jobs = []
        self.active_jobs = []
        self.reserved_count = 0
        self.total_req_res = 0
        self.events = []
        
        for job in workload_info['jobs']:
            job['type'] = 'arrival'
            timestamp = job['subtime']

            self.push_event(timestamp, job)
            
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