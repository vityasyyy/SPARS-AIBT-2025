import copy

class BaseAlgorithm():
    def __init__(self, ResourceManager, JobsManager, start_time, timeout=None):
        self.ResourceManager = ResourceManager
        self.JobsManager = JobsManager
        self.events = []
        self.current_time = start_time
        self.timeout = timeout
        self.resources_agenda = copy.deepcopy(self.ResourceManager.resources_agenda)
        self.available = []
        self.inactive = []
        self.compute_speeds = []
        self.allocated = []
        self.scheduled = []
        self.call_me_later = []
        self.do_once = False
        
    def push_event(self, timestamp, event):
        found = next((x for x in self.events if x['timestamp'] == timestamp), None)
        if found:
            found['events'].append(event)
        else:
            self.events.append({'timestamp': timestamp, 'events': [event]})
        self.events.sort(key=lambda x: x['timestamp'])
        
    def set_time(self, current_time):
        self.current_time = current_time

    def prep_schedule(self):
        self.events = []
        if self.current_time == 0 and self.do_once == False:
            self.push_event(self.current_time, {'type': 'change_dvfs_mode', 'node': [0,1,2,3,4,5,6,7], 'mode': 'overclock_1'})
            self.do_once = True
        self.compute_speeds = [node['compute_speed'] for node in self.ResourceManager.nodes]
        self.resources_agenda = copy.deepcopy(self.ResourceManager.resources_agenda)
        next_releases = self.resources_agenda
        machines = {m['id']: m['states']['switching_on']['transitions'] for m in self.ResourceManager.machines}
        
        for next_release in next_releases:
            for t in machines[next_release['node_id']]:
                if t['state'] == 'sleeping':
                    next_release['release_time'] += t['transition_time']
                        
        self.resources_agenda = sorted(
            next_releases, 
            key=lambda x: (x['release_time'], x['node_id'])
        )
        self.available = self.ResourceManager.get_available_nodes()
        self.inactive = self.ResourceManager.get_sleeping_nodes()
        self.allocated = []
        self.scheduled = []
        