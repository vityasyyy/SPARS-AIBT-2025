from HPCv3.RJMS.ResourceManager import ResourceManager
from HPCv3.RJMS.Scheduler import Scheduler
from HPCv3.RJMS.JobsManager import JobsManager
class RJMS:
    def __init__(self, platform_path, algorithm, start_time, timeout=None):

        self.ResourceManager = ResourceManager(platform_path)
        self.JobsManager = JobsManager()
        self.Scheduler = Scheduler(self.ResourceManager, self.JobsManager, algorithm, start_time, timeout)
        self.current_time = start_time
    
    def schedule(self, simulator_message):
        self.current_time = simulator_message['now']
        self.ResourceManager._update_node_duration(self.current_time)

        for _data in simulator_message['event_list']:
            events = _data['events']
            timestamp = _data['timestamp']
            for event in events:
                if event['type'] == 'arrival':
                    self.JobsManager.add_job_to_waiting_queue(event)
                    
                elif event['type'] == 'switch_off':
                    self.ResourceManager.switch_off(event['nodes'], timestamp)
                elif event['type'] == 'turn_off':
                    self.ResourceManager.turn_off(event['nodes'], timestamp)
                elif event['type'] == 'switch_on':
                    self.ResourceManager.switch_on(event['nodes'], timestamp)
                elif event['type'] == 'turn_on':
                    self.ResourceManager.turn_on(event['nodes'], timestamp)
                    
                elif event['type'] == 'execution_start':
                    self.ResourceManager.allocate(event['nodes'], event['job_id'], event['walltime'], self.current_time)
                    self.JobsManager.remove_job_from_waiting_queue(event['job_id'], 'execution_start')
                    self.ResourceManager.remove_reserved_nodes(event['job_id'])
                    
                elif event['type'] == 'execution_finished':
                    self.ResourceManager.release(event['nodes'])
                
                elif event['type'] == 'simulation_finished':
                    waiting_queue = self.JobsManager.waiting_queue
                    while waiting_queue:
                        job = waiting_queue[0]
                        self.JobsManager.remove_job_from_waiting_queue(job['job_id'], 'terminated')
                elif event['type'] == 'change_dvfs_mode':
                    self.ResourceManager.change_dvfs_mode(event['node'], event['mode'])
        
        self.ResourceManager.renew_resources_agenda(self.current_time)
        events = self.Scheduler.schedule(self.current_time)
        message = {'timestamp': self.current_time, 'event_list': events}
        return message

    
    
