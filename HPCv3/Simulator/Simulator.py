import json

from numpy import record
from HPCv3.Simulator.MachineMonitor import Monitor
from HPCv3.Simulator.PlatformControl import PlatformControl
class Simulator:
    def __init__(self, workload_path, platform_path, start_time):
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        self.PlatformControl = PlatformControl(self.platform_info, start_time)
        self.Monitor = Monitor(self.platform_info, start_time)
        self.current_time = start_time
        self.events = []
        self.is_running = False
        self.num_jobs = len(self.workload_info['jobs'])
        self.num_finished_jobs = 0
        
        self.push_event(start_time, {'type': 'simulation_start'})
        for job in self.workload_info['jobs']:
            job['type'] = 'arrival'
            timestamp = job['subtime'] + start_time

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
        
    def start_simulator(self):
        self.is_running = True
    
    def on_finish(self):
        self.is_running = False
        print(f"Simulation finished at time {self.current_time}.")
        self.Monitor.on_finish()
        message = {'now': self.current_time, 'event_list': [{'timestamp': self.current_time, 'events': [{'type': 'simulation_finished'}]}]}
        return message
        
    def proceed(self):
        if len(self.events) == 0:
            message = self.on_finish()
            return message
        
        self.current_time, events = self.events.pop(0).values()

        self.Monitor.record(mode='before', current_time=self.current_time)
        event_priority = {
            'turn_on': 0,
            'turn_off': 1,
            'execution_finished': 2,
            'execution_start': 3,
            'arrival': 4,
            'switch_off': 5,
            'switch_on': 6
        }
        
        events = sorted(events, key=lambda e: event_priority.get(e['type'], float('inf')))
        record_job_execution = []
        for event in events:
            self.event = event
            if self.event['type'] == 'switch_off':
                result_events = self.PlatformControl.switch_off(self.event['nodes'], self.current_time)
                for event_entry in result_events:
                    self.push_event(event_entry['timestamp'], event_entry['event'])

            elif self.event['type'] == 'turn_off':
                self.PlatformControl.turn_off(self.event['nodes'])
                
            elif self.event['type'] == 'switch_on':
                result_events = self.PlatformControl.switch_on(self.event['nodes'], self.current_time)
                for event_entry in result_events:
                    self.push_event(event_entry['timestamp'], event_entry['event'])
                
            elif self.event['type'] == 'turn_on':
                self.PlatformControl.turn_on(self.event['nodes'])
                
            elif self.event['type'] == 'arrival':
               pass
           
            elif self.event['type'] == 'execution_start':
                finish_time, event = self.PlatformControl.compute(self.event['nodes'], self.event, self.current_time)
                self.push_event(finish_time, event) 
                
            elif self.event['type'] == 'execution_finished':
                self.PlatformControl.release(self.event['nodes'])
                self.num_finished_jobs += 1
                record_job_execution.append(self.event)
            
            elif self.event['type'] == 'change_dvfs_mode':
                event = self.PlatformControl.change_dvfs_mode(self.event['node'], self.event['mode'])
                
                
        self.Monitor.record(mode='after', machines=self.PlatformControl.machines, current_time=self.current_time, record_job_execution=record_job_execution)
        
        if self.num_finished_jobs == self.num_jobs:
            message = self.on_finish()
            return message

        message = {'timestamp': self.current_time, 'events': events}
        return {'now': self.current_time, 'event_list': [message]}
