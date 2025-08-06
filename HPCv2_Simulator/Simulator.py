from HPCv2_Simulator.JobsManager import JobsManager
from HPCv2_Simulator.Monitor import SimMonitor
from HPCv2_Simulator.NodesManager import NodeManager
import json
import copy
from collections import defaultdict

class SPSimulator:
    def __init__(self, scheduler, platform_path, workload_path, start_time):        
        self.scheduler = scheduler
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)

        self.nb_res = self.platform_info['nb_res']
        self.machines = self.platform_info['machines']
        self.profiles = self.workload_info['profiles']

        self.sim_monitor = SimMonitor(self.nb_res, self.machines)
        self.jobs_manager = JobsManager(self.workload_info)
        self.node_manager = NodeManager(self.nb_res, self.sim_monitor, self.platform_info)
        
        self.current_time = start_time
        self.last_event_time = start_time
        self.event = None    
        self.total_req_res = 0
        self.is_running = False
    
    def print_energy_waste(self):
        index = 0
        sum = 0
        for node_energy_waste in self.sim_monitor.energy_waste:
            print(f'Energy wasted on node {index}: ', node_energy_waste)
            sum+=node_energy_waste
            index += 1
            
        print(f'Total energy consumption: {sum}')    
        
    def print_energy_consumption(self):
        index = 0
        sum = 0
        for node_energy_consumption in self.sim_monitor.energy_consumption:
            print(f'Energy consumption of node {index}: ', node_energy_consumption)
            sum+=node_energy_consumption
            index += 1
        print(f'Total energy consumption: {sum}')    
    
    def start_simulator(self):
        self.is_running = True
        self.scheduler.schedule()
    
    def execution_start(self, job, reserved_node, need_activation_node=[]):
        if len(reserved_node) + len(need_activation_node) < job['res']:
            print('not enough res')
            return
        
        for node in reserved_node:
            self.node_manager.reserved_resources.append({
                "node_index": node,
                "job_id": job['id']
            })
            
        self.node_manager.reserved_resources = sorted(self.node_manager.reserved_resources, key=lambda x: x["node_index"])

        job['reserved_nodes'] = reserved_node + need_activation_node
        job['type'] = 'execution_start'

        for index, _job in enumerate(self.jobs_manager.waiting_queue): 
            if _job['id'] == job['id']:
                self.jobs_manager.waiting_queue.pop(index)
                break
        
        self.jobs_manager.waiting_queue_ney.append(job)
        
        if len(need_activation_node) > 0:
            ts, e = self.node_manager.switch_on(need_activation_node, self.current_time, job, self.workload_info)
            # Switch on event will set release time to current_time + transition time
            
            # update the release time to current_time + transition time + job['walltime']
            for i in range(self.nb_res):
                if i in need_activation_node or i in reserved_node:
                    self.node_manager.resources_agenda[i]['release_time'] = self.current_time + self.node_manager.transition_time[1] + job['walltime']
            self.jobs_manager.push_event(ts, e) # Push turn on event
            self.jobs_manager.push_event(ts, job) # Push execution start event

        else:
            for i in range(self.nb_res):
                if i in need_activation_node or i in reserved_node:
                    self.node_manager.resources_agenda[i]['release_time'] = self.current_time + job['walltime']

            self.jobs_manager.push_event(self.current_time, job)
            
    def proceed(self):
        if len(self.jobs_manager.waiting_queue) > 0 and len(self.jobs_manager.events) == 0:
            self.jobs_manager.num_terminated_jobs = len(self.jobs_manager.waiting_queue)
            while self.jobs_manager.waiting_queue:
                job = self.jobs_manager.waiting_queue.pop(0)
                print(f"Job {job['id']} is terminated")
            return

        self.current_time, events = self.jobs_manager.events.pop(0).values()
        self.store_events = events
        # print(events)
        
        self.sim_monitor.update_energy_consumption(self.machines, self.current_time, self.last_event_time)
        self.node_manager.update_node_state_monitor(self.current_time, self.last_event_time)
        self.sim_monitor.update_idle_time(self.current_time, self.last_event_time)
        self.node_manager.update_inactive_resources_agenda(self.current_time)
        num_job_in_queue = len(self.jobs_manager.waiting_queue) + len(self.jobs_manager.waiting_queue_ney)
        self.sim_monitor.update_total_waiting_time(num_job_in_queue, self.current_time, self.last_event_time)

        self.sim_monitor.update_energy_waste()
        self.last_event_time = self.current_time

        for node_states in self.sim_monitor.node_state_monitor:
            acc = node_states['idle'] + node_states['switching_off'] + node_states['switching_on'] + node_states['computing'] + node_states['sleeping']
       
            if acc != self.current_time:
                input('bug occur')

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
        for event in events:
            self.event = event
            if self.event['type'] == 'switch_off':
                # ts = scheduled time when the node will be turned off
                # e = the 'turn_off' event to be triggered at time ts
                result = self.node_manager.switch_off(self.event['node'], self.current_time, self.event)
                if result is not None:
                    ts, e = result
                    self.jobs_manager.push_event(ts, e)


            elif self.event['type'] == 'turn_off':
                self.node_manager.turn_off(self.event['node'], self.current_time, self.event)
                
            elif self.event['type'] == 'switch_on':
                # ts = scheduled time when the node will be turned off
                # e = the 'turn_off' event to be triggered at time ts
                ts, e = self.node_manager.switch_on(self.event['node'], self.current_time, self.event, self.workload_info)
                self.jobs_manager.push_event(ts, e) # add the turn on event
                
            elif self.event['type'] == 'turn_on':
                self.node_manager.turn_on(self.event['node'], self.event, self.current_time)
                
            elif self.event['type'] == 'arrival':
                # Upon arrival, push the job into waiting queue
                self.jobs_manager.waiting_queue.append(self.event)
                # Sort the waiting queue based on submission time
                self.jobs_manager.waiting_queue.sort(key=lambda job: (job['subtime'], str(job['id'])))

            elif self.event['type'] == 'execution_start':
                new_waiting_queue_ney = []
                for d in self.jobs_manager.waiting_queue_ney:
                    if d['id'] != self.event['id']:
                        new_waiting_queue_ney.append(d)
                        
                self.jobs_manager.waiting_queue_ney = new_waiting_queue_ney
        
                allocated = self.event['reserved_nodes']
   
                
                self.node_manager.reserved_resources = [node for node in self.node_manager.reserved_resources if node['node_index'] not in allocated]
                self.node_manager.available_resources = [node for node in self.node_manager.available_resources if node not in allocated]
                self.node_manager.update_node_action(allocated, self.event, 'allocate', 'computing', self.current_time)

                finish_time = self.current_time + self.event['walltime']
                finish_event = {
                    'id': self.event['id'],
                    'res': self.event['res'],
                    'walltime': self.event['walltime'],
                    'type': 'execution_finished',
                    'subtime': self.event['subtime'],
                    'profile': self.event['profile'],
                    'allocated_resources': allocated
                }
                
            
                for i in range(self.nb_res):
                    if i in allocated:
                        self.node_manager.resources_agenda[i]['release_time'] = self.current_time + self.event['walltime']
                    
                self.jobs_manager.push_event(finish_time, finish_event)    
                
                finish_event['finish_time'] = finish_time
                self.jobs_manager.active_jobs.append(finish_event)
                
                self.jobs_manager.monitor_jobs.append({
                    'job_id': self.event['id'],
                    'workload_name': 'w0',
                    'profile': self.event['profile'],
                    'submission_time': self.event['subtime'],
                    'requested_number_of_resources': self.event['res'],
                    'requested_time': self.event['walltime'],
                    'success': 0,
                    'final_state': 'COMPLETED_WALLTIME_REACHED',
                    'starting_time': self.current_time,
                    'execution_time': self.event['walltime'],
                    'finish_time': finish_time,
                    'waiting_time': self.current_time - self.event['subtime'],
                    'turnaround_time': finish_time - self.event['subtime'],
                    'stretch': (finish_time - self.event['subtime']) / self.event['walltime'],
                    'allocated_resources': allocated,
                    'consumed_energy': -1
                })
            
            elif self.event['type'] == 'execution_finished':
                self.jobs_manager.num_jobs_finished += 1
                
                if self.jobs_manager.num_jobs_finished == self.jobs_manager.num_jobs:
                    self.sim_monitor.finish_time = self.current_time
                allocated = self.event['allocated_resources']
                for i in range(self.nb_res):
                    if i in allocated:
                        self.node_manager.resources_agenda[i]['release_time'] = 0
                        
                self.node_manager.available_resources.extend(allocated)
                self.node_manager.available_resources.sort()
                
                self.node_manager.update_node_action(allocated, self.event, 'release', 'idle', self.current_time)
                
                self.jobs_manager.active_jobs = [active_job for active_job in self.jobs_manager.active_jobs if active_job['id'] != self.event['id']]
                
                
        if self.jobs_manager.num_jobs_finished < self.jobs_manager.num_jobs:
            self.scheduler.schedule()
            
        if self.jobs_manager.num_jobs_finished == self.jobs_manager.num_jobs:
            for x in self.sim_monitor.nodes:
                if x[len(x)-1]['finish_time'] != self.current_time:
                    x[len(x)-1]['finish_time'] = self.current_time
            
            self.on_finish()
      
    def on_finish(self):
        self.is_running = False
        self.print_energy_consumption()
        self.print_energy_waste()

        for node_index, node in enumerate(self.sim_monitor.node_state_monitor):
            print('Node', node_index, ': ', node)