import json
import heapq
import pandas as pd
import copy

class MyDict:
    def __init__(self, _dict: dict):
        if isinstance(_dict, MyDict):
            self._dict = copy.deepcopy(_dict._dict)
        else:
            self._dict = _dict
        
    def __lt__(self, other):
        
        return self._dict['id'] < other._dict['id']
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value

class SPSimulator:
    def __init__(self, platform_path="platforms/spsim/platform.json", workload_path="workloads/simple_data.json"):        
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
            
        self.nb_res = self.platform_info['nb_res']
        self.machines = self.platform_info['machines']
        self.profiles = self.workload_info['profiles']
        
        self.jobs = []
        for job in self.workload_info['jobs']:
            job['type'] = 'arrival'
            
            heapq.heappush(self.jobs, (float(job['subtime']), MyDict(job)))
        
        self.sim_monitor = {
            "energy_consumption": [0] * len(self.machines),
            "start_idle": [0] * len(self.machines),
            "total_idle_time": [0] * len(self.machines)
        }
            
        
    def simulate_easy(self):
        current_time = 0
        available_resources = list(range(self.nb_res))
        schedule_queue = []
        waiting_queue = []
        monitor_jobs=[]
        active_jobs = []
        for event in self.jobs:
            event_time, event_detail = event
            heapq.heappush(schedule_queue, (event_time, MyDict(event_detail)))
        
        while schedule_queue or waiting_queue:      
            if schedule_queue:
                event_time, event = heapq.heappop(schedule_queue)
            else:
                event = waiting_queue.pop(0)
            
            current_time = event_time
            # calculated wasted time idle
            temp_index = 0
            for start_idle_res in self.sim_monitor['start_idle']:
                if start_idle_res == -1:
                    temp_index +=1
                    continue
                rate_energy_consumption_idle = self.machines[temp_index]['wattage_per_state'][1]
                idle_time = current_time - start_idle_res
                self.sim_monitor['energy_consumption'][temp_index] += (idle_time * rate_energy_consumption_idle)
                self.sim_monitor['total_idle_time'][temp_index] += (current_time - start_idle_res)
                
                
            for index_available_resource in available_resources:
                self.sim_monitor['start_idle'][index_available_resource] = current_time
            
            if event['type'] == 'arrival':
                if len(available_resources) >= event['res']:
                    
                    if waiting_queue:
                        estimated_finish_time = current_time + event['walltime']
                        estimated_next_job_start_time = 0
                        temp_available_resources = len(available_resources)
                        active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'], reverse=True)
                        
                        for active_job in active_jobs:
                            temp_available_resources += active_job['res']
                            if temp_available_resources >= event['res']:
                                estimated_next_job_start_time = active_job['finish_time']
                        # Sebelum push ke schedule queue, perlu cek dulu apakah bakal meningkatkan waiting time job ke-0 pada waiting_queue
                        # Meningkatkan waiting time apabila current_time + event['walltime'] < estimasi start time job ke-0 pada waiting_queue
                        # estimasi start time perlu mencatat active jobs, di mana setiap active jobs punya properti res dan finish time
                        # Start job ke-0 pada waiting queue adalah ketika job yang aktif, selesai satu per satu hingga available nodes > res job ke-0 pada waiting queue
                        
                        if estimated_finish_time < estimated_next_job_start_time:
                            event['type'] = 'execution_start'
                            heapq.heappush(schedule_queue, (current_time, MyDict(event)))
                        else:
                            waiting_queue.append(event)
                    else:
                        event['type'] = 'execution_start'
                        heapq.heappush(schedule_queue, (current_time, MyDict(event)))
                else:
                    waiting_queue.append(event)
                    
            elif event['type'] == 'execution_start':
                allocated = available_resources[:event['res']]
                available_resources = available_resources[event['res']:]

                finish_time = current_time + event['walltime']
                finish_event = {
                    'id': event['id'],
                    'res': event['res'],
                    'walltime': event['walltime'],
                    'type': 'execution_finished',
                    'subtime': event['subtime'],
                    'profile': event['profile'],
                    'allocated_resources': allocated
                }
                
                
                
                heapq.heappush(schedule_queue, (finish_time, MyDict(finish_event)))
                
                finish_event['finish_time'] = finish_time
                active_jobs.append(finish_event)
                
                for i in allocated:
                    self.sim_monitor['energy_consumption'][i] += (finish_time - current_time) * self.machines[i]['wattage_per_state'][3]
                    self.sim_monitor['start_idle'][i] = -1
                
                    
                    
                monitor_jobs.append({
                    'job_id': event['id'],
                    'workload_name': 'w0',
                    'profile': event['profile'],
                    'submission_time': event['subtime'],
                    'requested_number_of_resources': event['res'],
                    'requested_time': event['walltime'],
                    'success': 0,
                    'final_state': 'COMPLETED_WALLTIME_REACHED',
                    'starting_time': current_time,
                    'execution_time': event['walltime'],
                    'finish_time': finish_time,
                    'waiting_time': current_time - event['subtime'],
                    'turnaround_time': finish_time - event['subtime'],
                    'stretch': (finish_time - event['subtime']) / event['walltime'],
                    'allocated_resources': allocated,
                    'consumed_energy': -1
                })
            
            elif event['type'] == 'execution_finished':
                allocated = event['allocated_resources']
                available_resources.extend(allocated)
                available_resources.sort() 

                active_jobs = [active_job for active_job in active_jobs if active_job['id'] != event['id']]
                
                temp_available_resource = len(available_resources)
                
                while True:
                    is_pushed = False
                    sorted(waiting_queue)
                    # id
                    # subtime
                    
                    for k in range(0, len(waiting_queue)):
                        job = waiting_queue[k]
                        if temp_available_resource >= job['res']:
                            next_job = waiting_queue.pop(k)
                            next_job['type'] = 'execution_start'
                            temp_available_resource -= next_job['res']
                            heapq.heappush(schedule_queue, (current_time, MyDict(next_job)))
                            is_pushed = True
                            break
                    if is_pushed == False:
                        break
                    
        return monitor_jobs

sp_simulator = SPSimulator()
jobs_e = sp_simulator.simulate_easy()
jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)
jobs_e.to_csv('results/sp/easy_jobs.csv', index=False)
print(jobs_e)
print(sp_simulator.sim_monitor)