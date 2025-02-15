import json
import heapq
import pandas as pd
import copy
import numpy as np

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
            "total_idle_time": [0] * len(self.machines),
            'avg_waiting_time': 0,
            'waiting_event_count': 0,
            'finish_time': 0
        }
            
    def check_backfilling(self, current_time, event, temp_available_resources, active_jobs):
        # Sebelum push ke schedule queue, perlu cek dulu apakah bakal meningkatkan waiting time job ke-0 pada waiting_queue
        # Meningkatkan waiting time apabila current_time + event['walltime'] < estimasi start time job ke-0 pada waiting_queue
        # estimasi start time perlu mencatat active jobs, di mana setiap active jobs punya properti res dan finish time
        # Start job ke-0 pada waiting queue adalah ketika job yang aktif, selesai satu per satu hingga available nodes > res job ke-0 pada waiting queue
        estimated_finish_time = current_time + event['walltime']
        estimated_next_job_start_time = np.inf
 
        for active_job in active_jobs:
            temp_available_resources += active_job['res']
            if temp_available_resources >= event['res']:
                estimated_next_job_start_time = active_job['finish_time']
                break

        # Jika return true berarti diperbolehkan backfilling. 
        # Dengan logika, jika estimasi waktu finish job yang mau di-backfilling < estimasi start dari job berikutnya pada waiting queue maka boleh backfill
        # Masalahnya belum tentu jika job sekarang dimasukkan walaupung waktu finish > estimasi start job berikutnya maka akan mengganggu eksekusi job berikutnya
        # Contohnya 
        # Pada mesin dengan 4 node
        # Step 1: sekarang hanya ada 1 aktif job A yang memakan 3 resource estimasi selesai pada detik 50
        # Step 2: Sekarang ada job B yang tiba namun tidak bisa dieksekusi langsung karena memakan 2 resource. Maka estimasi start adalah pada detik 50 setelah job A release resource
        # Step 3: Sekarang Ingin dicek apakah job C yang memakan 1 resource dan estimasi selesai pada detik 60 bisa dibackfill
        # Namun dengan logika sekarang karna estimasi start job B adalah 50, dan estimasi job C adalah 60, maka Job C tidak boleh dibackfill walaupun sebenernya tidak akan mengganggu job B
        return estimated_finish_time < estimated_next_job_start_time
    def find_grouped_resources(self, resources, count):
        resources = sorted(resources)
        for i in range(len(resources) - count + 1):
            # Cek apakah resource dari i ke i+count berurutan
            if resources[i + count - 1] - resources[i] == count - 1:
                return resources[i:i + count]
        # Jika tidak ditemukan blok berurutan, ambil blok pertama seperti biasa
        return resources[:count]
        
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
            
            if event['id'] == 9:
                print('here')
            if event['type'] == 'arrival':
                if len(available_resources) >= event['res']:
                    if waiting_queue:
                        active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                        if self.check_backfilling(current_time, event, len(available_resources), active_jobs):
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
                allocated = self.find_grouped_resources(available_resources, event['res'])
                available_resources = [r for r in available_resources if r not in allocated]

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
                
                
                if finish_event['subtime'] != current_time:
                    self.sim_monitor['avg_waiting_time'] += (current_time - finish_event['subtime'])
                    self.sim_monitor['waiting_event_count'] += 1
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
                
                events_now = [(t, e) for t, e in schedule_queue if t == current_time]

                for _, event in events_now:
                    if event['type'] == 'execution_start':
                        temp_available_resource -= event['res']

                
                waiting_queue = sorted(waiting_queue)
                # tambahin perulangan gas gas insert job berdasarkan id atau submit time
   
                for _ in range(len(waiting_queue)):
                    job = waiting_queue[0]  # Selalu cek job pertama
                    if temp_available_resource >= job['res']:
                        popped_job = waiting_queue.pop(0)
                        popped_job['type'] = 'execution_start'
                        temp_available_resource -= popped_job['res']
                        heapq.heappush(schedule_queue, (current_time, MyDict(popped_job)))
                    else:
                        break



                    
                active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                while True:
                    is_pushed = False
                    for k in range(0, len(waiting_queue)):
                        job = waiting_queue[k]
                        if temp_available_resource >= job['res']:
                            if job['id'] == 9:
                                print('here')
                            if not self.check_backfilling(current_time, job, temp_available_resource, active_jobs):
                                continue
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
max_finish_time = max(job.get('finish_time', 0) for job in jobs_e)
jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_e.to_csv('results/sp/easy_jobs.csv', index=False)
print(jobs_e)
print(sp_simulator.sim_monitor)
print(sum(sp_simulator.sim_monitor['energy_consumption']))
print(sum(sp_simulator.sim_monitor['total_idle_time']))
print(sp_simulator.sim_monitor['avg_waiting_time']/sp_simulator.sim_monitor['waiting_event_count'])
print(max_finish_time)