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
        if 'current_time' in self._dict and 'current_time' in other._dict:
            if self._dict['current_time'] < other._dict['current_time']:
                return True
            elif self._dict['current_time'] > other._dict['current_time']:
                return False
        if self._dict['type'] != 'execution_finished' and self._dict['type'] != 'execution_start':
            return self._dict['id'] < other._dict['id']
        else:
            if self._dict['type'] == 'execution_finished' and other._dict['type'] == 'execution_start':
                return True
            elif self._dict['type'] == 'execution_start' and other._dict['type'] == 'execution_finished':
                return False
            else:
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
            'finish_time': 0,
            'nb_res': pd.DataFrame([{'current_time': 0, 'sleeping': 0, 'switching_on': 0, 'switching_off': 0, 'idle': 16, 'computing': 0, 'unavailable': 0}])
        }
            
    def check_backfilling(self, current_time, event, temp_available_resources, active_jobs, next_job, backfilled_node_count):
        # Sebelum push ke schedule queue, perlu cek dulu apakah bakal meningkatkan waiting time job ke-0 pada waiting_queue
        # Meningkatkan waiting time apabila current_time + event['walltime'] < estimasi start time job ke-0 pada waiting_queue
        # estimasi start time perlu mencatat active jobs, di mana setiap active jobs punya properti res dan finish time
        # Start job ke-0 pada waiting queue adalah ketika job yang aktif, selesai satu per satu hingga available nodes > res job ke-0 pada waiting queue
        temp_aval_res = temp_available_resources
        
        estimated_finish_time = current_time + event['walltime']
        last_job_active_job_finish_time_that_required_to_be_released = np.inf
 
        # last_job_active_job_finish_time_that_required_to_be_released = active_jobs[0]['finish_time']
        
        for active_job in active_jobs:
            temp_available_resources += active_job['res']
            if temp_available_resources >= next_job['res']:
                last_job_active_job_finish_time_that_required_to_be_released = active_job['finish_time']
                break
            
            

        # cari ketika waiting queue [0] start ada berapa node yg free
        
        for active_job in active_jobs:
            if active_job['finish_time'] <= last_job_active_job_finish_time_that_required_to_be_released:
                temp_aval_res += active_job['res']
            
            
        
        # Jika return true berarti diperbolehkan backfilling. 
        # Dengan logika, jika estimasi waktu finish job yang mau di-backfilling < estimasi start dari job berikutnya pada waiting queue maka boleh backfill
        # Masalahnya belum tentu jika job sekarang dimasukkan walaupung waktu finish > estimasi start job berikutnya maka akan mengganggu eksekusi job berikutnya
        # Contohnya 
        # Pada mesin dengan 4 node
        # Step 1: sekarang hanya ada 1 aktif job A yang memakan 3 resource estimasi selesai pada detik 50
        # Step 2: Sekarang ada job B yang tiba namun tidak bisa dieksekusi langsung karena memakan 2 resource. Maka estimasi start adalah pada detik 50 setelah job A release resource
        # Step 3: Sekarang Ingin dicek apakah job C yang memakan 1 resource dan estimasi selesai pada detik 60 bisa dibackfill
        # Namun dengan logika sekarang karna estimasi start job B adalah 50, dan estimasi job C adalah 60, maka Job C tidak boleh dibackfill walaupun sebenernya tidak akan mengganggu job B
        
        return estimated_finish_time < last_job_active_job_finish_time_that_required_to_be_released or temp_aval_res >= event['res'] + next_job['res']
    
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
                
            
       
            if event['type'] == 'arrival':
                if len(available_resources) >= event['res']:
                    if waiting_queue:
                        active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                        backfilled_node_count = 0
                        if self.check_backfilling(current_time, event, len(available_resources), active_jobs, waiting_queue[0], backfilled_node_count):
                            event['type'] = 'execution_start'
                            heapq.heappush(schedule_queue, (current_time, MyDict(event)))
                        else:
                            event['current_time'] = current_time
                            waiting_queue.append(event)
                    else:
                        event['type'] = 'execution_start'
                        heapq.heappush(schedule_queue, (current_time, MyDict(event)))
                else:
                    event['current_time'] = current_time
                    waiting_queue.append(event)
                    
            elif event['type'] == 'execution_start':
                # allocated = self.find_grouped_resources(available_resources, event['res'])
                # available_resources = [r for r in available_resources if r not in allocated]
                allocated = available_resources[:event['res']]
                available_resources = available_resources[event['res']:]
                mask = self.sim_monitor['nb_res']['current_time'] == current_time
                
                if self.sim_monitor['nb_res'].loc[mask].empty:
                    last_row = self.sim_monitor['nb_res'].iloc[-1].copy()

                    # Update the time
                    last_row['current_time'] = current_time
                    
                    # Add as a new row
                    self.sim_monitor['nb_res'].loc[len(self.sim_monitor['nb_res'])] = last_row

                    
                self.sim_monitor['nb_res'].loc[
                    self.sim_monitor['nb_res']['current_time'] == current_time, 
                    ['computing', 'idle']
                ] += [len(allocated), -len(allocated)]

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
                for index_available_resource in available_resources:
                    self.sim_monitor['start_idle'][index_available_resource] = current_time

                active_jobs = [active_job for active_job in active_jobs if active_job['id'] != event['id']]
                
                temp_available_resource = len(available_resources)
                
                mask = self.sim_monitor['nb_res']['current_time'] == current_time
                
                if self.sim_monitor['nb_res'].loc[mask].empty:
                    last_row = self.sim_monitor['nb_res'].iloc[-1].copy()

                    # Update the time
                    last_row['current_time'] = current_time
                    
                    # Add as a new row
                    self.sim_monitor['nb_res'].loc[len(self.sim_monitor['nb_res'])] = last_row
                    
                self.sim_monitor['nb_res'].loc[
                    self.sim_monitor['nb_res']['current_time'] == current_time, 
                    ['computing', 'idle']
                ] += [-len(allocated), len(allocated)]
              
                
                events_now = [(t, e) for t, e in schedule_queue if t == current_time]
                skipbf = False
                for aj in active_jobs:
                    if aj['finish_time'] == current_time:
                        skipbf=True
                        
                if skipbf == True:
                    continue
                    
                for _, _event in events_now:
                    if _event['type'] == 'execution_start':
                        temp_available_resource -= _event['res']

                
                waiting_queue = sorted(waiting_queue)
                # tambahin perulangan gas gas insert job berdasarkan id atau submit time

                    
                for _ in range(len(waiting_queue)):
                    job = waiting_queue[0]  # Selalu cek job pertama
                    if temp_available_resource >= job['res']:
                        popped_job = waiting_queue.pop(0)
                        popped_job['type'] = 'execution_start'
                        temp_available_resource -= popped_job['res']
                        if popped_job['id'] == 256:
                            print('here')
                        heapq.heappush(schedule_queue, (current_time, MyDict(popped_job)))
                    else:
                        break

                active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                backfilled_node_count = 0
                while True:
                    is_pushed = False
                    for k in range(0, len(waiting_queue)):
                        job = waiting_queue[k]
                        if temp_available_resource >= job['res']:
                            if not self.check_backfilling(current_time, job, temp_available_resource, active_jobs, waiting_queue[0], backfilled_node_count):
                                continue
                         
                            backfilled_node_count += job['res']
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

print('joule: ',sum(sp_simulator.sim_monitor['energy_consumption']))
print('idle_time: ',sum(sp_simulator.sim_monitor['total_idle_time']))
print('mean_waiting_time: ',sp_simulator.sim_monitor['avg_waiting_time'])
print('mean_waiting_time: ',sp_simulator.sim_monitor['avg_waiting_time']/500)
print('finish_time: ', max_finish_time)

sp_simulator.sim_monitor['nb_res'].to_csv('results/sp/easy_host.csv', index=False)