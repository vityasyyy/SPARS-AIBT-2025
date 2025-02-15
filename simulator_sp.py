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
        """
        Simulate job scheduling using the EASY backfilling approach.
        
        1) Sort or push all events (arrivals) into a schedule_queue.
        2) Repeatedly pop events in chronological order.
        3) If an arrival occurs, see if we can schedule immediately or must wait.
           - If we can schedule, check if that scheduling is feasible under backfilling (i.e. won't delay the priority job).
           - If feasible, schedule it; otherwise, push it into the waiting queue.
        4) If a job finishes, free resources, then attempt to schedule from the waiting queue (including backfill).
        5) Keep track of energy usage, idle times, waiting times, etc.
        
        :return: monitor_jobs (information about each job run in the simulation).
        """
        current_time = 0
        available_resources = list(range(self.nb_res))
        schedule_queue = []  # Will contain (event_time, event_dict)
        waiting_queue = []   # Jobs waiting to start
        monitor_jobs = []
        active_jobs = []
        
        # 1) Push the arrival events into schedule_queue
        for event_time, event_detail in self.jobs:
            heapq.heappush(schedule_queue, (event_time, MyDict(event_detail)))
        
        # Main simulation loop
        while schedule_queue or waiting_queue:
            if schedule_queue:
                event_time, event = heapq.heappop(schedule_queue)
            else:
                # If schedule_queue is empty, we take the next waiting job (though normally you'd wait for a finish)
                event = waiting_queue.pop(0)
                event_time = current_time  # we just keep the same time or update it accordingly
            
            current_time = event_time
            
            # Update energy consumption for idle nodes from the last time we set them idle
            for res_id in range(self.nb_res):
                start_idle_res = self.sim_monitor['start_idle'][res_id]
                if start_idle_res != -1:
                    # This resource has been idle since `start_idle_res`
                    rate_energy_consumption_idle = self.machines[res_id]['wattage_per_state'][1]  # Idle wattage
                    idle_time = current_time - start_idle_res
                    if idle_time > 0:
                        self.sim_monitor['energy_consumption'][res_id] += idle_time * rate_energy_consumption_idle
                        self.sim_monitor['total_idle_time'][res_id] += idle_time
                        # Mark resource as idle again from now
                        self.sim_monitor['start_idle'][res_id] = current_time

            # -------------------------
            # Handle arrival events
            # -------------------------
            if event['type'] == 'arrival':
                # If we have enough free resources for the job
                if len(available_resources) >= event['res']:
                    # If there's a waiting queue, we want to ensure we don't violate backfill constraints
                    if waiting_queue:
                        # Sort active jobs by their finishing time (if you need them in order)
                        active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                        
                        can_backfill = self.check_backfilling(
                            current_time, event, len(available_resources), active_jobs
                        )
                        if can_backfill:
                            # We can schedule it right now
                            event['type'] = 'execution_start'
                            heapq.heappush(schedule_queue, (current_time, MyDict(event)))
                        else:
                            # Otherwise, it must wait
                            waiting_queue.append(event)
                    else:
                        # No waiting queue, schedule immediately
                        event['type'] = 'execution_start'
                        heapq.heappush(schedule_queue, (current_time, MyDict(event)))
                else:
                    # Not enough free resources -> job must wait
                    waiting_queue.append(event)
            
            # -------------------------
            # Start execution
            # -------------------------
            elif event['type'] == 'execution_start':
                # Allocate resources for the job
                allocated = self.find_grouped_resources(available_resources, event['res'])
                # Remove those resources from the free list
                available_resources = [r for r in available_resources if r not in allocated]
                
                finish_time = current_time + event['walltime']
                finish_event = {
                    'id': event['id'],
                    'res': event['res'],
                    'walltime': event['walltime'],
                    'type': 'execution_finished',
                    'subtime': event['subtime'],
                    'profile': event['profile'],
                    'allocated_resources': allocated,
                    'finish_time': finish_time
                }
                
                # Update waiting time stats (for the job)
                if finish_event['subtime'] != current_time:
                    self.sim_monitor['avg_waiting_time'] += (current_time - finish_event['subtime'])
                    self.sim_monitor['waiting_event_count'] += 1
                
                # Push the finish event
                heapq.heappush(schedule_queue, (finish_time, MyDict(finish_event)))
                
                # Keep track in active_jobs
                active_jobs.append(finish_event)
                
                # Update resource energy consumption for busy (execution) time
                for r_id in allocated:
                    # wattage_per_state[3] could be "full usage" or "max usage" watt
                    self.sim_monitor['energy_consumption'][r_id] += (finish_time - current_time) * self.machines[r_id]['wattage_per_state'][3]
                    # Mark resource as non-idle
                    self.sim_monitor['start_idle'][r_id] = -1
                
                # Collect data for monitor
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
            
            # -------------------------
            # Finish execution
            # -------------------------
            elif event['type'] == 'execution_finished':
                allocated = event['allocated_resources']
                # Free the resources
                available_resources.extend(allocated)
                available_resources.sort()
                
                # Remove from active_jobs
                active_jobs = [aj for aj in active_jobs if aj['id'] != event['id']]
                
                # The number of resources currently free
                temp_available_resource = len(available_resources)
                
                # Check if there are any other events scheduled at the same current_time
                # that might also start. We want to subtract resources for them too.
                events_now = [(t, e) for (t, e) in schedule_queue if t == current_time]
                for _, e_now in events_now:
                    if e_now['type'] == 'execution_start':
                        temp_available_resource -= e_now['res']
                
               
                    
                # First, try to schedule waiting jobs in order (FCFS)
                waiting_queue = sorted(waiting_queue, key=lambda x: x['id'])  # or subtime if you prefer
                for _ in range(len(waiting_queue)):
                    job = waiting_queue[0]  # Selalu cek job pertama
                    if temp_available_resource >= job['res']:
                        popped_job = waiting_queue.pop(0)
                        popped_job['type'] = 'execution_start'
                        temp_available_resource -= popped_job['res']
                        heapq.heappush(schedule_queue, (current_time, MyDict(popped_job)))
                    else:
                        break
                    
                # Try to start as many from the front of waiting queue as possible
                for _ in range(len(waiting_queue)):
                    if not waiting_queue:
                        break
                    job = waiting_queue[0]
                    if temp_available_resource >= job['res']:
                        # Attempt to schedule
                        job_can_backfill = self.check_backfilling(current_time, job, temp_available_resource, active_jobs)
                        if job_can_backfill:
                            waiting_queue.pop(0)
                            job['type'] = 'execution_start'
                            heapq.heappush(schedule_queue, (current_time, MyDict(job)))
                            temp_available_resource -= job['res']
                        else:
                            # If we fail backfilling check, we skip this job for now
                            # (We do not pop it out, might break or continue)
                            break
                    else:
                        break
                
                # Now, do a second pass to see if any smaller jobs behind can backfill 
                # if the first waiting job can’t start. (Classic EASY tries to fill with smaller jobs behind.)
                # This is the typical "backfill pass"
                active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                
                while True:
                    is_pushed = False
                    for idx, job in enumerate(waiting_queue):
                        if temp_available_resource >= job['res']:
                            # Try backfill
                            job_can_backfill = self.check_backfilling(current_time, job, temp_available_resource, active_jobs)
                            if job_can_backfill:
                                # This job can start
                                job = waiting_queue.pop(idx)
                                job['type'] = 'execution_start'
                                heapq.heappush(schedule_queue, (current_time, MyDict(job)))
                                temp_available_resource -= job['res']
                                
                                is_pushed = True
                                break
                            # else can't backfill -> skip it, check next job
                        # else not enough resources -> check next
                    if not is_pushed:
                        # No job was scheduled in this pass, so end loop
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