import heapq
import copy
from .env import MyDict

# t = waktu pada terjadinya event
# T = waktu job terakhir selesai dieksekusi

# Ctm = 0 ketika node-m tidak melakukan komputasi pada waktu t
# Ctm = 1 ketika node-m melakukan komputasi pada waktu t

# Utm = state dari node ke-m pada waktu t
# 0 -> aktif
# 1 -> sleep
# 2 -> switching on
# 3 -> switching off

# Pa = consumption rate aktif
# Ps = sleep
# Pson = switching on
# Psof = switching off

# node * consumption rate


# R1(k, k+1) = consumption rate waktu event ke-k + consumption rate waktu event ke-k+1

# R2(k,k+1) = total jumlah job pada waiting_queue pada waktu event ke-k + total jumlah job pada waiting_queue pada waktu event ke-k+1

# M = Jumlah node

# J(k, k+1) = jika ada job submission time di antara waktu event ke-k dan event ke-k+1 maka dikali Δt

def reward1(self):
    count_son = len(self.on_off_resources)
    count_sof = len(self.off_on_resources)
    count_s = len(self.inactive_resources)
    
    total_consumption_rate_son = count_son * 190
    total_consumption_rate_sof = count_sof * 9
    total_consumption_rate_s = count_s * 9
    total_consumption_rate_a = (self.nb_res - count_son - count_sof - count_s) * 190
    
        
    return total_consumption_rate_son + total_consumption_rate_sof + total_consumption_rate_s + total_consumption_rate_a 
    
def reward2(self):
    return len(self.waiting_queue)

def max_jwt(self, waiting_queue, executed_jobs, next_time, delta_t):
    result = 0
    
    for job in waiting_queue:
        if job['subtime'] <= next_time and job['subtime'] >= self.current_time:
           result += delta_t 

    for job in executed_jobs:
        if job['subtime'] <= next_time and job['subtime'] >= self.current_time:
           result += delta_t 
           
    if result == 0:
        return 1
    
    return result

def step(self, timeout):
    if self.schedule_queue:
        event_time, event = heapq.heappop(self.schedule_queue)
    else:
        event = self.waiting_queue.pop(0)
    
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
        

    for index_available_resource in self.available_resources:
        self.sim_monitor['start_idle'][index_available_resource] = current_time
    
    if event['type'] == 'switch_off':
        valid_switch_off = [item for item in event['node'] if item in self.available_resources]
        
        if len(valid_switch_off) == 0:
            return
        
        self.available_resources = [item for item in self.available_resources if item not in valid_switch_off]
        self.on_off_resources.extend(valid_switch_off)
        self.on_off_resources = sorted(self.on_off_resources)
        self.update_nb_res(current_time, event, event['type'], valid_switch_off)
        
        heapq.heappush(self.schedule_queue, (current_time + self.transition_time[0], MyDict({'node': copy.deepcopy(valid_switch_off), 'type': 'turn_off' })))
        
    elif event['type'] == 'turn_off':
        self.inactive_resources.extend(event['node'])
        self.inactive_resources = sorted(self.inactive_resources)
        
        self.on_off_resources = [item for item in self.on_off_resources if item not in event['node']]
        self.on_off_resources = sorted(self.on_off_resources)
        self.update_nb_res(current_time, event, event['type'], event['node'])
        
    elif event['type'] == 'switch_on':
        # kayaknya perlu nambahin valid switch on
        self.inactive_resources = [item for item in self.inactive_resources if item not in event['node']]
        self.off_on_resources.extend(event['node'])
        self.off_on_resources = sorted(self.off_on_resources)
        self.update_nb_res(current_time, event, event['type'], event['node'])
        
        heapq.heappush(self.schedule_queue, (current_time + self.transition_time[1], MyDict({'node': copy.deepcopy(event['node']), 'type': 'turn_on' })))
        
    elif event['type'] == 'turn_on':
        self.available_resources.extend(event['node'])
        self.available_resources = sorted(self.available_resources)
        self.off_on_resources = [item for item in self.off_on_resources if item not in event['node']]
        self.off_on_resources = sorted(self.off_on_resources)
        self.update_nb_res(current_time, event, event['type'], event['node'])
        
    elif event['type'] == 'arrival':
        if len(self.available_resources) + len(self.inactive_resources) >= event['res']:
            if self.waiting_queue:
                self.active_jobs = sorted(self.active_jobs, key=lambda x: x['finish_time'])
                backfilled_node_count = 0
                if self.check_backfilling(current_time, event, len(self.available_resources) + len(self.inactive_resources), self.active_jobs, self.waiting_queue[0], backfilled_node_count):
                    event['type'] = 'execution_start'
                    if len(self.available_resources) < event['res']:
                        heapq.heappush(self.schedule_queue, (current_time + self.transition_time[1], MyDict(event)))
                    else:
                        heapq.heappush(self.schedule_queue, (current_time, MyDict(event)))
                else:
                    event['current_time'] = current_time
                    self.waiting_queue.append(event)
            else:
                event['type'] = 'execution_start'
                heapq.heappush(self.schedule_queue, (current_time, MyDict(event)))
        else:
            event['current_time'] = current_time
            self.waiting_queue.append(event)
            
    elif event['type'] == 'execution_start':
        if event['res'] > len(self.available_resources):
            if event.has_key('reserve') == False:
                self.reserved_count += event['res']
                event['reserve'] = True
            if len(self.off_on_resources) < event['res']:
                inactive_node_sorted = sorted(self.inactive_resources)
                available_node_sorted = sorted(self.available_resources)
                merged = inactive_node_sorted + available_node_sorted
                merged = sorted(merged)
                activated_nodes = []

                for node in merged:
                    if len(activated_nodes) < event['res']:
                        activated_nodes.append(node)
                
                intersection = list(set(self.inactive_resources) & set(activated_nodes))
                heapq.heappush(self.schedule_queue, (current_time, MyDict({'type': 'switch_on', 'node': copy.deepcopy(intersection)})))
            heapq.heappush(self.schedule_queue, (current_time + self.transition_time[1], MyDict(event)))
            
            return
        

        if event['res'] <= len(self.available_resources) and (self.inactive_resources and min(self.inactive_resources) < self.available_resources[event['res'] - 1]):
            inactive_node_sorted = sorted(self.inactive_resources)
            available_node_sorted = sorted(self.available_resources)
            merged = inactive_node_sorted + available_node_sorted
            merged = sorted(merged)
            activated_nodes = []

            for node in merged:
                if len(activated_nodes) < event['res']:
                    activated_nodes.append(node)
            
            intersection = list(set(self.inactive_resources) & set(activated_nodes))
            
            heapq.heappush(self.schedule_queue, (current_time, MyDict({'type': 'switch_on', 'node': copy.deepcopy(intersection)})))
            
            if event.has_key('reserve') == False:
                self.reserved_count += event['res']
                event['reserve'] = True
            heapq.heappush(self.schedule_queue, (current_time + self.transition_time[1], MyDict(event)))
            
            return
        
        if event.has_key('reserve') and event['reserve'] == True:
            self.reserved_count -= event['res']
            
        allocated = self.available_resources[:event['res']]
        self.available_resources = self.available_resources[event['res']:]
        
        self.update_nb_res(current_time, event, 'allocate', allocated)

        self.executed_jobs.append(event)
        
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
            
        heapq.heappush(self.schedule_queue, (finish_time, MyDict(finish_event)))
        
        finish_event['finish_time'] = finish_time
        self.active_jobs.append(finish_event)
        
        for i in allocated:
            self.sim_monitor['energy_consumption'][i] += (finish_time - current_time) * self.machines[i]['wattage_per_state'][3]
            self.sim_monitor['start_idle'][i] = -1
        
        self.monitor_jobs.append({
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
        self.available_resources.extend(allocated)
        self.available_resources.sort() 
        for index_available_resource in self.available_resources:
            self.sim_monitor['start_idle'][index_available_resource] = current_time

        self.active_jobs = [active_job for active_job in self.active_jobs if active_job['id'] != event['id']]
        
        temp_available_resource_2 = len(self.available_resources)
        temp_available_resource = len(self.available_resources) + len(self.inactive_resources) + len(self.off_on_resources) - self.reserved_count
        check_if_need_activation = temp_available_resource - temp_available_resource_2
        
        self.update_nb_res(current_time, event, 'release', allocated)
        
        skipbf = False
        for aj in self.active_jobs:
            if aj['finish_time'] == current_time:
                skipbf=True
                
        if skipbf == True:
            return
        
        self.waiting_queue = sorted(self.waiting_queue)

        
        for _ in range(len(self.waiting_queue)):
            job = self.waiting_queue[0]
            if temp_available_resource >= job['res']:
                popped_job = self.waiting_queue.pop(0)
                popped_job['type'] = 'execution_start'
                temp_available_resource -= popped_job['res']
                self.reserved_count += popped_job['res']
                popped_job['reserve'] = True
                heapq.heappush(self.schedule_queue, (current_time, MyDict(popped_job)))
            else:
                break

        self.active_jobs = sorted(self.active_jobs, key=lambda x: x['finish_time'])
        backfilled_node_count = 0
        while True:
            is_pushed = False
            for k in range(0, len(self.waiting_queue)):
                job = self.waiting_queue[k]
                if temp_available_resource >= job['res']:
                    if not self.check_backfilling(current_time, job, temp_available_resource, self.active_jobs, self.waiting_queue[0], backfilled_node_count):
                        continue
                    
                    backfilled_node_count += job['res']
                    next_job = self.waiting_queue.pop(k)
                    next_job['type'] = 'execution_start'
                    temp_available_resource -= next_job['res']
                    
                    heapq.heappush(self.schedule_queue, (current_time, MyDict(next_job)))
                    is_pushed = True
                    break
            if is_pushed == False:
                break
    
        if temp_available_resource < check_if_need_activation:
            
            heapq.heappush(self.schedule_queue, (current_time, MyDict({'type':'switch_on', 'node': self.inactive_resources[:check_if_need_activation - temp_available_resource - len(self.off_on_resources)]})))

    mask = self.sim_monitor['nb_res']['time'] == current_time
    has_idle = (self.sim_monitor['nb_res'].loc[mask, 'idle'] > 0).any()
    if has_idle:
        heapq.heappush(self.schedule_queue, (current_time + timeout, MyDict({'type':'switch_off', 'node': copy.deepcopy(self.available_resources)})))
        

    
def simulate_easy(self, timeout):
    for event in self.jobs:
        event_time, event_detail = event
        heapq.heappush(self.schedule_queue, (event_time, MyDict(event_detail)))
    
    heapq.heappush(self.schedule_queue, (timeout, MyDict({'node': copy.deepcopy(self.available_resources), 'type': 'switch_off'})))
    

    while self.schedule_queue or self.waiting_queue:      
        self.step(timeout)
        self.step_count += 1
        if len(self.schedule_queue) != 0 and len(self.waiting_queue) != 0:
            # MAKE ACTION HERE -> IT WILL IMMEDIATELY 
            copied_instance = copy.deepcopy(self)
            copied_instance.step(timeout)
            reward1 = self.reward1() + copied_instance.reward1()
            reward2 = self.reward2() + copied_instance.reward2()
            delta_t = copied_instance.current_time - self.current_time
            jwt = self.max_jwt(copied_instance.waiting_queue, copied_instance.executed_jobs, copied_instance.current_time, delta_t)
            
            alpha = 0.5
            beta = 0.5
            final_reward = -alpha * (reward1 / self.nb_res * 190 * delta_t) - beta * (reward2/jwt)
            
            
            print(f'STEP-{self.step_count} ========== {final_reward}')
            
    return self.monitor_jobs