import heapq
import copy
from .env import MyDict
import torch
import torch.optim as optim
import torch.nn as nn



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

def fcfsbf(self, temp_available_resource):
    self.waiting_queue = sorted(self.waiting_queue)
        
    for _ in range(len(self.waiting_queue)):
        job = self.waiting_queue[0]
        if temp_available_resource >= job['res']:
            popped_job = self.waiting_queue.pop(0)
            popped_job['type'] = 'execution_start'
            temp_available_resource -= popped_job['res']
            popped_job['reserve'] = True
            self.waiting_queue_ney.append(popped_job)
            self.waiting_queue_ney = sorted(self.waiting_queue_ney)
            heapq.heappush(self.schedule_queue, (self.current_time, MyDict(popped_job)))
        else:
            break

    self.active_jobs = sorted(self.active_jobs, key=lambda x: x['finish_time'])
    backfilled_node_count = 0
    while True:
        is_pushed = False
        for k in range(0, len(self.waiting_queue)):
            job = self.waiting_queue[k]
            if temp_available_resource >= job['res']:
                if not self.check_backfilling(self.current_time, job, temp_available_resource, self.active_jobs, self.waiting_queue[0], backfilled_node_count):
                    continue
                
                backfilled_node_count += job['res']
                next_job = self.waiting_queue.pop(k)
                next_job['type'] = 'execution_start'
                temp_available_resource -= next_job['res']
                self.waiting_queue_ney.append(next_job)
                self.waiting_queue_ney = sorted(self.waiting_queue_ney)
                heapq.heappush(self.schedule_queue, (self.current_time, MyDict(next_job)))
                is_pushed = True
                break
        if is_pushed == False:
            break
        
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
    return len(self.waiting_queue) + len(self.waiting_queue_ney)

# def reward2(self):
#     result = 0
#     for job in self.waiting_queue:
#         result += self.current_time - job['subtime']
#     for job in self.waiting_queue_ney:
#         result += self.current_time - job['subtime']
#     return result

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

def step(self, event):
    temp_index = 0
    for node_action in self.sim_monitor['nodes_action']:
        if node_action['state'] == 'sleeping':
            rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][0]
        elif node_action['state'] == 'idle':
            rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][1]
        elif node_action['state'] == 'computing':
            rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][2]
        elif node_action['state'] == 'switching_on':
            rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][3]
        elif node_action['state'] == 'switching_off':
            rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][4]
        
        rate_energy_consumption = self.machines[temp_index]['wattage_per_state'][1]
        duration = self.current_time - node_action['time']
        self.sim_monitor['energy_consumption'][temp_index] += (duration * rate_energy_consumption)
        
        self.sim_monitor['nodes_action'][temp_index]['time'] = self.current_time
    
        temp_index +=1

    
    if event['type'] == 'turn_off':
        for node in event['node']:
            self.sim_monitor['nodes_action'][node]['state'] = 'sleeping'
            self.sim_monitor['nodes_action'][node]['time'] = self.current_time
            
        self.inactive_resources.extend(event['node'])
        self.inactive_resources = sorted(self.inactive_resources)
        self.on_off_resources = [item for item in self.on_off_resources if item not in event['node']]
        self.on_off_resources = sorted(self.on_off_resources)
        self.update_nb_res(self.current_time, event, event['type'], event['node'])
        
    elif event['type'] == 'turn_on':
        for node in event['node']:
            self.sim_monitor['nodes_action'][node]['state'] = 'idle'
            self.sim_monitor['nodes_action'][node]['time'] = self.current_time
        self.available_resources.extend(event['node'])
        self.available_resources = sorted(self.available_resources)
        self.off_on_resources = [item for item in self.off_on_resources if item not in event['node']]
        self.off_on_resources = sorted(self.off_on_resources)
        self.update_nb_res(self.current_time, event, event['type'], event['node'])
        
        temp_available_resource = len(self.available_resources)
        self.fcfsbf(temp_available_resource)
        
        
    elif event['type'] == 'arrival':
        self.total_req_res += event['res']
        self.arrival_count += 1
        if len(self.available_resources) + len(self.inactive_resources) >= event['res']:
            if self.waiting_queue:
                self.active_jobs = sorted(self.active_jobs, key=lambda x: x['finish_time'])
                backfilled_node_count = 0
                if self.check_backfilling(self.current_time, event, len(self.available_resources) + len(self.inactive_resources), self.active_jobs, self.waiting_queue[0], backfilled_node_count):
                    event['type'] = 'execution_start'
                    if len(self.available_resources) < event['res']:
                        heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict(event)))
                    else:
                        heapq.heappush(self.schedule_queue, (self.current_time, MyDict(event)))
                else:
                    event['current_time'] = self.current_time
                    self.waiting_queue.append(event)
            else:
                event['type'] = 'execution_start'
                heapq.heappush(self.schedule_queue, (self.current_time, MyDict(event)))
        else:
            event['current_time'] = self.current_time
            self.waiting_queue.append(event)
            
    elif event['type'] == 'execution_start':
        new_waiting_queue_ney = []
        for d in self.waiting_queue_ney:
            if d['id'] != event['id']:
                new_waiting_queue_ney.append(d)
        self.waiting_queue_ney = new_waiting_queue_ney


        if event['res'] > len(self.available_resources):
            self.waiting_queue.append(event)
            return
        
            
        allocated = self.available_resources[:event['res']]
        self.available_resources = self.available_resources[event['res']:]
        
        for node in allocated:
            self.sim_monitor['nodes_action'][node]['state'] = 'computing'
            self.sim_monitor['nodes_action'][node]['time'] = self.current_time
            
        self.update_nb_res(self.current_time, event, 'allocate', allocated)

        self.executed_jobs.append(event)
        
        finish_time = self.current_time + event['walltime']
        finish_event = {
            'id': event['id'],
            'res': event['res'],
            'walltime': event['walltime'],
            'type': 'execution_finished',
            'subtime': event['subtime'],
            'profile': event['profile'],
            'allocated_resources': allocated
        }
        
        if finish_event['subtime'] != self.current_time:
            self.sim_monitor['avg_waiting_time'] += (self.current_time - finish_event['subtime'])
            self.sim_monitor['waiting_event_count'] += 1
            
        heapq.heappush(self.schedule_queue, (finish_time, MyDict(finish_event)))
        
        finish_event['finish_time'] = finish_time
        self.active_jobs.append(finish_event)
        
        
        self.monitor_jobs.append({
            'job_id': event['id'],
            'workload_name': 'w0',
            'profile': event['profile'],
            'submission_time': event['subtime'],
            'requested_number_of_resources': event['res'],
            'requested_time': event['walltime'],
            'success': 0,
            'final_state': 'COMPLETED_WALLTIME_REACHED',
            'starting_time': self.current_time,
            'execution_time': event['walltime'],
            'finish_time': finish_time,
            'waiting_time': self.current_time - event['subtime'],
            'turnaround_time': finish_time - event['subtime'],
            'stretch': (finish_time - event['subtime']) / event['walltime'],
            'allocated_resources': allocated,
            'consumed_energy': -1
        })
    
    elif event['type'] == 'execution_finished':
        allocated = event['allocated_resources']
        self.available_resources.extend(allocated)
        self.available_resources.sort() 
        
        for node in allocated:
            self.sim_monitor['nodes_action'][node]['state'] = 'idle'
            self.sim_monitor['nodes_action'][node]['time'] = self.current_time

        self.update_nb_res(self.current_time, event, 'release', allocated)
        
        self.active_jobs = [active_job for active_job in self.active_jobs if active_job['id'] != event['id']]
        
        temp_available_resource = len(self.available_resources) + len(self.off_on_resources)
        
        
        skipbf = False
        for aj in self.active_jobs:
            if aj['finish_time'] == self.current_time:
                skipbf=True
                
        if skipbf == True:
            return
        
        self.fcfsbf(temp_available_resource)
        
    
def simulate_easy(self):
    for event in self.jobs:
        event_time, event_detail = event
        heapq.heappush(self.schedule_queue, (event_time, MyDict(event_detail)))

    while self.schedule_queue or self.waiting_queue:
        if self.schedule_queue:
            event_time, event = heapq.heappop(self.schedule_queue)
        else:
            event = self.waiting_queue.pop(0)
        
        self.current_time = event_time
        
        self.step(event)
        self.step_count += 1
        print(f'=============== {self.step_count} ===============')
        print('event type: ',event['type'])
        if event.has_key('id'):
            print('event id: ', event['id'])
        print('current_time: ',self.current_time)
        if len(self.schedule_queue) != 0 or len(self.waiting_queue) != 0:
            
            # waiting_queue_ney are the job popped out of waiting_queue and added into schedule queue with event as execution start at hasnt been executed yet.add()
            
            total_waiting_time = 0
            needed_res_for_execution = 0
            if len(self.waiting_queue) > 0:
                for job in self.waiting_queue:
                    total_waiting_time += self.current_time - job['subtime']
            if len(self.waiting_queue_ney) > 0:
                for job in self.waiting_queue_ney:
                    total_waiting_time += self.current_time - job['subtime']
                    needed_res_for_execution += job['res']
            
                
             
            # Global Feature:
            # 1. num of job in waiting queue + waiting queue ney
            # 2. total waiting time
            # 3. total requested resource for all job in waiting queue
            # 4. num of idle resource
            # 5. num of sleeping resource
            # 6. num of resource needed for job ready for execution
            
            # Node Feature:
            # 1. state: 0 sleeping, 1 idle
            # 2. Transition cost
            # 3. Transition time
            
            
            gfb1 = len(self.waiting_queue) + len(self.waiting_queue_ney)
            gfb2 = total_waiting_time
            gfb3 = self.total_req_res
            gfb4 = len(self.available_resources)
            gfb5 = len(self.inactive_resources)
            gfb6 = needed_res_for_execution
            

            global_feature = [gfb1, gfb2, gfb3, gfb4, gfb5, gfb6]
            
            nodes = self.available_resources + self.inactive_resources
            nodes.sort()
            
            if len(nodes) == 0:
                continue
            
            nodes_features = []
            for node in nodes:
                node_features = copy.deepcopy(global_feature)

                if node in self.available_resources:
                    node_features.append(1) # idle = 1
                    node_features.append(9) # switch off consumption rate = 9
                    node_features.append(5) # transition time = 5
                elif node in self.inactive_resources:
                    node_features.append(0) #inactive = 1
                    node_features.append(190) # switch on consumption rate = 190
                    node_features.append(5) # transition time = 5
                
                nodes_features.append(node_features)
                
            feature = torch.tensor(nodes_features, dtype=torch.float32)
            
            output = self.model(feature)
            if len(nodes) == 0:
                print('here')
                
            action = output.squeeze(0).tolist()


            if isinstance(action, (int, float)):
                action = [action]

            index = 0
            switch_off = []
            switch_on = []
            for node in nodes:
                if node in self.available_resources and action[index] >= 0.5:
                    switch_off.append(node)
                    self.sim_monitor['nodes_action'][node]['state'] = 'switching_off'
                    self.sim_monitor['nodes_action'][node]['time'] = self.current_time
                elif node in self.inactive_resources and action[index] >= 0.5:
                    self.sim_monitor['nodes_action'][node]['state'] = 'switching_on'
                    self.sim_monitor['nodes_action'][node]['time'] = self.current_time
                    switch_on.append(node)
            
            #switching off
            if len(switch_off) > 0:
                self.available_resources = [item for item in self.available_resources if item not in switch_off]
                self.on_off_resources.extend(switch_off)
                self.on_off_resources = sorted(self.on_off_resources)
                self.update_nb_res(self.current_time, event, 'switch_off', switch_off)
                heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[0], MyDict({'node': copy.deepcopy(switch_off), 'type': 'turn_off' })))
                
            #switching on
            if len(switch_on) > 0:
                self.inactive_resources = [item for item in self.inactive_resources if item not in switch_on]
                self.off_on_resources.extend(switch_on)
                self.off_on_resources = sorted(self.off_on_resources)
                self.update_nb_res(self.current_time, event, 'switch_on', switch_on)
                heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict({'node': copy.deepcopy(switch_on), 'type': 'turn_on' })))
                
            
            print(f'action : switch on {len(switch_on)} nodes, switch off {len(switch_off)} nodes')
            copied_instance = copy.deepcopy(self)
            
            # for i in range(1):
            dostep = False
            if copied_instance.schedule_queue:
                n_event_time, n_event = heapq.heappop(copied_instance.schedule_queue)
                dostep = True
            elif copied_instance.waiting_queue:
                n_event = copied_instance.waiting_queue.pop(0)
                dostep = True
                
            if dostep:
                copied_instance.current_time = n_event_time
                copied_instance.step(n_event)
            
            reward1 = self.reward1() + copied_instance.reward1()
            reward2 = self.reward2() + copied_instance.reward2()
            delta_t = copied_instance.current_time - self.current_time
           
            if delta_t == 0:
                print('here')

            jwt = self.max_jwt(copied_instance.waiting_queue, copied_instance.executed_jobs, copied_instance.current_time, delta_t)
            
            alpha = 0.5
            beta = 0.5
            fr1 = -alpha * (reward1 / (self.nb_res * 190 * delta_t))
            fr2 = -beta * (reward2/jwt)
            # fr2 = -beta * total_waiting_time * 10
            final_reward = fr1 + fr2
            

            print('reward ec: ',fr1)
            print('reward wt: ',fr2)
            print('final reward: ',final_reward)
            # print(action)
            
            loss = torch.tensor(final_reward, dtype=torch.float32, requires_grad=True)
            self.optimizer.zero_grad()  
            loss.backward() 
            self.optimizer.step()  
    
    for x in self.sim_monitor['nodes']:
        if x[len(x)-1]['finish_time'] != self.current_time:
            x[len(x)-1]['finish_time'] = self.current_time
            
            
    return self.monitor_jobs