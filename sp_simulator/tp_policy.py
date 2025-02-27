import heapq
import copy
from .env import MyDict

def simulate_easy(self, timeout):
    self.reserved_list = []
    for event in self.jobs:
        event_time, event_detail = event
        heapq.heappush(self.schedule_queue, (event_time, MyDict(event_detail)))
    
    heapq.heappush(self.schedule_queue, (timeout, MyDict({'node': copy.deepcopy(self.available_resources), 'type': 'switch_off'})))
    

    while self.schedule_queue or self.waiting_queue:      
        if self.schedule_queue:
            event_time, event = heapq.heappop(self.schedule_queue)
        else:
            event = self.waiting_queue.pop(0)
        
        self.current_time = event_time
        self.update_energy_consumption()
       
        if event['type'] == 'switch_off':
            valid_switch_off = [item for item in event['node'] if item in self.available_resources]
            
            if len(valid_switch_off) == 0:
                continue
            
            self.available_resources = [item for item in self.available_resources if item not in valid_switch_off]
            self.on_off_resources.extend(valid_switch_off)
            self.on_off_resources = sorted(self.on_off_resources)
            self.update_node_action(valid_switch_off, event, 'switch_off', 'switching_off')
            
            heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[0], MyDict({'node': copy.deepcopy(valid_switch_off), 'type': 'turn_off' })))
            
        elif event['type'] == 'turn_off':
            self.inactive_resources.extend(event['node'])
            self.inactive_resources = sorted(self.inactive_resources)
            
            self.on_off_resources = [item for item in self.on_off_resources if item not in event['node']]
            self.on_off_resources = sorted(self.on_off_resources)
            self.update_node_action(event['node'], event, 'turn_off', 'sleeping')
            
        elif event['type'] == 'switch_on':
            self.inactive_resources = [item for item in self.inactive_resources if item not in event['node']]
            self.off_on_resources.extend(event['node'])
            self.off_on_resources = sorted(self.off_on_resources)
            self.update_node_action(event['node'], event, 'switch_on', 'switching_on')
            
            heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict({'node': copy.deepcopy(event['node']), 'type': 'turn_on' })))
            
        elif event['type'] == 'turn_on':
            self.available_resources.extend(event['node'])
            self.available_resources = sorted(self.available_resources)
            self.off_on_resources = [item for item in self.off_on_resources if item not in event['node']]
            self.off_on_resources = sorted(self.off_on_resources)
            self.update_node_action(event['node'], event, 'turn_on', 'idle')
            
        elif event['type'] == 'arrival':
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
            if event['res'] > len(self.available_resources):
                if event.has_key('reserve') == False:
                    self.reserved_count += event['res']
                    event['reserve'] = True
                heapq.heappush(self.schedule_queue, (self.current_time + self.transition_time[1], MyDict(event)))
                continue
            
            if event.has_key('reserve') and event['reserve'] == True:
                self.reserved_count -= event['res']
            
            allocated = self.available_resources[:event['res']]
            self.available_resources = self.available_resources[event['res']:]
            
            self.update_node_action(allocated, event, 'allocate', 'computing')

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
            
            pre_switch_time = finish_time - self.transition_time[1] 
            pre_switch_event = {
                'type': 'pre_switch_on_check',
                'finish_time': finish_time,
                'id': event['id'],
                'res': event['res']
            }
            heapq.heappush(self.schedule_queue, (pre_switch_time, MyDict(pre_switch_event)))
            
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

            self.active_jobs = [active_job for active_job in self.active_jobs if active_job['id'] != event['id']]
            
            temp_available_resource_2 = len(self.available_resources)
            temp_available_resource = len(self.available_resources) + len(self.inactive_resources) + len(self.off_on_resources) - self.reserved_count
            check_if_need_activation = temp_available_resource - temp_available_resource_2
            
            self.update_nb_res(self.current_time, event, 'release', allocated)
            
            skipbf = False
            for aj in self.active_jobs:
                if aj['finish_time'] == self.current_time:
                    skipbf=True
                    
            if skipbf == True:
                continue
            
            self.waiting_queue = sorted(self.waiting_queue)

            for _ in range(len(self.waiting_queue)):
                job = self.waiting_queue[0]
                if temp_available_resource >= job['res']:
                    popped_job = self.waiting_queue.pop(0)
                    popped_job['type'] = 'execution_start'
                    temp_available_resource -= popped_job['res']
                    self.reserved_count += popped_job['res']
                    popped_job['reserve'] = True
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
                        
                        heapq.heappush(self.schedule_queue, (self.current_time, MyDict(next_job)))
                        is_pushed = True
                        break
                if is_pushed == False:
                    break
        
            if temp_available_resource < check_if_need_activation:
                
                heapq.heappush(self.schedule_queue, (self.current_time, MyDict({'type':'switch_on', 'node': self.inactive_resources[:check_if_need_activation - temp_available_resource - len(self.off_on_resources)]})))
        
        elif event['type'] == 'pre_switch_on_check':
            temp_available_resource_2 = len(self.available_resources)
            temp_available_resource = len(self.available_resources) + len(self.inactive_resources) - self.reserved_count + len(self.off_on_resources)
            check_if_need_activation = temp_available_resource - temp_available_resource_2
            _active_jobs = copy.deepcopy(self.active_jobs)

            for aj in self.active_jobs:
                if self.current_time < aj['finish_time'] <= self.current_time + self.transition_time[1]:
                    temp_available_resource += aj['res']
                    
            for rl in self.reserved_list:
                if self.current_time - self.transition_time[1] < rl['reserve_time'] < self.current_time:
                    temp_available_resource -= rl['res']
                    
            _waiting_queue = copy.deepcopy(self.waiting_queue)
            _waiting_queue = sorted(_waiting_queue)
            
            _waiting_queue = [item for item in _waiting_queue if item['id'] not in {reserved['id'] for reserved in self.reserved_list}]
            
            skip = False
            for t,e in self.schedule_queue:
                if t == self.current_time and e['type'] == 'pre_switch_on_check':
                    skip = True
                    break
            if skip:
                continue
            
            for _ in range(len(_waiting_queue)):
                job = _waiting_queue[0]
                if temp_available_resource >= job['res']:
                    temp_available_resource -= job['res']
                    job['reserve_time']= self.current_time
                    self.reserved_list.append(job)
                    _waiting_queue.pop(0)
                else:
                    break

            backfilled_node_count = 0
            
            _active_jobs = copy.deepcopy(self.active_jobs)
            _active_jobs = [item for item in _active_jobs if item['finish_time'] > self.current_time + 5]
            
            while True:
                is_pushed = False
                for k in range(0, len(_waiting_queue)):
                    job = _waiting_queue[k]
                    if temp_available_resource >= job['res']:
                        if not self.check_backfilling(self.current_time+self.transition_time[1], job, temp_available_resource, _active_jobs, _waiting_queue[0], backfilled_node_count):
                            continue

                        backfilled_node_count += job['res']
                        temp_available_resource -= job['res']
                        job['reserve_time']= self.current_time
                        self.reserved_list.append(job)
                        is_pushed = True
                        break
                if is_pushed == False:
                    break
            if temp_available_resource < check_if_need_activation:
                
                heapq.heappush(self.schedule_queue, (self.current_time, MyDict({'type':'switch_on', 'node': self.inactive_resources[:check_if_need_activation - temp_available_resource]})))

        mask = self.sim_monitor['nb_res']['time'] == self.current_time
        has_idle = (self.sim_monitor['nb_res'].loc[mask, 'idle'] > 0).any()
        if has_idle:
            heapq.heappush(self.schedule_queue, (self.current_time + timeout, MyDict({'type':'switch_off', 'node': copy.deepcopy(self.available_resources)})))
            
    for x in self.sim_monitor['nodes']:
        if x[len(x)-1]['finish_time'] != self.current_time:
            x[len(x)-1]['finish_time'] = self.current_time
            
    common_jobs = [res for res in self.reserved_list if any(res['id'] == job['job_id'] for job in self.monitor_jobs)]

    filtered_jobs = [
        res for res in common_jobs
        if any(res['reserve_time'] + 5 != job['starting_time'] for job in self.monitor_jobs if res['id'] == job['job_id'])]

    fault_list = []
    for _job in filtered_jobs:
        new_dict = {
            'id': _job['id'],
            'reserve_time': _job['reserve_time'],
        }
        target_dict = next((item for item in self.monitor_jobs if item['job_id'] == _job['id']), None)
        new_dict['starting_time'] = target_dict['starting_time']
        new_dict['finish_time'] = target_dict['finish_time']
        fault_list.append(new_dict)
    
    return self.monitor_jobs, fault_list
