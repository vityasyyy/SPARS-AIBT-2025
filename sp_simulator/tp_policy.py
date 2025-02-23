import heapq
import copy
from .env import MyDict
import pandas as pd

def simulate_easy(self, timeout):
    current_time = 0
    available_resources = list(range(self.nb_res))
    inactive_resources = []
    on_off_resources = []
    off_on_resources = []
    schedule_queue = []
    waiting_queue = []
    monitor_jobs=[]
    active_jobs = []
    reserved_count = 0

    reserved_list = []
    for event in self.jobs:
        event_time, event_detail = event
        heapq.heappush(schedule_queue, (event_time, MyDict(event_detail)))
    
    heapq.heappush(schedule_queue, (timeout, MyDict({'node': copy.deepcopy(available_resources), 'type': 'switch_off'})))
    

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
        
        if event['type'] == 'switch_off':
            valid_switch_off = [item for item in event['node'] if item in available_resources]
            
            if len(valid_switch_off) == 0:
                continue
            
            available_resources = [item for item in available_resources if item not in valid_switch_off]
            on_off_resources.extend(valid_switch_off)
            on_off_resources = sorted(on_off_resources)
            self.update_nb_res(current_time, event, event['type'], valid_switch_off)
            
            heapq.heappush(schedule_queue, (current_time + self.transition_time[0], MyDict({'node': copy.deepcopy(valid_switch_off), 'type': 'turn_off' })))
            
        elif event['type'] == 'turn_off':
            inactive_resources.extend(event['node'])
            inactive_resources = sorted(inactive_resources)
            
            on_off_resources = [item for item in on_off_resources if item not in event['node']]
            on_off_resources = sorted(on_off_resources)
            self.update_nb_res(current_time, event, event['type'], event['node'])
            
        elif event['type'] == 'switch_on':
            inactive_resources = [item for item in inactive_resources if item not in event['node']]
            off_on_resources.extend(event['node'])
            off_on_resources = sorted(off_on_resources)
            self.update_nb_res(current_time, event, event['type'], event['node'])
            
            heapq.heappush(schedule_queue, (current_time + self.transition_time[1], MyDict({'node': copy.deepcopy(event['node']), 'type': 'turn_on' })))
            
        elif event['type'] == 'turn_on':
            available_resources.extend(event['node'])
            available_resources = sorted(available_resources)
            off_on_resources = [item for item in off_on_resources if item not in event['node']]
            off_on_resources = sorted(off_on_resources)
            self.update_nb_res(current_time, event, event['type'], event['node'])
            
        elif event['type'] == 'arrival':
            if len(available_resources) + len(inactive_resources) >= event['res']:
                if waiting_queue:
                    active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                    backfilled_node_count = 0
                    if self.check_backfilling(current_time, event, len(available_resources) + len(inactive_resources), active_jobs, waiting_queue[0], backfilled_node_count):
                        event['type'] = 'execution_start'
                        if len(available_resources) < event['res']:
                            heapq.heappush(schedule_queue, (current_time + self.transition_time[1], MyDict(event)))
                        else:
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
            if event['res'] > len(available_resources):
                if event.has_key('reserve') == False:
                    reserved_count += event['res']
                    event['reserve'] = True
                heapq.heappush(schedule_queue, (current_time + self.transition_time[1], MyDict(event)))
                continue
            
            if event.has_key('reserve') and event['reserve'] == True:
                reserved_count -= event['res']
                
            allocated = available_resources[:event['res']]
            available_resources = available_resources[event['res']:]
            
            self.update_nb_res(current_time, event, 'allocate', allocated)

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
            
            pre_switch_time = finish_time - self.transition_time[1] 
            pre_switch_event = {
                'type': 'pre_switch_on_check',
                'finish_time': finish_time,
                'id': event['id'],
                'res': event['res']
            }
            heapq.heappush(schedule_queue, (pre_switch_time, MyDict(pre_switch_event)))
            
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
            
            temp_available_resource_2 = len(available_resources)
            temp_available_resource = len(available_resources) + len(inactive_resources) + len(off_on_resources) - reserved_count
            check_if_need_activation = temp_available_resource - temp_available_resource_2
            
            self.update_nb_res(current_time, event, 'release', allocated)
            skipbf = False
            for aj in active_jobs:
                if aj['finish_time'] == current_time:
                    skipbf=True
                    
            if skipbf == True:
                continue
            
            waiting_queue = sorted(waiting_queue)

            for _ in range(len(waiting_queue)):
                job = waiting_queue[0]
                if temp_available_resource >= job['res']:
                    popped_job = waiting_queue.pop(0)
                    popped_job['type'] = 'execution_start'
                    temp_available_resource -= popped_job['res']
                    reserved_count += popped_job['res']
                    popped_job['reserve'] = True
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
        
            if temp_available_resource < check_if_need_activation:
                
                heapq.heappush(schedule_queue, (current_time, MyDict({'type':'switch_on', 'node': inactive_resources[:check_if_need_activation - temp_available_resource - len(off_on_resources)]})))
        
        elif event['type'] == 'pre_switch_on_check':
            temp_available_resource_2 = len(available_resources)
            temp_available_resource = len(available_resources) + len(inactive_resources) - reserved_count + len(off_on_resources)
            check_if_need_activation = temp_available_resource - temp_available_resource_2
            _active_jobs = copy.deepcopy(active_jobs)

            for aj in active_jobs:
                if current_time < aj['finish_time'] <= current_time + self.transition_time[1]:
                    temp_available_resource += aj['res']
                    
            for rl in reserved_list:
                if current_time - self.transition_time[1] < rl['reserve_time'] < current_time:
                    temp_available_resource -= rl['res']
                    
            _waiting_queue = copy.deepcopy(waiting_queue)
            _waiting_queue = sorted(_waiting_queue)
            
            _waiting_queue = [item for item in _waiting_queue if item['id'] not in {reserved['id'] for reserved in reserved_list}]
            
            skip = False
            for t,e in schedule_queue:
                if t == current_time and e['type'] == 'pre_switch_on_check':
                    skip = True
                    break
            if skip:
                continue
            
            for _ in range(len(_waiting_queue)):
                job = _waiting_queue[0]
                if temp_available_resource >= job['res']:
                    temp_available_resource -= job['res']
                    job['reserve_time']= current_time
                    reserved_list.append(job)
                    _waiting_queue.pop(0)
                else:
                    break

            backfilled_node_count = 0
            
            _active_jobs = copy.deepcopy(active_jobs)
            _active_jobs = [item for item in _active_jobs if item['finish_time'] > current_time + 5]
            
            while True:
                is_pushed = False
                for k in range(0, len(_waiting_queue)):
                    job = _waiting_queue[k]
                    if temp_available_resource >= job['res']:
                        if not self.check_backfilling(current_time+self.transition_time[1], job, temp_available_resource, _active_jobs, _waiting_queue[0], backfilled_node_count):
                            continue

                        backfilled_node_count += job['res']
                        temp_available_resource -= job['res']
                        job['reserve_time']= current_time
                        reserved_list.append(job)
                        is_pushed = True
                        break
                if is_pushed == False:
                    break
            if temp_available_resource < check_if_need_activation:
                
                heapq.heappush(schedule_queue, (current_time, MyDict({'type':'switch_on', 'node': inactive_resources[:check_if_need_activation - temp_available_resource]})))

        mask = self.sim_monitor['nb_res']['time'] == current_time
        has_idle = (self.sim_monitor['nb_res'].loc[mask, 'idle'] > 0).any()
        if has_idle:
            heapq.heappush(schedule_queue, (current_time + timeout, MyDict({'type':'switch_off', 'node': copy.deepcopy(available_resources)})))
            

    common_jobs = [res for res in reserved_list if any(res['id'] == job['job_id'] for job in monitor_jobs)]

    filtered_jobs = [
        res for res in common_jobs
        if any(res['reserve_time'] + 5 != job['starting_time'] for job in monitor_jobs if res['id'] == job['job_id'])]

    fault_list = []
    for _job in filtered_jobs:
        new_dict = {
            'id': _job['id'],
            'reserve_time': _job['reserve_time'],
        }
        target_dict = next((item for item in monitor_jobs if item['job_id'] == _job['id']), None)
        new_dict['starting_time'] = target_dict['starting_time']
        new_dict['finish_time'] = target_dict['finish_time']
        fault_list.append(new_dict)
    
    return monitor_jobs, fault_list
