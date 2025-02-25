import heapq
import copy
from .env import MyDict


def simulate_easy(sp_simulator, timeout):
    current_time = 0
    available_resources = list(range(sp_simulator.nb_res))
    inactive_resources = []
    on_off_resources = []
    off_on_resources = []
    schedule_queue = []
    waiting_queue = []
    monitor_jobs=[]
    active_jobs = []
    reserved_count = 0

    for event in sp_simulator.jobs:
        event_time, event_detail = event
        heapq.heappush(schedule_queue, (event_time, MyDict(event_detail)))
    
    heapq.heappush(schedule_queue, (timeout, MyDict({'node': copy.deepcopy(available_resources), 'type': 'switch_off'})))
    

    while schedule_queue or waiting_queue:      
        if schedule_queue:
            event_time, event = heapq.heappop(schedule_queue)
        else:
            event = waiting_queue.pop(0)
        
        current_time = event_time

      
      
        if event['type'] == 'switch_off':
            valid_switch_off = [item for item in event['node'] if item in available_resources]
            
            if len(valid_switch_off) == 0:
                continue
            
            available_resources = [item for item in available_resources if item not in valid_switch_off]
            on_off_resources.extend(valid_switch_off)
            on_off_resources = sorted(on_off_resources)
            sp_simulator.update_nb_res(current_time, event, event['type'], valid_switch_off)
            
            heapq.heappush(schedule_queue, (current_time + sp_simulator.transition_time[0], MyDict({'node': copy.deepcopy(valid_switch_off), 'type': 'turn_off' })))
            
        elif event['type'] == 'turn_off':
            inactive_resources.extend(event['node'])
            inactive_resources = sorted(inactive_resources)
            
            on_off_resources = [item for item in on_off_resources if item not in event['node']]
            on_off_resources = sorted(on_off_resources)
            sp_simulator.update_nb_res(current_time, event, event['type'], event['node'])
            
        elif event['type'] == 'switch_on':
            # kayaknya perlu nambahin valid switch on
            inactive_resources = [item for item in inactive_resources if item not in event['node']]
            off_on_resources.extend(event['node'])
            off_on_resources = sorted(off_on_resources)
            sp_simulator.update_nb_res(current_time, event, event['type'], event['node'])
            
            heapq.heappush(schedule_queue, (current_time + sp_simulator.transition_time[1], MyDict({'node': copy.deepcopy(event['node']), 'type': 'turn_on' })))
            
        elif event['type'] == 'turn_on':
            available_resources.extend(event['node'])
            available_resources = sorted(available_resources)
            off_on_resources = [item for item in off_on_resources if item not in event['node']]
            off_on_resources = sorted(off_on_resources)
            sp_simulator.update_nb_res(current_time, event, event['type'], event['node'])
            
        elif event['type'] == 'arrival':
            if len(available_resources) + len(inactive_resources) >= event['res']:
                if waiting_queue:
                    active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                    backfilled_node_count = 0
                    if sp_simulator.check_backfilling(current_time, event, len(available_resources) + len(inactive_resources), active_jobs, waiting_queue[0], backfilled_node_count):
                        event['type'] = 'execution_start'
                        if len(available_resources) < event['res']:
                            heapq.heappush(schedule_queue, (current_time + sp_simulator.transition_time[1], MyDict(event)))
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
                if len(off_on_resources) < event['res']:
                    inactive_node_sorted = sorted(inactive_resources)
                    available_node_sorted = sorted(available_resources)
                    merged = inactive_node_sorted + available_node_sorted
                    merged = sorted(merged)
                    activated_nodes = []

                    for node in merged:
                        if len(activated_nodes) < event['res']:
                            activated_nodes.append(node)
                    
                    intersection = list(set(inactive_resources) & set(activated_nodes))
                    heapq.heappush(schedule_queue, (current_time, MyDict({'type': 'switch_on', 'node': copy.deepcopy(intersection)})))
                heapq.heappush(schedule_queue, (current_time + sp_simulator.transition_time[1], MyDict(event)))
                continue
            

            if event['res'] <= len(available_resources) and (inactive_resources and min(inactive_resources) < available_resources[event['res'] - 1]):
                inactive_node_sorted = sorted(inactive_resources)
                available_node_sorted = sorted(available_resources)
                merged = inactive_node_sorted + available_node_sorted
                merged = sorted(merged)
                activated_nodes = []

                for node in merged:
                    if len(activated_nodes) < event['res']:
                        activated_nodes.append(node)
                
                intersection = list(set(inactive_resources) & set(activated_nodes))
                
                heapq.heappush(schedule_queue, (current_time, MyDict({'type': 'switch_on', 'node': copy.deepcopy(intersection)})))
                
                if event.has_key('reserve') == False:
                    reserved_count += event['res']
                    event['reserve'] = True
                heapq.heappush(schedule_queue, (current_time + sp_simulator.transition_time[1], MyDict(event)))
                continue
            
            if event.has_key('reserve') and event['reserve'] == True:
                reserved_count -= event['res']
                
            allocated = available_resources[:event['res']]
            available_resources = available_resources[event['res']:]
            
            sp_simulator.update_nb_res(current_time, event, 'allocate', allocated)

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
                sp_simulator.sim_monitor['avg_waiting_time'] += (current_time - finish_event['subtime'])
                sp_simulator.sim_monitor['waiting_event_count'] += 1
                
            heapq.heappush(schedule_queue, (finish_time, MyDict(finish_event)))
            
            finish_event['finish_time'] = finish_time
            active_jobs.append(finish_event)
            
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
                sp_simulator.sim_monitor['start_idle'][index_available_resource] = current_time

            active_jobs = [active_job for active_job in active_jobs if active_job['id'] != event['id']]
            
            temp_available_resource_2 = len(available_resources)
            temp_available_resource = len(available_resources) + len(inactive_resources) + len(off_on_resources) - reserved_count
            check_if_need_activation = temp_available_resource - temp_available_resource_2
            
            sp_simulator.update_nb_res(current_time, event, 'release', allocated)
            
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
                        if not sp_simulator.check_backfilling(current_time, job, temp_available_resource, active_jobs, waiting_queue[0], backfilled_node_count):
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

        mask = sp_simulator.sim_monitor['nb_res']['time'] == current_time
        has_idle = (sp_simulator.sim_monitor['nb_res'].loc[mask, 'idle'] > 0).any()
        if has_idle:
            heapq.heappush(schedule_queue, (current_time + timeout, MyDict({'type':'switch_off', 'node': copy.deepcopy(available_resources)})))
    
            
    return monitor_jobs