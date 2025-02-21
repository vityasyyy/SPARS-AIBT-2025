from .env import MyDict
import heapq
import pandas as pd

def simulate_easy(sp_simulator):
    current_time = 0
    available_resources = list(range(sp_simulator.nb_res))
    schedule_queue = []
    waiting_queue = []
    monitor_jobs=[]
    active_jobs = []
    for event in sp_simulator.jobs:
        event_time, event_detail = event
        heapq.heappush(schedule_queue, (event_time, MyDict(event_detail)))
    
    while schedule_queue or waiting_queue:      
        if schedule_queue:
            event_time, event = heapq.heappop(schedule_queue)
        else:
            event = waiting_queue.pop(0)
        
        current_time = event_time
        temp_index = 0
        for start_idle_res in sp_simulator.sim_monitor['start_idle']:
            if start_idle_res == -1:
                temp_index +=1
                continue
            rate_energy_consumption_idle = sp_simulator.machines[temp_index]['wattage_per_state'][1]
            idle_time = current_time - start_idle_res
            sp_simulator.sim_monitor['energy_consumption'][temp_index] += (idle_time * rate_energy_consumption_idle)
            sp_simulator.sim_monitor['total_idle_time'][temp_index] += (current_time - start_idle_res)
            
            
        for index_available_resource in available_resources:
            sp_simulator.sim_monitor['start_idle'][index_available_resource] = current_time
            
        
    
        if event['type'] == 'arrival':
            if len(available_resources) >= event['res']:
                if waiting_queue:
                    active_jobs = sorted(active_jobs, key=lambda x: x['finish_time'])
                    backfilled_node_count = 0
                    if sp_simulator.check_backfilling(current_time, event, len(available_resources), active_jobs, waiting_queue[0], backfilled_node_count):
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
            allocated = available_resources[:event['res']]
            available_resources = available_resources[event['res']:]
            allocated_len = len(allocated)
            
    
            mask = sp_simulator.sim_monitor['nb_res']['time'] == current_time

            if mask.sum() == 0:
                last_row = sp_simulator.sim_monitor['nb_res'].iloc[-1].copy()
                last_row['current_time'] = current_time
                sp_simulator.sim_monitor['nb_res'] = pd.concat([sp_simulator.sim_monitor['nb_res'], last_row.to_frame().T], ignore_index=True)
                mask = sp_simulator.sim_monitor['nb_res']['time'] == current_time

            matching_indices = sp_simulator.sim_monitor['nb_res'].index[mask].tolist()
            if matching_indices:
                row_idx = matching_indices[0]
                sp_simulator.sim_monitor['nb_res'].at[row_idx, 'computing'] += allocated_len
                sp_simulator.sim_monitor['nb_res'].at[row_idx, 'idle'] -= allocated_len

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
            
            for i in allocated:
                sp_simulator.sim_monitor['energy_consumption'][i] += (finish_time - current_time) * sp_simulator.machines[i]['wattage_per_state'][3]
                sp_simulator.sim_monitor['start_idle'][i] = -1
            
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
            
            temp_available_resource = len(available_resources)
            
            mask = sp_simulator.sim_monitor['nb_res']['time'] == current_time
            
            if sp_simulator.sim_monitor['nb_res'].loc[mask].empty:
                last_row = sp_simulator.sim_monitor['nb_res'].iloc[-1].copy()
                last_row['current_time'] = current_time
        
                sp_simulator.sim_monitor['nb_res'].loc[len(sp_simulator.sim_monitor['nb_res'])] = last_row
                
            sp_simulator.sim_monitor['nb_res'].loc[
                sp_simulator.sim_monitor['nb_res']['time'] == current_time, 
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

            for _ in range(len(waiting_queue)):
                job = waiting_queue[0] 
                if temp_available_resource >= job['res']:
                    popped_job = waiting_queue.pop(0)
                    popped_job['type'] = 'execution_start'
                    temp_available_resource -= popped_job['res']
            
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
    
            
    return monitor_jobs