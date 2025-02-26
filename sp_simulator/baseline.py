from .env import MyDict
import heapq
import pandas as pd

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
        
        self.update_energy_consumption()
    
        if event['type'] == 'arrival':
            if len(self.available_resources) >= event['res']:
                if self.waiting_queue:
                    self.active_jobs = sorted(self.active_jobs, key=lambda x: x['finish_time'])
                    backfilled_node_count = 0
                    if self.check_backfilling(self.current_time, event, len(self.available_resources), self.active_jobs, self.waiting_queue[0], backfilled_node_count):
                        event['type'] = 'execution_start'
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
            allocated = self.available_resources[:event['res']]
            self.available_resources = self.available_resources[event['res']:]
            allocated_len = len(allocated)
            
            self.update_node_action(allocated, event, 'allocate', 'computing')
            
            mask = self.sim_monitor['nb_res']['time'] == self.current_time

            if mask.sum() == 0:
                last_row = self.sim_monitor['nb_res'].iloc[-1].copy()
                last_row['current_time'] = self.current_time
                self.sim_monitor['nb_res'] = pd.concat([self.sim_monitor['nb_res'], last_row.to_frame().T], ignore_index=True)
                mask = self.sim_monitor['nb_res']['time'] == self.current_time

            matching_indices = self.sim_monitor['nb_res'].index[mask].tolist()
            if matching_indices:
                row_idx = matching_indices[0]
                self.sim_monitor['nb_res'].at[row_idx, 'computing'] += allocated_len
                self.sim_monitor['nb_res'].at[row_idx, 'idle'] -= allocated_len

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

            self.update_node_action(allocated, event, 'release', 'idle')
            
            self.active_jobs = [active_job for active_job in self.active_jobs if active_job['id'] != event['id']]
            
            temp_available_resource = len(self.available_resources)
            
            mask = self.sim_monitor['nb_res']['time'] == self.current_time
            
            if self.sim_monitor['nb_res'].loc[mask].empty:
                last_row = self.sim_monitor['nb_res'].iloc[-1].copy()
                last_row['current_time'] = self.current_time
        
                self.sim_monitor['nb_res'].loc[len(self.sim_monitor['nb_res'])] = last_row
                
            self.sim_monitor['nb_res'].loc[
                self.sim_monitor['nb_res']['time'] == self.current_time, 
                ['computing', 'idle']
            ] += [-len(allocated), len(allocated)]
            
            
            events_now = [(t, e) for t, e in self.schedule_queue if t == self.current_time]
            skipbf = False
            for aj in self.active_jobs:
                if aj['finish_time'] == self.current_time:
                    skipbf=True
                    
            if skipbf == True:
                continue
                
            for _, _event in events_now:
                if _event['type'] == 'execution_start':
                    temp_available_resource -= _event['res']

            
            self.waiting_queue = sorted(self.waiting_queue)

            for _ in range(len(self.waiting_queue)):
                job = self.waiting_queue[0] 
                if temp_available_resource >= job['res']:
                    popped_job = self.waiting_queue.pop(0)
                    popped_job['type'] = 'execution_start'
                    temp_available_resource -= popped_job['res']
            
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
    
            
    return self.monitor_jobs