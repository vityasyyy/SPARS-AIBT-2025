import json
import heapq
from multiprocessing import heap
from matplotlib.style import available
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
        priority_type = {'turn_on', 'turn_off', 'switch_on', 'switch_off'}

        if self._dict['type'] in priority_type:
            return True
        elif other._dict['type'] in priority_type:
            return False
        elif self._dict['type'] not in priority_type and other._dict['type'] in priority_type:
            return True
       
        
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
        
    def has_key(self, key):
        return key in self._dict

class SPSimulator:
    def __init__(self, platform_path="platforms/spsim/platform.json", workload_path="workloads/simple_data_100.json"):        
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
            
        self.nb_res = self.platform_info['nb_res']
        self.machines = self.platform_info['machines']
        self.profiles = self.workload_info['profiles']
        self.transition_time = [self.platform_info['switch_off_time'], self.platform_info['switch_on_time']]
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
            'nb_res': pd.DataFrame([{'time': 0, 
                                'sleeping': 0, 
                                'sleeping_nodes': [], 
                                'switching_on': 0, 
                                'switching_on_nodes':[], 
                                'switching_off': 0,
                                'switching_off_nodes': [],  
                                'idle': 16,
                                'idle_nodes': list(range(self.nb_res)),
                                'computing': 0, 
                                'computing_nodes': [],
                                'unavailable': 0}]),
            'nodes': [
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}],
                        [{'type': 'idle', 'starting_time': 0, 'finish_time': 0}]
                    ]
        }
    
    def update_nb_res(self, current_time, event, _type, nodes):
        mask = self.sim_monitor['nb_res']['time'] == current_time
        
        for node_index in nodes:
            node_history = self.sim_monitor['nodes'][node_index]
            if node_history[len(node_history)-1]['type'] != _type:
                node_history[len(node_history)-1]['finish_time'] = current_time
                if _type == 'release':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'allocate':
                    node_history.append({'type': 'computing', 'starting_time': current_time, 'finish_time': event['walltime'] + current_time})
                elif _type == 'switch_off':
                    node_history.append({'type': 'switching_off', 'starting_time': current_time, 'finish_time': 1 + current_time})
                elif _type == 'switch_on':
                    node_history.append({'type': 'switching_on', 'starting_time': current_time, 'finish_time': 1 + current_time})
                elif _type == 'turn_off':
                    node_history.append({'type': 'sleeping', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'turn_on':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
                
        if mask.sum() == 0:
            last_row = self.sim_monitor['nb_res'].iloc[-1].copy()
            last_row['time'] = current_time
            self.sim_monitor['nb_res'] = pd.concat([self.sim_monitor['nb_res'], last_row.to_frame().T], ignore_index=True)
            mask = self.sim_monitor['nb_res']['time'] == current_time
        
        row_idx = self.sim_monitor['nb_res'].index[mask].tolist()[0]
        nodes_len = len(nodes)
        
        if _type == 'release':
            # pass the nodes that want to be released
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'computing'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = _nodes
            
            self.sim_monitor['nb_res'].at[row_idx, 'computing_nodes'] = [
                item for item in self.sim_monitor['nb_res'].at[row_idx, 'computing_nodes']
                if item.get('job_id') != event['id']
            ]
        
        elif _type == 'allocate':
            self.sim_monitor['nb_res'].at[row_idx, 'computing'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] -= nodes_len
            
            self.sim_monitor['nb_res'].at[row_idx, 'computing_nodes'] += [{'job_id': event['id'], 'nodes': nodes,'starting_time': current_time, 'finish_time': current_time+event['walltime']}]
            
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'])
            
        elif _type == 'switch_off':
            # pass the nodes that want to be switched off
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] = _nodes
            
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'])
            
        elif _type == 'turn_off':
            # pass the nodes that finally turned off
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] = _nodes
               
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'switching_off_nodes'])
            
        elif _type == 'switch_on':
              # pass the nodes that want to be switched off
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] = _nodes
            
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'sleeping_nodes'])

        elif _type == 'turn_on':
                # pass the nodes that finally turned off
            self.sim_monitor['nb_res'].at[row_idx, 'idle'] += nodes_len
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on'] -= nodes_len
            
            _nodes = self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor['nb_res'].at[row_idx, 'idle_nodes'] = _nodes 
            
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] = [item for item in self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] if item not in nodes]
            self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'] = sorted(self.sim_monitor['nb_res'].at[row_idx, 'switching_on_nodes'])
            
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
                # kayaknya perlu nambahin valid switch on
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
              
                events_now = [(t, e) for t, e in schedule_queue if t == current_time]
                skipbf = False
                for aj in active_jobs:
                    if aj['finish_time'] == current_time:
                        skipbf=True
                        
                if skipbf == True:
                    continue
                    
                # for _, _event in events_now:
                #     if _event['type'] == 'execution_start':
                #         temp_available_resource -= _event['res']

                
                waiting_queue = sorted(waiting_queue)
                # tambahin perulangan gas gas insert job berdasarkan id atau submit time
                
                for _ in range(len(waiting_queue)):
                    job = waiting_queue[0]  # Selalu cek job pertama
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

    


            # Filter nodes and remove empty dictionaries

            mask = self.sim_monitor['nb_res']['time'] == current_time
            has_idle = (self.sim_monitor['nb_res'].loc[mask, 'idle'] > 0).any()
            if has_idle:

                heapq.heappush(schedule_queue, (current_time + timeout, MyDict({'type':'switch_off', 'node': copy.deepcopy(available_resources)})))
                # masukin trigger type turning off pada waktu current time + timeout
                # tapi jangan lupa kalau ada ternyata belum lewat timeout ada event baru masuk dan node tersebut digunakan berarti gjd trigger dong
                # berarti trigger itu adalah list of dict
                # di mana setiap dict ada time dan list node idle
                # ketika yang di list node idle itu dialokasikan, brrti di list node idle hapus node yang dialokasikan ke job
                # ketika list node idle habis, berarti trigger tersebut dihilangkan dari muka bumi
                
                # triggerer dipush ke schedule queue saja
                
                # ini baru triggerer untuk kalo ada available resource dan trus dimatikan
                # perlu bikin handler untuk nambahin data ke triggerer ketika node yang di sleeping perlu dihidupkan
                # temp aval res tinggal ditambah len(sleeping node), trus bikin temp total node needed, ketika temp total node needed > len(idle) brrti kan perlu ngidupin sleeping
                
                # berarti perlu bikin variable untuk tracking inactive nodes
                
                # kalo perlu ngidupin brrti tambahin triggerer
                
                # triggerer dibagi jadi 2, transisi on off, transisi off on, finally off, finally on
                # atau mungkin pas transisi on off langsung tambahin aja finally off ke monitor nb_res
                

             
        return monitor_jobs

timeout = 30
sp_simulator = SPSimulator()
jobs_e = sp_simulator.simulate_easy(timeout)
max_finish_time = max(job.get('finish_time', 0) for job in jobs_e)
jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_e.to_csv('results/sp/easy_jobs_t30.csv', index=False)
sp_simulator.sim_monitor['nb_res'].to_csv('results/sp/easy_host_t30.csv', index=False)

nodes_data = sp_simulator.sim_monitor['nodes']

flattened_data = []
for idx, sublist in enumerate(nodes_data):
    for item in sublist:
        item['allocated_resources'] = idx
        flattened_data.append(item)

df = pd.DataFrame(flattened_data)

df = df[['allocated_resources', 'type', 'starting_time', 'finish_time']]
df = df[df['type'] != 'computing']

def set_job_id(row):
    if row['type'] == 'idle':
        return -1
    elif row['type'] == 'switching_off':
        return -2
    elif row['type'] == 'switching_on':
        return -3
    elif row['type'] == 'sleeping':
        return -4
    return 0 


def calculate_metrics(nodes_monitor):
    total_joule = nodes_monitor.apply(
        lambda row: (row['finish_time'] - row['starting_time']) * len(row['allocated_resources'].split()) * (
            95 if row['type'] == 'idle' else
            9 if row['type'] == 'sleeping' else
            190 if row['type'] == 'computing' else
            9 if row['type'] == 'switching_off' else
            190 if row['type'] == 'switching_on' else 0
        ), axis=1
    ).sum()

    print(total_joule)
    

df['job_id'] = df.apply(set_job_id, axis=1)

grouped_df = df.groupby(['type', 'starting_time', 'finish_time']).agg({'allocated_resources': lambda x: ' '.join(map(str, x)), 'job_id': 'first'}).reset_index()

grouped_df = grouped_df.sort_values(by=['starting_time', 'finish_time'])
grouped_df['submission_time'] = grouped_df['starting_time']
jobs_e = jobs_e[['job_id', 'allocated_resources', 'starting_time', 'finish_time', 'submission_time']]
jobs_e['type'] = 'computing'

final_df = pd.concat([grouped_df, jobs_e], ignore_index=True)

final_df = final_df.sort_values(by=['starting_time', 'finish_time'])

calculate_metrics(final_df)

final_df.to_csv('results/sp/easy_nodes_t30.csv', index=False)
final_df.to_json('results/sp/easy_nodes_t30.json', orient='records', lines=True)