from matplotlib.style import available
import pandas as pd
import copy

class NodeManager:
    def __init__(self, nb_res, sim_monitor, platform_info):
        self.nb_res = nb_res
        self.available_resources = list(range(nb_res))
        self.reserved_resources = []
        self.inactive_resources = []
        self.on_off_resources = []
        self.off_on_resources = []
        self.resources_agenda = [{'release_time': 0, 'node': i} for i in range(self.nb_res)]
        
        self.transition_time = [platform_info['switch_off_time'], platform_info['switch_on_time']]
        
        self.sim_monitor = sim_monitor
    
    def update_inactive_resources_agenda(self, current_time):
        for agenda in self.resources_agenda:
            node = agenda['node']
            if node in self.inactive_resources:
                agenda['release_time'] = current_time + self.transition_time[1]


    def get_not_allocated_resources(self):
        reserved_node_indices = [res["node_index"] for res in self.reserved_resources]
        available_resources = self.available_resources
        available_resources = [resource for resource in available_resources if resource not in reserved_node_indices]
        
        inactive_resources = self.inactive_resources
        inactive_resources = [resource for resource in inactive_resources if resource not in reserved_node_indices]
        
        not_allocated_resources = self.available_resources + self.inactive_resources
        not_allocated_resources = [resource for resource in not_allocated_resources if resource not in reserved_node_indices]
        not_allocated_resources = sorted(not_allocated_resources)
        
        return available_resources, inactive_resources
        
    def update_node_state_monitor(self, current_time, last_event_time):
        for index, node_action in enumerate(self.sim_monitor.nodes_action):
            state = node_action['state']
            last_time = max(node_action['time'], last_event_time) 
            duration = current_time - last_time  

            self.sim_monitor.node_state_monitor[index][state] += duration
            
    def update_node_action(self, allocated, event, event_type, target_state, current_time):
        # UPDATE MONITOR
        for node in allocated:
            self.sim_monitor.nodes_action[node]['state'] = target_state
            self.sim_monitor.nodes_action[node]['time'] = current_time
        
        # UPDATE NODE MANAGER
        self.update_node_state_log(current_time, event, event_type, allocated)
        
    def update_node_state_log(self, current_time, event, _type, nodes):
        mask = self.sim_monitor.node_state_log['time'] == current_time
        
        for node_index in nodes:
            node_history = self.sim_monitor.nodes[node_index]
            if node_history[len(node_history)-1]['type'] != _type:
                node_history[len(node_history)-1]['finish_time'] = current_time
                if _type == 'release':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'allocate':
                    node_history.append({'type': 'computing', 'starting_time': current_time, 'finish_time': event['walltime'] + current_time})
                elif _type == 'switch_off':
                    node_history.append({'type': 'switching_off', 'starting_time': current_time, 'finish_time': self.transition_time[0] + current_time})
                elif _type == 'switch_on':
                    node_history.append({'type': 'switching_on', 'starting_time': current_time, 'finish_time': self.transition_time[1] + current_time})
                elif _type == 'turn_off':
                    node_history.append({'type': 'sleeping', 'starting_time': current_time, 'finish_time': current_time})
                elif _type == 'turn_on':
                    node_history.append({'type': 'idle', 'starting_time': current_time, 'finish_time': current_time})
          
        
        if mask.sum() == 0:
            last_row = self.sim_monitor.node_state_log.iloc[-1].copy()
            last_row['time'] = current_time
            self.sim_monitor.node_state_log = pd.concat([self.sim_monitor.node_state_log, last_row.to_frame().T], ignore_index=True)
            mask = self.sim_monitor.node_state_log['time'] == current_time
        
        row_idx = self.sim_monitor.node_state_log.index[mask].tolist()[0]
        nodes_len = len(nodes)
        
        if _type == 'release':
            self.sim_monitor.node_state_log.at[row_idx, 'idle'] += nodes_len
            self.sim_monitor.node_state_log.at[row_idx, 'computing'] -= nodes_len
            
            _nodes = self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] = _nodes
            
            self.sim_monitor.node_state_log.at[row_idx, 'computing_nodes'] = [
                item for item in self.sim_monitor.node_state_log.at[row_idx, 'computing_nodes']
                if item.get('job_id') != event['id']
            ]
        
        elif _type == 'allocate':
            self.sim_monitor.node_state_log.at[row_idx, 'computing'] += nodes_len
            self.sim_monitor.node_state_log.at[row_idx, 'idle'] -= nodes_len
            
            self.sim_monitor.node_state_log.at[row_idx, 'computing_nodes'] += [{'job_id': event['id'], 'nodes': nodes,'starting_time': current_time, 'finish_time': current_time+event['walltime']}]
            
            self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'])
            
        elif _type == 'switch_off':
            self.sim_monitor.node_state_log.at[row_idx, 'switching_off'] += nodes_len
            self.sim_monitor.node_state_log.at[row_idx, 'idle'] -= nodes_len
            
            _nodes = self.sim_monitor.node_state_log.at[row_idx, 'switching_off_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.node_state_log.at[row_idx, 'switching_off_nodes'] = _nodes
            
            self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] = [item for item in self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] if item not in nodes]
            self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] = sorted(self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'])
            
        elif _type == 'turn_off':
            self.sim_monitor.node_state_log.at[row_idx, 'sleeping'] += nodes_len
            self.sim_monitor.node_state_log.at[row_idx, 'switching_off'] -= nodes_len
            
            _nodes = self.sim_monitor.node_state_log.at[row_idx, 'sleeping_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.node_state_log.at[row_idx, 'sleeping_nodes'] = _nodes
               
            self.sim_monitor.node_state_log.at[row_idx, 'switching_off_nodes'] = [item for item in self.sim_monitor.node_state_log.at[row_idx, 'switching_off_nodes'] if item not in nodes]
            self.sim_monitor.node_state_log.at[row_idx, 'switching_off_nodes'] = sorted(self.sim_monitor.node_state_log.at[row_idx, 'switching_off_nodes'])
            
        elif _type == 'switch_on':
            self.sim_monitor.node_state_log.at[row_idx, 'switching_on'] += nodes_len
            self.sim_monitor.node_state_log.at[row_idx, 'sleeping'] -= nodes_len
            
            _nodes = self.sim_monitor.node_state_log.at[row_idx, 'switching_on_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.node_state_log.at[row_idx, 'switching_on_nodes'] = _nodes
            
            self.sim_monitor.node_state_log.at[row_idx, 'sleeping_nodes'] = [item for item in self.sim_monitor.node_state_log.at[row_idx, 'sleeping_nodes'] if item not in nodes]
            self.sim_monitor.node_state_log.at[row_idx, 'sleeping_nodes'] = sorted(self.sim_monitor.node_state_log.at[row_idx, 'sleeping_nodes'])

        elif _type == 'turn_on':
            self.sim_monitor.node_state_log.at[row_idx, 'idle'] += nodes_len
            self.sim_monitor.node_state_log.at[row_idx, 'switching_on'] -= nodes_len
            
            _nodes = self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] + nodes
            _nodes = _nodes
            _nodes = sorted(_nodes)
            self.sim_monitor.node_state_log.at[row_idx, 'idle_nodes'] = _nodes 
            
            self.sim_monitor.node_state_log.at[row_idx, 'switching_on_nodes'] = [item for item in self.sim_monitor.node_state_log.at[row_idx, 'switching_on_nodes'] if item not in nodes]
            self.sim_monitor.node_state_log.at[row_idx, 'switching_on_nodes'] = sorted(self.sim_monitor.node_state_log.at[row_idx, 'switching_on_nodes'])

        
    def switch_off(self, node, current_time, event):
        valid_switch_off = [item for item in node if item in self.available_resources]
        
        if len(valid_switch_off) == 0:
            return
        
        self.available_resources = [item for item in self.available_resources if item not in valid_switch_off]
        self.on_off_resources.extend(valid_switch_off)
        self.on_off_resources = sorted(self.on_off_resources)
        self.update_node_action(valid_switch_off, event, 'switch_off', 'switching_off', current_time)
        

        for i in range(self.nb_res):
            if i in valid_switch_off:
                self.resources_agenda[i]['release_time'] = current_time + self.transition_time[0] + self.transition_time[1]

        ts = current_time + self.transition_time[0]
        e = {'type': 'turn_off', 'node': copy.deepcopy(valid_switch_off)}

        return ts, e
    
    def turn_off(self, node, current_time, event):
        self.inactive_resources.extend(node)
        self.inactive_resources = sorted(self.inactive_resources)
                
        self.on_off_resources = [item for item in self.on_off_resources if item not in node]
        self.on_off_resources = sorted(self.on_off_resources)
        
        for i in range(self.nb_res):
            if i in node:
                self.resources_agenda[i]['release_time'] = current_time 

        self.update_node_action(node, event, 'turn_off', 'sleeping', current_time)
    
    def switch_on(self, node, current_time, event, workload_info):
        self.inactive_resources = [item for item in self.inactive_resources if item not in node]
        self.off_on_resources.extend(node)
        self.off_on_resources = sorted(self.off_on_resources)
        self.update_node_action(node, event, 'switch_on', 'switching_on', current_time)
        
        reserved_node_indices = [res["node_index"] for res in self.reserved_resources]
        for i in range(self.nb_res):
            # If i is in switch_on node
            if i in node:
                # If i is in reserved node (based on index)
                if i in reserved_node_indices:
                    reserved = next((res for res in self.reserved_resources if res['node_index'] == i), None)
                    if reserved is not None:
                        job_id = reserved['job_id']
                        job_detail = next((job for job in workload_info['jobs'] if job['id'] == job_id), None)
                        if job_detail is not None:
                            self.resources_agenda[i]['release_time'] = (
                                current_time + self.transition_time[1] + job_detail['walltime']
                            )
                        else:
                            self.resources_agenda[i]['release_time'] = current_time + self.transition_time[1]
                    else:
                        self.resources_agenda[i]['release_time'] = current_time + self.transition_time[1]
                else:
                    self.resources_agenda[i]['release_time'] = current_time + self.transition_time[1]


        
        ts = current_time + self.transition_time[1]
        e = {'type': 'turn_on', 'node': copy.deepcopy(node)}
        return ts, e
    
    def turn_on(self, node, event, current_time):
        self.available_resources.extend(node)
        self.available_resources = sorted(self.available_resources)
        self.off_on_resources = [item for item in self.off_on_resources if item not in node]
        self.off_on_resources = sorted(self.off_on_resources)
        
        for i in range(self.nb_res):
            if i in node:
                self.resources_agenda[i]['release_time'] = current_time
                
        self.update_node_action(node, event, 'turn_on', 'idle', current_time)

    def prioritize_lowest_node(self, num_needed_res):
        reserved_node = []
        need_activation_node = []
        
        reserved_node_indices = [res["node_index"] for res in self.reserved_resources]
        for node_index in range(self.nb_res):
            if (node_index in self.available_resources or node_index in self.inactive_resources) and node_index not in reserved_node_indices:
                reserved_node.append(node_index)

                if node_index in self.inactive_resources:
                    need_activation_node.append(node_index)
                if len(reserved_node) == num_needed_res:
                    return reserved_node, need_activation_node
            
        return reserved_node, need_activation_node