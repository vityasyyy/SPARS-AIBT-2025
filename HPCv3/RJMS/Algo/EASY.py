from .FCFS import FCFS
from itertools import combinations

class EASY(FCFS):
    def schedule(self):
        super().prep_schedule()
        super().FCFS()
        
        self.backfill()
        
        if self.timeout is not None:
            super().timeout_policy()
            
        return self.events
    
    def find_node_combination(self, p_start_t, compute_demand, nodes, next_releases, x):
        n = len(nodes)
        if x > n:
            return None

        best_combo = None
        best_finish_time = 0
        node_ids = [node['node_id'] for node in nodes]
        for combo in combinations(nodes, x):
            total_compute_power = sum(node['compute_speed'] for node in combo)

            max_activation_delay = max(
                entry['release_time']
                for entry in next_releases
                if entry['node_id'] in node_ids
            )

            finish_time = max_activation_delay + (compute_demand / total_compute_power)

            if finish_time <= p_start_t and finish_time > best_finish_time:
                best_finish_time = finish_time
                best_combo = combo

        return best_combo
     
    def backfill(self):

        waiting_queue = [job for job in self.JobsManager.waiting_queue if job['job_id'] not in self.scheduled]
        if len(waiting_queue) > 2:
            p_job = waiting_queue[0]
                
            backfilling_queue = waiting_queue[1:]
            
            reserved_nodes = self.ResourceManager.get_reserved_nodes()
            
            not_reserved_nodes = [node for node in self.ResourceManager.nodes if node['id'] not in reserved_nodes]
            
            next_releases = self.resources_agenda
            machines = {m['id']: m['states']['switching_on']['transitions'] for m in self.ResourceManager.machines}
            
            for next_release in next_releases:
                for t in machines[next_release['node_id']]:
                    if t['state'] == 'sleeping':
                        next_release['release_time'] += t['transition_time']
                            
            next_releases = sorted(
                next_releases, 
                key=lambda x: (x['release_time'], x['node_id'])
            )
            
            if len(next_releases) < p_job['res']:
                return
            
            last_host = next_releases[p_job['res'] - 1]
            p_start_t = last_host['release_time']
                    
            candidates = [r['node_id'] for r in next_releases if r['release_time'] <= p_start_t]
            head_job_reservation = candidates[-p_job['res']:]
            
            not_reserved_nodes = [r for r in not_reserved_nodes if r['id'] not in head_job_reservation]

            for job in backfilling_queue:
                not_computing_resources = [node['id'] for node in self.ResourceManager.nodes if node['job_id'] is None and node['id'] not in self.allocated]
                not_reserved = [h for h in not_computing_resources if h not in head_job_reservation]

                available_resources_not_reserved = [r for r in self.available if r in not_reserved]
                inactive_resources_not_reserved = [r for r in self.inactive if r in not_reserved]

                if job['res'] <= len(not_reserved):
                    if job['res'] <= len(available_resources_not_reserved):
                        allocated_nodes = available_resources_not_reserved[:job['res']]
                        event = {
                            'job_id': job['job_id'],
                            'walltime': job['walltime'],
                            'res': job['res'],
                            'type': 'execution_start',
                            'nodes': allocated_nodes
                        }
                        self.available = [node for node in self.available if node not in allocated_nodes]
                        self.allocated.extend(allocated_nodes)
                        super().push_event(self.current_time, event)
                        """should update releases agenda, do the same for others'
                        """
                    elif job['res'] <= len(available_resources_not_reserved) + len(inactive_resources_not_reserved):
                        count_avail = len(available_resources_not_reserved)
                        allocated_nodes = available_resources_not_reserved
                        num_need_activation = job['res'] - count_avail
                        to_activate = inactive_resources_not_reserved[:num_need_activation]
                        reserved_nodes = allocated_nodes + to_activate
                        self.allocated.extend(reserved_nodes)
                        super().push_event(self.current_time, {'type': 'switch_on', 'nodes': to_activate})
                        self.available = [node for node in self.available if node not in allocated_nodes]
                        self.inactive = self.inactive[num_need_activation:]
                        
                        highest_transition_time = 0
                        for machine_transition in self.ResourceManager.machines_transition:
                            if machine_transition['node_id'] in reserved_nodes:
                                for transition in machine_transition['transitions']:
                                    if transition['from'] == 'switching_off' and transition['to'] == 'sleeping' and transition['transition_time'] > highest_transition_time:
                                        highest_transition_time = transition['transition_time']
                                        
                        compute_demand = job['walltime'] * job['res']
                        
                        self.JobsManager.add_job_to_scheduled_queue(job['job_id'], reserved_nodes, highest_transition_time)
                        self.ResourceManager.reserve_nodes(job['job_id'], reserved_nodes)
                else:
                    not_computing_resources = [node['id'] for node in self.ResourceManager.nodes if node['job_id'] is None and node['id'] not in self.allocated]
                    
                    compute_demand = job['walltime'] * job['res']
                    free_nodes = [{'node_id': node['id'], 'compute_speed': node['compute_speed']} for node in self.ResourceManager.nodes if node['id'] in not_computing_resources]
                    
                    combo = self.find_node_combination(p_start_t, compute_demand, free_nodes, next_releases, job['res'])
                    
                    if combo == None:
                        continue
                    to_activate = [node for node in combo if node in self.inactive]
                    
                    if len(to_activate) == 0:
                        allocated_nodes = self.available[:job['res']]
                        event = {
                            'job_id': job['job_id'],
                            'walltime': job['walltime'],
                            'res': job['res'],
                            'type': 'execution_start',
                            'nodes': allocated_nodes
                        }
                        self.allocated.extend(allocated_nodes)
                        self.available = self.available[job['res']:]
                        super().push_event(self.current_time, event)
                    else:
                        count_available_nodes = len(self.available)
                        allocated_nodes = self.available
                        num_need_activation = job['res'] - count_available_nodes
                        to_activate = self.inactive[:num_need_activation]
                        reserved_nodes = allocated_nodes + to_activate
                        self.allocated.extend(reserved_nodes)
                        super().push_event(self.current_time, {'type': 'switch_on', 'nodes': to_activate})
                        self.available = []
                        self.inactive = self.inactive[num_need_activation:]
                        self.JobsManager.add_job_to_scheduled_queue(job['id'])
                        self.ResourceManager.reserve_nodes(job['id'], reserved_nodes)