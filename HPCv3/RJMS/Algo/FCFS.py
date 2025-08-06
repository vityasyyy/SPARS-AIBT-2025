from HPCv3.RJMS.Algo.BaseAlgorithm import BaseAlgorithm


class FCFS(BaseAlgorithm):
    def schedule(self):
        super().prep_schedule()
        self.FCFS()
        return self.events
        
    def FCFS(self):
        waiting_queue = [job for job in self.JobsManager.waiting_queue if job['job_id'] not in self.scheduled]
        for job in waiting_queue:
            if job['res'] <= len(self.available) + len(self.inactive):
                if job['res'] <= len(self.available):
                    allocated_nodes = self.available[:job['res']]
                    event = {
                        'job_id': job['job_id'],
                        'walltime': job['walltime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': allocated_nodes
                    }
                    self.available = self.available[job['res']:]
                    self.allocated.extend(allocated_nodes)
                    compute_demand = job['walltime'] * job['res']
                    compute_power = sum(self.compute_speeds[i] for i in allocated_nodes)
                    finish_time = self.current_time + (compute_demand / compute_power)
                    for node in self.resources_agenda:
                        if node['node_id'] in allocated_nodes:
                            node['release_time'] = finish_time
                    self.scheduled.append(job['job_id'])
                    super().push_event(self.current_time, event)
                else:
                    count_avail = len(self.available)
                    num_need_activation = job['res'] - count_avail
                    to_activate = self.inactive[:num_need_activation]
                    reserved_nodes = self.available + to_activate
                    super().push_event(self.current_time, {'type': 'switch_on', 'nodes': to_activate})
                    self.allocated.extend(reserved_nodes)
                    self.available = []
                    self.inactive = self.inactive[num_need_activation:]
                    
                    """should consider activation delay"""
                    compute_demand = job['walltime'] * job['res']
                    compute_power = sum(self.compute_speeds[i] for i in reserved_nodes)
                    finish_time = self.current_time + (compute_demand / compute_power)
                    
                    for node in self.resources_agenda:
                        if node['node_id'] in allocated_nodes:
                            node['release_time'] = finish_time
                            
                    self.JobsManager.add_job_to_scheduled_queue(job['id'])
                    self.ResourceManager.reserve_nodes(job['id'], reserved_nodes)
            else:
                break
