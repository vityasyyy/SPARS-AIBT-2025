from .fcfs_scheduler import FCFSScheduler

class EasyScheduler(FCFSScheduler):
    def schedule(self):
        super().schedule()
        if len(self.simulator.waiting_queue) >= 2:
            self.backfill()
            
    def backfill(self):
        p_job = self.simulator.waiting_queue[0]
            
        backfilling_queue = self.simulator.waiting_queue[1:]
        
        not_reserved_resources = sorted(set(self.simulator.available_resources + self.simulator.inactive_resources) - set(self.simulator.reserved_resources))
        next_releases = []
        for nrs in not_reserved_resources:
            next_releases.append({'release_time': 0, 'node': nrs})
            
        for job in self.simulator.active_jobs:
            for node in job['allocated_resources']:
                next_releases.append({'release_time': job['finish_time'], 'node': node})
        
        next_releases = sorted(
            next_releases, 
            key=lambda x: (x['release_time'], x['node'])
        )
        
        if len(next_releases) < p_job['res']:
            return
        
        last_host = next_releases[p_job['res'] - 1]
        p_start_t = last_host['release_time']
        
        candidates = [r['node'] for r in next_releases if r['release_time'] <= p_start_t]
        reservation = candidates[-p_job['res']:]
        
        not_reserved_resources = [r for r in not_reserved_resources if r not in reservation]

        for job in backfilling_queue:
            available = self.simulator.get_not_allocated_resources()
            not_reserved = [h for h in available if h not in reservation]

            if job['res'] <= len(not_reserved):
                reserved_node, need_activation_node = self.simulator.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
            elif job['walltime'] and job['walltime'] + self.simulator.current_time <= p_start_t and job['res'] <= len(available):
                reserved_node, need_activation_node = self.simulator.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)