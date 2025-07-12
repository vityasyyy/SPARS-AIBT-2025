from functools import total_ordering
from re import sub
from unicodedata import is_normalized

from nbformat import current_nbformat
from .easy_scheduler import EasyScheduler
import torch as T

class RLScheduler(EasyScheduler):
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
    
    def get_arrival_time(self, is_normalized = True):
        submission_times = self.simulator.monitor_jobs['submission_time']
        if len(submission_times) == 0:
            return 0
        if len(submission_times) == 1:
            return submission_times[0]
        
        submission_times = T.tensor(submission_times)
        submission_times -= submission_times[0].item()
        max_time = submission_times[-1]
        submission_times_r = T.roll(submission_times, 1)
        submission_times -= submission_times_r
        arrival_rate = T.mean(submission_times[1:])
        if is_normalized:
            arrival_rate /= max_time
        return arrival_rate
    
    def get_mean_walltime_in_queue(self, is_normalized=True):
        walltime_in_queue = [job['walltime'] for job in self.simulator.waiting_queue]
        walltime_in_queue = T.tensor(walltime_in_queue, dtype=T.float32)
        if len(walltime_in_queue) == 0:
            return 0
        mean_walltime = T.mean(walltime_in_queue)
        if is_normalized:
            mean_walltime /= T.max(walltime_in_queue)
        return mean_walltime
    
    def get_mean_waittime_in_queue(self, is_normalized=True):
        subtimes = [job['submission_time'] for job in self.simulator.waiting_queue]
        if len(subtimes) == 0:
            return 0
        subtimes = T.tensor(subtimes, dtype=T.float32)
        wait_times = self.simulator.current_time-subtimes
        mean_wait_times = T.mean(wait_times)
        if is_normalized:
            mean_wait_times /= T.max(wait_times)
        return mean_wait_times

    
    def get_wasted_energy(self, is_normalized=True):
        wasted_energy = self.simulator.sim_monitor.energy_waste
        if is_normalized:
            all_energy = self.simulator.sim_monitor.energy_consumption
            total_energy = T.sum(T.tensor(all_energy, dtype=T.float32))
            wasted_energy /= total_energy
        return wasted_energy
    
    def get_host_sleeping(self):
        host_states = [(0 if (node['state'] == 'sleeping') else 1 for node in self.simulator.sim_monitor.nodes_action)]
        host_states = T.tensor(host_states)
        return host_states
    
    def get_host_idle(self):
        host_states = [(0 if (node['state'] == 'idle') else 1 for node in self.simulator.sim_monitor.nodes_action)]
        host_states = T.tensor(host_states)
        return host_states
        
    def get_current_idle_time(self):
        current_idle_time = self.simulator.sim_monitor.idle_time
        current_idle_time = T.tensor(current_idle_time)
        return current_idle_time
    
    def get_host_wasted_energy(self):
        wasted_energy = [[0. for _ in range(self.simulator.nb_res)]]
        for idx in range(self.nb_res):
            wasted_energy[0][idx] = self.simulator.sim_monitor.energy_waste[idx]
            if is_normalized:
                wasted_energy[0][idx] = self.simulator.sim_monitor.energy_consumption[idx]
        wasted_energy = T.tensor(wasted_energy)
        return wasted_energy
    
    def get_switching_time(self):
        switching_time = [[0. for _ in range(self.simulator.nb_res)]]
        for index, node in enumerate(self.simulator.sim_monitor.node_state_monitor):
            switching_time[0][index] = node['switching_off'] + node['switching_on']
            if is_normalized:
                switching_time[0][index] /= self.simulator.current_time
        switching_time = T.tensor(switching_time)
        return switching_time
    
    def get_remaining_runtime_percent(self):
        remaining_runtime = [0 for _ in range(self.simulator.nb_res)]
        for job in self.simulator.jobs_monitor.active_jobs:
            for node in job['allocated_resources']:
                remaining_runtime[node] = job['finish_time'] - self.simulator.current_time
                remaining_runtime[node] /= job['walltime']
        remaining_runtime = T.tensor([remaining_runtime])
        return remaining_runtime
    
    def get_features(self):
        num_sim_features = 5
        simulator_features = T.zeros((num_sim_features,), dtype=T.float32)
        simulator_features[0] = len(self.simulator.waiting_queue)
        simulator_features[1] = self.get_arrival_rate(is_normalized = True)