from HPCv3.Simulator.Algo.BaseAlgorithm import BaseAlgorithm


class FCFS(BaseAlgorithm):
    def schedule(self, new_state, waiting_queue, scheduled_queue):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue)
        self.FCFS()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFS(self):
        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled]
        for job in waiting_queue:
            if job['res'] <= len(self.available) + len(self.inactive):
                if job['res'] <= len(self.available):
                    allocated_nodes = self.available[:job['res']]

                    self.available = self.available[job['res']:]
                    self.allocated.extend(allocated_nodes)
                    compute_demand = job['walltime'] * job['res']
                    compute_power = sum(
                        node['compute_speed'] for node in self.state if node in allocated_nodes)
                    finish_time = self.current_time + \
                        (compute_demand / compute_power)
                    for node in self.state:
                        if node['id'] in allocated_nodes:
                            node['release_time'] = finish_time
                    self.scheduled.append(job['job_id'])

                    allocated_nodes = [node['id'] for node in allocated_nodes]
                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'walltime': job['walltime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': allocated_nodes
                    }
                    super().push_event(self.current_time, event)
                else:
                    count_avail = len(self.available)
                    num_need_activation = job['res'] - count_avail
                    to_activate = self.inactive[:num_need_activation]
                    reserved_nodes = self.available + to_activate

                    allocated_nodes = reserved_nodes
                    self.allocated.extend(reserved_nodes)
                    self.available = []
                    self.inactive = self.inactive[num_need_activation:]

                    """should consider activation delay"""
                    highest_transition_time = 0
                    for node in self.state:
                        if node['id'] in reserved_nodes:
                            for transition in node['transitions']:
                                if transition['from'] == 'switching_off' and transition['to'] == 'sleeping' and transition['transition_time'] > highest_transition_time:
                                    highest_transition_time = transition['transition_time']

                    compute_demand = job['walltime'] * job['res']
                    compute_power = sum(
                        node['compute_speed'] for node in self.state if node in reserved_nodes)
                    start_predict_time = self.current_time + highest_transition_time
                    finish_time = start_predict_time + \
                        (compute_demand / compute_power)

                    for node in self.state:
                        if node['id'] in allocated_nodes:
                            node['release_time'] = finish_time

                    reserved_nodes = [node['id'] for node in reserved_nodes]
                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'walltime': job['walltime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': reserved_nodes
                    }
                    super().push_event(self.current_time, {
                        'type': 'switch_on', 'nodes': to_activate})
                    super().push_event(start_predict_time, event)

            else:
                break
