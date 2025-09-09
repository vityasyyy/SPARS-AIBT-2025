from SPARS.Simulator.Algo.BaseAlgorithm import BaseAlgorithm


class FCFSAuto(BaseAlgorithm):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        self.FCFSAuto()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFSAuto(self):
        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled]
        for job in waiting_queue:
            if job['res'] <= len(self.available) + len(self.inactive):
                if job['res'] <= len(self.available):
                    allocated_nodes = self.available[:job['res']]
                    allocated_ids = [node['id'] for node in allocated_nodes]
                    self.available = self.available[job['res']:]
                    self.allocated.extend(allocated_nodes)
                    compute_demand = job['reqtime'] * job['res']
                    compute_power = sum(
                        node['compute_speed'] for node in self.state if node in allocated_nodes)
                    finish_time = self.current_time + \
                        (compute_demand / compute_power)

                    agenda_map = {ra["id"]: ra for ra in self.resources_agenda}

                    for node in self.state:
                        if node["id"] in allocated_ids:
                            agenda_map[node["id"]
                                       ]["release_time"] = finish_time
                    self.scheduled.append(job['job_id'])

                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'runtime': job['runtime'],
                        'reqtime': job['reqtime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': allocated_ids
                    }
                    super().push_event(self.current_time, event)
                else:
                    count_avail = len(self.available)
                    num_need_activation = job['res'] - count_avail
                    to_activate = self.inactive[:num_need_activation]

                    reserved_nodes = self.available + to_activate

                    allocated_nodes = reserved_nodes
                    allocated_ids = [node['id'] for node in allocated_nodes]
                    self.allocated.extend(reserved_nodes)
                    self.available = []
                    self.inactive = self.inactive[num_need_activation:]

                    """should consider activation delay"""

                    highest_release_time = max(
                        (ra["release_time"]
                         for ra in self.resources_agenda if ra["id"] in allocated_ids),
                        default=0
                    )

                    compute_demand = job['reqtime']
                    compute_power = min(
                        node['compute_speed'] for node in self.state if node in reserved_nodes)
                    start_predict_time = highest_release_time
                    finish_time = start_predict_time + \
                        (compute_demand / compute_power)

                    agenda_map = {ra["id"]: ra for ra in self.resources_agenda}

                    for node in self.state:
                        if node["id"] in allocated_ids:
                            agenda_map[node["id"]
                                       ]["release_time"] = finish_time

                    reserved_nodes = [node['id'] for node in reserved_nodes]

                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'runtime': job['runtime'],
                        'reqtime': job['reqtime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': reserved_nodes
                    }
                    to_activate = [node['id'] for node in to_activate]
                    super().push_event(self.current_time, {
                        'type': 'switch_on', 'nodes': to_activate})
                    super().push_event(start_predict_time, event)

            else:
                break
