from SPARS.Simulator.Algo.BaseAlgorithm import BaseAlgorithm


class FCFSNormal(BaseAlgorithm):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        self.FCFSNormal()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFSNormal(self):
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
                break
