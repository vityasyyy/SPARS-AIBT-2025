from .fcfs_auto_switch_on import FCFSAuto
from itertools import combinations


class EASYAuto(FCFSAuto):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        super().FCFSAuto()

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
            total_compute_power = min(node['compute_speed'] for node in combo)

            max_activation_delay = max(
                entry['release_time']
                for entry in next_releases
                if entry['id'] in node_ids
            )

            finish_time = max_activation_delay + \
                (compute_demand / total_compute_power)

            if finish_time <= p_start_t and finish_time > best_finish_time:
                best_finish_time = finish_time
                best_combo = combo

        return best_combo

    def backfill(self):

        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled]
        if len(waiting_queue) > 2:
            p_job = waiting_queue[0]

            backfilling_queue = waiting_queue[1:]

            reserved_nodes = self.allocated

            not_reserved_nodes = [
                node for node in self.state if node['id'] not in reserved_nodes]

            next_releases = self.resources_agenda

            next_releases = sorted(
                next_releases,
                key=lambda x: (x['release_time'], x['id'])
            )

            if len(next_releases) < p_job['res']:
                return

            last_host = next_releases[p_job['res'] - 1]
            p_start_t = last_host['release_time']

            candidates = [r['id']
                          for r in next_releases if r['release_time'] <= p_start_t]
            head_job_reservation = candidates[-p_job['res']:]

            not_reserved_nodes = [
                r for r in not_reserved_nodes if r['id'] not in head_job_reservation]

            for job in backfilling_queue:

                allocated_ids = [node['id'] for node in self.allocated]
                not_computing_resources = [
                    node['id'] for node in self.state if node['job_id'] is None and node['id'] not in allocated_ids]
                not_reserved = [
                    h for h in not_computing_resources if h not in head_job_reservation]

                available_resources_not_reserved = [
                    r for r in self.available if r in not_reserved]
                inactive_resources_not_reserved = [
                    r for r in self.inactive if r in not_reserved]

                if job['res'] <= len(not_reserved):
                    if job['res'] <= len(available_resources_not_reserved):
                        allocated_nodes = available_resources_not_reserved[:job['res']]
                        allocated_nodes = [node['id']
                                           for node in allocated_nodes]

                        event = {
                            'job_id': job['job_id'],
                            'subtime': job['subtime'],
                            'runtime': job['runtime'],
                            'reqtime': job['reqtime'],
                            'res': job['res'],
                            'type': 'execution_start',
                            'nodes': allocated_nodes
                        }
                        self.available = [
                            node for node in self.available if node not in allocated_nodes]
                        self.allocated.extend(allocated_nodes)
                        if event['job_id'] == 39:
                            print('x')
                        super().push_event(self.current_time, event)
                        """should update releases agenda, do the same for others'
                        """
                    elif job['res'] <= len(available_resources_not_reserved) + len(inactive_resources_not_reserved):
                        count_avail = len(available_resources_not_reserved)
                        num_need_activation = job['res'] - count_avail
                        to_activate = inactive_resources_not_reserved[:num_need_activation]

                        allocated_nodes = available_resources_not_reserved
                        reserved_nodes = allocated_nodes + to_activate

                        self.allocated.extend(reserved_nodes)

                        self.available = [
                            node for node in self.available if node not in allocated_nodes]

                        self.inactive = self.inactive[num_need_activation:]

                        allocated_ids = {
                            d["id"] for d in allocated_nodes}

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

                        for node in self.state:
                            if node['id'] in allocated_nodes:
                                node['release_time'] = finish_time

                        reserved_nodes = [node['id']
                                          for node in reserved_nodes]

                        event = {
                            'job_id': job['job_id'],
                            'subtime': job['subtime'],
                            'runtime': job['runtime'],
                            'reqtime': job['reqtime'],
                            'res': job['res'],
                            'type': 'execution_start',
                            'nodes': reserved_nodes
                        }
                        if event['job_id'] == 39:
                            print('x')
                        to_activate = [node['id'] for node in to_activate]

                        super().push_event(self.current_time, {
                            'type': 'switch_on', 'nodes': to_activate})

                        super().push_event(start_predict_time, event)
                else:
                    allocated_ids = [node['id']
                                     for node in self.allocated]
                    not_computing_resources = [
                        node['id'] for node in self.state if node['job_id'] is None and node['id'] not in allocated_ids]

                    compute_demand = job['reqtime']
                    free_nodes = [{'node_id': node['id'], 'compute_speed': node['compute_speed']}
                                  for node in self.state if node['id'] in not_computing_resources]

                    combo = self.find_node_combination(
                        p_start_t, compute_demand, free_nodes, next_releases, job['res'])

                    if combo == None:
                        continue
                    to_activate = [
                        node for node in combo if node in self.inactive]

                    if len(to_activate) == 0:
                        allocated_nodes = self.available[:job['res']]
                        self.allocated.extend(allocated_nodes)
                        allocated_ids = [node['id']
                                         for node in allocated_nodes]

                        event = {
                            'job_id': job['job_id'],
                            'subtime': job['subtime'],
                            'runtime': job['runtime'],
                            'reqtime': job['reqtime'],
                            'res': job['res'],
                            'type': 'execution_start',
                            'nodes': allocated_ids
                        }
                        if event['job_id'] == 39:
                            print('x')

                        self.available = self.available[job['res']:]
                        super().push_event(self.current_time, event)
                    else:
                        count_available_nodes = len(self.available)
                        allocated_nodes = self.available
                        num_need_activation = job['res'] - \
                            count_available_nodes
                        to_activate = self.inactive[:num_need_activation]
                        reserved_nodes = allocated_nodes + to_activate
                        self.allocated.extend(reserved_nodes)
                        self.available = []
                        self.inactive = self.inactive[num_need_activation:]

                        allocated_ids = {
                            d["id"] for d in allocated_nodes}  # use set for faster lookup

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

                        for node in self.state:
                            if node['id'] in allocated_nodes:
                                node['release_time'] = finish_time

                        reserved_nodes = [node['id']
                                          for node in reserved_nodes]

                        event = {
                            'job_id': job['job_id'],
                            'subtime': job['subtime'],
                            'runtime': job['runtime'],
                            'reqtime': job['reqtime'],
                            'res': job['res'],
                            'type': 'execution_start',
                            'nodes': reserved_nodes
                        }
                        if event['job_id'] == 39:
                            print('x')
                        super().push_event(self.current_time, {
                            'type': 'switch_on', 'nodes': to_activate})
                        super().push_event(start_predict_time, event)
