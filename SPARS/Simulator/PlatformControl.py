from SPARS.Simulator.Machine import Machine
import logging
logger = logging.getLogger("runner")


class PlatformControl:
    def __init__(self, platform_info, start_time):
        self.machines = Machine(platform_info, start_time)
        self.resources_agenda = self.resources_agenda = [{'release_time': 0, 'id': i}
                                                         for i in range(len(self.machines.nodes))]

        self.active_jobs = []

    def get_state(self):
        return self.machines.nodes

    def update_resources_agenda_global(self, current_time):
        for resource_agenda in self.resources_agenda:
            node = self.machines.nodes[resource_agenda['id']]
            node_transitions = self.machines.machines_transition[resource_agenda['id']]
            if node['state'] == 'sleeping':
                for transition in node_transitions['transitions']:
                    if transition['from'] == 'switching_on' and transition['to'] == 'active':
                        resource_agenda['release_time'] = current_time + \
                            transition['transition_time']
            elif node['state'] == 'active' and node['job_id'] is None:
                resource_agenda['release_time'] = current_time

    def update_resource_agenda(self, node_ids, release_time):
        for resource_agenda in self.resources_agenda:
            if resource_agenda['id'] in node_ids:
                resource_agenda['release_time'] = release_time

    def compute(self, node_ids, job, current_time):
        if len(node_ids) < job['res']:
            raise RuntimeError(
                f"Allocated nodes {node_ids} is not sufficient for job {job['job_id']}, requested resources={job['res']}")
        success = self.machines.allocate(node_ids, job['job_id'])

        if not success:
            logger.info(f'Job {job} failed to execute')
            return None
        compute_power = min(node['compute_speed']
                            for node in self.machines.nodes if node['id'] in node_ids)

        actual_compute_demand = job['runtime']
        actual_finish_time = current_time + \
            (actual_compute_demand / compute_power)

        requested_compute_demand = job['reqtime']
        requested_finish_time = current_time + \
            (requested_compute_demand / compute_power)
        event = {'job_id': job['job_id'], 'type': 'execution_finished', 'nodes': node_ids,
                 'start_time': current_time, 'subtime': job['subtime'], 'start_time': current_time, 'actual_finish_time': actual_finish_time, 'req_finish_time': requested_finish_time}

        finish_time = min(requested_finish_time, actual_finish_time)
        self.update_resource_agenda(node_ids, finish_time)
        return finish_time, event

    def change_dvfs_mode(self, node_ids, mode):
        self.machines.change_dvfs_mode(node_ids, mode)
        return {'type': 'change_dvfs_mode', 'node': node_ids, 'mode': mode}

    def release(self, event, current_time):
        terminated = False  # under request
        if event['req_finish_time'] < event['actual_finish_time']:
            terminated = True
        self.machines.release(event['nodes'])
        self.update_resource_agenda(event['nodes'], current_time)

        return terminated

    def turn_on(self, node_ids):
        self.machines.turn_on(node_ids)

    def turn_off(self, node_ids):
        self.machines.turn_off(node_ids)

    def switch_off(self, node_ids, current_time):
        self.machines.switch_off(node_ids)

        timestamp_map = {}

        for node_id in node_ids:
            for machine_transition in self.machines.machines_transition:
                if machine_transition['node_id'] == node_id:
                    for transition in machine_transition['transitions']:
                        if transition['from'] == 'switching_off' and transition['to'] == 'sleeping':
                            transition_time = transition['transition_time']

                    timestamp = current_time + transition_time

                    if timestamp not in timestamp_map:
                        timestamp_map[timestamp] = []
                    timestamp_map[timestamp].append(node_id)
                    break

        result = []
        for timestamp, nodes in timestamp_map.items():
            result.append({
                'event': {'type': 'turn_off', 'nodes': nodes},
                'timestamp': timestamp
            })

        return result

    def switch_on(self, node_ids, current_time):
        self.machines.switch_on(node_ids)

        timestamp_map = {}

        for node_id in node_ids:
            for machine_transition in self.machines.machines_transition:
                if machine_transition['node_id'] == node_id:
                    for transition in machine_transition['transitions']:
                        if transition['from'] == 'switching_off' and transition['to'] == 'sleeping':
                            transition_time = transition['transition_time']
                    timestamp = current_time + transition_time

                    if timestamp not in timestamp_map:
                        timestamp_map[timestamp] = []
                    timestamp_map[timestamp].append(node_id)
                    break

        result = []
        for timestamp, nodes in timestamp_map.items():
            result.append({
                'event': {'type': 'turn_on', 'nodes': nodes},
                'timestamp': timestamp
            })

        return result
