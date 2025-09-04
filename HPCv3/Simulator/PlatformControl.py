from HPCv3.Simulator.Machine import Machine


class PlatformControl:
    def __init__(self, platform_info, start_time):
        self.machines = Machine(platform_info, start_time)

        self.active_jobs = []

    def get_state(self):
        return self.machines.nodes

    def compute(self, node_ids, job, current_time):
        self.machines.allocate(node_ids, job['job_id'])
        compute_demand = job['walltime'] * job['res']
        compute_power = sum(node['compute_speed']
                            for node in self.machines.nodes if node['id'] in node_ids)
        finish_time = current_time + (compute_demand / compute_power)
        event = {'job_id': job['job_id'], 'type': 'execution_finished', 'nodes': node_ids,
                 'start_time': current_time, 'submission_time': job['subtime']}
        return finish_time, event

    def change_dvfs_mode(self, node_ids, mode):
        self.machines.change_dvfs_mode(node_ids, mode)
        return {'type': 'change_dvfs_mode', 'node': node_ids, 'mode': mode}

    def release(self, node_ids):
        self.machines.release(node_ids)

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
