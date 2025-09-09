import json


class ResourceManager:
    def __init__(self, platform_path):
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)

        self.machines = self.platform_info['machines']
        self.resources_agenda = [{'release_time': 0, 'node_id': i}
                                 for i in range(len(self.machines))]

        self.nodes = []

        self.machines_transition = []
        for machine in self.machines:
            node_transitions = []
            for from_state, data in machine["states"].items():
                for trans in data.get("transitions", []):
                    node_transitions.append({
                        "from": from_state,
                        "to": trans["state"],
                        "transition_time": trans["transition_time"]
                    })
            self.machines_transition.append({
                "node_id": machine["id"],
                "transitions": node_transitions
            })

        for machine in self.machines:
            dvfs_mode = machine['dvfs_mode']
            active_state = machine['states']['active']
            dvfs_profile = machine['dvfs_profiles'][dvfs_mode]

            power = dvfs_profile['power'] if active_state['power'] == 'from_dvfs' else active_state['power']
            compute_speed = dvfs_profile['compute_speed'] if active_state[
                'compute_speed'] == 'from_dvfs' else active_state['compute_speed']

            node = {
                'id': machine['id'],
                'state': 'active',
                'dvfs_mode': dvfs_mode,
                'power': power,
                'compute_speed': compute_speed,
                'transitions': active_state['transitions'],
                'can_run_jobs': active_state['can_run_jobs'],
                'job_id': None,
                'start_time': 0,
                'duration': 0
            }

            self.nodes.append(node)

        self.reserved_nodes = []

    def get_switching_on_nodes(self):
        switching_on = []
        for node in self.nodes:
            if node['state'] == 'switching_on':
                switching_on.append(node['id'])

        return switching_on

    def get_switching_off_nodes(self):
        switching_off = []
        for node in self.nodes:
            if node['state'] == 'switching_off':
                switching_off.append(node['id'])
        return switching_off

    def renew_resources_agenda(self, current_time):
        for resource_agenda in self.resources_agenda:
            if resource_agenda['release_time'] < current_time:
                resource_agenda['release_time'] = current_time

    def reserve_nodes(self, job_id, node_ids):
        self.reserved_nodes.append({'job_id': job_id, 'nodes': node_ids})

    def remove_reserved_nodes(self, job_id):
        self.reserved_nodes = [
            rn for rn in self.reserved_nodes if rn['job_id'] != job_id
        ]

    def get_available_nodes(self):
        available_nodes = []

        for node in self.nodes:
            if node['can_run_jobs'] and node['job_id'] == None:
                available_nodes.append(node['id'])

        return available_nodes

    def get_reserved_nodes(self):
        reserved_nodes = []
        for reserved_node in self.reserved_nodes:
            reserved_nodes.extend(reserved_node['nodes'])

        return reserved_nodes

    def get_sleeping_nodes(self):
        sleeping_nodes = []

        for node in self.nodes:
            if node['state'] == 'sleeping':
                sleeping_nodes.append(node['id'])

        return sleeping_nodes

    def change_dvfs_mode(self, nodes, mode):
        for node_id in nodes:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        for node in self.nodes:
            if node['id'] in nodes:
                if node['state'] != 'active':
                    raise RuntimeError(
                        f"Node {node['id']} must be in 'active' state to change DVFS mode")
                if mode not in self.machines[node['id']]['dvfs_profiles']:
                    raise ValueError(
                        f"Invalid DVFS mode '{mode}' for node {node['id']}")
                node['dvfs_mode'] = mode
                profile = self.machines[node['id']]['dvfs_profiles'][mode]
                node['power'] = profile['power']
                node['compute_speed'] = profile['compute_speed']

    def _update_node_duration(self, current_time):
        for node in self.nodes:
            node['duration'] = current_time - node['start_time']

    def _update_node_state(self, node, new_state):
        if new_state not in node['transitions']:
            raise RuntimeError(
                f"Invalid state transition from '{node['state']}' to '{new_state}' on node {node['id']}")
        node['state'] = new_state
        state_def = self.machines[node['id']]['states'][new_state]
        if state_def['power'] == 'from_dvfs':
            dvfs_profile = self.machines[node['id']
                                         ]['dvfs_profiles'][node['dvfs_mode']]
            node['power'] = dvfs_profile['power']
            node['compute_speed'] = dvfs_profile['compute_speed']
        else:
            node['power'] = state_def['power']
            node['compute_speed'] = state_def['compute_speed']
        node['transitions'] = state_def['transitions']
        node['can_run_jobs'] = state_def['can_run_jobs']

    def switch_on(self, node_ids, current_time):
        for node_id in node_ids:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        for node in self.nodes:
            if node['id'] in node_ids:
                if node['state'] == 'sleeping':
                    node['state'] = 'switching on'
                    node['job_id'] = None
                    node['start_time'] = current_time
                    node['duration'] = 0

                    resource_agenda = next(
                        (ra for ra in self.resources_agenda if ra['node_id'] == node_id), None)

                    for machine_transition in self.machines_transition:
                        if machine_transition['node_id'] == node_id:
                            for transition in machine_transition['transitions']:
                                if transition['from'] == 'switching_off' and transition['to'] == 'sleeping':
                                    transition_time = transition['transition_time']
                            resource_agenda['release_time'] = current_time + \
                                transition_time

                else:
                    raise RuntimeError(
                        f"Node {node['id']} cannot be switched on — state is not sleeping")

    def turn_on(self, node_ids, current_time):
        for node_id in node_ids:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        for node in self.nodes:
            if node['id'] in node_ids:
                if node['state'] == 'switching on':
                    node['state'] = 'active'
                    node['job_id'] = None
                    node['start_time'] = current_time
                    node['duration'] = 0
                    node['can_run_jobs'] = True

                    resource_agenda = next(
                        (ra for ra in self.resources_agenda if ra['node_id'] == node_id), None)

                    if resource_agenda:
                        resource_agenda['release_time'] = current_time
                else:
                    raise RuntimeError(
                        f"Node {node['id']} cannot be turned on — state is not switching on")

    def turn_off(self, node_ids, current_time):
        for node_id in node_ids:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        for node in self.nodes:
            if node['id'] in node_ids:
                if node['state'] == 'switching off':
                    node['state'] = 'sleeping'
                    node['job_id'] = None
                    node['start_time'] = current_time
                    node['duration'] = 0

                    resource_agenda = next(
                        (ra for ra in self.resources_agenda if ra['node_id'] == node_id), None)

                    if resource_agenda:
                        resource_agenda['release_time'] = current_time
                else:
                    raise RuntimeError(
                        f"Node {node['id']} cannot be turned off — state is not switching off")

    def switch_off(self, node_ids, current_time):
        for node_id in node_ids:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        for node in self.nodes:
            if node['id'] in node_ids:
                if node['state'] == 'active':
                    node['state'] = 'switching off'
                    node['job_id'] = None
                    node['start_time'] = current_time
                    node['duration'] = 0
                    node['can_run_jobs'] = False
                    node_id = node['id']

                    resource_agenda = next(
                        (ra for ra in self.resources_agenda if ra['node_id'] == node_id), None)

                    machine = next(
                        (m for m in self.machines if m['id'] == node_id), None)

                    if resource_agenda and machine:
                        machine_switching_off_transitions = machine['states']['switching_off']['transitions']
                        for machine_switching_off_transition in machine_switching_off_transitions:
                            if machine_switching_off_transition['state'] == 'sleeping':
                                transition_time = machine_switching_off_transition['transition_time']
                                break
                        resource_agenda['release_time'] = current_time + \
                            transition_time

                elif node['state'] == 'active' and node['job_id'] is not None:
                    raise RuntimeError(
                        f"Node {node['id']} cannot be switched off — it is computing")
                elif node['state'] != 'active':
                    raise RuntimeError(
                        f"Node {node['id']} cannot be switched off — state is not 'idle'")

    def allocate(self, node_ids, job_id, walltime, current_time):
        for node_id in node_ids:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        compute_demand = walltime * len(node_ids)
        compute_power = sum(node['compute_speed']
                            for node in self.nodes if node['id'] in node_ids)
        finish_time = current_time + (compute_demand / compute_power)

        for node in self.nodes:
            if node['id'] in node_ids:
                if node['can_run_jobs']:
                    node['job_id'] = job_id

                    resource_agenda = next(
                        (ra for ra in self.resources_agenda if ra['node_id'] == node['id']), None)

                    if resource_agenda:
                        resource_agenda['release_time'] = finish_time
                else:
                    raise RuntimeError(
                        f"Node {node['id']} cannot be allocated for computation — state is not 'idle'")

    def release(self, nodes):
        for node_id in nodes:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        for node in self.nodes:
            if node['id'] in nodes:
                if node['state'] != 'active':
                    raise RuntimeError(
                        f"Node {node['id']} cannot be released — state is not 'active'")
                elif node['state'] == 'active' and node['job_id'] == None:
                    raise RuntimeError(
                        f"Node {node['id']} cannot be released — state is 'active' but not computing")
                node['job_id'] = None
