import pandas as pd

class SimMonitor:
    def __init__(self, nb_res, machines):
        self.energy_consumption = [0] * len(machines)
        self.nodes_action = [{'state': 'idle', 'time': 0} for _ in range(nb_res)]
        self.idle_time = [0] * len(machines)
        self.total_waiting_time = 0
        self.finish_time = 0
        self.node_state_log = pd.DataFrame([{
            'time': 0, 'sleeping': 0, 'sleeping_nodes': [], 'switching_on': 0, 'switching_on_nodes': [],
            'switching_off': 0, 'switching_off_nodes': [], 'idle': nb_res, 'idle_nodes': list(range(nb_res)),
            'computing': 0, 'computing_nodes': [], 'unavailable': 0
        }])
        self.nodes = [[{'type': 'idle', 'starting_time': 0, 'finish_time': 0}] for _ in range(nb_res)]
        self.node_state_monitor = [{'idle': 0, 'switching_off': 0, 'switching_on': 0, 'computing': 0, 'sleeping': 0} for _ in range(nb_res)]
        self.energy_waste = [0] * len(machines)
    
    def update_energy_waste(self):
        for node_index, node in enumerate(self.node_state_monitor):
            self.energy_waste[node_index] = (node["idle"] * 190) + (node["switching_off"] * 9) + (node["switching_on"] * 190)
    
    def update_energy_consumption(self, machines, current_time, last_event_time):
        for index, node_action in enumerate(self.nodes_action):
            state = node_action['state']
            
            state_mapping = {
                'sleeping': 0,
                'idle': 1,
                'computing': 2,
                'switching_on': 3,
                'switching_off': 4
            }
            
            rate_energy_consumption = machines[index]['wattage_per_state'][state_mapping[state]]

            last_time = max(node_action['time'], last_event_time)
            duration = current_time - last_time  

            self.energy_consumption[index] += duration * rate_energy_consumption
            
    def update_idle_time(self, current_time, last_event_time):
        for index, node_action in enumerate(self.nodes_action):
            if node_action['state'] == 'idle':
                last_time = max(node_action['time'], last_event_time) 
        
                self.idle_time[index] += current_time - last_time
                
    def update_total_waiting_time(self, num_job_in_queue, current_time, last_event_time):
        self.total_waiting_time += num_job_in_queue * (current_time - last_event_time )
      