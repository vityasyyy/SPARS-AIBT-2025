from .proactive import Proactive

class SmartBackfill(Proactive):
    def __init__(self, state, waiting_queue, start_time, jobs_manager, timeout=None):
        super().__init__(state, waiting_queue, start_time, jobs_manager, timeout)
        if self.state:
            # Extract power info from the platform definition
            machine_info = self.state[0]
            self.power_active = machine_info['dvfs_profiles']['base']['power']
            self.power_shutdown = machine_info['states']['switching_off']['power']
            self.power_boot = machine_info['states']['switching_on']['power']
            self.time_shutdown = machine_info['states']['switching_off']['transitions'][0]['transition_time']
            self.time_boot = machine_info['states']['switching_on']['transitions'][0]['transition_time']
        print("EnergyAwareScheduler Initialized.")
        
    def _is_worth_to_backfill(self, job, nodes, shadow_time):
        
