from .env import SPSimulator

class FCFSScheduler:
    def __init__(self, simulator: SPSimulator):
        self.simulator = simulator
        
    def schedule(self):
        for job in self.simulator.waiting_queue[:]:
            if job['res'] <= len(self.simulator.get_not_allocated_resources()):
                reserved_node, need_activation_node = self.simulator.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
            else:
                break