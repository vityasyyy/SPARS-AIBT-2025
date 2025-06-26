class FCFSScheduler:
    def __init__(self, simulator):
        self.simulator = simulator
        
    def schedule(self):

        for job in self.simulator.jobs_manager.waiting_queue[:]:
            not_allocated_resources = self.simulator.node_manager.get_not_allocated_resources()
            if self.simulator.current_time == 52:
                print('debug')
            if job['res'] <= len(not_allocated_resources):
                reserved_node, need_activation_node = self.simulator.node_manager.prioritize_lowest_node(job['res'])
                self.simulator.execution_start(job, reserved_node, need_activation_node)
            else:
                break