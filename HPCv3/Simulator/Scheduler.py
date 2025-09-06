from HPCv3.Simulator.Algo.fcfs_auto_switch_on import FCFSAuto
from HPCv3.Simulator.Algo.fcfs_normal import FCFSNormal

ALGO_MAP = {
    'fcfs_auto': FCFSAuto,
    'fcfs_normal': FCFSNormal
}


class Scheduler:
    def __init__(self, state, waiting_queue, algorithm, start_time, timeout=None):
        AlgorithmClass = ALGO_MAP[algorithm.lower()]
        self.algorithm = AlgorithmClass(
            state,
            waiting_queue,
            start_time,
            timeout
        )

    def schedule(self, current_time, new_state, waiting_queue, scheduled_queue):
        self.algorithm.set_time(current_time)
        events = self.algorithm.schedule(
            new_state, waiting_queue, scheduled_queue)
        return events
