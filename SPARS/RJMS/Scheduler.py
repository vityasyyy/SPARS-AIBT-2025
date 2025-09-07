from SPARS.RJMS.Algo.SmartFCFS import SmartFCFS
from SPARS.RJMS.Algo.FCFS import FCFS
from SPARS.RJMS.Algo.EASY import EASY

ALGO_MAP = {
    'fcfs': FCFS,
    'easy': EASY,
    'smart-fcfs': SmartFCFS
}


class Scheduler:
    def __init__(self, ResourceManager, JobsManager, algorithm_name, start_time, timeout=None):
        AlgorithmClass = ALGO_MAP[algorithm_name.lower()]
        self.algorithm = AlgorithmClass(
            ResourceManager,
            JobsManager,
            start_time,
            timeout
        )

    def schedule(self, current_time):
        self.algorithm.set_time(current_time)
        events = self.algorithm.schedule()
        return events
