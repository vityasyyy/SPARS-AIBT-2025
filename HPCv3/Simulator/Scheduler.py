from HPCv3.Simulator.Algo.fcfs import FCFS


ALGO_MAP = {
    'fcfs': FCFS,
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
