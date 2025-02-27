import heapq
import copy
from .env import MyDict


def simulate_easy(self, timeout):

    for event in self.jobs:
        event_time, event_detail = event
        heapq.heappush(self.schedule_queue, (event_time, MyDict(event_detail)))
    
    self.start_simulator(timeout)
    

    while self.schedule_queue or self.waiting_queue:      
        self.proceed(timeout)
    
    return self.monitor_jobs