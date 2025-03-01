from scheduler_sp.env import SPSimulator
from scheduler_sp.fcfs_scheduler import FCFSScheduler
from scheduler_sp.easy_scheduler import EasyScheduler
import pandas as pd

def run_simulation(scheduler):
    simulator = SPSimulator(scheduler, model=None)
    scheduler.simulator = simulator
    
    while simulator.schedule_queue:
        simulator.proceed()
    
    return simulator.monitor_jobs, simulator.sim_monitor

fcfs_scheduler = FCFSScheduler(None)
easy_scheduler = EasyScheduler(None)

jobs_f, sim_f = run_simulation(fcfs_scheduler)
jobs_e, sim_e = run_simulation(easy_scheduler)

print('~~~ FCFS SCHEDULER ~~~')
print('total idle time: ',sim_f['total_idle_time'])
print('mean idle time: ',sim_f['total_idle_time'])
print('finish time: ',sim_f['finish_time'])
print('total waiting time: ',sim_f['total_waiting_time'])
print('mean waiting time: ',sim_f['total_waiting_time']/len(jobs_f))

print('~~~ EASY SCHEDULER ~~~')
print('total idle time: ',sim_e['total_idle_time'])
print('mean idle time: ',sim_e['total_idle_time'])
print('finish time: ',sim_e['finish_time'])
print('total waiting time: ',sim_e['total_waiting_time'])
print('mean waiting time: ',sim_e['total_waiting_time']/len(jobs_e))

jobs_f = pd.DataFrame(jobs_f)
jobs_f['allocated_resources'] = jobs_f['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_f.to_csv('results/cssp/baseline/fcfs_jobs.csv', index=False)

jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)
jobs_e.to_csv('results/cssp/baseline/easy_jobs.csv', index=False)