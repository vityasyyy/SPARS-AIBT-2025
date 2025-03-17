import torch
from scheduler_sp.env import SPSimulator
from scheduler_sp.rl_scheduler import RLScheduler
import pandas as pd
from actor import HPCNodeManager
from critic import HPCCritic

def run_simulation(scheduler):
    simulator = SPSimulator(scheduler)
    scheduler.simulator = simulator
    
    while simulator.schedule_queue or simulator.jobs_monitor.waiting_queue:
        simulator.proceed()
    
    return simulator.jobs_monitor.monitor_jobs, simulator.sim_monitor

actor = torch.load('untrained/hpc_actor.pth', map_location=torch.device('cpu'))
critic = torch.load('untrained/hpc_critic.pth', map_location=torch.device('cpu'))
rl_scheduler = RLScheduler(None, actor, critic)

jobs_e, sim_e = run_simulation(rl_scheduler)


print('~~~ EASY SCHEDULER ~~~')
print('total idle time: ',sum(sim_e.idle_time))
print('mean idle time: ',sim_e.idle_time)
print('finish time: ',sim_e.finish_time)
print('total waiting time: ',sim_e.total_waiting_time)
print('mean waiting time: ',sim_e.total_waiting_time/len(jobs_e))

jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)
jobs_e.to_csv('results/cssp/baseline/easy_jobs.csv', index=False)