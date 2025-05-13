import torch
from scheduler_sp.env import SPSimulator
from scheduler_sp.rl_scheduler_2 import RLScheduler
import pandas as pd
from nn_generator.model_generator_2 import HPCNodeManager


def run_simulation(scheduler, platform_filepath, workload_filepath):
    simulator = SPSimulator(scheduler, platform_path=platform_filepath, workload_path=workload_filepath)
    scheduler.simulator = simulator
    
    while simulator.events or simulator.jobs_monitor.waiting_queue:
        simulator.proceed()
    
    return simulator.jobs_monitor.monitor_jobs, simulator.sim_monitor

agent = torch.load('nn_generator/hpc_node_manager.pth', map_location=torch.device('cpu'))
rl_scheduler = RLScheduler(None, agent)
workload_filepath = "workloads/simple_data_10.json"
jobs_e, sim_e = run_simulation(rl_scheduler, "platforms/spsim/platform.json", workload_filepath)


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
jobs_e.to_csv('results/cssp/rl/easy_jobs.csv', index=False)