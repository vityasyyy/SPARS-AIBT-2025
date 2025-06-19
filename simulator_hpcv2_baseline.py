from HPCv2_Simulator.Simulator import SPSimulator
from HPCv2_Scheduler.easy_scheduler import EasyScheduler
import pandas as pd

def run_simulation(scheduler, platform_filepath, workload_filepath):
    simulator = SPSimulator(scheduler, platform_path=platform_filepath, workload_path=workload_filepath)
    scheduler.simulator = simulator
    
    while simulator.jobs_manager.events or simulator.jobs_manager.waiting_queue:
        simulator.proceed()
    
    return simulator.jobs_manager, simulator.sim_monitor

easy_scheduler = EasyScheduler(None)

workload_filepath = "workloads/simple_data_1000.json"
platform_filepath = "platforms/spsim/platform.json"
jobs_e, sim_e = run_simulation(easy_scheduler, platform_filepath, workload_filepath)

print('~~~ EASY SCHEDULER ~~~')
print('total idle time: ',sum(sim_e.idle_time))
print('mean idle time: ',sim_e.idle_time)
print('finish time: ',sim_e.finish_time)
print('total waiting time: ',sim_e.total_waiting_time)
print('mean waiting time: ',sim_e.total_waiting_time/len(jobs_e.monitor_jobs))
print('energy consumption: ', sim_e.energy_consumption)
print('energy waste: ', sum(sim_e.energy_waste))

jobs_e = pd.DataFrame(jobs_e.monitor_jobs)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)
jobs_e.to_csv('results/cssp/baseline/easy_jobs_simple_1000.csv', index=False)