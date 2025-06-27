from HPCv2_Simulator.Simulator import SPSimulator
from HPCv2_Scheduler.easy_scheduler import EasyScheduler
import pandas as pd
from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor, HostMonitor, HostPowerStateSwitchMonitor, Monitor, SchedulerMonitor
from batsim_utils.shutdown_policy import TimeoutPolicy
from scheduler_batsim.backfilling import EASYScheduler

fp_hpcv2_platform = "platforms/spsim/platform_validate.json"
fp_batsimpy_platform = "platforms/batsim/platform_validate.xml"
fp_workload = "workloads/validate_data_f1.json"
timeout = 30

def run_simulation(scheduler, shutdown_policy):
    simulator = SimulatorHandler()
    scheduler = scheduler(simulator)
    policy = shutdown_policy(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    jobs_mon = JobMonitor(simulator)
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)
    h_mon = HostMonitor(simulator)
    hpss_mon = HostPowerStateSwitchMonitor(simulator)
    # mon = Monitor(simulator)
    sch_mon = SchedulerMonitor(simulator)
    # 2) Start simulation
    simulator.start(platform=fp_batsimpy_platform,
                    workload=fp_workload,
                    verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        # First Fit policy
        scheduler.schedule()

        # proceed directly to the next event because the shutdown_policy is event-based.
        simulator.proceed_time()

    simulator.close()

    # 4) Return/Dump statistics
    return jobs_mon, sim_mon, host_mon, e_mon, h_mon, hpss_mon, sch_mon

jobs_wt, sim_wt, host_wt, e_wt, h_wt, hpss_wt, sch_wt = run_simulation(EASYScheduler, lambda s: TimeoutPolicy(timeout, s))


print('jul: ',sim_wt.info['consumed_joules'])
print('idle_time: ',sim_wt.info['time_idle'])
print('mean_idle_time: ',sim_wt.info['time_idle']/16)
print('waiting_time: ',sum(jobs_wt.info['waiting_time']))
print('mean_waiting_time: ',sim_wt.info['mean_waiting_time'])
print('energy waste: ',sim_wt.info['energy_waste'])
print('finish_time', max(jobs_wt.info['finish_time']))


jobs_wt = jobs_wt.to_dataframe()
host_wt = host_wt.to_dataframe()
jobs_wt.to_csv(f'result_validate/batsimpy_easy_jobs_t{timeout}.csv', index=False)
sim_wt.to_csv(f'result_validate/batsimpy_easy_sims_t{timeout}.csv')
host_wt.to_csv(f'result_validate/batsimpy_easy_host_t{timeout}.csv', index=False)
e_wt.to_csv(f'result_validate/batsimpy_easy_e_t{timeout}.csv')
h_wt.to_csv(f'result_validate/batsimpy_easy_h_t{timeout}.csv')
hpss_wt.to_csv(f'result_validate/batsimpy_easy_hpss_t{timeout}.csv')
# mon_wt.to_csv(f'result_validate/batsimpy_easy_m_t{timeout}.csv')
sch_wt.to_csv(f'result_validate/batsimpy_easy_sch_t{timeout}.csv')


### HPC V2
def process_node_job_data(nodes_data, jobs):
    job_type_map = {'idle': -1, 'switching_off': -2, 'switching_on': -3, 'sleeping': -4}
    
    flattened_data = []
    for idx, sublist in enumerate(nodes_data):
        for item in sublist:
            item['allocated_resources'] = idx
            item['job_id'] = job_type_map.get(item['type'], 0)
            flattened_data.append(item)
    
    df = pd.DataFrame(flattened_data)
    df = df[['allocated_resources', 'type', 'starting_time', 'finish_time', 'job_id']]
    df = df[df['type'] != 'computing']
    
    grouped_df = df.groupby(['type', 'starting_time', 'finish_time']).agg({
        'allocated_resources': lambda x: ' '.join(map(str, x)), 
        'job_id': 'first'
    }).reset_index()
    
    grouped_df = grouped_df.sort_values(by=['starting_time', 'finish_time'])
    grouped_df['submission_time'] = grouped_df['starting_time']
    
    jobs = jobs.loc[:, ['job_id', 'allocated_resources', 'starting_time', 'finish_time', 'submission_time']].copy()
    jobs.loc[:, 'type'] = 'computing'
    
    final_df = pd.concat([grouped_df, jobs], ignore_index=True).sort_values(by=['starting_time', 'finish_time'])
    return final_df

def run_simulation(scheduler, platform_filepath, workload_filepath, timeout):
    simulator = SPSimulator(scheduler, platform_path=platform_filepath, workload_path=workload_filepath, timeout = timeout)
    scheduler.simulator = simulator
    simulator.start_simulator()
    while simulator.is_running:
        simulator.proceed()
    
    return simulator.jobs_manager, simulator.sim_monitor


easy_scheduler = EasyScheduler(None)
workload_filepath = "workloads/validate_data_f1.json"
platform_filepath = "platforms/spsim/platform_validate.json"
timeout = 30
jobs_e, sim_e = run_simulation(easy_scheduler, platform_filepath, workload_filepath, timeout)


print('~~~ EASY SCHEDULER ~~~')
print('total idle time: ',sum(sim_e.idle_time))
print('idle time: ',sim_e.idle_time)
print('finish time: ',sim_e.finish_time)
print('total waiting time: ',sim_e.total_waiting_time)
print('mean waiting time: ',sim_e.total_waiting_time/len(jobs_e.monitor_jobs))
print('energy consumption: ', sim_e.energy_consumption)
print('energy waste: ', sum(sim_e.energy_waste))

jobs_e = pd.DataFrame(jobs_e.monitor_jobs)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_e.to_csv(f'result_validate/hpcv2_jobs_t{timeout}.csv', index=False)

sim_e.node_state_log.to_csv(f'result_validate/hpcv2_hosts_t{timeout}.csv', index=False)

nodes_e = process_node_job_data(sim_e.nodes, jobs_e)
nodes_e.to_csv(f'result_validate/hpcv2_nodes_t{timeout}.csv', index=False)