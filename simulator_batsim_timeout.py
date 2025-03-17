import time
from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor
from batsim_utils.shutdown_policy import TimeoutPolicy
from scheduler_batsim.backfilling import EASYScheduler

def run_simulation(scheduler, shutdown_policy):
    simulator = SimulatorHandler()
    scheduler = scheduler(simulator)
    policy = shutdown_policy(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    jobs_mon = JobMonitor(simulator)
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform="platforms/batsim/platform.xml",
                    workload="workloads/workload_mg_5.json",
                    verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        # First Fit policy
        scheduler.schedule()

        # proceed directly to the next event because the shutdown_policy is event-based.
        simulator.proceed_time()

    simulator.close()

    # 4) Return/Dump statistics
    return jobs_mon, sim_mon, host_mon, e_mon

timeout = 100
jobs_wt, sim_wt, host_wt, e_wt = run_simulation(EASYScheduler, lambda s: TimeoutPolicy(timeout, s))

print(f"Execution time: {execution_time:.6f} seconds")

print('jul: ',sim_wt.info['consumed_joules'])
print('idle_time: ',sim_wt.info['time_idle'])
print('mean_idle_time: ',sim_wt.info['time_idle']/16)
print('waiting_time: ',sum(jobs_wt.info['waiting_time']))
print('mean_waiting_time: ',sim_wt.info['mean_waiting_time'])
print('energy waste: ',sim_wt.info['energy_waste'])
print('finish_time', max(jobs_wt.info['finish_time']))


jobs_wt = jobs_wt.to_dataframe()
host_wt = host_wt.to_dataframe()
jobs_wt.to_csv(f'results/batsim/timeout/easy_jobs_t{timeout}.csv', index=False)
host_wt.to_csv(f'results/batsim/timeout/easy_host_t{timeout}.csv', index=False)

