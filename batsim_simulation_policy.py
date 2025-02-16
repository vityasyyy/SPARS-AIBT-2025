from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor
from batsim_utils.shutdown_policy import TimeoutPolicy
from scheduler.backfilling import EASYScheduler

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
                    workload="workloads/simple_data.json",
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


jobs_t5, sim_t5, host_t5, e_t5 = run_simulation(EASYScheduler, lambda s: TimeoutPolicy(30, s))

jobs_t5 = jobs_t5.to_dataframe()
host_t5 = host_t5.to_dataframe()
jobs_t5.to_csv('results/batsim/easy_jobs_t5.csv', index=False)
host_t5.to_csv('results/batsim/easy_host_t5.csv', index=False)