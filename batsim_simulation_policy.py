from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor
from batsim_utils.shutdown_policy import TimeoutPolicy


def run_simulation(shutdown_policy):
    simulator = SimulatorHandler()
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
        for job in simulator.queue:
            available = simulator.platform.get_not_allocated_hosts()
            if job.res <= len(available):
                allocation = [h.id for h in available[:job.res]]
                simulator.allocate(job.id, allocation)

        # proceed directly to the next event because the shutdown_policy is event-based.
        simulator.proceed_time()

    simulator.close()

    # 4) Return/Dump statistics
    return jobs_mon, sim_mon, host_mon, e_mon

jobs_none, sim_none, host_none, e_none = run_simulation(lambda s: None) # Without shutdown
jobs_none = jobs_none.to_dataframe()
print(jobs_none)

# jobs_t1, sim_t1, host_t1, e_t1 = run_simulation(lambda s: TimeoutPolicy(5, s)) # Timeout (1)
# print(jobs_t1)

# jobs_t5, sim_t5, host_t5, e_t5 = run_simulation(lambda s: TimeoutPolicy(5, s)) # Timeout (5)s
# print(jobs_t5)