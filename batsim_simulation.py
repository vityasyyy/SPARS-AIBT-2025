import batsim_py
from scheduler.fcfs import FCFSScheduler
from scheduler.backfilling import EASYScheduler
from evalys.jobset import JobSet

def run_simulation(scheduler):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    jobs_mon = batsim_py.monitors.JobMonitor(simulator)
    sim_mon = batsim_py.monitors.SimulationMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform="platforms/batsim/platform.xml",
                    workload="workloads/simple_data.json",
                    verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        simulator.proceed_time()  # proceed directly to the next event.
    simulator.close()

    # 4) Return/Dump statistics
    return jobs_mon, sim_mon
    
jobs_f, sim_f = run_simulation(FCFSScheduler)
jobs_e, sim_e = run_simulation(EASYScheduler)

print(sim_e.info['consumed_joules'])
print(sum(jobs_e.info['waiting_time']))
print(max(jobs_e.info['finish_time']))

fcfs, easy = jobs_f.to_dataframe(), jobs_e.to_dataframe()
print(JobSet(fcfs))
# fcfs['allocated_resources'] = fcfs['allocated_resources'].apply(lambda x: ','.join(map(str, x)))
# easy['allocated_resources'] = easy['allocated_resources'].apply(lambda x: ','.join(map(str, x)))

fcfs.to_csv('results/batsim/fcfs_jobs.csv', index=False)
easy.to_csv('results/batsim/easy_jobs.csv', index=False)

fcfs, easy = sim_f.to_dataframe(), sim_e.to_dataframe()
fcfs.to_csv('results/batsim/fcfs_sims.csv', index=False)
easy.to_csv('results/batsim/easy_sims.csv', index=False)