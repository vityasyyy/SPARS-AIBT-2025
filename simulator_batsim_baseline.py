import batsim_py
from scheduler_batsim.fcfs import FCFSScheduler
from scheduler_batsim.backfilling import EASYScheduler
from evalys.jobset import JobSet
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor

def run_simulation(scheduler):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    jobs_mon = JobMonitor(simulator)
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform="platforms/batsim/platform.xml",
                    workload="workloads/simple_data_1000.json",
                    verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        simulator.proceed_time()  # proceed directly to the next event.
    simulator.close()

    # 4) Return/Dump statistics
    return jobs_mon, sim_mon, host_mon, e_mon
    
# # jobs_f, sim_f, host_f, energy_f = run_simulation(FCFSScheduler)
# print('~~~ FCFS SCHEDULER ~~~')
# print('jul: ',sim_f.info['consumed_joules'])
# print('idle_time: ',sim_f.info['time_idle'])
# print('mean_idle_time: ',sim_f.info['time_idle']/16)
# print('waiting_time: ',sum(jobs_f.info['waiting_time']))
# print('mean_waiting_time: ',sim_f.info['mean_waiting_time'])
# print('finish_time', max(jobs_f.info['finish_time']))
# fcfs, easy = jobs_f.to_dataframe(), jobs_e.to_dataframe()
# print(JobSet(fcfs))
# # fcfs['allocated_resources'] = fcfs['allocated_resources'].apply(lambda x: ','.join(map(str, x)))
# # easy['allocated_resources'] = easy['allocated_resources'].apply(lambda x: ','.join(map(str, x)))

# fcfs.to_csv('results/batsim/baseline/fcfs_jobs.csv', index=False)



jobs_e, sim_e, host_e, energy_e= run_simulation(EASYScheduler)
print('~~~ EASY SCHEDULER ~~~')
print('jul: ',sim_e.info['consumed_joules'])
print('idle_time: ',sim_e.info['time_idle'])
print('mean_idle_time: ',sim_e.info['time_idle']/16)
print('waiting_time: ',sum(jobs_e.info['waiting_time']))
print('mean_waiting_time: ',sim_e.info['mean_waiting_time'])
print('finish_time', max(jobs_e.info['finish_time']))
easy = jobs_e.to_dataframe()
easy.to_csv('results/batsim/baseline/easy_jobs_simple_1000.csv', index=False)
# fcfs, easy = sim_f.to_dataframe(), sim_e.to_dataframe()
# fcfs.to_csv('results/batsim/baseline/fcfs_sims.csv', index=False)
# easy = sim_e.to_dataframe()
# easy.to_csv('results/batsim/baseline/easy_sims.csv', index=False)