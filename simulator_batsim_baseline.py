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
