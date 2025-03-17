import batsim_py
from scheduler_batsim.backfilling import EASYScheduler

from scheduler_sp.env import SPSimulator
from scheduler_sp.easy_scheduler import EasyScheduler
import pandas as pd
import time

from workloads_generator import ProblemGenerator
import json

from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor
import matplotlib.pyplot as plt

def run_simulation_sp(scheduler, platform_filepath, workload_filepath):
    simulator = SPSimulator(scheduler, platform_path=platform_filepath, workload_path=workload_filepath)
    scheduler.simulator = simulator
    
    while simulator.schedule_queue or simulator.jobs_monitor.waiting_queue:
        simulator.proceed()
    
    return simulator.jobs_monitor.monitor_jobs, simulator.sim_monitor

def run_simulation_batsim(scheduler, platform_filepath, workload_filepath):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)

    # Instantiate monitors to collect simulation statistics
    jobs_mon = JobMonitor(simulator)
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # Start simulation
    simulator.start(platform=platform_filepath,
                    workload=workload_filepath,
                    verbosity="information")

    # Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        simulator.proceed_time()  # proceed directly to the next event.
    simulator.close()

    # Return statistics
    return jobs_mon, sim_mon, host_mon, e_mon

problem_generator = ProblemGenerator(num_jobs=1000, max_node=8)
easy_scheduler_sp = EasyScheduler(None)

sp_execution_times = []
batsim_execution_times = []
steps = 100
workload_id = 1
failed_count = 0

def benchmarking():
    global workload_id, failed_count
    while workload_id <= steps:
        print(f'~~~ WORKLOAD {workload_id}~~~')
        print(f'Failed Count: {failed_count}')
        # Generate problem
        workload_filepath = f"workloads/workload_find_error_{workload_id}.json"
        workloads = problem_generator.generate()

        workloads = {
            "nb_res": 16,
            "jobs": workloads,
            "profiles": {
                "100": {
                    "cpu": 10000000000000000000000,
                    "com": 0,
                    "type": "parallel_homogeneous"
                }
            }
        }
        
        with open(workload_filepath, "w") as json_file:
            json.dump(workloads, json_file, indent=4)
            
        # SP-sim
        start_time_sp = time.perf_counter()
        jobs_e_sp, sim_e_sp = run_simulation_sp(easy_scheduler_sp, "platforms/spsim/platform.json", workload_filepath)
        end_time_sp = time.perf_counter()
        execution_time_sp = end_time_sp - start_time_sp
  
        # Batsim
        start_time_batsim = time.perf_counter()
        jobs_e_batsim, sim_e_batsim, host_e_batsim, energy_e_batsim = run_simulation_batsim(EASYScheduler, "platforms/batsim/platform.xml", workload_filepath)
        end_time_batsim = time.perf_counter()
        execution_time_batsim = end_time_batsim - start_time_batsim
        
       
        different_result = False

        # Check energy waste
        sp_energy_waste = sum(sim_e_sp.energy_waste)
        batsim_energy_waste = sim_e_batsim.info['energy_waste']
        if sp_energy_waste != batsim_energy_waste:
            print(f"[ERROR] Energy waste mismatch: SP={sp_energy_waste}, Batsim={batsim_energy_waste}")
            different_result = True

        # Check finish time
        sp_finish_time = sim_e_sp.finish_time
        batsim_finish_time = max(jobs_e_batsim.info['finish_time'])
        if sp_finish_time != batsim_finish_time:
            print(f"[ERROR] Finish time mismatch: SP={sp_finish_time}, Batsim={batsim_finish_time}")
            different_result = True

        # Check idle time
        sp_idle_time = sum(sim_e_sp.idle_time)
        batsim_idle_time = sim_e_batsim.info['time_idle']
        if sp_idle_time != batsim_idle_time:
            print(f"[ERROR] Idle time mismatch: SP={sp_idle_time}, Batsim={batsim_idle_time}")
            different_result = True

        # Check total waiting time
        sp_waiting_time = sim_e_sp.total_waiting_time
        batsim_waiting_time = sum(jobs_e_batsim.info['waiting_time'])
        if sp_waiting_time != batsim_waiting_time:
            print(f"[ERROR] Total waiting time mismatch: SP={sp_waiting_time}, Batsim={batsim_waiting_time}")
            different_result = True

        # Print error message with workload ID
        if different_result:
            print(f"[ERROR] DIFFERENT RESULT ON WORKLOAD: {workload_id}")
            with open(f"workloads/workload_found_error_{failed_count}.json", "w") as json_file:
                json.dump(workloads, json_file, indent=4)
            jobs_e_sp_df = pd.DataFrame(jobs_e_sp)
            jobs_e_sp_df['allocated_resources'] = jobs_e_sp_df['allocated_resources'].apply(
                lambda x: f' '.join(map(str, x))
            )
            jobs_e_sp_df.to_csv(f'results/cssp/baseline/easy_jobs_error_{failed_count}.csv', index=False)

            jobs_e_batsim_df = jobs_e_batsim.to_dataframe()
            jobs_e_batsim_df.to_csv(f'results/batsim/baseline/easy_jobs_error_{failed_count}.csv', index=False)
            failed_count += 1
        else:
            sp_execution_times.append(execution_time_sp)
            batsim_execution_times.append(execution_time_batsim)
            workload_id += 1

    print(f"Total failed workloads: {failed_count}")

benchmarking()
avg_sp_et = sum(sp_execution_times)/steps
avg_batsim_et = sum(batsim_execution_times)/steps
print('avg sp executin time: ', avg_sp_et)
print('avg batsim executin time: ', avg_batsim_et)
print('failed_count: ', failed_count)
# Plot SP-Sim results (dot marker)
plt.scatter(range(len(sp_execution_times)), sp_execution_times, marker='x', color='blue', label="SP-Sim")

# Plot Batsim results (plus marker)
plt.scatter(range(len(batsim_execution_times)), batsim_execution_times, marker='+', color='red', label="Batsim")

# Labels and title
plt.xlabel("Workload ID")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison: SP-Sim vs Batsim")

max_y_value = max(max(sp_execution_times, default=0), max(batsim_execution_times, default=0))
plt.ylim(0, max_y_value + 0.05)

plt.legend()

# Save the figure
plt.savefig("execution_time_comparison.png", dpi=300)

# Show the plot
plt.show()
