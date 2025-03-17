import batsim_py
from scheduler_batsim.backfilling import EASYScheduler
from batsim_utils.shutdown_policy import TimeoutPolicy

from scheduler_sp.env import SPSimulator
from scheduler_sp.easy_scheduler import EasyScheduler
import pandas as pd
import time
import json
import matplotlib.pyplot as plt

from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor
from workloads_generator import ProblemGenerator


def run_simulation_batsim(scheduler, platform_filepath, workload_filepath):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)
    
    # Instantiate monitors
    jobs_mon = JobMonitor(simulator)
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # Start simulation
    simulator.start(platform=platform_filepath,
                    workload=workload_filepath,
                    verbosity="information")

    while simulator.is_running:
        scheduler.schedule()
        simulator.proceed_time()
    simulator.close()

    return jobs_mon, sim_mon, host_mon, e_mon


def run_simulation_batsim_t(scheduler, platform_filepath, workload_filepath, shutdown_policy):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)
    policy = shutdown_policy(simulator)

    # Instantiate monitors
    jobs_mon = JobMonitor(simulator)
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # Start simulation
    simulator.start(platform=platform_filepath,
                    workload=workload_filepath,
                    verbosity="information")

    while simulator.is_running:
        scheduler.schedule()
        simulator.proceed_time()
    simulator.close()

    return jobs_mon, sim_mon, host_mon, e_mon


problem_generator = ProblemGenerator(num_jobs=1000, max_node=8)

steps = 100
workload_id = 1
failed_count = 0

# Store metrics for plotting
waiting_times_t = []
waiting_times_b = []
energy_waste_t = []
energy_waste_b = []

finish_times_t = []
finish_times_b = []

def benchmarking():
    global workload_id, failed_count
    while workload_id <= steps:
        print(f'~~~ WORKLOAD {workload_id} ~~~')

        # Generate workload
        workload_filepath = f"workloads/workload_mg_{workload_id}.json"
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

        # Run simulations for Timeout Policy
        jobs_t, sim_t, host_t, energy_t = run_simulation_batsim_t(
            EASYScheduler, "platforms/batsim/platform.xml", workload_filepath, lambda s: TimeoutPolicy(100, s))
        energy_waste_t.append(sim_t.info['energy_waste'])
        waiting_times_t.append(jobs_t.info['waiting_time'])
        finish_times_t.append(max(jobs_t.info['finish_time']))
        
        # Run simulations for Baseline Policy
        jobs_b, sim_b, host_b, energy_b = run_simulation_batsim(
            EASYScheduler, "platforms/batsim/platform.xml", workload_filepath)
        energy_waste_b.append(sim_b.info['energy_waste'])
        waiting_times_b.append(jobs_b.info['waiting_time'])
        finish_times_b.append(max(jobs_b.info['finish_time']))

        

        workload_id += 1

    print(f"Total failed workloads: {failed_count}")

benchmarking()

# Compute and print average finish times
avg_finish_time_timeout = sum(finish_times_t) / len(finish_times_t) if finish_times_t else 0
avg_finish_time_baseline = sum(finish_times_b) / len(finish_times_b) if finish_times_b else 0

# Compute and print average energy waste
avg_energy_waste_timeout = sum(energy_waste_t) / len(energy_waste_t) if energy_waste_t else 0
avg_energy_waste_baseline = sum(energy_waste_b) / len(energy_waste_b) if energy_waste_b else 0

# Compute and print average waiting time
avg_waiting_time_timeout = sum(waiting_times_t) / len(waiting_times_t) if waiting_times_t else 0
avg_waiting_time_baseline = sum(waiting_times_b) / len(waiting_times_b) if waiting_times_b else 0

# Print results
print("\n=== Simulation Results ===")
print(f"Average Finish Time (Timeout Policy): {avg_finish_time_timeout:.2f} seconds")
print(f"Average Finish Time (Baseline Policy): {avg_finish_time_baseline:.2f} seconds")

print(f"Average Energy Waste (Timeout Policy): {avg_energy_waste_timeout:.2f} Joules")
print(f"Average Energy Waste (Baseline Policy): {avg_energy_waste_baseline:.2f} Joules")

print(f"Average Waiting Time (Timeout Policy): {avg_waiting_time_timeout:.2f} seconds")
print(f"Average Waiting Time (Baseline Policy): {avg_waiting_time_baseline:.2f} seconds")
print("==========================\n")


# Plot Waiting Time
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(waiting_times_t) + 1), waiting_times_t, label="Timeout Policy", marker='o')
plt.plot(range(1, len(waiting_times_b) + 1), waiting_times_b, label="Baseline Policy", marker='s')
plt.xlabel("Workload ID")
plt.ylabel("Waiting Time (s)")
plt.title("Comparison of Waiting Time")
plt.legend()
plt.grid()
plt.savefig("waiting_time_comparison.png")

# Plot Energy Waste
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(energy_waste_t) + 1), energy_waste_t, label="Timeout Policy", marker='o')
plt.plot(range(1, len(energy_waste_b) + 1), energy_waste_b, label="Baseline Policy", marker='s')
plt.xlabel("Workload ID")
plt.ylabel("Energy Waste (Joules)")
plt.title("Comparison of Energy Waste")
plt.legend()
plt.grid()
plt.savefig("energy_waste_comparison.png")

plt.show()
