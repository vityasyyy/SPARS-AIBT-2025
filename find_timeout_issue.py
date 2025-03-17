import time
import json
from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor
from batsim_utils.shutdown_policy import TimeoutPolicy
from scheduler_batsim.backfilling import EASYScheduler
from workloads_generator import ProblemGenerator

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
                    workload="workloads/workload_find_timeout_issue.json",
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

# Timeout untuk kebijakan shutdown
TIMEOUT = 100

# Inisialisasi generator workload
problem_generator = ProblemGenerator(num_jobs=1000, max_node=8)

def find_issue():
   while True:
        """ Mencari potensi masalah dengan workload yang dibuat secara dinamis. """
        
        workload_id = int(time.time())  # Gunakan timestamp sebagai ID unik untuk workload
        workload_filepath = f"workloads/workload_find_timeout_issue.json"
        
        # Membuat workload
        workloads = problem_generator.generate()
        workloads_data = {
            "nb_res": 16,
            "jobs": workloads,
            "profiles": {
                "100": {
                    "cpu": 1e22,  # Menggunakan notasi ilmiah untuk angka besar
                    "com": 0,
                    "type": "parallel_homogeneous"
                }
            }
        }

        # Menyimpan workload ke dalam file JSON
        with open(workload_filepath, "w") as json_file:
            json.dump(workloads_data, json_file, indent=4)

        # Menjalankan simulasi dengan workload yang baru dibuat
        jobs_mon, sim_mon, host_mon, e_mon = run_simulation(EASYScheduler, lambda s: TimeoutPolicy(TIMEOUT, s))

        
find_issue()