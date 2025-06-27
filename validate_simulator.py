from HPCv2_Simulator.Simulator import SPSimulator
from HPCv2_Scheduler.easy_scheduler import EasyScheduler
import pandas as pd
from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor, HostPowerStateSwitchMonitor
from batsim_utils.shutdown_policy import TimeoutPolicy
from scheduler_batsim.backfilling import EASYScheduler
import numpy as np
import json

class ProblemGenerator:
    def __init__(self, lambda_arrival=0.08, mu_execution=50, sigma_execution=2, mu_noise=0, sigma_noise=1, num_jobs=None, max_node=4):
        self.lambda_arrival = lambda_arrival
        self.mu_execution = mu_execution
        self.sigma_execution = sigma_execution
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
        self.max_node = max_node
        self.num_jobs = num_jobs if num_jobs is not None else max(1, int(np.random.normal(10, 2)))

    def generate(self):
        interarrival_times = np.random.exponential(1 / self.lambda_arrival, self.num_jobs)
        arrival_times = np.cumsum(interarrival_times)
        requested_execution_times = np.random.normal(self.mu_execution, self.sigma_execution, self.num_jobs)
        
        # noise = np.random.normal(self.mu_noise, self.sigma_noise, self.num_jobs)
        # actual_execution_times = np.maximum(0, requested_execution_times + noise)
        num_nodes_required = np.clip(np.random.normal(4, 1, self.num_jobs), 1, self.max_node)
        workloads = []
            
        for i in range(self.num_jobs):
            if arrival_times[i] == 30:
               arrival_times[i] += 1 
            workloads.append({
                "id": i + 1,
                'res': int(num_nodes_required[i]),
                'subtime': round(float(arrival_times[i])),
                'walltime': round(float(requested_execution_times[i])),
                'profile': '100',
                'user_id': 0
            })

        for i in range(len(workloads) - 1):
            id_current = workloads[i]['id']
            id_next = workloads[i + 1]['id']
            subtime_current = workloads[i]['subtime']
            subtime_next = workloads[i + 1]['subtime']

            if str(id_current) > str(id_next) and subtime_current == subtime_next:
                workloads[i + 1]['subtime'] += 1 

        return workloads

num_jobs = 10
max_node = 3
problem_generator = ProblemGenerator(num_jobs=num_jobs, max_node=max_node)

workload_filepath = "workloads/validate_data.json"

while True:
    workloads = problem_generator.generate()

    workloads = {
        "nb_res": max_node,
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


    ### BATSIM

    def run_simulation(scheduler, shutdown_policy):
        simulator = SimulatorHandler()
        scheduler = scheduler(simulator)
        policy = shutdown_policy(simulator)

        # 1) Instantiate monitors to collect simulation statistics
        jobs_mon = JobMonitor(simulator)
        sim_mon = SimulationMonitor(simulator)
        host_mon = HostStateSwitchMonitor(simulator)
        host_power_mon = HostPowerStateSwitchMonitor(simulator)
        e_mon = ConsumedEnergyMonitor(simulator)

        # 2) Start simulation
        simulator.start(platform="platforms/batsim/platform_validate.xml",
                        workload="workloads/validate_data.json",
                        verbosity="information")

        # 3) Schedule all jobs
        while simulator.is_running:
            # First Fit policy
            scheduler.schedule()

            # proceed directly to the next event because the shutdown_policy is event-based.
            simulator.proceed_time()

        simulator.close()

        # 4) Return/Dump statistics
        return jobs_mon, sim_mon, host_mon, e_mon, host_power_mon

    timeout = 30
    jobs_wt, sim_wt, host_wt, e_wt, hpm = run_simulation(EASYScheduler, lambda s: TimeoutPolicy(timeout, s))


    print('jul: ',sim_wt.info['consumed_joules'])
    print('idle_time: ',sim_wt.info['time_idle'])
    print('mean_idle_time: ',sim_wt.info['time_idle']/16)
    print('waiting_time: ',sum(jobs_wt.info['waiting_time']))
    print('mean_waiting_time: ',sim_wt.info['mean_waiting_time'])
    print('energy waste: ',sim_wt.info['energy_waste'])
    print('finish_time', max(jobs_wt.info['finish_time']))
    
    jobswtft = max(jobs_wt.info['finish_time'])

    jobs_wt = jobs_wt.to_dataframe()
    host_wt = host_wt.to_dataframe()


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
    workload_filepath = "workloads/validate_data.json"
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


    
    if jobswtft != sim_e.finish_time:
        jobs_wt.to_csv(f'result_validate/batsimpy_jobs_t{timeout}.csv', index=False)
        host_wt.to_csv(f'result_validate/batsimpy_hosts_t{timeout}.csv', index=False)
        hpm.to_csv(f'result_validate/batsimpy_hpss_t{timeout}.csv', )
        
        jobs_e.to_csv(f'result_validate/hpcv2_jobs_t{timeout}.csv', index=False)
        sim_e.node_state_log.to_csv(f'result_validate/hpcv2_hosts_t{timeout}.csv', index=False)
        nodes_e = process_node_job_data(sim_e.nodes, jobs_e)
        nodes_e.to_csv(f'result_validate/hpcv2_nodes_t{timeout}.csv', index=False)
        break