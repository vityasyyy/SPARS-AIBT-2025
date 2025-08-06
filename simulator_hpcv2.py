import argparse
import pandas as pd
import torch
import sys

from HPCv2_Simulator.Simulator import SPSimulator
from HPCv2_Scheduler.fcfs_scheduler import FCFSScheduler
from HPCv2_Scheduler.easy_scheduler import EasyScheduler
from HPCv2_Scheduler.smart_fcfs_scheduler import SmartFCFSScheduler
from HPCv2_Scheduler.smart_easy_scheduler import SmartEasyScheduler
from HPCv2_Utils.data_mapper import process_node_job_data
from HPCv2_Scheduler.easy_rl_scheduler import RLScheduler
from nn_generator.model_generator_2 import HPCNodeManager


def run_simulation(scheduler, platform_filepath, workload_filepath):
    simulator = SPSimulator(scheduler, platform_path=platform_filepath, workload_path=workload_filepath)
    scheduler.simulator = simulator
    simulator.start_simulator()
    while simulator.is_running:
        simulator.proceed()
    return simulator.jobs_manager, simulator.sim_monitor

def main():
    parser = argparse.ArgumentParser(description="Run HPCv2 simulation with specified scheduler.")
    parser.add_argument('--workload', type=str, default='workloads/validate_data.json', help='Path to workload file (default: workloads/validate_data.json)')
    parser.add_argument('--platform', type=str, default='platforms/spsim/platform_validate.json', help='Path to platform file (default: platforms/spsim/platform_validate.json)')
    parser.add_argument('--timeout', type=int, help='Simulation timeout in seconds')
    parser.add_argument('--out', type=str, required=True, help='Output directory (required)')
    parser.add_argument('--scheduler', type=str, required=True, choices=['easy', 'fcfs', 'easy-rl', 'smart-fcfs', 'smart-easy'], help='Scheduler type: easy or fcfs (required)')
    parser.add_argument('--node_manager', type=str, help='Path to node manager file (required if scheduler=easy-rl)')

    args = parser.parse_args()
    if args.scheduler == 'easy-rl' and args.node_manager is None:
        print("Error: --node_manager is required when --scheduler is 'easy-rl'")
        sys.exit(1)
    
    if args.node_manager:
        node_manager = torch.load(args.node_manager, map_location=torch.device('cpu'))
    
    # Pilih scheduler
    if args.scheduler == 'easy':
        scheduler = EasyScheduler(None, timeout=args.timeout)
    elif args.scheduler == 'easy-rl':
        scheduler = RLScheduler(None, node_manager=node_manager)
    elif args.scheduler == 'fcfs':
        scheduler = FCFSScheduler(None , timeout=args.timeout)
    elif args.scheduler == 'smart-fcfs':
        scheduler = SmartFCFSScheduler(None , timeout=args.timeout)
    elif args.scheduler == 'smart-easy':
        scheduler = SmartEasyScheduler(None , timeout=args.timeout)
    else:
        raise ValueError("Unsupported scheduler")

    # Jalankan simulasi
    jobs, sim = run_simulation(scheduler, args.platform, args.workload)

    print(f"~~~ {args.scheduler.upper()} SCHEDULER ~~~")
    print('total idle time: ', sum(sim.idle_time))
    print('mean idle time: ', sim.idle_time)
    print('finish time: ', sim.finish_time)
    print('total waiting time: ', sim.total_waiting_time)
    print('mean waiting time: ', sim.total_waiting_time / len(jobs.monitor_jobs))
    print('energy consumption: ', sim.energy_consumption)
    print('energy waste: ', sum(sim.energy_waste))

    # Simpan hasil
    jobs_df = pd.DataFrame(jobs.monitor_jobs)
    sn_df = pd.DataFrame(sim.nodes)
    jobs_df.to_csv('idkjobs.csv')
    sn_df.to_csv('idksn.csv')
    jobs_df['allocated_resources'] = jobs_df['allocated_resources'].apply(lambda x: ' '.join(map(str, x)))

    
    nodes_df = process_node_job_data(sim.nodes, jobs_df)
    
    
    output_dir_jobs = f"results/hpcv2/{args.out}_{args.scheduler}_jobs_{'t'+str(args.timeout) if args.timeout else 'baseline'}.csv"
    output_dir_hosts = f"results/hpcv2/{args.out}_{args.scheduler}_hosts_{'t'+str(args.timeout) if args.timeout else 'baseline'}.csv"
    output_dir_nodes = f"results/hpcv2/{args.out}_{args.scheduler}_nodes_{'t'+str(args.timeout) if args.timeout else 'baseline'}.csv"
    
    jobs_df.to_csv(output_dir_jobs, index=False)
    sim.node_state_log.to_csv(output_dir_hosts, index=False)
    nodes_df.to_csv(output_dir_nodes, index=False)


if __name__ == '__main__':
    main()
