import argparse
import os
from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor, HostPowerStateSwitchMonitor
from batsim_utils.shutdown_policy import TimeoutPolicy
from scheduler_batsim.backfilling import EASYScheduler  
from scheduler_batsim.fcfs import FCFSScheduler  

def get_scheduler_class(name):
    schedulers = {
        "easy": EASYScheduler,
        "fcfs": FCFSScheduler
    }
    return schedulers.get(name.lower())

def run_simulation(scheduler_class, shutdown_policy_class, platform_path, workload_path):
    simulator = SimulatorHandler()
    scheduler = scheduler_class(simulator)
    if shutdown_policy_class is not None:
        policy = shutdown_policy_class(simulator)

    jobs_mon = JobMonitor(simulator)
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)
    hpss_mon = HostPowerStateSwitchMonitor(simulator)

    ct = simulator._SimulatorHandler__current_time

    
    simulator.start(platform=platform_path,
                    workload=workload_path,
                    verbosity="information")

    while simulator.is_running:
        if simulator._SimulatorHandler__current_time == ct:
            input('Press Enter to continue...')
        ct = simulator._SimulatorHandler__current_time
        scheduler.schedule()
        simulator.proceed_time()

    simulator.close()
    return jobs_mon, sim_mon, host_mon, e_mon, hpss_mon

def parse_args():
    parser = argparse.ArgumentParser(description="Run Batsim simulation with given scheduler and config.")
    parser.add_argument('--scheduler', type=str, required=True, help="Scheduler name (e.g., easy)")
    parser.add_argument('--timeout', type=int, default=None, help="Shutdown timeout in seconds (optional)")
    parser.add_argument('--platform', type=str, required=True, help="Path to platform XML")
    parser.add_argument('--workload', type=str, required=True, help="Path to workload JSON")
    parser.add_argument('--output', type=str, required=True, help="Output identifier for directory structure")
    return parser.parse_args()

def main():
    args = parse_args()

    scheduler_class = get_scheduler_class(args.scheduler)
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    if args.timeout is not None:
        shutdown_policy = lambda sim: TimeoutPolicy(args.timeout, sim)
    else:
        shutdown_policy = lambda sim: None

    jobs_mon, sim_mon, host_mon, e_mon, hpss_mon = run_simulation(
        scheduler_class,
        shutdown_policy,
        args.platform,
        args.workload
    )

    # Output directory formatting
    
    output_dir_jobs = f"results/batsimpy/{args.output}_{args.scheduler}_jobs_{'t'+str(args.timeout) if args.timeout else 'baseline'}.csv"
    output_dir_hosts = f"results/batsimpy/{args.output}_{args.scheduler}_hosts_{'t'+str(args.timeout) if args.timeout else 'baseline'}.csv"
   

    # Save CSVs
    jobs_df = jobs_mon.to_dataframe()
    host_df = hpss_mon.to_dataframe()
    jobs_df.to_csv(os.path.join(output_dir_jobs), index=False)
    host_df.to_csv(os.path.join(output_dir_hosts), index=False)

    # Print summary
    print('Total energy (joules):', sim_mon.info['consumed_joules'])
    print('Total idle time:', sim_mon.info['time_idle'])
    print('Mean idle time:', sim_mon.info['time_idle'] / 16)
    print('Total waiting time:', sum(jobs_mon.info['waiting_time']))
    print('Mean waiting time:', sim_mon.info['mean_waiting_time'])
    print('Energy waste:', sim_mon.info['energy_waste'])
    print('Finish time:', max(jobs_mon.info['finish_time']))

if __name__ == "__main__":
    main()
