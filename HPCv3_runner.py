import argparse
from HPCv3.Simulator.Simulator import Simulator
from HPCv3.RJMS.RJMS import RJMS
import pandas as pd
import ast
from datetime import datetime

def process_node_job_data(nodes_data, jobs):
    # Define mapping for non-active states
    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4
    }
    
    # Process node logs: propagate dvfs_mode and collect intervals
    node_intervals = []
    for node in nodes_data:
        node_id = node['id']
        state_history = node['state_history']
        current_dvfs = None
        for interval in state_history:
            # Propagate dvfs_mode: use last known if not present
            if 'dvfs_mode' in interval:
                current_dvfs = interval['dvfs_mode']
            interval['dvfs_mode'] = current_dvfs
            # Skip zero-length intervals
            if interval['start_time'] < interval['finish_time']:
                node_intervals.append({
                    'node_id': node_id,
                    'state': interval['state'],
                    'dvfs_mode': interval['dvfs_mode'],
                    'start_time': interval['start_time'],
                    'finish_time': interval['finish_time']
                })
    
    node_intervals_df = pd.DataFrame(node_intervals)
    
    # Explode job logs to one row per node
    jobs_exploded = jobs.copy()
    jobs_exploded['nodes'] = jobs_exploded['nodes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    ) # Convert string to list
    jobs_exploded = jobs_exploded.explode('nodes')
    jobs_exploded = jobs_exploded.rename(columns={'nodes': 'node_id'})
    jobs_exploded = jobs_exploded[['job_id', 'node_id', 'start_time', 'finish_time']]
    
    # Process active intervals: merge with job logs to assign job_id
    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy()
    active_merged = pd.merge(
        active_df, 
        jobs_exploded, 
        on=['node_id', 'start_time', 'finish_time'], 
        how='left'
    )
    active_merged['job_id'] = active_merged['job_id'].fillna(-1)  # Idle if no matching job
    
    # Process non-active intervals: map states to job_id
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy()
    non_active_df['job_id'] = non_active_df['state'].map(mapping_non_active).fillna(-1)
    
    # Combine active and non-active intervals
    combined = pd.concat([active_merged, non_active_df])
    
    # Group by state, dvfs_mode, time interval, and job_id; aggregate nodes
    combined['node_id'] = combined['node_id'].astype(int)  # Ensure node_id is int for sorting
    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'start_time', 'finish_time', 'job_id']
    ).agg(
        nodes=('node_id', lambda x: ' '.join(map(str, sorted(x))))
    )  # <- this was missing
    grouped = grouped.reset_index()

    
    # Sort by start_time and select columns
    grouped = grouped.sort_values(by=['start_time', 'finish_time'])
    result = grouped[['dvfs_mode', 'state', 'start_time', 'finish_time', 'nodes', 'job_id']]
    
    return result


def run_simulation(simulator, rjms, human_readable):
    simulator.start_simulator()
    while simulator.is_running:
        simulator_message = simulator.proceed()
        if human_readable:
            print(
                "[simulator send message to RJMS]",
                {
                    'now': datetime.fromtimestamp(simulator_message['now']).strftime("%Y-%m-%d %H:%M:%S"),
                    'event_list': [
                        {
                            **event,
                            'timestamp': datetime.fromtimestamp(event['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                        }
                        for event in simulator_message['event_list']
                    ]
                }
            )
        else:
            print("[simulator send message to RJMS]", simulator_message)

        
        scheduler_message = rjms.schedule(simulator_message)
        if human_readable:
            print(
                "[RJMS send message to simulator]",
                {
                    **scheduler_message,
                    'timestamp': datetime.fromtimestamp(scheduler_message['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                    if 'timestamp' in scheduler_message else None,
                    'event_list': [
                        {
                            **event,
                            'timestamp': datetime.fromtimestamp(event['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                        }
                        for event in scheduler_message['event_list']
                    ]
                }
            )
        else:
            print("[RJMS send message to simulator]", scheduler_message)
        for _data in scheduler_message['event_list']:
            timestamp = _data['timestamp']
            events = _data['events']
            for event in events:
                simulator.push_event(timestamp, event)
                
    # Assuming simulator.Monitor.nodes_state_log is already populated
    raw_node_log = pd.DataFrame(simulator.Monitor.states_hist)
    raw_node_log.to_csv('results/hpcv3/raw_node_log.csv', index=False)
    raw_job_log = pd.DataFrame(simulator.Monitor.jobs_execution_log)
    raw_job_log.to_csv('results/hpcv3/raw_job_log.csv', index=False)
    node_log = process_node_job_data(simulator.Monitor.states_hist, raw_job_log)
    node_log.to_csv('results/hpcv3/node_log.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Run HPCv3 simulation.')
    parser.add_argument('--workload_path', type=str, required=True, help='Path to workload JSON file.')
    parser.add_argument('--platform_path', type=str, required=True, help='Path to platform JSON file.')
    parser.add_argument('--algorithm', type=str, required=True, help='Scheduling algorithm name.')
    parser.add_argument('--timeout', type=int, required=False, help='Simulation timeout in seconds (optional).')
    parser.add_argument("--start_time", type=str, required=False, help="Time in format YYYY-MM-DD HH:MM:SS (optional). If not provided, starts at 0.")
    
    args = parser.parse_args()

    if args.start_time:
        start_time = int(datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S").timestamp())
        human_readable = True
    else:
        start_time = 0
        human_readable = False

    simulator = Simulator(args.workload_path, args.platform_path, start_time)
    rjms = RJMS(args.platform_path, args.algorithm, start_time, args.timeout)
    run_simulation(simulator, rjms, human_readable)

if __name__ == '__main__':
    main()
