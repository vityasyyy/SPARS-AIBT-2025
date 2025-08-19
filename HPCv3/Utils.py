import pandas as pd
import ast
import os

def process_node_job_data(nodes_data, jobs):
    """ MAP DATA"""

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4
    }
    
    node_intervals = []
    for node in nodes_data:
        node_id = node['id']
        state_history = node['state_history']
        current_dvfs = None
        for interval in state_history:
            if 'dvfs_mode' in interval:
                current_dvfs = interval['dvfs_mode']
            interval['dvfs_mode'] = current_dvfs

            if interval['start_time'] < interval['finish_time']:
                node_intervals.append({
                    'node_id': node_id,
                    'state': interval['state'],
                    'dvfs_mode': interval['dvfs_mode'],
                    'start_time': interval['start_time'],
                    'finish_time': interval['finish_time']
                })
    
    node_intervals_df = pd.DataFrame(node_intervals)
    
    jobs_exploded = jobs.copy()
    print(jobs_exploded)
    jobs_exploded['nodes'] = jobs_exploded['nodes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    jobs_exploded = jobs_exploded.explode('nodes')
    jobs_exploded = jobs_exploded.rename(columns={'nodes': 'node_id'})
    
    if 'submission_time' not in jobs_exploded.columns:
        jobs_exploded['submission_time'] = pd.NA    
        
    jobs_exploded = jobs_exploded[['job_id', 'node_id', 'submission_time', 'start_time', 'finish_time']]
    
    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy()
    active_merged = pd.merge(
        active_df, 
        jobs_exploded, 
        on=['node_id', 'start_time', 'finish_time'], 
        how='left'
    )
    active_merged['job_id'] = active_merged['job_id'].fillna(-1)
    
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy()
    non_active_df['job_id'] = non_active_df['state'].map(mapping_non_active).fillna(-1)
    non_active_df['submission_time'] = pd.NA
    
    combined = pd.concat([active_merged, non_active_df])
    
    combined['node_id'] = combined['node_id'].astype(int)  
    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'submission_time', 'start_time', 'finish_time', 'job_id']
    ).agg(
        nodes=('node_id', lambda x: ' '.join(map(str, sorted(x))))
    ) 
    grouped = grouped.reset_index()

    
    grouped = grouped.sort_values(by=['start_time', 'finish_time'])
    result = grouped[['dvfs_mode', 'state', 'submission_time', 'start_time', 'finish_time', 'nodes', 'job_id']]
    
    return result


def log_output(simulator, output_folder):
    os.makedirs(f'{output_folder}', exist_ok=True)
    # Assuming simulator.Monitor.nodes_state_log is already populated
    raw_node_log = pd.DataFrame(simulator.Monitor.states_hist)
    raw_node_log.to_csv(f'{output_folder}/raw_node_log.csv', index=False)
    raw_job_log = pd.DataFrame(simulator.Monitor.jobs_execution_log)
    raw_job_log.to_csv(f'{output_folder}/raw_job_log.csv', index=False)
    node_log = process_node_job_data(simulator.Monitor.states_hist, raw_job_log)
    node_log.to_csv(f'{output_folder}/node_log.csv', index=False)