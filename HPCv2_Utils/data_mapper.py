import pandas as pd

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