import pandas as pd
from sp_simulator.env import SPSimulator
from sp_simulator.t_policy import simulate_easy

timeout = 30
sp_simulator = SPSimulator(model = None)
jobs_e = simulate_easy(sp_simulator, timeout)
max_finish_time = max(job.get('finish_time', 0) for job in jobs_e)
jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_e.to_csv(f'results/sp/timeout/easy_jobs_t{timeout}.csv', index=False)
sp_simulator.sim_monitor['nb_res'].to_csv(f'results/sp/timeout/easy_host_t{timeout}.csv', index=False)

nodes_data = sp_simulator.sim_monitor['nodes']

flattened_data = []
for idx, sublist in enumerate(nodes_data):
    for item in sublist:
        item['allocated_resources'] = idx
        flattened_data.append(item)

df = pd.DataFrame(flattened_data)

df = df[['allocated_resources', 'type', 'starting_time', 'finish_time']]
df = df[df['type'] != 'computing']

def set_job_id(row):
    if row['type'] == 'idle':
        return -1
    elif row['type'] == 'switching_off':
        return -2
    elif row['type'] == 'switching_on':
        return -3
    elif row['type'] == 'sleeping':
        return -4
    return 0 

def calculate_energy(nodes_monitor):
    total_joule = nodes_monitor.apply(
        lambda row: (row['finish_time'] - row['starting_time']) * len(row['allocated_resources'].split()) * (
            95 if row['type'] == 'idle' else
            9 if row['type'] == 'sleeping' else
            190 if row['type'] == 'computing' else
            9 if row['type'] == 'switching_off' else
            190 if row['type'] == 'switching_on' else 0
        ), axis=1
    ).sum()

    print('total joule: ', total_joule)

def calculate_total_time(nodes_monitor):
    types = ['idle', 'sleeping', 'computing', 'switching_on', 'switching_off']
    total_times = {}

    for t in types:
        total_times[t] = nodes_monitor[nodes_monitor['type'] == t].apply(
            lambda row: (row['finish_time'] - row['starting_time']) * len(row['allocated_resources'].split()),
            axis=1
        ).sum()
    
    for t, time in total_times.items():
        print(f'{t} time: {time}')

    return total_times

def calculate_waiting_time(jobs_monitor):
    total_waiting_time = (jobs_monitor['starting_time'] - jobs_monitor['submission_time']).sum()
    print('total waiting time: ', total_waiting_time)
    print('avg waiting time: ', total_waiting_time/len(jobs_monitor))

def calculate_last_finish(jobs_monitor):
    last_finish_time = jobs_monitor['finish_time'].max()
    print('last finish time: ', last_finish_time)

df['job_id'] = df.apply(set_job_id, axis=1)

grouped_df = df.groupby(['type', 'starting_time', 'finish_time']).agg({'allocated_resources': lambda x: ' '.join(map(str, x)), 'job_id': 'first'}).reset_index()

grouped_df = grouped_df.sort_values(by=['starting_time', 'finish_time'])
grouped_df['submission_time'] = grouped_df['starting_time']
jobs_e = jobs_e[['job_id', 'allocated_resources', 'starting_time', 'finish_time', 'submission_time']]
jobs_e['type'] = 'computing'

final_df = pd.concat([grouped_df, jobs_e], ignore_index=True)

final_df = final_df.sort_values(by=['starting_time', 'finish_time'])

calculate_energy(final_df)
calculate_total_time(final_df)
calculate_waiting_time(jobs_e)
calculate_last_finish(jobs_e)

final_df.to_csv(f'results/sp/timeout/easy_nodes_t{timeout}.csv', index=False)
final_df.to_json(f'results/sp/timeout/easy_nodes_t{timeout}.json', orient='records', lines=True)