import pandas as pd
from sp_simulator.env import SPSimulator
from sp_simulator.rl2 import simulate_easy, step, reward1, reward2, max_jwt, fcfsbf
from types import MethodType
import torch
from trash_ai_generator import HPCNodeManager

model = torch.load("untrained/hpc_node_manager.pth", map_location=torch.device('cpu'))
model.train()

sp_simulator = SPSimulator(model)

sp_simulator.simulate_easy = MethodType(simulate_easy, sp_simulator)
sp_simulator.step = MethodType(step, sp_simulator)
sp_simulator.reward1 = MethodType(reward1, sp_simulator)
sp_simulator.reward2 = MethodType(reward2, sp_simulator)
sp_simulator.max_jwt = MethodType(max_jwt, sp_simulator)
sp_simulator.fcfsbf = MethodType(fcfsbf, sp_simulator)


jobs_e = sp_simulator.simulate_easy()

max_finish_time = max(job.get('finish_time', 0) for job in jobs_e)
jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_e.to_csv(f'results/sp/rl/easy_jobs.csv', index=False)
sp_simulator.sim_monitor['nb_res'].to_csv(f'results/sp/rl/easy_host.csv', index=False)

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

df['job_id'] = df.apply(set_job_id, axis=1)

grouped_df = df.groupby(['type', 'starting_time', 'finish_time']).agg({'allocated_resources': lambda x: ' '.join(map(str, x)), 'job_id': 'first'}).reset_index()

grouped_df = grouped_df.sort_values(by=['starting_time', 'finish_time'])
grouped_df['submission_time'] = grouped_df['starting_time']
jobs_e = jobs_e[['job_id', 'allocated_resources', 'starting_time', 'finish_time', 'submission_time']]
jobs_e['type'] = 'computing'

final_df = pd.concat([grouped_df, jobs_e], ignore_index=True)

final_df = final_df.sort_values(by=['starting_time', 'finish_time'])


final_df.to_csv(f'results/sp/rl/easy_nodes.csv', index=False)
final_df.to_json(f'results/sp/rl/easy_nodes.json', orient='records', lines=True)