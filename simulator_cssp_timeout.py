from scheduler_sp.env import SPSimulator
from scheduler_sp.easy_scheduler import EasyScheduler
import pandas as pd

timeout = 30

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



def run_simulation(scheduler):
    simulator = SPSimulator(scheduler, timeout = timeout)
    scheduler.simulator = simulator
    
    while simulator.schedule_queue:
        simulator.proceed()
    
    return simulator.monitor_jobs, simulator.sim_monitor

easy_scheduler = EasyScheduler(None)
jobs_e, sim_e = run_simulation(easy_scheduler)

print('~~~ EASY SCHEDULER ~~~')
print('total idle time: ',sum(sim_e['idle_time']))
print('mean idle time: ',sim_e['idle_time'])
print('finish time: ',sim_e['finish_time'])
print('total waiting time: ',sim_e['total_waiting_time'])
print('mean waiting time: ',sim_e['total_waiting_time']/len(jobs_e))


jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_e.to_csv(f'results/cssp/timeout/easy_jobs_t{timeout}.csv', index=False)

sim_e['nb_res'].to_csv(f'results/cssp/timeout/easy_host_t{timeout}.csv', index=False)

nodes_e = process_node_job_data(sim_e['nodes'], jobs_e)
nodes_e.to_csv(f'results/cssp/timeout/easy_nodes_t{timeout}.csv', index=False)