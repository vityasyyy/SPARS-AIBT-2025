import pandas as pd

jobs_easy_batsim = pd.read_csv(f'results/batsim/baseline/easy_jobs_error_1.csv')

jobs_easy_batsim['job_id'] = jobs_easy_batsim['job_id'].astype(str)
jobs_easy_batsim['profile'] = jobs_easy_batsim['profile'].astype(str)

jobs_easy_batsim['finish_time'] = pd.to_numeric(jobs_easy_batsim['finish_time'])
jobs_easy_batsim['starting_time'] = pd.to_numeric(jobs_easy_batsim['starting_time'])

for i in range(len(jobs_easy_batsim) - 1):
    row = jobs_easy_batsim.iloc[i]
    next_row = jobs_easy_batsim.iloc[i + 1]

    if row['finish_time'] == next_row['finish_time'] and row['starting_time'] > next_row['starting_time']:
        print('released separately', row['finish_time'])
        
    
    if row['finish_time'] == next_row['finish_time'] and row['starting_time'] <= next_row['starting_time']:

        print('released together', row['finish_time'])