import pandas as pd
from sp_simulator.env import SPSimulator
from sp_simulator.baseline import simulate_easy

sp_simulator = SPSimulator(model = None)
jobs_e = simulate_easy(sp_simulator)
max_finish_time = max(job.get('finish_time', 0) for job in jobs_e)
jobs_e = pd.DataFrame(jobs_e)
jobs_e['allocated_resources'] = jobs_e['allocated_resources'].apply(
    lambda x: f' '.join(map(str, x))
)

jobs_e.to_csv('results/sp/baseline/easy_jobs.csv', index=False)

sp_simulator.print_energy_consumption()
print('mean_waiting_time: ',sp_simulator.sim_monitor['avg_waiting_time'])
print('mean_waiting_time: ',sp_simulator.sim_monitor['avg_waiting_time']/100)
print('finish_time: ', max_finish_time)

sp_simulator.sim_monitor['nb_res'].to_csv('results/sp/baseline/easy_host.csv', index=False)