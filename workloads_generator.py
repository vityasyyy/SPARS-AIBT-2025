import numpy as np
import json
import argparse
import math

class ProblemGenerator:
    def __init__(self, lambda_arrival=0.2, mu_execution=50, sigma_execution=2,
                 mu_noise=0, sigma_noise=1, num_jobs=None, max_node=8):
        self.lambda_arrival = lambda_arrival
        self.mu_execution = mu_execution
        self.sigma_execution = sigma_execution
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
        self.max_node = max_node
        self.num_jobs = num_jobs if num_jobs is not None else max(1, int(np.random.normal(10, 2)))

    def generate(self):
        interarrival_times = np.random.exponential(1 / self.lambda_arrival, self.num_jobs)
        arrival_times = np.cumsum(interarrival_times)
        requested_execution_times = np.random.normal(self.mu_execution, self.sigma_execution, self.num_jobs)
        noise = np.random.normal(self.mu_noise, self.sigma_noise, self.num_jobs)

        actual_execution_times = np.maximum(0, requested_execution_times + noise)
        num_nodes_required = np.clip(np.random.normal(math.ceil(self.max_node/2), 1, self.num_jobs), 1, self.max_node)
        workloads = []

        for i in range(self.num_jobs):
            workloads.append({
                "id": i + 1,
                'res': int(num_nodes_required[i]),
                'subtime': round(float(arrival_times[i])),
                'walltime': round(float(requested_execution_times[i])),
                'profile': '100',
                'user_id': 0
            })

        for i in range(len(workloads) - 1):
            id_current = workloads[i]['id']
            id_next = workloads[i + 1]['id']
            subtime_current = workloads[i]['subtime']
            subtime_next = workloads[i + 1]['subtime']

            if str(id_current) > str(id_next) and subtime_current == subtime_next:
                workloads[i + 1]['subtime'] += 1

        return workloads

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic workload JSON.")
    parser.add_argument("--num_jobs", type=int, required=True, help="Number of jobs to generate")
    parser.add_argument("--max_node", type=int, required=True, help="Maximum number of nodes per job")
    

    args = parser.parse_args()

    problem_generator = ProblemGenerator(num_jobs=args.num_jobs, max_node=args.max_node)
    workloads = problem_generator.generate()

    output_data = {
        "nb_res": args.max_node,
        "jobs": workloads,
        "profiles": {
            "100": {
                "cpu": 10000000000000000000000,
                "com": 0,
                "type": "parallel_homogeneous"
            }
        }
    }
    workload_filepath = f'workloads/wl_nj{args.num_jobs}_mn{args.max_node}.json'
    with open(workload_filepath, "w") as json_file:
        json.dump(output_data, json_file, indent=4)