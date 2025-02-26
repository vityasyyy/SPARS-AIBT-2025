import numpy as np
import json
import math

class ProblemGenerator:
    def __init__(self, lambda_arrival=0.5, mu_execution=150, sigma_execution=2, mu_noise=0, sigma_noise=1, num_jobs=None, max_node=8):
        self.lambda_arrival = lambda_arrival
        self.mu_execution = mu_execution
        self.sigma_execution = sigma_execution
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
        self.max_node = max_node
        self.num_jobs = num_jobs if num_jobs is not None else max(
            1, int(np.random.normal(10, 2)))

    def generate(self):
        interarrival_times = np.random.exponential(
            1 / self.lambda_arrival, self.num_jobs)
        arrival_times = np.cumsum(interarrival_times)
        requested_execution_times = np.random.normal(
            self.mu_execution, self.sigma_execution, self.num_jobs)
        noise = np.random.normal(
            self.mu_noise, self.sigma_noise, self.num_jobs)

        actual_execution_times = np.maximum(
            0, requested_execution_times + noise)
        expected_execution_times = requested_execution_times
        num_nodes_required = np.random.normal(5, 1, self.num_jobs)
        num_nodes_required = np.clip(num_nodes_required, 1, self.max_node)
        workloads = []
        for i in range(self.num_jobs):
            workloads.append({
                "id": i+1,
                'res': int(num_nodes_required[i]),
                'subtime': round(float(arrival_times[i])),
                'walltime': round(float(requested_execution_times[i])),
                'profile': '100',
                'user_id': 0
                # 'actual_execution_time': float(min(actual_execution_times[i], expected_execution_times[i])),
                # 'scheduled': False
            })

        return workloads

num_jobs=10
max_node=8
problem_generator = ProblemGenerator(num_jobs=100, max_node=8)

workload_filepath = "workloads/simple_data_100.json"
workloads = problem_generator.generate()

workloads = {
    "nb_res": max_node,
    "jobs": workloads,
    "profiles": {
        "100": {
            "cpu": 10000000000000000000000,
            "com": 0,
            "type": "parallel_homogeneous"
        }
    }
}

with open(workload_filepath, "w") as json_file:
    json.dump(workloads, json_file, indent=4)