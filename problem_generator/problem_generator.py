import numpy as np


class ProblemGenerator:
    def __init__(self, lambda_arrival=0.5, mu_execution=150, sigma_execution=2, mu_noise=0, sigma_noise=1, num_jobs=None):
        self.lambda_arrival = lambda_arrival
        self.mu_execution = mu_execution
        self.sigma_execution = sigma_execution
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
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
        num_nodes_required = np.random.normal(10, 5, self.num_jobs)

        upcoming_jobs = []
        for i in range(self.num_jobs):
            upcoming_jobs.append({
                'arrival': float(arrival_times[i]),
                'requested_execution_time': float(requested_execution_times[i]),
                'nodes': int(num_nodes_required[i]),
                'actual_execution_time': float(min(actual_execution_times[i], expected_execution_times[i])),
                'scheduled': False
            })

        return upcoming_jobs
