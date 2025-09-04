# gym_env.py
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import logging

# import your real Simulator and RJMS
from HPCv3.Simulator.Simulator import Simulator
from HPCv3.RJMS.RJMS import RJMS


logger = logging.getLogger("runner")


class HPCGymEnv(gym.Env):
    """
    Gymnasium environment that ONLY wraps Simulator + RJMS.
    Responsibilities:
      - advance_system(): run sim -> rjms -> apply rjms events -> return features (pre-action)
      - apply_action(action): translate agent action -> apply to sim -> compute reward (prev vs next)
      - step(action): helper for Gym compatibility -> advance_system + apply_action
    No agent/critic/memory inside.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, simulator):
        super().__init__()

        self.simulator = simulator

    def step(self, rl_agent):
        need_rl = False

        while not need_rl and self.simulator.is_running:
            events = self.simulator.proceed()
            for event_list in events['event_list']:
                for event in event_list['events']:
                    if event['type'] == 'CALL_RL':
                        need_rl = True
                        break
                if need_rl:
                    break

        state = self.simulator.PlatformControl.get_state()
        features = self.feature_extraction(state)

    def reset(self, workload_path, platform_path,
              start_time, algorithm, agent):
        self.simulator = Simulator(workload_path, platform_path,
                                   start_time, algorithm, rl=True, agent=agent)

    def feature_extraction(self, state):
        # === GLOBAL FEATURES ===
        # print("1. jumlah jobs di queue:", len(self.simulator.queue))
        # print("2. arrival rate:")
        # submission_time = self.job_monitor.info["submission_time"]
        # print("---overall arrival rate (not&normalized):",get_arrival_rate(submission_time, False), get_arrival_rate(submission_time, True))
        # print("---last 100 jobs arrival rate (not&normalized):",get_arrival_rate(submission_time[-100:], False), get_arrival_rate(submission_time[-100:], True) )
        # hosts = self.simulator.platform.hosts
        # print("3. mean runtime jobs di queue:", get_mean_runtime_nodes_in_queue(self.simulator, normalized=False), get_mean_runtime_nodes_in_queue(self.simulator, normalized=False))
        # exit()
        # queue = self.simulator.queue
        # print("3. current mean waiting time:", get_mean_waittime_queue(queue, current_time, False), get_mean_waittime_queue(queue, current_time, True))
        # print("4. Wasted energy (Joule):", get_wasted_energy(self.energy_monitor, self.host_monitor, False), get_wasted_energy(self.energy_monitor, self.host_monitor, True))
        # print("5. mean requested walltime jobs in queue:", get_mean_walltime_in_queue(queue, False), get_mean_walltime_in_queue(queue,True))
        job_num = len(self.simulator.jobs_manager.waiting_queue)
        arrival_rate = self.simulator.Monitor.jobs_submission_log / \
            (self.simulator.current_time - self.simulator.start_time)
        mean_runtime_jobs_in_queue = sum(
            self.simulator.jobs_manager.waiting_queue['walltime']) / len(self.simulator.jobs_manager.waiting_queue)
        total_energy_waste = sum(self.simulator.Monitor.energy['energy_waste'])
        mean_requested_walltime_jobs_in_queue = mean_runtime_jobs_in_queue

        # === NODE FEATURES ===
        # print("NODE FEATURES")
        # print("1. ON/OFF")
        host_on_off = -
        # print("2. ACTIVE/IDLE")
        host_active_idle = get_host_active_idle(self.simulator.platform)
        node_features[:, 1] = host_active_idle
        # print("3. Running Idle TIME")
        current_idle_time = get_current_idle_time(self.host_monitor)
        node_features[:, 2] = current_idle_time
        # print("4. remaining time (percent) of job in nodes")
        remaining_runtime_percent = get_remaining_runtime_percent(
            list(self.simulator.platform.hosts), self.job_infos, self.simulator.current_time)
        node_features[:, 3] = remaining_runtime_percent
        # print("5. wasted energy / consumed joules")
        wasted_energy, normalized_wasted_energy = get_host_wasted_energy(
            self.host_monitor, False), get_host_wasted_energy(self.host_monitor, True)
        node_features[:, 4] = normalized_wasted_energy
        # print("6. time switching state/ time computing? or total time perhaps?")
        switching_time, normalized_switching_time = get_switching_time(
            self.host_monitor, False), get_switching_time(self.host_monitor, True, self.simulator.current_time)
        node_features[:, 5] = normalized_switching_time
        return simulator_features, node_features
        num_nodes = len(state)
        num_active = sum(
            1 for node in state if node['state'] == 'active' and node['job_id'] is not None)
        num_idle = sum(
            1 for node in state if node['state'] == 'idle' and node['job_id'] is not None)
        num_sleeping = sum(1 for node in state if node['state'] == 'sleeping')
        num_switching_on = sum(
            1 for node in state if node['state'] == 'switching_on')
        num_switching_off = sum(
            1 for node in state if node['state'] == 'switching_off')

        features = np.array([num_nodes, num_active, num_idle,
                             num_sleeping, num_switching_on, num_switching_off], dtype=np.float32)

        return features
