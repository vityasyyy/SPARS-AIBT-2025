# gym_env.py
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import logging
import torch as T
from typing import Tuple

# import your real Simulator and RJMS
from HPCv3.Gym.utils import Reward
from HPCv3.Simulator.Simulator import Simulator
from HPCv3.RJMS.RJMS import RJMS

from HPCv3.Gym.utils import feature_extraction


logger = logging.getLogger("runner")


def get_feasible_mask(states):
    # fm[:, 0] = dibiarkan, dummy
    # fm[:, 1] = boleh matikan/tidak
    # fm[:, 2] = boleh hidupkan/tidak
    feasible_mask = np.ones((len(states), 3), dtype=np.float32)
    is_switching_off = np.asarray(
        [host['state'] == 'switching_off' for host in states])
    is_switching_on = np.asarray(
        [host['state'] == 'switching_on' for host in states])
    is_switching = np.logical_or(is_switching_off, is_switching_on)
    is_idle = np.asarray(
        [host['state'] == 'active' and host['job_id'] is None for host in states])
    is_sleeping = np.asarray(
        [host['state'] == 'sleeping' for host in states])
    is_allocated = np.asarray(
        [host['state'] == 'active' and host['job_id'] is None for host in states])

    # can it be switched off
    is_really_idle = np.logical_and(is_idle, np.logical_not(is_allocated))
    feasible_mask[:, 1] = np.logical_and(
        np.logical_not(is_switching), is_really_idle)

    # can it be switched on
    feasible_mask[:, 2] = np.logical_and(
        np.logical_not(is_switching), is_sleeping)
    # return cuma 2 action, update 15-09-2022
    return feasible_mask[:, 1:]


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

    def action_translator(self, state, actions):

        binary_actions = [0 if pair[0] > pair[1] else 1 for pair in actions[0]]
        switch_off = []
        switch_on = []
        for _, action in enumerate(binary_actions):
            if action == 0 and state[_]['state'] == 'active' and state[_]['job_id'] is None:
                switch_off.append(_)
            elif action == 1 and state[_]['state'] == 'sleeping':
                switch_on.append(_)

        return switch_off, switch_on

    def advance(self):

        while self.simulator.is_running:
            events = self.simulator.proceed()
            scheduler_message = self.simulator.scheduler.schedule(self.simulator.current_time, self.simulator.PlatformControl.get_state(
            ), self.simulator.jobs_manager.waiting_queue, self.simulator.jobs_manager.scheduled_queue)

            for _data in scheduler_message:
                timestamp = _data['timestamp']
                _events = _data['events']
                for event in _events:
                    self.simulator.push_event(timestamp, event)

            for event_list in events['event_list']:
                for event in event_list['events']:
                    if event['type'] == 'CALL_RL':
                        return

    def step(self, actions):
        state = self.simulator.PlatformControl.get_state()
        logger.info(f"Action taken: {actions}")
        switch_off, switch_on = self.action_translator(state, actions)
        # logger.info(f"Switch off: {switch_off}")
        # logger.info(f"Switch on: {switch_on}")

        self.simulator.push_event(self.simulator.current_time, {
                                  'type': 'switch_off', 'nodes': switch_off})
        self.simulator.push_event(self.simulator.current_time, {
                                  'type': 'switch_on', 'nodes': switch_on})

        need_rl = False

        while not need_rl and self.simulator.is_running:
            events = self.simulator.proceed()
            scheduler_message = self.simulator.scheduler.schedule(self.simulator.current_time, self.simulator.PlatformControl.get_state(
            ), self.simulator.jobs_manager.waiting_queue, self.simulator.jobs_manager.scheduled_queue)

            for _data in scheduler_message:
                timestamp = _data['timestamp']
                _events = _data['events']
                for event in _events:
                    self.simulator.push_event(timestamp, event)

            for event_list in events['event_list']:
                for event in event_list['events']:
                    if event['type'] == 'CALL_RL':
                        need_rl = True
                        break
                if need_rl:
                    break

        reward = Reward.calculate_reward(self.simulator.Monitor)
        done = not self.simulator.is_running
        next_features = feature_extraction(self.simulator)

        # Reshape for Critic (add batch dimension)
        next_features = np.concatenate(next_features)
        next_features = next_features.reshape(1, 16, 11)

        return next_features, reward, done

    def reset(self, workload_path, platform_path,
              start_time, algorithm, agent):
        self.simulator = Simulator(workload_path, platform_path,
                                   start_time, algorithm, rl=True, agent=agent)
