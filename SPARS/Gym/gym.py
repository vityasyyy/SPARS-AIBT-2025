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
from SPARS.Gym.utils import Reward, get_feasible_mask
from SPARS.Simulator.Simulator import Simulator
from SPARS.RJMS.RJMS import RJMS

from SPARS.Gym.utils import feature_extraction

CPU_DEVICE = T.device("cpu")

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

    def __init__(self, simulator, device=CPU_DEVICE):
        super().__init__()

        self.simulator = simulator
        self.device = device

    def action_translator(self, state, logits):
        """
        Decide which nodes to switch on/off to reach the desired active count.

        Args:
            state: iterable of dicts with keys at least {'id', 'state', 'job_id'}
                - state['state'] in {'active', 'sleeping', ...}
                - state['job_id'] is None if the active node is idle
            num_active_nodes: desired number of active nodes (int)

        Returns:
            (switch_off_ids, switch_on_ids) as two lists of node ids.
            If not enough eligible nodes are found, returns as many as possible.
        """
        # Current active count
        #
        logits = T.sigmoid(
            logits)
        num_active_nodes = logits * self.simulator.Monitor.num_nodes
        current_active = sum(1 for n in state if n.get('state') == 'active')

        # Positive => need to turn on `delta` sleeping nodes.
        # Negative => need to turn off `-delta` idle active nodes.
        delta = current_active - num_active_nodes

        switch_on, switch_off = [], []
        delta = int(delta)
        if delta > 0:
            # Need more active nodes: wake sleeping nodes
            for n in state:
                if n.get('state') == 'active' and n.get('job_id') is None:
                    switch_off.append(n['id'])
                    if len(switch_off) == delta:
                        break

        elif delta < 0:
            # Too many active nodes: put idle active nodes to sleep
            need = -delta
            for n in state:
                if n.get('state') == 'sleeping':
                    switch_on.append(n['id'])
                    if len(switch_on) == need:
                        break

        return switch_off, switch_on

    def advance(self):
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

        observation = self.get_observation()

        return observation

    def step(self, actions):
        state = self.simulator.PlatformControl.get_state()
        logger.info(f"Action taken: {actions}")
        switch_off, switch_on = self.action_translator(state, actions)
        # logger.info(f"Switch off: {switch_off}")
        # logger.info(f"Switch on: {switch_on}")
        if len(switch_off) > 0:
            self.simulator.push_event(self.simulator.current_time, {
                'type': 'switch_off', 'nodes': switch_off})
        if len(switch_on) > 0:
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
        reward_function = Reward()
        reward = reward_function.calculate_reward(
            self.simulator.Monitor, self.simulator.jobs_manager.waiting_queue, self.simulator.current_time)
        done = not self.simulator.is_running
        observation = self.get_observation()

        return observation, reward, done

    def reset(self, workload_path, platform_path,
              start_time, algorithm):
        self.simulator = Simulator(workload_path, platform_path,
                                   start_time, algorithm, rl=True)

    def get_observation(self):
        num_nodes = self.simulator.Monitor.num_nodes

        states = self.simulator.PlatformControl.get_state()
        mask = get_feasible_mask(states)
        features = feature_extraction(self.simulator)

        features = features.reshape(1, 11)
        features_ = T.from_numpy(features).to(self.device).float()

        mask = np.asanyarray(mask)
        mask = mask.reshape(1, num_nodes, 2)
        mask_ = T.from_numpy(mask).to(self.device).float()

        observation = (features_, mask_)

        return observation
