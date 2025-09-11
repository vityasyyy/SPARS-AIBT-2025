from SPARS.Gym.utils import learn
from SPARS.Utils import get_logger, log_output
from SPARS.Simulator.Simulator import Simulator, run_simulation
from SPARS.Gym.gym import HPCGymEnv
from datetime import datetime
from RL_Agent.SPARS.agent import ActorCriticMLP
# from model.memory import PPOMemory
import torch as T
import numpy as np
# === Configure logging here (runner) ===

logger = get_logger("runner", level="INFO", log_file="results/simulation.log")

# === Configuration ===
# Path to platform JSON file
workload_path = "debug_runs/run_0019/workloads/generated.json"
platform_path = "debug_runs/run_0019/platforms/generated.json"  # Path to output folder
output_path = "debug_runs/run_0019/results/generated"  # Scheduling algorithm name
algorithm = "easy_auto"  # Simulation timeout in seconds (optional)
# Time in format YYYY-MM-DD HH:MM:SS (optional). If not provided, starts at 0.
timeout = 12
start_time = None
RL = False  # Reinforcement Learning mode
overrun_policy = 'continue'

# === Initialize simulator ===
if start_time is not None:
    start_time = int(datetime.strptime(
        start_time, "%Y-%m-%d %H:%M:%S").timestamp())
    human_readable = True
else:
    start_time = 0
    human_readable = False


# === Main ===
if RL:
    # === RL parameters ===
    learning_rate = 0.0003
    device = "cuda"
    epoch = 10
    num_nodes = 16
    obs_dim = 11
    act_dim = 1

    model = ActorCriticMLP(obs_dim, act_dim, num_nodes, device)
    model_opt = T.optim.Adam(model.parameters(), lr=learning_rate)

    simulator = Simulator(workload_path, platform_path,
                          start_time, algorithm, overrun_policy, rl=True, timeout=timeout)

    env = HPCGymEnv(simulator, device)

    for _ in range(epoch):
        env.reset(workload_path, platform_path, start_time, algorithm)
        env.simulator.start_simulator()
        observation = env.get_observation()

        memory_features = []
        memory_masks = []
        memory_actions = []
        memory_rewards = []
        while env.simulator.is_running:
            features_, mask_ = observation
            featurs_ = features_.to(device)
            logits, values = model(features_)

            next_observation, reward, done = env.step(logits)

            memory_actions.append(logits.detach())
            memory_features.append(features_.detach())
            memory_masks.append(mask_.detach())
            memory_rewards.append(reward.detach())

            saved_experiences = (memory_actions, memory_features, memory_masks,
                                 memory_rewards)

            learn(model, model_opt, done,
                  saved_experiences, next_observation)

            observation = next_observation

    log_output(env.simulator, output_path)
else:
    simulator = Simulator(workload_path, platform_path,
                          start_time, algorithm, overrun_policy, timeout=timeout)
    run_simulation(simulator, output_path)
