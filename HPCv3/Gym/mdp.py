# mdp.py
import os
import numpy as np
import torch as T

# Adjust this import to the actual location of your env file
# If you named the env file "GymEnv.py" and the class inside is GymEnv,
# change the import accordingly.
from HPCv3.Gym.gym_env import HPCGymEnv   # <- adjust path/name if needed

from agent.agent import Agent
from agent.critic import Critic
from agent.memory import PPOMemory

DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")


class PPOTrainer:
    """
    PPOTrainer collects trajectories using the split-step pattern:
      1) pre_features = env.advance_system()
      2) action, prob, value = agent(...)   # agent inference (outside env internals)
      3) next_features, reward, done, info = env.apply_action(action)
    Training (update_policy) is called explicitly by user/runner when desired.
    """

    def __init__(
        self,
        env: HPCGymEnv,
        agent: Agent,
        critic: Critic,
        memory: PPOMemory,
        lr=3e-4,
        clip_epsilon=0.2,
        epochs=4,
        gamma=0.99,
        lam=0.95,
        device=DEVICE,
    ):
        self.env = env
        self.agent = agent.to(device)
        self.critic = critic.to(device)
        self.memory = memory

        self.device = device
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam

        self.agent_opt = T.optim.Adam(self.agent.parameters(), lr=self.lr)
        self.critic_opt = T.optim.Adam(self.critic.parameters(), lr=self.lr)

    # ----------------------
    # Data collection
    # ----------------------
    def collect_trajectory(self, max_steps=1000):
        """
        Collect transitions using:
         pre_features = env.advance_system()
         agent.inference(pre_features) -> action, prob, value
         next_features, reward, done = env.apply_action(action)
        Store samples to self.memory using your PPOMemory API.
        """
        pre_features = self.env.reset()
        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < max_steps:
            # 1) Advance: sim -> rjms -> update -> pre_features
            pre_features = self.env.advance_system()

            # 2) Agent inference (actor + critic)
            obs_tensor = T.tensor(
                pre_features, dtype=T.float32).unsqueeze(0).to(self.device)
            mask = T.ones(1, 1, dtype=T.float32).to(self.device)

            with T.no_grad():
                # The agent forward signature in your project was: agent(obs_tensor, mask) -> probs, entropy
                probs, _ = self.agent(obs_tensor, mask)
                action_probs = probs.squeeze(0)
                action = T.multinomial(action_probs, 1).item()
                selected_prob = action_probs[action].item()

                value = self.critic(obs_tensor).item()

            # 3) Apply action and observe post-action state and reward
            next_features, reward, done, info = self.env.apply_action(action)

            # 4) Store transition in memory (match your PPOMemory.store_memory signature)
            # I use the same keys you used earlier:
            self.memory.store_memory(
                state=pre_features,
                mask=np.ones(1),
                action=action,
                probs=selected_prob,
                vals=value,
                reward=reward,
                done=done,
            )

            ep_reward += reward
            step += 1

            # Optionally update policy when memory ready
            if hasattr(self.memory, "ready_to_train") and self.memory.ready_to_train():
                self.update_policy()

        return ep_reward, step

    # ----------------------
    # Policy update
    # ----------------------
    def update_policy(self):
        """
        Pull batches from memory and run PPO-style updates.
        This uses a generic interface: memory.generate_batches() -> (states, actions, old_probs, vals, rewards, dones, batches)
        Adapt this to your PPOMemory API if it differs.
        """
        data = self.memory.generate_batches()
        # Expecting: states, actions, old_probs, vals, rewards, dones, batches
        try:
            states, actions, old_probs, vals, rewards, dones, batches = data
        except Exception:
            # If memory.generate_batches has a different signature, adapt here.
            raise RuntimeError(
                "PPOMemory.generate_batches() returned unexpected format. Adjust mdp.update_policy() accordingly.")

        # Compute returns & advantages (simple GAE-like)
        returns = []
        advantages = []
        running_return = 0
        running_adv = 0
        next_value = 0

        # Convert lists to numpy for easier handling if necessary
        # Here we compute naive discounted returns and advantages per time reversed
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * \
                running_return * (1.0 - dones[t])
            returns.insert(0, running_return)
            # advantage estimate using TD residual (simple)
            td_error = rewards[t] + self.gamma * \
                next_value * (1.0 - dones[t]) - vals[t]
            running_adv = td_error + self.gamma * \
                self.lam * running_adv * (1.0 - dones[t])
            advantages.insert(0, running_adv)
            next_value = vals[t]

        # To tensors
        states = T.tensor(np.vstack(states), dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.device)
        old_probs = T.tensor(old_probs, dtype=T.float32).to(self.device)
        advantages = T.tensor(advantages, dtype=T.float32).to(self.device)
        returns = T.tensor(returns, dtype=T.float32).to(self.device)

        # PPO epochs
        for _ in range(self.epochs):
            # mask placeholder if your agent needs it
            mask = T.ones(states.size(0), 1, dtype=T.float32).to(self.device)

            new_probs, entropy = self.agent(states, mask)
            new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            values = self.critic(states).squeeze(1)

            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = T.clamp(ratio, 1 - self.clip_epsilon, 1 +
                            self.clip_epsilon) * advantages
            actor_loss = -T.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_loss = -0.01 * entropy.mean()

            loss = actor_loss + critic_loss + entropy_loss

            self.agent_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.agent_opt.step()
            self.critic_opt.step()

        # Clear memory after update (or handle inside memory)
        if hasattr(self.memory, "clear_memory"):
            self.memory.clear_memory()

    # ----------------------
    # Convenience helpers
    # ----------------------
    def save(self, path):
        state = {
            "agent_state_dict": self.agent.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }
        T.save(state, path)

    def load(self, path, map_location=None):
        ckpt = T.load(
            path, map_location=self.device if map_location is None else map_location)
        self.agent.load_state_dict(ckpt["agent_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])


def build_trainer_from_config(
    workload_path,
    platform_path,
    algorithm,
    start_time,
    timeout,
    agent_kwargs: dict,
    critic_kwargs: dict,
    memory_kwargs: dict,
    trainer_kwargs: dict,
    device=DEVICE,
):
    """
    Factory to build env + agent + critic + memory + trainer.
    Returns: (env, trainer) ; no execution happens here.
    """
    env = HPCGymEnv(workload_path, platform_path, algorithm,
                    start_time=start_time, timeout=timeout)

    # Construct agent / critic using provided kwargs
    agent = Agent(**agent_kwargs).to(device)
    critic = Critic(**critic_kwargs).to(device)
    memory = PPOMemory(**memory_kwargs)

    trainer = PPOTrainer(env=env, agent=agent, critic=critic,
                         memory=memory, device=device, **trainer_kwargs)
    return env, trainer
