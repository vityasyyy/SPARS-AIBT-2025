# runner.py
import json
import os
from datetime import datetime
from typing import Any, Dict

import torch as T

from SPARS.Utils import get_logger, log_output
from SPARS.Simulator.Simulator import Simulator, run_simulation

# IMPORTANT: load Gym config BEFORE importing the env so monkey-patches apply
from SPARS.Gym import config          # your pluggable Gym setup
from SPARS.Gym import utils as G
from SPARS.Gym.gym import HPCGymEnv

from RL_Agent.SPARS.agent import ActorCriticMLP  # (kept as in your original)

DEFAULT_CFG_PATH = "simulator_config.yaml"


def _load_config(path: str = DEFAULT_CFG_PATH) -> Dict[str, Any]:
    """
    Load YAML or JSON config from a fixed path. If the file doesn't exist,
    fall back to internal defaults (keeps the runner usable out of the box).
    """
    import pathlib
    p = pathlib.Path(path)
    if p.exists():
        if p.suffix.lower() in {".yml", ".yaml"}:
            import yaml  # requires PyYAML
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback defaults (edit these if you want different out-of-the-box behavior)
    return {
        "paths": {
            "workload": "workloads/generated.json",
            "platform": "platforms/generated.json",
            "output":   "results/generated",
        },
        "run": {
            "algorithm": "fcfs_normal",
            "overrun_policy": "continue",
            "timeout": None,
            "start_time": 0,  # epoch int, or "now", or "YYYY-MM-DD HH:MM:SS"
        },
        "rl": {
            "enabled": False,
            "type": "discrete",   # "discrete" | "continuous"
            "dt": 1800,           # required for discrete
            "device": "auto",     # "auto" | "cpu" | "cuda"
            "learning_rate": 3e-4,
            "epochs": 10,
            "num_nodes": 16,
            "obs_dim": 11,
            "act_dim": 1,
        },
        "logging": {
            "level": "INFO",
            "file": "results/simulation.log",
        },
    }


def _choose_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if T.cuda.is_available() else "cpu"
    return pref


def _parse_start_time(value) -> int:
    """
    Accepts:
      - int/float epoch
      - "now"
      - "YYYY-MM-DD HH:MM:SS"
    Returns epoch seconds (int).
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if value.lower() == "now":
            return int(datetime.now().timestamp())
        try:
            t = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            return int(t.timestamp())
        except ValueError:
            raise ValueError(
                "run.start_time must be epoch int, 'now', or 'YYYY-MM-DD HH:MM:SS'"
            )
    raise TypeError("Unsupported start_time type")


# ---------------------------
# Helpers for flexible agent construction (ONLY addition)
# ---------------------------
def _load_object(spec: str):
    """Load 'pkg.mod:Obj' or 'pkg.mod.Obj' into a Python object."""
    import importlib
    if ":" in spec:
        mod, name = spec.split(":", 1)
    else:
        mod, _, name = spec.rpartition(".")
        if not mod:
            raise ValueError(f"Bad import path: {spec}")
    return getattr(importlib.import_module(mod), name)


def _instantiate_with_flexible_kwargs(cls, params: dict, *, positional_first: str | None = None):
    """
    Instantiate `cls` with kwargs in `params`. If the constructor needs a first positional
    argument (e.g., optimizer 'params'), set positional_first='params'.
    Filters unknown kwargs automatically when possible.
    """
    import inspect
    params = dict(params or {})

    def _call(p: dict):
        if positional_first and positional_first in p:
            pf = p.pop(positional_first)
            try:
                return cls(pf, **p)
            finally:
                p[positional_first] = pf
        return cls(**p)

    try:
        return _call(params)
    except TypeError:
        # Filter unknown kwargs unless ctor accepts **kwargs
        sig = None
        try:
            sig = inspect.signature(cls.__init__)
            has_varkw = any(
                a.kind == inspect.Parameter.VAR_KEYWORD for a in sig.parameters.values())
            if has_varkw:
                raise
            allowed = {k for k in sig.parameters if k != "self"}
            filtered = {k: v for k, v in params.items() if k in allowed}
            return _call(filtered)
        except Exception:
            raise


def _build_agent(rl_cfg: dict, device: str):
    """
    Build agent and optimizer ENTIRELY from cfg['rl']['agent'] with flexible params.
    - No hard-coded keys like obs_dim/act_dim are injected.
    - 'device' handling:
        * if agent.params.device == "auto" -> resolve with _choose_device
        * if agent.params.device missing   -> set to resolved device
        * if agent ctor doesn't accept 'device', it's filtered; if it's an nn.Module,
          we still move it to the device afterward.
    """
    agent_cfg = rl_cfg.get("agent") or {}

    # ----- Agent class -----
    AgentClass = _load_object(agent_cfg.get(
        "class", "RL_Agent.SPARS.agent:ActorCriticMLP"))
    params = dict(agent_cfg.get("params") or {})

    cfg_device = params.get("device", rl_cfg.get("device", "auto"))
    final_device = _choose_device(
        cfg_device if cfg_device is not None else "auto")

    if "device" not in params or str(params.get("device")).lower() == "auto":
        params["device"] = final_device

    model = _instantiate_with_flexible_kwargs(AgentClass, params)

    # Ensure nn.Module is moved even if ctor ignored 'device'
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            model.to(final_device)
    except Exception:
        pass

    # ----- Optimizer -----
    opt_cfg = agent_cfg.get("optimizer") or {}
    OptClass = _load_object(opt_cfg.get("class", "torch.optim:Adam"))

    opt_params = dict(opt_cfg.get("params") or {})
    if "lr" not in opt_params and "learning_rate" in rl_cfg:
        opt_params["lr"] = float(rl_cfg["learning_rate"])

    optimizer = _instantiate_with_flexible_kwargs(
        OptClass,
        {"params": model.parameters() if hasattr(model, "parameters")
         else model, **opt_params},
        positional_first="params",
    )

    return model, optimizer
# ---------------------------


def main():
    cfg = _load_config(DEFAULT_CFG_PATH)

    # --- Logging ---
    logger = get_logger(
        "runner",
        level=cfg["logging"].get("level", "INFO"),
        log_file=cfg["logging"].get("file", "results/simulation.log"),
    )

    # --- Config Unpack ---
    workload_path = cfg["paths"]["workload"]
    platform_path = cfg["paths"]["platform"]
    output_path = cfg["paths"]["output"]

    algorithm = cfg["run"]["algorithm"]
    overrun_policy = cfg["run"].get("overrun_policy", "continue")
    timeout = cfg["run"].get("timeout", None)
    start_time = _parse_start_time(cfg["run"].get("start_time", 0))

    rl_enabled = bool(cfg["rl"].get("enabled", False))
    rl_type = cfg["rl"].get("type", "discrete") if rl_enabled else None
    rl_dt = cfg["rl"].get("dt", None) if rl_type == "discrete" else None
    device = _choose_device(cfg["rl"].get("device", "auto"))

    # === RL parameters ===
    learning_rate = float(cfg["rl"].get("learning_rate", 3e-4))
    epochs = int(cfg["rl"].get("epochs", 10))
    num_nodes = int(cfg["rl"].get("num_nodes", 16))
    obs_dim = int(cfg["rl"].get("obs_dim", 11))
    act_dim = int(cfg["rl"].get("act_dim", 1))

    if rl_enabled and rl_type == "discrete" and rl_dt is None:
        raise RuntimeError("Discrete RL requires rl.dt in the config file.")

    if rl_enabled:
        # Build simulator from config (no CLI/args)
        simulator = Simulator.from_config(
            cfg,
            rl_kwargs={"rl_type": rl_type, "rl_dt": rl_dt},
        )
        env = HPCGymEnv(simulator, device)

        # === ONLY change below: agent is now built from config flexibly ===
        model, model_opt = _build_agent(cfg["rl"], device)

        for _ in range(epochs):
            # reset per epoch
            simulator = Simulator.from_config(
                cfg,
                rl_kwargs={"rl_type": rl_type, "rl_dt": rl_dt},
            )
            env.reset(simulator)
            env.simulator.start_simulator()
            observation = env.get_observation()

            memory_features = []
            memory_masks = []
            memory_actions = []
            memory_rewards = []

            while env.simulator.is_running:
                features_, mask_ = observation
                features_ = features_.to(device)

                # your policy/value forward

                # --- SPARS ---
                # logits, values = model(features_)

                # --- Thomas Reshape ---
                features_reshaped = features_.reshape(1, num_nodes, 11)
                logits, values = model(features_reshaped)

                next_observation, reward, done = env.step(logits)

                # store experience (detach from graph)
                memory_actions.append(logits.detach())
                memory_features.append(features_.detach())
                memory_masks.append(mask_.detach())
                memory_rewards.append(reward.detach() if isinstance(reward, T.Tensor)
                                      else T.tensor(float(reward)))

                saved_experiences = (
                    memory_actions, memory_features, memory_masks, memory_rewards
                )

                # learner chosen in SPARS.Gym.config (utils.G.learn)
                G.learn(model, model_opt, done,
                        saved_experiences, next_observation)

                observation = next_observation

        log_output(env.simulator, output_path)

        # --- Save agent checkpoint ---
        os.makedirs(output_path, exist_ok=True)
        ckpt = {
            "agent_class": f"{model.__class__.__module__}:{model.__class__.__name__}",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_opt.state_dict(),
            "rl_config": cfg.get("rl", {}),
            "epochs_trained": epochs,
        }
        ckpt_path = os.path.join(output_path, "agent_checkpoint.pt")
        T.save(ckpt, ckpt_path)
        logger.info(f"Saved agent checkpoint to: {ckpt_path}")

    else:
        simulator = Simulator.from_config(cfg)
        run_simulation(simulator, output_path)


if __name__ == "__main__":
    main()
