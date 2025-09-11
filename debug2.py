# debug.py
# Combined runner + workload generator + platform generator + 100-case spammer
# with your requested randomization rules (no argparse, RL OFF).
# ---------------------------------------------------------------------------

import os
import sys
import json
import math
import random
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Optional integration with SPARS logging; fall back to stdlib logging
# -----------------------------------------------------------------------------
try:
    from SPARS.Utils import get_logger as spars_get_logger, log_output as spars_log_output
except Exception:
    spars_get_logger = None
    spars_log_output = None

import logging


def _fallback_get_logger(name: str, level: str = "INFO", log_file: str = None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def get_logger(name: str, level: str = "INFO", log_file: str = None):
    if spars_get_logger is not None:
        try:
            return spars_get_logger(name, level=level, log_file=log_file)
        except Exception:
            pass
    return _fallback_get_logger(name, level, log_file)


def log_output(simulator, output_path: str):
    """Prefer SPARS.Utils.log_output; else leave a breadcrumb file."""
    if spars_log_output is not None:
        try:
            return spars_log_output(simulator, output_path)
        except Exception:
            pass
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "_debug_fallback_log_output.txt"), "a", encoding="utf-8") as f:
        f.write(f"Run completed at {datetime.now().isoformat()}\n")

# -----------------------------------------------------------------------------
# Workloads generator (merged)
# -----------------------------------------------------------------------------


class ProblemGenerator:
    def __init__(
        self,
        lambda_arrival: float = 0.2,
        mu_execution: float = 80,
        sigma_execution: float = 30,
        mu_noise: float = 0,
        sigma_noise: float = 1,
        num_jobs: int = None,
        max_node: int = 8,
    ):
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
            1 / max(self.lambda_arrival, 1e-6), self.num_jobs)
        arrival_times = np.cumsum(interarrival_times)
        requested_execution_times = np.random.normal(
            self.mu_execution, max(self.sigma_execution, 1e-6), self.num_jobs)
        noise = np.random.normal(self.mu_noise, max(
            self.sigma_noise, 1e-6), self.num_jobs)

        requested_execution_times = np.maximum(requested_execution_times, 1.0)
        actual_execution_times = np.maximum(
            requested_execution_times + noise, 1.0)
        num_nodes_required = np.clip(
            np.random.normal(math.ceil(self.max_node / 2), 1, self.num_jobs),
            1, self.max_node
        )
        workloads = []
        for i in range(self.num_jobs):
            workloads.append({
                "job_id": i + 1,
                "res": int(num_nodes_required[i]),
                "subtime": round(float(arrival_times[i])),
                "reqtime": round(float(requested_execution_times[i])),
                "runtime": round(float(actual_execution_times[i])),
                "profile": "100",
                "user_id": 0
            })
        return workloads


def build_workload_json(max_node_for_jobs: int, workloads: list) -> Dict[str, Any]:
    return {
        # keep nb_res aligned to the maximum nodes a job can request
        "nb_res": max_node_for_jobs,
        "jobs": workloads,
        "profiles": {
            "100": {
                "cpu": 10000000000000000000000,
                "com": 0,
                "type": "parallel_homogeneous"
            }
        }
    }

# -----------------------------------------------------------------------------
# Platform generator (merged)
# -----------------------------------------------------------------------------


def generate_machine(
    machine_id: int,
    base_power: float,
    switching_off_power: float,
    switching_on_power: float,
    sleeping_power: float,
    switching_on_time: int,
    switching_off_time: int
) -> Dict[str, Any]:
    base_speed = 1.0
    underclock_power = base_power * 0.7
    underclock_speed = base_speed * 0.7
    overclock_power = base_power * 1.3
    overclock_speed = base_speed * 1.3

    machine = {
        "id": machine_id,
        "dvfs_profiles": {
            "underclock_1": {"power": underclock_power, "compute_speed": underclock_speed},
            "base": {"power": base_power, "compute_speed": base_speed},
            "overclock_1": {"power": overclock_power, "compute_speed": overclock_speed},
        },
        "dvfs_mode": "base",
        "states": {
            "active": {
                "power": "from_dvfs",
                "compute_speed": "from_dvfs",
                "can_run_jobs": True,
                "transitions": [{"state": "switching_off", "transition_time": 0}],
            },
            "switching_off": {
                "power": switching_off_power,
                "compute_speed": "from_dvfs",
                "can_run_jobs": False,
                "transitions": [{"state": "sleeping", "transition_time": switching_off_time}],
            },
            "switching_on": {
                "power": switching_on_power,
                "compute_speed": "from_dvfs",
                "can_run_jobs": False,
                "transitions": [{"state": "active", "transition_time": switching_on_time}],
            },
            "sleeping": {
                "power": sleeping_power,
                "compute_speed": "from_dvfs",
                "can_run_jobs": False,
                "transitions": [{"state": "switching_on", "transition_time": 0}],
            },
        },
    }
    return machine


def generate_cluster(num_node: int) -> Dict[str, Any]:
    machines = [generate_machine(i, 210, 9, 9, 9, 5, 5)
                for i in range(num_node)]
    return {"machines": machines}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_json(path: str, data: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def unique_case_paths(base_dir: str, case_id: int) -> Tuple[str, str, str]:
    """returns (workload_path, platform_path, result_dir) for each case"""
    case_dir = os.path.join(base_dir, f"run_{case_id:04d}")
    workload_path = os.path.join(case_dir, "workloads", "generated.json")
    platform_path = os.path.join(case_dir, "platforms", "generated.json")
    result_dir = os.path.join(case_dir, "results")
    return workload_path, platform_path, result_dir

# -----------------------------------------------------------------------------
# Single-case runner (RL is always OFF here)
# -----------------------------------------------------------------------------


def run_one_case(
    case_id: int,
    base_dir: str,
    logger: logging.Logger
):
    """
    Randomization rules per your request:
      - timeout: random int [10, 30]
      - algorithm: random choice from ["easy_auto","easy_normal","fcfs_auto","fcfs_normal"]
      - platform_nodes: random int [2, 32]
      - job_max_nodes = platform_nodes // 2 (>=1)
      - RL: always OFF
      - number of cases: fixed elsewhere to 100
    """
    # per-case random seed
    seed = random.randint(1, 10_000_000)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    # paths
    workload_path, platform_path, result_dir = unique_case_paths(
        base_dir, case_id)
    ensure_dir(os.path.dirname(workload_path))
    ensure_dir(os.path.dirname(platform_path))
    ensure_dir(result_dir)

    # random selections
    timeout = random.randint(10, 30)
    algorithm = random.choice(
        ["easy_auto", "easy_normal", "fcfs_auto", "fcfs_normal"])
    platform_nodes = random.randint(2, 32)
    job_max_nodes = max(1, platform_nodes // 2)

    # workloads random params (kept sane)
    jobs = 100
    lambda_arrival = max(0.05, np.random.uniform(0.05, 0.5))
    mu_exec = np.random.uniform(40, 120)
    sigma_exec = np.random.uniform(10, 50)
    mu_noise = np.random.uniform(-5, 5)
    sigma_noise = np.random.uniform(0.5, 5.0)

    # build workload
    pg = ProblemGenerator(
        lambda_arrival=lambda_arrival,
        mu_execution=mu_exec,
        sigma_execution=sigma_exec,
        mu_noise=mu_noise,
        sigma_noise=sigma_noise,
        num_jobs=jobs,
        max_node=job_max_nodes
    )
    workloads = pg.generate()
    workload_json = build_workload_json(job_max_nodes, workloads)
    write_json(workload_path, workload_json)

    # build platform
    cluster = generate_cluster(platform_nodes)
    write_json(platform_path, cluster)

    logger.info(
        f"[CASE {case_id}] seed={seed} alg={algorithm} timeout={timeout}s "
        f"platform_nodes={platform_nodes} job_max_nodes={job_max_nodes} jobs={jobs} -> start"
    )

    # start_time semantics from your runner: 0 (no human-readable)
    start_time = 0
    overrun_policy = "continue"

    try:
        from SPARS.Simulator.Simulator import Simulator, run_simulation
        simulator = Simulator(
            workload_path, platform_path, start_time, algorithm,
            overrun_policy, timeout=timeout
        )
        run_simulation(simulator, result_dir)
        log_output(simulator, result_dir)
        logger.info(f"[CASE {case_id}] SUCCESS")
        return True, ""
    except Exception as e:
        err_txt = f"[CASE {case_id}] ERROR: {type(e).__name__}: {e}"
        logger.error(err_txt)
        tb_path = os.path.join(result_dir, "exception_traceback.txt")
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(err_txt + "\n\n")
            f.write(traceback.format_exc())
        return False, err_txt

# -----------------------------------------------------------------------------
# Main spammer (exactly 100 cases, no args)
# -----------------------------------------------------------------------------


def main():
    TOTAL_CASES = 100
    BASE_DIR = "debug_runs"
    ensure_dir(BASE_DIR)

    logger = get_logger("debug_spammer", level="INFO",
                        log_file=os.path.join(BASE_DIR, "debug_spam.log"))

    success = 0
    failures = 0
    summary_csv = os.path.join(BASE_DIR, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as fcsv:
        fcsv.write("case_id,success,error\n")
        for case_id in range(1, TOTAL_CASES + 1):
            ok, err = run_one_case(case_id, BASE_DIR, logger)
            if ok:
                success += 1
                fcsv.write(f"{case_id},1,\n")
            else:
                failures += 1
                err_clean = err.replace("\n", " ").replace(",", ";")
                fcsv.write(f"{case_id},0,{err_clean}\n")

            if case_id % 10 == 0:
                logger.info(
                    f"Progress: {case_id}/{TOTAL_CASES} done | OK={success} FAIL={failures}")

    logger.info(
        f"ALL DONE. total={TOTAL_CASES} success={success} failures={failures}")
    logger.info(f"Summary CSV: {summary_csv}")
    logger.info(f"Logs: {os.path.join(BASE_DIR, 'debug_spam.log')}")


if __name__ == "__main__":
    main()
