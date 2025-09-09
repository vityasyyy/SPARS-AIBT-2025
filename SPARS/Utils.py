import pandas as pd
import ast
import os
# logger_setup.py
import logging
import os
from typing import Optional

# --- Optional: add a TRACE level (between NOTSET=0 and DEBUG=10) ---
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


logging.Logger.trace = _trace  # type: ignore[attr-defined]

# --- Factory ---


def get_logger(
    name: Optional[str] = None,
    level: str | int = "INFO",
    log_file: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create a configured logger.
    - level: "DEBUG"/"INFO"/... or int (e.g., 10) or "TRACE"
    - log_file: if given, also logs to a file (UTF-8)
    """
    logger = logging.getLogger(name if name else __name__)
    if logger.handlers:  # avoid duplicate handlers when called multiple times
        return logger

    # Resolve level (env var wins if set)
    env_level = os.getenv("LOG_LEVEL")
    level_str = (env_level or str(level)).upper()
    level_value = {
        "TRACE": TRACE,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(level_str, logging.INFO)
    logger.setLevel(level_value)
    logger.propagate = propagate

    # Shared formatter
    fmt = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    ch = logging.StreamHandler()
    # you can choose a different threshold per handler
    ch.setLevel(level_value)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level_value)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def _to_int_series(s):
    # handle lists, strings like "3", floats like 3.0, and None
    return pd.to_numeric(s, errors="coerce").astype("Int64")  # nullable int


def _to_float_series(s):
    # unify time columns to float (seconds). If you use datetime, convert both sides to datetime64[ns] instead.
    return pd.to_numeric(s, errors="coerce").astype("float64")


def process_node_job_data(nodes_data, jobs):
    """MAP DATA"""

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4
    }

    # ---- build node intervals ----
    node_intervals = []
    for node in nodes_data:
        node_id = node['id']
        state_history = node['state_history']
        current_dvfs = None
        for interval in state_history:
            if 'dvfs_mode' in interval:
                current_dvfs = interval['dvfs_mode']
            interval['dvfs_mode'] = current_dvfs

            if interval['start_time'] < interval['finish_time']:
                node_intervals.append({
                    'node_id': node_id,
                    'state': interval['state'],
                    'dvfs_mode': interval['dvfs_mode'],
                    'start_time': interval['start_time'],
                    'finish_time': interval['finish_time']
                })

    node_intervals_df = pd.DataFrame(node_intervals)

    # ---- explode jobs to (job_id, node_id) rows ----
    jobs_exploded = jobs.copy()

    # nodes column may be stringified list -> list
    jobs_exploded['nodes'] = jobs_exploded['nodes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    jobs_exploded = jobs_exploded.explode(
        'nodes').rename(columns={'nodes': 'node_id'})

    # unify submission time column: prefer 'submission_time', else 'subtime', else NA
    if 'submission_time' in jobs_exploded.columns:
        subcol = 'submission_time'
    elif 'subtime' in jobs_exploded.columns:
        subcol = 'subtime'
        jobs_exploded['submission_time'] = jobs_exploded['subtime']
    else:
        subcol = 'submission_time'
        jobs_exploded['submission_time'] = pd.NA

    # ensure 'terminated' exists; if missing, set NA
    if 'terminated' not in jobs_exploded.columns:
        jobs_exploded['terminated'] = pd.NA

    jobs_exploded = jobs_exploded[[
        'job_id', 'node_id', 'submission_time', 'start_time', 'finish_time', 'terminated']]

    # ---- ACTIVE intervals (join with jobs on exact node/time interval) ----
    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy(
    )
    active_merged = pd.merge(
        active_df,
        jobs_exploded,
        on=['node_id', 'start_time', 'finish_time'],
        how='left'
    )
    active_merged['job_id'] = active_merged['job_id'].fillna(-1)

    # ---- NON-ACTIVE intervals (map job_id codes; no submission_time / terminated) ----
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy(
    )
    non_active_df['job_id'] = non_active_df['state'].map(
        mapping_non_active).fillna(-1)
    non_active_df['submission_time'] = pd.NA
    non_active_df['terminated'] = pd.NA

    # ---- combine & group nodes into space-separated strings ----
    combined = pd.concat([active_merged, non_active_df], ignore_index=True)
    combined['node_id'] = combined['node_id'].astype(int)

    grouped = (
        combined
        .groupby(['state', 'dvfs_mode', 'submission_time', 'start_time', 'finish_time', 'job_id'], dropna=False)
        .agg(
            nodes=('node_id', lambda x: ' '.join(map(str, sorted(x)))),
            terminated=('terminated', 'first')  # carry job flag
        )
        .reset_index()
        .sort_values(by=['start_time', 'finish_time'])
    )

    # final column order
    result = grouped[['dvfs_mode', 'state', 'submission_time',
                      'start_time', 'finish_time', 'nodes', 'job_id', 'terminated']]
    return result


def process_node_job_data(nodes_data, jobs):
    """ MAP DATA"""

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4
    }

    # --- build node intervals ---
    node_intervals = []
    for node in nodes_data:
        node_id = node['id']
        state_history = node['state_history']
        current_dvfs = None
        for interval in state_history:
            if 'dvfs_mode' in interval:
                current_dvfs = interval['dvfs_mode']
            interval['dvfs_mode'] = current_dvfs

            if interval['start_time'] < interval['finish_time']:
                node_intervals.append({
                    'node_id': node_id,
                    'state': interval['state'],
                    'dvfs_mode': interval['dvfs_mode'],
                    'start_time': interval['start_time'],
                    'finish_time': interval['finish_time']
                })

    node_intervals_df = pd.DataFrame(node_intervals)

    # --- jobs explode by nodes ---
    jobs_exploded = jobs.copy()

    # normalize nodes column to list
    jobs_exploded['nodes'] = jobs_exploded['nodes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    jobs_exploded = jobs_exploded.explode(
        'nodes').rename(columns={'nodes': 'node_id'})

    # normalize submission time column name
    if 'submission_time' not in jobs_exploded.columns:
        if 'subtime' in jobs_exploded.columns:
            jobs_exploded = jobs_exploded.rename(
                columns={'subtime': 'submission_time'})
        else:
            jobs_exploded['submission_time'] = pd.NA

    # ensure terminated exists (if not present, default NA)
    if 'terminated' not in jobs_exploded.columns:
        jobs_exploded['terminated'] = pd.NA

    jobs_exploded = jobs_exploded[['job_id', 'node_id', 'submission_time',
                                   'start_time', 'finish_time', 'terminated']]

    # --- active intervals joined with jobs ---
    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy(
    )
    active_merged = pd.merge(
        active_df,
        jobs_exploded,
        on=['node_id', 'start_time', 'finish_time'],
        how='left'
    )
    active_merged['job_id'] = active_merged['job_id'].fillna(-1)
    # keep terminated as-is; if no match it will be NaN/NA

    # --- non-active intervals get mapped job_id, NA submission_time/terminated ---
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy(
    )
    non_active_df['job_id'] = non_active_df['state'].map(
        mapping_non_active).fillna(-1)
    non_active_df['submission_time'] = pd.NA
    non_active_df['terminated'] = pd.NA

    # --- combine ---
    combined = pd.concat([active_merged, non_active_df], ignore_index=True)
    combined['node_id'] = combined['node_id'].astype(int)

    # --- group nodes into intervals (aggregate node ids; propagate terminated) ---
    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'submission_time',
            'start_time', 'finish_time', 'job_id'],
        dropna=False
    ).agg(
        nodes=('node_id', lambda x: ' '.join(map(str, sorted(x)))),
        terminated=('terminated', lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())
                    if s.notna().any() else pd.NA)
    ).reset_index()

    grouped = grouped.sort_values(by=['start_time', 'finish_time'])

    result = grouped[['dvfs_mode', 'state', 'submission_time',
                      'start_time', 'finish_time', 'nodes', 'job_id', 'terminated']]

    return result


def log_output(simulator, output_folder):
    os.makedirs(f'{output_folder}', exist_ok=True)
    # Assuming simulator.Monitor.nodes_state_log is already populated
    raw_node_log = pd.DataFrame(simulator.Monitor.states_hist)
    raw_node_log.to_csv(f'{output_folder}/raw_node_log.csv', index=False)
    raw_job_log = pd.DataFrame(simulator.Monitor.jobs_execution_log)
    raw_job_log.to_csv(f'{output_folder}/raw_job_log.csv', index=False)
    node_log = process_node_job_data(
        simulator.Monitor.states_hist, raw_job_log)
    node_log.to_csv(f'{output_folder}/node_log.csv', index=False)
