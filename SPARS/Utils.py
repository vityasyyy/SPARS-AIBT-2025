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
    """ MAP DATA"""

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4
    }

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

    jobs_exploded = jobs.copy()
    jobs_exploded['nodes'] = jobs_exploded['nodes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    jobs_exploded = jobs_exploded.explode('nodes')
    jobs_exploded = jobs_exploded.rename(columns={'nodes': 'node_id'})

    if 'submission_time' not in jobs_exploded.columns:
        jobs_exploded['submission_time'] = pd.NA

    jobs_exploded = jobs_exploded[[
        'job_id', 'node_id', 'submission_time', 'start_time', 'finish_time']]

    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy(
    )
    active_merged = pd.merge(
        active_df,
        jobs_exploded,
        on=['node_id', 'start_time', 'finish_time'],
        how='left'
    )
    active_merged['job_id'] = active_merged['job_id'].fillna(-1)

    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy(
    )
    non_active_df['job_id'] = non_active_df['state'].map(
        mapping_non_active).fillna(-1)
    non_active_df['submission_time'] = pd.NA

    combined = pd.concat([active_merged, non_active_df])

    combined['node_id'] = combined['node_id'].astype(int)
    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'submission_time',
            'start_time', 'finish_time', 'job_id']
    ).agg(
        nodes=('node_id', lambda x: ' '.join(map(str, sorted(x))))
    )
    grouped = grouped.reset_index()

    grouped = grouped.sort_values(by=['start_time', 'finish_time'])
    result = grouped[['dvfs_mode', 'state', 'submission_time',
                      'start_time', 'finish_time', 'nodes', 'job_id']]

    return result


def process_node_job_data(nodes_data, jobs):
    """ MAP DATA"""

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4
    }

    # --- Build node intervals
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

    # --- Normalize node_intervals_df dtypes
    node_intervals_df['node_id'] = _to_int_series(node_intervals_df['node_id'])
    node_intervals_df['start_time'] = _to_float_series(
        node_intervals_df['start_time'])
    node_intervals_df['finish_time'] = _to_float_series(
        node_intervals_df['finish_time'])

    # --- Prepare jobs and explode nodes
    jobs_exploded = jobs.copy()

    # ensure 'nodes' is a list
    if 'nodes' in jobs_exploded.columns:
        jobs_exploded['nodes'] = jobs_exploded['nodes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        jobs_exploded = jobs_exploded.explode('nodes')
        jobs_exploded = jobs_exploded.rename(columns={'nodes': 'node_id'})
    else:
        # if no 'nodes' column, create an empty one to avoid KeyError
        jobs_exploded['node_id'] = pd.NA

    if 'submission_time' not in jobs_exploded.columns:
        jobs_exploded['submission_time'] = pd.NA

    # --- Normalize jobs_exploded dtypes (must match node_intervals_df)
    jobs_exploded['node_id'] = _to_int_series(jobs_exploded['node_id'])
    jobs_exploded['start_time'] = _to_float_series(jobs_exploded['start_time'])
    jobs_exploded['finish_time'] = _to_float_series(
        jobs_exploded['finish_time'])
    # keep job_id as nullable int as well
    if 'job_id' in jobs_exploded.columns:
        jobs_exploded['job_id'] = pd.to_numeric(
            jobs_exploded['job_id'], errors='coerce').astype('Int64')

    jobs_exploded = jobs_exploded[[
        'job_id', 'node_id', 'submission_time', 'start_time', 'finish_time']]

    # --- Split active vs non-active
    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy(
    )
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy(
    )

    # --- Merge active intervals with jobs (exact match on node_id & times)
    active_merged = pd.merge(
        active_df,
        jobs_exploded,
        on=['node_id', 'start_time', 'finish_time'],
        how='left',
        indicator=True
    )

    # fill job_id=-1 for unmatched, keep dtype as Int64 then cast to int for output later if desired
    active_merged['job_id'] = active_merged['job_id'].fillna(
        -1).astype('Int64')

    # --- Non-active: map to negative codes
    non_active_df = non_active_df.assign(
        job_id=non_active_df['state'].map(
            mapping_non_active).fillna(-1).astype('Int64'),
        submission_time=pd.NA
    )

    # --- Combine
    combined = pd.concat(
        [df for df in (active_merged, non_active_df)
         if not df.empty and not df.isna().all().all()],
        ignore_index=True
    )

    # node_id as plain int for grouping/printing
    combined['node_id'] = combined['node_id'].astype('Int64')

    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'submission_time',
            'start_time', 'finish_time', 'job_id'],
        dropna=False
    ).agg(
        nodes=('node_id', lambda x: ' '.join(
            map(str, sorted([int(v) for v in x.dropna()]))))
    ).reset_index()

    grouped = grouped.sort_values(by=['start_time', 'finish_time'])

    # If you want plain int for job_id in the final result:
    grouped['job_id'] = grouped['job_id'].astype(int)

    result = grouped[['dvfs_mode', 'state', 'submission_time',
                      'start_time', 'finish_time', 'nodes', 'job_id']]
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
