import warnings
import pandas as pd
import ast
import os
# logger_setup.py
import logging
import os
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

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


def parse_nodes(x):
    # handle NaN/None/empty strings
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            return list(v) if isinstance(v, (list, tuple)) else v
        except Exception:
            return []
    return x  # as-is


def _ensure_jobs_shape(jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure jobs has the columns you later rely on, with compatible dtypes.
    Works even if jobs is None or entirely empty.
    """
    expected = {
        'job_id': 'Int64',
        'nodes': 'object',
        'submission_time': 'Int64',
        'start_time': 'Int64',
        'finish_time': 'Int64',
        'terminated': 'boolean',
    }

    if jobs is None or not isinstance(jobs, pd.DataFrame) or jobs.empty:
        # Build a skeleton empty DF with expected columns/dtypes
        return pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in expected.items()})

    # Make sure all expected columns exist; keep existing values if present
    for c, dt in expected.items():
        if c not in jobs.columns:
            jobs[c] = pd.Series(pd.NA, index=jobs.index, dtype=dt)
        else:
            # Best-effort cast; ignore if incompatible to avoid crashing
            try:
                jobs[c] = jobs[c].astype(dt)
            except Exception:
                pass
    return jobs


def process_node_job_data(nodes_data, jobs):
    """ MAP DATA — robust to empty jobs CSV """

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4
    }

    # --- build node intervals ---
    node_intervals = []
    for node in (nodes_data or []):
        node_id = node['id']
        state_history = node.get('state_history', [])
        current_dvfs = None
        for interval in state_history:
            if 'dvfs_mode' in interval:
                current_dvfs = interval['dvfs_mode']
            interval['dvfs_mode'] = current_dvfs
            if interval['start_time'] < interval['finish_time']:
                node_intervals.append({
                    'node_id':   node_id,
                    'state':     interval['state'],
                    'dvfs_mode': interval['dvfs_mode'],
                    'start_time': interval['start_time'],
                    'finish_time': interval['finish_time']
                })

    node_intervals_df = pd.DataFrame(node_intervals, columns=[
                                     'node_id', 'state', 'dvfs_mode', 'start_time', 'finish_time'])

    # --- jobs explode by nodes (robust to empty/missing columns) ---
    jobs = _ensure_jobs_shape(jobs)
    jobs_exploded = jobs.copy()

    # normalize nodes column to list and explode
    jobs_exploded['nodes'] = jobs_exploded['nodes'].map(parse_nodes)
    jobs_exploded = jobs_exploded.explode('nodes')
    jobs_exploded = jobs_exploded.rename(columns={'nodes': 'node_id'})

    # normalize submission time column name (keep yours, but ensure presence already handled)
    if 'submission_time' not in jobs_exploded.columns and 'subtime' in jobs_exploded.columns:
        jobs_exploded = jobs_exploded.rename(
            columns={'subtime': 'submission_time'})

    # ensure the columns you slice/merge on exist
    for c, dt in [('job_id', 'Int64'),
                  ('node_id', 'Int64'),
                  ('submission_time', 'Int64'),
                  ('start_time', 'Int64'),
                  ('finish_time', 'Int64'),
                  ('terminated', 'boolean')]:
        if c not in jobs_exploded.columns:
            jobs_exploded[c] = pd.Series(
                pd.NA, index=jobs_exploded.index, dtype=dt)
        else:
            try:
                jobs_exploded[c] = jobs_exploded[c].astype(dt)
            except Exception:
                pass

    jobs_exploded = jobs_exploded[[
        'job_id', 'node_id', 'submission_time', 'start_time', 'finish_time', 'terminated']]

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

    # --- non-active intervals ---
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy(
    )
    non_active_df['job_id'] = non_active_df['state'].map(
        mapping_non_active).fillna(-1)
    non_active_df['submission_time'] = pd.NA
    non_active_df['terminated'] = pd.NA

    # --- combine ---
    combined = pd.concat([active_merged, non_active_df], ignore_index=True)
    if not combined.empty:
        combined['node_id'] = combined['node_id'].astype('Int64')

    # --- group nodes into intervals ---
    if combined.empty:
        return pd.DataFrame(columns=['dvfs_mode', 'state', 'submission_time', 'start_time', 'finish_time', 'nodes', 'job_id', 'terminated'])

    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'submission_time',
            'start_time', 'finish_time', 'job_id'],
        dropna=False
    ).agg(
        nodes=('node_id', lambda x: ' '.join(
            map(str, sorted([int(i) for i in x.dropna().tolist()])))),
        terminated=('terminated', lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())
                    if s.notna().any() else pd.NA)
    ).reset_index()

    grouped = grouped.sort_values(by=['start_time', 'finish_time'])

    result = grouped[['dvfs_mode', 'state', 'submission_time',
                      'start_time', 'finish_time', 'nodes', 'job_id', 'terminated']]
    return result


def build_waiting_time_df(jobs_execution_log: list) -> pd.DataFrame:
    """
    Convert jobs_execution_log (list of dict) into a DataFrame with:
    job_id, subtime, start_time, finish_time, waiting_time (start_time - subtime).

    Handles both numeric timestamps and datetime-like strings.
    """
    df = pd.DataFrame(jobs_execution_log)
    required = {'job_id', 'subtime', 'start_time', 'finish_time'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    sub = df['subtime']
    start = df['start_time']

    if not (pd.api.types.is_numeric_dtype(sub) and pd.api.types.is_numeric_dtype(start)):
        sub_dt = pd.to_datetime(sub, errors='coerce')
        start_dt = pd.to_datetime(start, errors='coerce')
        waiting = (start_dt - sub_dt).dt.total_seconds()
    else:
        waiting = start - sub

    out = df.loc[:, ['job_id', 'subtime', 'start_time', 'finish_time']].copy()
    out['waiting_time'] = waiting
    return out


def write_waiting_time_log(simulator, output_folder: str, filename: str = "waiting_time_log.csv") -> str:
    """
    Build waiting-time DataFrame from simulator.Monitor.jobs_execution_log
    and write it to <output_folder>/<filename>. Returns the file path.
    """
    os.makedirs(output_folder, exist_ok=True)
    wt_df = build_waiting_time_df(simulator.Monitor.jobs_execution_log)
    path = os.path.join(output_folder, filename)
    wt_df.to_csv(path, index=False)
    return path


def build_energy_df(energy_log: list) -> pd.DataFrame:
    """
    Convert simulator.Monitor.energy (list[dict]) into a DataFrame with columns:
    id, energy_consumption, energy_effective, energy_waste.
    """
    df = pd.DataFrame(energy_log)
    required = {'id', 'energy_consumption', 'energy_effective', 'energy_waste'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns in energy log: {sorted(missing)}")

    out = df.loc[:, ['id', 'energy_consumption',
                     'energy_effective', 'energy_waste']].copy()

    # (Optional) coerce to numeric in case inputs are strings
    for col in ['energy_consumption', 'energy_effective', 'energy_waste']:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    return out


def write_energy_log(simulator, output_folder: str, filename: str = "energy_log.csv") -> str:
    """
    Build energy DataFrame from simulator.Monitor.energy and write it to CSV.
    Returns the file path.
    """
    os.makedirs(output_folder, exist_ok=True)
    energy_df = build_energy_df(simulator.Monitor.energy)
    path = os.path.join(output_folder, filename)
    energy_df.to_csv(path, index=False)
    return path


def build_metrics_df(jobs_execution_log: list, energy_log: list) -> pd.DataFrame:
    """
    Return a 1-row DataFrame with:
      total_waiting_time, total_energy_waste
    waiting_time is computed as start_time - subtime (seconds if datetimes).
    """
    # reuse existing builders
    wt_df = build_waiting_time_df(
        jobs_execution_log) if jobs_execution_log else pd.DataFrame(columns=['waiting_time'])
    en_df = build_energy_df(energy_log) if energy_log else pd.DataFrame(
        columns=['energy_waste'])

    total_waiting = pd.to_numeric(wt_df.get('waiting_time', pd.Series(
        dtype=float)), errors='coerce').sum(min_count=1)
    total_waste = pd.to_numeric(en_df.get('energy_waste', pd.Series(
        dtype=float)), errors='coerce').sum(min_count=1)

    # if no data or all NaN, make them 0.0
    if pd.isna(total_waiting):
        total_waiting = 0.0
    if pd.isna(total_waste):
        total_waste = 0.0

    return pd.DataFrame([{
        'total_waiting_time': float(total_waiting),
        'total_energy_waste': float(total_waste),
    }])


def write_metrics_log(simulator, output_folder: str, filename: str = "metrics.csv") -> str:
    """
    Build metrics DataFrame and write it to <output_folder>/<filename>.
    """
    os.makedirs(output_folder, exist_ok=True)
    metrics_df = build_metrics_df(
        simulator.Monitor.jobs_execution_log, simulator.Monitor.energy)
    path = os.path.join(output_folder, filename)
    metrics_df.to_csv(path, index=False)
    return path


def log_output(simulator, output_folder):
    os.makedirs(f'{output_folder}', exist_ok=True)

    raw_node_log = pd.DataFrame(simulator.Monitor.states_hist)
    raw_node_log.to_csv(f'{output_folder}/raw_node_log.csv', index=False)

    raw_job_log = pd.DataFrame(simulator.Monitor.jobs_execution_log)
    raw_job_log.to_csv(f'{output_folder}/raw_job_log.csv', index=False)

    write_waiting_time_log(simulator, output_folder)
    write_energy_log(simulator, output_folder)
    write_metrics_log(simulator, output_folder)

    node_log = process_node_job_data(
        simulator.Monitor.states_hist, raw_job_log)
    node_log.to_csv(f'{output_folder}/node_log.csv', index=False)
