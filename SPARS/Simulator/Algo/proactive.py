from .easy_auto_switch_on import EASYAuto
from collections import deque
from statistics import pstdev
import math


class Proactive(EASYAuto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # GLOBAL sliding window of recent cluster demand (jobs waiting + scheduled)
        self.global_demand_window = deque(maxlen=40)

        # Per-node state for adaptive timeout
        # node_timeouts[node_id] = {
        #   "demand_window": deque(0/1),
        #   "current_timeout": float,
        #   "last_active_time": float
        # }
        self.node_timeouts = {}

        # Tunables (tweak these)
        self.global_window_size = 40
        self.node_window_size = 20
        self.base_timeout = getattr(self, "timeout", 60) or 60  # fallback
        self.min_timeout = 10
        self.max_timeout = 600
        self.node_history_min = (
            5  # need this many points before trusting node predictor
        )
        self.global_history_min = (
            6  # need this many points before trusting global predictor
        )

        # weighting between node-local and global signals
        self.node_weight = 0.6
        self.global_weight = 0.4

        # smoothing for timeout updates (0..1 old weight)
        self.timeout_smooth = 0.85

        # prediction horizon (how many steps ahead to project)
        self.prediction_horizon = 3

    # ---------------------------
    # schedule entrypoint (hook)
    # ---------------------------
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        # 1 - standard setup
        super().prep_schedule(
            new_state, waiting_queue, scheduled_queue, resources_agenda
        )

        # 2 - run FCFS/backfill etc.
        self.FCFSAuto()
        self.backfill()

        # 3 - update demand history and run node-level adaptive timeout (which uses global predictor)
        self.update_global_demand_history()
        self.update_node_histories()
        if getattr(self, "timeout", None) is not None:
            self.proactive_timeout_policy_per_node_with_global()

        return self.events

    # ---------------------------
    # helpers: prediction functions
    # ---------------------------
    def _linreg_slope(self, window):
        """Return slope (per-step) computed with simple least-squares. If variance 0, slope = 0."""
        n = len(window)
        if n < 2:
            return 0.0
        # compute means
        mean_t = (n - 1) / 2.0
        mean_y = sum(window) / n
        # numerator and denominator
        num = 0.0
        den = 0.0
        for i, y in enumerate(window):
            dt = i - mean_t
            num += dt * (y - mean_y)
            den += dt * dt
        if den == 0.0:
            return 0.0
        return num / den

    def _predict_from_window(self, window, horizon=1):
        """
        Predict `horizon` steps ahead from a 1-D numeric window using mean + slope*horizon.
        Returns a float (can be >1 for global demand); caller should clamp as needed.
        """
        if not window:
            return 0.0
        mean_y = sum(window) / len(window)
        slope = self._linreg_slope(window)
        return mean_y + slope * horizon

    # ---------------------------
    # history updaters
    # ---------------------------
    def update_global_demand_history(self):
        """
        Compute a numeric 'current demand' representing how many nodes are requested
        by waiting/scheduled jobs and append to global window.
        """
        # Note: adapt this to your queue semantics. Using waiting_queue + scheduled_queue is safer.
        waiting_sum = sum(
            job.get("res", 0) for job in getattr(self, "waiting_queue", [])
        )
        scheduled_sum = sum(
            job.get("res", 0) for job in getattr(self, "scheduled_queue", [])
        )
        current_demand = waiting_sum + scheduled_sum

        # append
        if not hasattr(self, "global_demand_window"):
            self.global_demand_window = deque(maxlen=self.global_window_size)
        self.global_demand_window.append(current_demand)

    def update_node_histories(self):
        """
        Ensure each node has an entry in node_timeouts and append busy/idle signal (1/0).
        """
        now = self.current_time
        for node in getattr(self, "state", []):
            nid = node["id"]
            if nid not in self.node_timeouts:
                self.node_timeouts[nid] = {
                    "demand_window": deque(maxlen=self.node_window_size),
                    "current_timeout": float(self.base_timeout),
                    "last_active_time": now,
                }

            info = self.node_timeouts[nid]
            is_busy = node.get("job_id") is not None
            info["demand_window"].append(1 if is_busy else 0)
            if is_busy:
                info["last_active_time"] = now

    # ---------------------------
    # main algorithm
    # ---------------------------
    def proactive_timeout_policy_per_node_with_global(self):
        """
        Core algorithm:
        - Predict short-term global demand (jobs -> node fraction).
        - For each idle & active node, predict node-local utilization.
        - Combine node-local and global predictions to compute an adaptive timeout for that node.
        - Manage timers (self.timeout_list): add timers for candidates, remove timers for nodes that are no longer candidates,
          fire immediate shutdowns when timers already expired.
        """
        now = self.current_time
        state_by_id = {n["id"]: n for n in getattr(self, "state", [])}
        total_nodes = max(1, len(state_by_id))

        # 1) Global predicted demand (#nodes required) and normalized global utilization (0..1+)
        global_window = list(getattr(self, "global_demand_window", []))
        if len(global_window) >= self.global_history_min:
            predicted_global_demand = self._predict_from_window(
                global_window, horizon=self.prediction_horizon
            )
        else:
            predicted_global_demand = (
                sum(global_window) / max(1, len(global_window))
                if global_window
                else 0.0
            )

        # normalize to cluster capacity => fraction [0, inf); we'll clamp later
        predicted_global_util = predicted_global_demand / total_nodes
        # keep it reasonable
        predicted_global_util = max(0.0, predicted_global_util)

        # 2) Build candidate idle nodes: active, no job, not reserved
        active_nodes = [
            n for n in getattr(self, "state", []) if n.get("state") == "active"
        ]
        idle_candidates = [
            n
            for n in active_nodes
            if n.get("job_id") is None and not n.get("reserved", False)
        ]

        # prepare timeout bookkeeping
        existing_timeouts = {
            t["node_id"]: t["time"] for t in getattr(self, "timeout_list", [])
        }
        new_timeout_list = []
        next_earliest = None
        switch_off = []

        for node in idle_candidates:
            nid = node["id"]
            info = self.node_timeouts.setdefault(
                nid,
                {
                    "demand_window": deque(maxlen=self.node_window_size),
                    "current_timeout": float(self.base_timeout),
                    "last_active_time": now,
                },
            )

            # Node-local predicted utilization (0..1) based on its 0/1 window
            node_window = list(info["demand_window"])
            if len(node_window) >= self.node_history_min:
                # predict fraction of time busy in near future
                predicted_node_util = self._predict_from_window(
                    node_window, horizon=self.prediction_horizon
                )
            else:
                predicted_node_util = (
                    sum(node_window) / max(1, len(node_window)) if node_window else 0.0
                )

            # clamp node util to [0,1]
            predicted_node_util = min(max(predicted_node_util, 0.0), 1.0)

            # 3) Combine node-local + global signal
            combined_signal = (self.node_weight * predicted_node_util) + (
                self.global_weight * predicted_global_util
            )
            # clamp combined roughly to [0, 3] so scaling below stays reasonable
            combined_signal = max(0.0, min(combined_signal, 3.0))

            # 4) Map combined signal -> timeout factor
            # design: low combined -> aggressive shutdown; high combined -> keep node longer
            if combined_signal <= 0.05:
                factor = 0.4
            elif combined_signal < 0.2:
                factor = 0.7
            elif combined_signal < 0.5:
                factor = 1.0
            else:
                factor = min(3.0, 1.0 + combined_signal * 1.5)

            new_timeout = self.base_timeout * factor

            # 5) Smooth update to avoid jitter
            smoothed = (self.timeout_smooth * info["current_timeout"]) + (
                (1.0 - self.timeout_smooth) * new_timeout
            )
            node_timeout = max(self.min_timeout, min(smoothed, self.max_timeout))
            info["current_timeout"] = node_timeout

            # 6) Compute expiry time (use last_active_time so timeout counts from last job end)
            t_exp = info["last_active_time"] + node_timeout

            # If timer already existed for this node, update or reuse; if expired -> immediate switch
            prev_t = existing_timeouts.get(nid)
            if t_exp <= now:
                # already past expiry (long idle) -> immediate shutdown
                switch_off.append(nid)
            else:
                new_timeout_list.append({"node_id": nid, "time": t_exp})
                if next_earliest is None or t_exp < next_earliest:
                    next_earliest = t_exp

        # 7) replace timeout_list with the new timers only for current candidates
        self.timeout_list = new_timeout_list

        # 8) Fire off switch_off if any
        if switch_off:
            # dedupe just in case
            switch_off = list(dict.fromkeys(switch_off))
            self.push_event(now, {"type": "switch_off", "nodes": switch_off})

        # 9) schedule wakeup for next earliest timer if necessary
        if (
            next_earliest is not None
            and getattr(self, "next_timeout_at", None) != next_earliest
        ):
            self.push_event(next_earliest, {"type": "call_me_later"})
            self.next_timeout_at = next_earliest

        # Debugging output (optional — remove in production)
        print(
            f"[Adaptive] global_pred={predicted_global_demand:.2f} ({predicted_global_util:.2f}), "
            f"idle_candidates={len(idle_candidates)}, next={next_earliest}, switch_off={switch_off}"
        )
