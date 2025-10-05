from .easy_auto_switch_on import EASYAuto


class Proactive(EASYAuto):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        """
        The main scheduling loop.
        """
        # 1. Standard setup from the parent class
        super().prep_schedule(
            new_state, waiting_queue, scheduled_queue, resources_agenda
        )

        # 2. Run the FCFS and backfilling logic as usual
        self.FCFSAuto()
        self.backfill()

        # 3. Call our new, overridden, and smarter timeout policy
        if self.timeout is not None:
            self.timeout_policy()

        return self.events

    def timeout_policy(self):
        self.proactive_timeout_policy()

    def proactive_timeout_policy(self):
        """
        This is the new, smarter timeout policy that OVERRIDES the one in BaseAlgorithm.
        It combines queue-awareness with the original's safe, stateful machinery.
        """
        # --- START: Your Proactive Queue-Aware Logic ---
        # 1. Calculate the real-time demand from jobs currently waiting.
        waiting_queue_demand = sum(job["res"] for job in self.scheduled_queue)
        # 2. Count all nodes that are currently in the 'active' state.
        active_nodes = [n for n in self.state if n["state"] == "active"]
        active_nodes_count = len(active_nodes)

        # 3. Determine the true surplus of nodes.
        surplus = active_nodes_count - waiting_queue_demand

        # If there is no surplus, we should not shut down ANY nodes.
        if surplus <= 0:
            print(
                f"DEBUG: active_nodes_count={active_nodes_count}, waiting_queue_demand={waiting_queue_demand}"
            )
            print("Waiting queue:", self.scheduled_queue)
            print("No surplus, skipping timeout processing.")
            # Clear any existing timers because the situation has changed.
            self.timeout_list = []
            return

        # 4. Identify which specific idle nodes are our safe-to-shutdown candidates.
        print("DEBUG: Surplus detected:", surplus)
        idle_nodes = [
            n for n in active_nodes if n["job_id"] is None and not n["reserved"]
        ]
        candidate_nodes_to_switch_off = idle_nodes[:surplus]
        candidate_ids = {n["id"] for n in candidate_nodes_to_switch_off}
        # --- END: Your Proactive Queue-Aware Logic ---

        # --- START: Original Stateful Machinery (Now Guided by Our Logic) ---
        now = self.current_time
        t_exp = now + self.timeout

        timeout_node_ids = {t["node_id"] for t in self.timeout_list}

        # **MODIFIED STEP**: Only add new timers for nodes that are in our safe candidate list.
        for node in candidate_nodes_to_switch_off:
            if node["id"] not in timeout_node_ids:
                self.timeout_list.append({"node_id": node["id"], "time": t_exp})
                timeout_node_ids.add(node["id"])

        # This section safely manages the lifecycle of the timers.
        state_by_id = {n["id"]: n for n in self.state}
        switch_off = []
        keep_timeouts = []
        next_earliest = None

        for t in self.timeout_list:
            node = state_by_id.get(t["node_id"])

            # **MODIFIED STEP**: If a node is no longer a candidate, remove its timer.
            if not node or node["id"] not in candidate_ids:
                continue

            # If the timer has not expired, keep it.
            if now < t["time"] or node["state"] != "active":
                keep_timeouts.append(t)
                if next_earliest is None or t["time"] < next_earliest:
                    next_earliest = t["time"]
                continue

            # If the timer has expired, schedule the node for shutdown.
            if node["job_id"] is None:
                switch_off.append(node["id"])

        self.timeout_list = keep_timeouts

        if switch_off:
            self.push_event(now, {"type": "switch_off", "nodes": switch_off})

        if (
            next_earliest is not None
            and getattr(self, "next_timeout_at", None) != next_earliest
        ):
            self.push_event(next_earliest, {"type": "call_me_later"})
            self.next_timeout_at = next_earliest
