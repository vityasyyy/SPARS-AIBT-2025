from .easy_auto_switch_on import EASYAuto


class Proactive(EASYAuto):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(
            new_state, waiting_queue, scheduled_queue, resources_agenda
        )
        super().FCFSAuto()
        super().backfill()
        if self.timeout_policy() None
        return self.events

    def proactive_timeout_policy(self):
        waiting_queue_demand = sum(job["res"] for job in self.waiting_queue)
        active_nodes = [n for n in self.state if n["state"] == "active"]
        active_nodes_count = len(active_nodes)

        surplus = active_nodes_count - waiting_queue_demand

        if surplus <= 0:
            return

        idle_nodes = [
            n for n in active_nodes if n["job_id"] is None and not n["reserved"]
        ]
        candidate_nodes_to_switch_off = idle_nodes[
            :surplus
        ]  # candidate to switch off un-needed nodes

        node_ids_to_switch_off = [n["id"] for n in candidate_nodes_to_switch_off]
        if node_ids_to_switch_off:
            self.push_event(
                self.current_time + self.timeout,
                {"type": "switch_off", "nodes": node_ids_to_switch_off},
            )
