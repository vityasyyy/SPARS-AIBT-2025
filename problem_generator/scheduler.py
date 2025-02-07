import heapq


class Scheduler:
    def __init__(self, total_nodes=4):
        self.total_nodes = total_nodes

    def generate_schedule(self, interarrival_times, arrival_times,
                          requested_execution_times, actual_execution_times,
                          num_nodes_required):
        # Urutkan job berdasarkan waktu kedatangan (FCFS)
        jobs = sorted([
            {
                'arrival': arrival,
                'requested_execution': req_exec,
                'actual_execution': act_exec,
                'nodes': nodes,
                'scheduled': False
            } for arrival, req_exec, act_exec, nodes in zip(
                arrival_times,
                requested_execution_times,
                actual_execution_times,
                num_nodes_required
            )
        ], key=lambda x: x['arrival'])

        for i, job in enumerate(jobs, start=1):
            job['id'] = i

        node_heap = [0.0] * self.total_nodes
        heapq.heapify(node_heap)
        schedule_results = []
        current_time = 0
        index = 0

        while index < len(jobs):
            if jobs[index]['scheduled']:
                index += 1
                continue

            job = jobs[index]
            required_nodes = job['nodes']

            # Cari waktu mulai terdekat untuk job ini
            node_release_times = heapq.nsmallest(required_nodes, node_heap)
            start_time = max(job['arrival'], max(node_release_times))

            # Backfilling: Cari job yang bisa diisi di gap sebelum start_time
            if start_time > current_time:
                gap_duration = start_time - current_time

                # Cari job yang bisa masuk ke gap
                for bf_idx in range(index + 1, len(jobs)):
                    bf_job = jobs[bf_idx]

                    if (not bf_job['scheduled'] and
                        bf_job['arrival'] <= current_time and
                        bf_job['actual_execution'] <= gap_duration and
                            bf_job['nodes'] <= self.total_nodes):

                        # Cek ketersediaan node untuk backfill job
                        bf_node_release = heapq.nsmallest(
                            bf_job['nodes'], node_heap)
                        bf_start = max(bf_job['arrival'], max(bf_node_release))

                        if bf_start <= current_time:
                            # Jadwalkan backfill job
                            for _ in range(bf_job['nodes']):
                                heapq.heappop(node_heap)
                            for _ in range(bf_job['nodes']):
                                heapq.heappush(
                                    node_heap, current_time + bf_job['actual_execution'])

                            schedule_results.append({
                                **bf_job,
                                'start': current_time,
                                'actual_finish': current_time + bf_job['actual_execution'],
                                'expected_finish': current_time + bf_job['requested_execution'],
                                'waiting': current_time - bf_job['arrival']
                            })
                            bf_job['scheduled'] = True
                            break

            # Jadwalkan job utama
            for _ in range(required_nodes):
                heapq.heappop(node_heap)
            for _ in range(required_nodes):
                heapq.heappush(node_heap, start_time + job['actual_execution'])

            schedule_results.append({
                **job,
                'start': start_time,
                'actual_finish': start_time + job['actual_execution'],
                'expected_finish': start_time + job['requested_execution'],
                'waiting': start_time - job['arrival']
            })

            current_time = max(current_time, start_time)
            job['scheduled'] = True
            index += 1

        return sorted(schedule_results, key=lambda x: x['start'])
