import csv
from collections import defaultdict
import argparse

def parse_machine_range(machine_range):
    machines = []
    parts = machine_range.strip().split()
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            machines.extend(range(start, end + 1))
        else:
            machines.append(int(part))
    return machines


def parse_nodes(resource_str):
    return parse_machine_range(resource_str)


def read_machine_states(file_path):
    events = defaultdict(list)
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            time = int(row[0])
            machine_range = row[1]
            new_state = int(row[2])
            machines = parse_machine_range(machine_range)
            events[time].append((machines, new_state))
    return events



def read_jobs(file_path):
    jobs = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                job_id = int(row[0])
            except ValueError:
                print(f"Skipping row with invalid job_id: {row}")
                continue
            sub_time = int(float(row[3]))
            start_time = int(float(row[8]))
            finish_time = int(float(row[10]))
            nodes = parse_machine_range(row[14])
            jobs.append({
                'job_id': job_id,
                'submission_time': sub_time,
                'starting_time': start_time,
                'finish_time': finish_time,
                'nodes': nodes
            })
    return jobs



def simulate(machine_events, jobs):
    all_times = set(machine_events.keys())
    for job in jobs:
        all_times.add(job['starting_time'])
        all_times.add(job['finish_time'])

    timeline = sorted(all_times)
    state = {} 
    results = []

    for current_time in timeline:
        switching_on = []
        switching_off = []
        sleeping = []
        idle = []
        computing = []

        if current_time in machine_events:
            for machines, new_state in machine_events[current_time]:
                for m in machines:
                    state[m] = new_state

        all_machine_ids = set()
        for event_list in machine_events.values():
            for machines, _ in event_list:
                all_machine_ids.update(machines)


        for job in jobs:
            all_machine_ids.update(job['nodes'])

        active_jobs = []
        for job in jobs:
            if job['starting_time'] <= current_time < job['finish_time']:
                active_jobs.append({
                    'job_id': job['job_id'],
                    'starting_time': job['starting_time'],
                    'finish_time': job['finish_time'],
                    'nodes': job['nodes']
                })

        node_to_job = {}
        for job in active_jobs:
            for n in job['nodes']:
                node_to_job[n] = job

        all_nodes = sorted(all_machine_ids)
        for m in all_nodes:
            s = state.get(m, 1)
            if s == -2:
                switching_off.append(m)
            elif s == -1:
                switching_on.append(m)
            elif s == 0:
                sleeping.append(m)
            elif s == 1:
                if m in node_to_job:
                    computing.append(m)
                else:
                    idle.append(m)

        computing_details = []
        for job in active_jobs:
            computing_details.append({
                'job_id': job['job_id'],
                'starting_time': job['starting_time'],
                'finish_time': job['finish_time'],
                'nodes': job['nodes']
            })

        results.append({
            'time': current_time,
            'sleeping': len(sleeping),
            'sleeping_nodes': sleeping,
            'switching_on': len(switching_on),
            'switching_on_nodes': switching_on,
            'switching_off': len(switching_off),
            'switching_off_nodes': switching_off,
            'idle': len(idle),
            'idle_nodes': idle,
            'computing': len(computing),
            'computing_nodes': computing_details
        })

    return results

def write_output(results, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'time',
            'sleeping', 'sleeping_nodes',
            'switching_on', 'switching_on_nodes',
            'switching_off', 'switching_off_nodes',
            'idle', 'idle_nodes',
            'computing', 'computing_nodes'
        ]
        writer.writerow(header)
        for row in results:
            writer.writerow([
                row['time'],
                row['sleeping'], row['sleeping_nodes'],
                row['switching_on'], row['switching_on_nodes'],
                row['switching_off'], row['switching_off_nodes'],
                row['idle'], row['idle_nodes'],
                row['computing'], row['computing_nodes']
            ])

def parse_args():
    parser = argparse.ArgumentParser(description="Simulasi status node dari machine events dan job file.")
    parser.add_argument("--machine_events", "-m", required=True, help="Path ke file machine events (CSV)")
    parser.add_argument("--jobs", "-j", required=True, help="Path ke file jobs (CSV)")
    parser.add_argument("--result", "-r", required=True, help="Path ke output file CSV")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    machine_events = read_machine_states(args.machine_events)
    jobs = read_jobs(args.jobs)
    results = simulate(machine_events, jobs)
    write_output(results, args.result)