import csv
import ast
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert NSV1 CSV to NSV2 CSV format.")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV (NSV1 format)")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV (NSV2 format)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    # Step 1: Read input CSV
    input_data = []
    with open(INPUT_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_data.append(row)

    # Step 2: Track node events
    node_timelines = defaultdict(list)

    for i, row in enumerate(input_data):
        time = int(row['time'])
        next_time = int(input_data[i + 1]['time']) if i + 1 < len(input_data) else time

        # Handle states: idle, sleeping, switching_on, switching_off
        for state_name, job_id in [
            ('idle_nodes', -1),
            ('sleeping_nodes', -4),
            ('switching_on_nodes', -3),
            ('switching_off_nodes', -2)
        ]:
            nodes = ast.literal_eval(row[state_name])
            for node in nodes:
                node_timelines[node].append((time, next_time, job_id))

        # Handle computing jobs
        if row['computing'] != '0':
            jobs = ast.literal_eval(row['computing_nodes'])
            for job in jobs:
                job_id = job['job_id']
                start = job['starting_time']
                end = job['finish_time']
                for node in job['nodes']:
                    node_timelines[node].append((start, end, job_id))

    # Step 3: Combine rows with same state, time, and job_id
    combined_rows = []
    visited = set()

    for node, events in node_timelines.items():
        for event in events:
            key = (event[0], event[1], event[2])
            if key not in visited:
                group_nodes = [
                    n for n, evs in node_timelines.items()
                    if key in evs
                ]
                visited.add(key)
                combined_rows.append({
                    'type': 'computing' if key[2] > 0 else {
                        -1: 'idle',
                        -2: 'switching_off',
                        -3: 'switching_on',
                        -4: 'sleeping'
                    }[key[2]],
                    'starting_time': key[0],
                    'finish_time': key[1],
                    'allocated_resources': ' '.join(map(str, sorted(group_nodes))),
                    'job_id': key[2]
                })

    # Step 4: Sort and write to CSV
    combined_rows.sort(key=lambda x: (x['starting_time'], int(x['allocated_resources'].split()[0])))

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'type', 'starting_time', 'finish_time', 'allocated_resources', 'job_id'
        ])
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Output written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()