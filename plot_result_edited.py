import csv
import ast
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot dual/triple Gantt chart for HPCv2, BatsimPy, and optionally Batsched.")
    parser.add_argument('--hpcv2', required=True, help="CSV file for HPCv2 schedule (NSV2 format)")
    parser.add_argument('--batsimpy', required=True, help="CSV file for BatsimPy schedule (NSV2 format)")
    parser.add_argument('--batsched', required=False, help="CSV file for Batsched schedule (NSV2 format, optional)")
    parser.add_argument('--output', default=None, help="Output PNG file path (default: plt/val/dual_gantt_t{timeout}.png)")
    parser.add_argument('--timeout', type=int, default=30, help="Timeout value for labeling (default: 30)")
    return parser.parse_args()

def plot_timeline(ax, timeline, title, xticks, max_time):
    job_colors = {}
    has_submission = any('submission_time' in event for event in timeline)
    for event in timeline:
        nodes = sorted(event['allocated_resources'])
        groups = []
        while nodes:
            start = nodes[0]
            end = start
            while nodes and nodes[0] == end:
                end += 1
                nodes.pop(0)
            groups.append((start, end - start))

        for y, height in groups:
            if event['job_id'] in colors:
                color = colors[event['job_id']]
            else:
                if event['job_id'] not in job_colors:
                    job_colors[event['job_id']] = np.random.rand(3,)
                color = job_colors[event['job_id']]

            ax.broken_barh([(event['starting_time'], event['finish_time'] - event['starting_time'])],
                           (y, height), facecolors=color)

            if event['job_id'] > 0:
                ax.text((event['starting_time'] + event['finish_time']) / 2, y + height / 2,
                        str(event['job_id']), ha='center', va='center',
                        color='white' if np.mean(color) < 0.5 else 'black')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90, fontsize=8)
    ax.set_ylabel('Nodes')
    if has_submission:
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(10))
    else:
        ax.set_yticks(range(8))
        ax.set_yticklabels(range(8))
    ax.set_xlim(0, max_time)
    ax.grid(True)
    ax.set_title(title)

def read_timeline(file_path):
    data = pd.read_csv(file_path)
    timeline = []
    has_submission = 'submission_time' in data.columns
    for _, row in data.iterrows():
        nodes = list(map(int, row['allocated_resources'].split()))
        entry = {
            'starting_time': float(row['starting_time']),
            'finish_time': float(row['finish_time']),
            'allocated_resources': nodes,
            'type': row['type'],
            'job_id': int(row['job_id'])
        }
        if has_submission:
            entry['submission_time'] = float(row['submission_time'])
        timeline.append(entry)
    return timeline

def main():
    args = parse_arguments()

    timeline_hpcv2 = read_timeline(args.hpcv2)
    timeline_batsim = read_timeline(args.batsimpy)

    if args.batsched:
        timeline_batsched = read_timeline(args.batsched)
        max_time = max(
            max([ev['finish_time'] for ev in timeline_hpcv2]),
            max([ev['finish_time'] for ev in timeline_batsim]),
            max([ev['finish_time'] for ev in timeline_batsched])
        )
    else:
        timeline_batsched = None
        max_time = max(
            max([ev['finish_time'] for ev in timeline_hpcv2]),
            max([ev['finish_time'] for ev in timeline_batsim])
        )

    xticks_hpcv2 = sorted(set([e['starting_time'] for e in timeline_hpcv2] + [e['finish_time'] for e in timeline_hpcv2]))
    xticks_batsim = sorted(set([e['starting_time'] for e in timeline_batsim] + [e['finish_time'] for e in timeline_batsim]))

    if timeline_batsched:
        xticks_batsched = sorted(set([e['starting_time'] for e in timeline_batsched] + [e['finish_time'] for e in timeline_batsched]))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(150, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(150, 7))

    plot_timeline(ax1, timeline_hpcv2, "HPCv2 Schedule", xticks_hpcv2, max_time)
    plot_timeline(ax2, timeline_batsim, "Batsimpy Schedule", xticks_batsim, max_time)

    if timeline_batsched:
        plot_timeline(ax3, timeline_batsched, "Batsched Schedule", xticks_batsched, max_time)
        ax3.set_xlabel('Time')
    else:
        ax2.set_xlabel('Time')

    for event in timeline_hpcv2:
        if event['job_id'] > 0:
            ax1.plot(event['submission_time'], 9, 'ro')
            ax1.text(event['submission_time'], 8.5, str(event['job_id']), ha='center', va='center', color='red')
        
    legend_patches = [
        mpatches.Patch(color='gray', label='Idle (-1)'),
        mpatches.Patch(color='red', label='Switching Off (-2)'),
        mpatches.Patch(color='green', label='Switching On (-3)'),
        mpatches.Patch(color='lightblue', label='Sleeping (-4)'),
        mpatches.Patch(color='blue', label='Computing Jobs')
    ]
    ax1.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()

    output_path = args.output if args.output else f"plt/val/comparison_gantt_t{args.timeout}.png"
    plt.savefig(output_path)
    print(f"Gantt chart saved to: {output_path}")


# Global color mapping
colors = {
    -1: 'gray',
    -2: 'red',
    -3: 'green',
    -4: 'lightblue'
}

if __name__ == "__main__":
    main()
