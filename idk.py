import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Data parsing (using pandas to read CSV)
data = pd.read_csv('results/sp/easy_nodes_t30.csv')

# Prepare data structures
all_nodes = set()
timeline = []

# Process data and find all unique nodes
for index, row in data.iterrows():
    nodes = list(map(int, row['allocated_resources'].split()))  # Adjust the column name as needed
    all_nodes.update(nodes)
    timeline.append({
        'starting_time': float(row['starting_time']),  # Adjust the column name as needed
        'finish_time': float(row['finish_time']),  # Adjust the column name as needed
        'allocated_resources': nodes,
        'type': row['type'],  # Adjust the column name as needed
        'job_id': int(row['job_id'])  # Adjust the column name as needed
    })

# Create figure
fig, ax = plt.subplots(figsize=(15, 10))
max_time = max([ev['finish_time'] for ev in timeline])
max_node = max(all_nodes)

# Color mapping
colors = {
    -1: 'gray',
    -2: 'red',
    -3: 'green',
    -4: 'lightblue'
}

# Plot each event
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
        color = colors.get(event['job_id'], np.random.rand(3,))
        ax.broken_barh([(event['starting_time'], event['finish_time'] - event['starting_time'])], 
                      (y, height), facecolors=color, edgecolor='black')
        
        if event['job_id'] > 0:
            ax.text((event['starting_time'] + event['finish_time'])/2, y + height/2, 
                    str(event['job_id']), ha='center', va='center',
                    color='white' if np.mean(colors.get(event['job_id'], [0,0,0])) < 0.5 else 'black')

# Configure plot
ax.set_xlabel('Time')
ax.set_ylabel('allocated_resources')
ax.set_yticks(np.arange(max_node + 1) + 0.5)
ax.set_yticklabels(np.arange(max_node + 1))
ax.set_xlim(0, max_time)
ax.grid(True)

# Create legend
legend_patches = [
    mpatches.Patch(color='gray', label='Idle (-1)'),
    mpatches.Patch(color='red', label='Switching Off (-2)'),
    mpatches.Patch(color='green', label='Switching On (-3)'),
    mpatches.Patch(color='lightblue', label='Sleeping (-4)'),
    mpatches.Patch(color='blue', label='Computing Jobs')
]
ax.legend(handles=legend_patches, loc='upper right')

plt.title('Machine Scheduling Gantt Chart')
plt.tight_layout()
plt.show()
