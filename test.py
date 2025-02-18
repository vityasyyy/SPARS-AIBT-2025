import pandas as pd

data = [
    {
        "time": 0.0,
        "sleeping": 0,
        "sleeping_nodes": [],
        "switching_on": 0,
        "switching_on_nodes": [],
        "switching_off": 0,
        "switching_off_nodes": [],
        "idle": 16,
        "idle_nodes": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ],
        "computing": 0,
        "computing_nodes": [
            {
                "job_id": 1,
                "nodes": [0, 1, 2, 3],
                "starting_time": 2.0,
                "finish_time": 151.0
            },
            {
                "job_id": 2,
                "nodes": [4, 5, 6, 7, 8],
                "starting_time": 3.0,
                "finish_time": 153.0
            },
            {
                "job_id": 3,
                "nodes": [9, 10, 11, 12],
                "starting_time": 4.0,
                "finish_time": 154.0
            },
            {
                "job_id": 11,
                "nodes": [13, 14],
                "starting_time": 21.0,
                "finish_time": 171.0
            }
        ],
        "unavailable": 0
    },
    {
        "time": 2.0,
        "sleeping": 0,
        "sleeping_nodes": [],
        "switching_on": 0,
        "switching_on_nodes": [],
        "switching_off": 0,
        "switching_off_nodes": [],
        "idle": 12,
        "idle_nodes": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "computing": 4,
        "computing_nodes": [
            {
                "job_id": 1,
                "nodes": [0, 1, 2, 3],
                "starting_time": 2.0,
                "finish_time": 151.0
            },
            {
                "job_id": 2,
                "nodes": [4, 5, 6, 7, 8],
                "starting_time": 3.0,
                "finish_time": 153.0
            },
            {
                "job_id": 3,
                "nodes": [9, 10, 11, 12],
                "starting_time": 4.0,
                "finish_time": 154.0
            },
            {
                "job_id": 11,
                "nodes": [13, 14],
                "starting_time": 21.0,
                "finish_time": 171.0
            }
        ],
        "unavailable": 0
    }
]

# Function to convert the given data to the required format
def convert_data(data):
    result = []
    prev_idle_nodes = []
    prev_idle_time = 0

    for i, record in enumerate(data):
        # Process idle nodes
        if record["idle"] > 0:
            for node in record["idle_nodes"]:
                # Check if node is idle and update idle time until job starts
                if node not in prev_idle_nodes:
                    result.append({
                        "starting_time": prev_idle_time,
                        "finish_time": record["time"],
                        "nodes": [node],
                        "type": "idle",
                        "job_id": -1
                    })
            prev_idle_nodes = record["idle_nodes"]
            prev_idle_time = record["time"]
        
        # Process computing nodes (jobs)
        if record["computing"] > 0:
            for job in record["computing_nodes"]:
                for node in job["nodes"]:
                    # Update idle time for nodes that are starting a job
                    result.append({
                        "starting_time": prev_idle_time,
                        "finish_time": job["starting_time"],
                        "nodes": [node],
                        "type": "idle",
                        "job_id": -1
                    })
                    
                    result.append({
                        "starting_time": job["starting_time"],
                        "finish_time": job["finish_time"],
                        "nodes": job["nodes"],
                        "type": "computing",
                        "job_id": job["job_id"]
                    })

        # Handle switching on or off if required
        if record["switching_on"] > 0 or record["switching_off"] > 0:
            for node in record["switching_on_nodes"]:
                result.append({
                    "starting_time": record["time"],
                    "finish_time": record["time"] + 1,
                    "nodes": [node],
                    "type": "switching_on",
                    "job_id": -1
                })
            for node in record["switching_off_nodes"]:
                result.append({
                    "starting_time": record["time"],
                    "finish_time": record["time"] + 1,
                    "nodes": [node],
                    "type": "switching_off",
                    "job_id": -1
                })

    return result

# Generate the final processed data
converted_data = convert_data(data)
converted_data = pd.DataFrame(converted_data)

# Save the data to CSV
converted_data.to_csv('results/sp/processed_data.csv', index=False)

# Print the results for inspection
print(converted_data)
