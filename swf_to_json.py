import argparse
import json
from os import error

def parse_swf(input_path):
    jobs = []
    max_res = 0

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(';'):
                continue

            fields = line.split()
            if len(fields) < 18:
                error("Invalid line format: {}".format(line))
                continue 

            job_id = int(fields[0])
            subtime = int(fields[1])
            walltime = int(fields[3])
            req_procs = int(fields[4])
            user_id = int(fields[11])

            if req_procs == -1:
                req_procs = 0
            if walltime == -1:
                walltime = 0
                
            if req_procs > max_res:
                max_res = req_procs

            jobs.append({
                "job_id": job_id,
                "res": req_procs,
                "subtime": subtime,
                "walltime": walltime,
                "user_id": user_id
            })

    return max_res, jobs


def main():
    parser = argparse.ArgumentParser(description="Convert SWF 2.2 file to JSON job format.")
    parser.add_argument("--input", help="Path to input SWF file")
    parser.add_argument("--output", help="Path to output JSON file")
    args = parser.parse_args()

    max_res, jobs = parse_swf(args.input)

    result = {
        "nb_res": max_res,
        "jobs": jobs
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Converted {len(jobs)} jobs. Max resources requested: {max_res}")


if __name__ == "__main__":
    main()
