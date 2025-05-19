
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['WANDB_MODE']='disabled'


import argparse
import os
import subprocess
import sys

import yaml

GPUS_PER_NODE = 8


def dict2args(d):
    args = []
    for k, v in d.items():
        args.append(f"--{k}")
        if isinstance(v, list):
            for x in v:
                args.append(str(x))
        else:
            args.append(str(v))
    return args


def rm_torchrun_task(nodes, config, training_code):
    try:
        distribute_config = {
            "nproc_per_node": GPUS_PER_NODE,
            "nnodes": nodes,
            "node_rank": int(os.environ['RANK']),
            "master_addr": os.environ['MASTER_ADDR'],
            "master_port": int(os.environ['MASTER_PORT'])
        }
    except:
        distribute_config = {
            "nproc_per_node": GPUS_PER_NODE,
            "nnodes": nodes,
            "node_rank": 0,
            "master_addr": "127.0.0.1",
            "master_port": 6000
        }
    command = [
        "torchrun"
    ] + dict2args(distribute_config) + [training_code] + dict2args(config)
    print(" ".join(command))
    subprocess.run(command)


def main():
    parser = argparse.ArgumentParser("Launch a Reward Modeling experiment")
    parser.add_argument("-c", "--config", required=True, help="Configuration YAML")
    parser.add_argument("-d", "--working", required=True, help="Working directory")
    parser.add_argument("-f", "--file", default='train/InternVL-2/run_rm.py', help="training code path")
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Launch through slurm using the given number of GPUs",
    )
    args = parser.parse_args()

    os.makedirs(args.working, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f.read())

    config["output_dir"] = args.working

    rm_torchrun_task(args.gpus // GPUS_PER_NODE, config, args.file)



if __name__ == "__main__":
    main()
