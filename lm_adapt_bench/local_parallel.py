import os
import sys
import subprocess
import argparse
import time
import logging
from typing import List

def main():
    parser = argparse.ArgumentParser(description="Entropy Bench Local Parallel Launcher")
    parser.add_argument("--gpus", required=True, help="Comma-separated list of GPU IDs (e.g., 0,1,2,3)")
    parser.add_argument("--cli-module", default="lm_adapt_bench.cli", help="Module to run")
    
    # We'll capture all other unknown args to pass to the cli
    args, unknown = parser.parse_known_args()
    
    gpu_ids = args.gpus.split(",")
    world_size = len(gpu_ids)
    
    processes = []
    
    # Setup logging for the launcher
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [Launcher] %(levelname)s: %(message)s')
    logger = logging.getLogger("launcher")
    
    logger.info(f"Launching {world_size} processes on GPUs: {gpu_ids}")
    
    for rank, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        # Construct the command
        cmd = [
            sys.executable, "-m", args.cli_module,
            "--rank", str(rank),
            "--world-size", str(world_size),
        ] + unknown
        
        logger.info(f"Rank {rank} (GPU {gpu_id}): {' '.join(cmd)}")
        
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        
        # Stagger start slightly
        time.sleep(2)

    try:
        # Wait for all processes to complete
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        logger.warning("Launcher interrupted, killing processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        logger.info("All processes terminated.")

    logger.info("Local parallel run completed.")

if __name__ == "__main__":
    main()
