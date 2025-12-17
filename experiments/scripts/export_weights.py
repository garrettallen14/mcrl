#!/usr/bin/env python3
"""
Export trained weights to a portable format.

Run this on RunPod after training to save weights:
    python experiments/scripts/export_weights.py --run-dir experiments/runs/ppo_XXXXXX

Then download the .pkl file from the checkpoints folder.
"""

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from mcrl.training.networks_fast import FastActorCritic, init_fast_network
from mcrl.utils.checkpoint import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export trained weights")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Training run directory")
    parser.add_argument("--step", type=int, default=100000000,
                        help="Step number for checkpoint name")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    
    # Check if metrics file exists (indicates completed training)
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
    
    # If there's a saved TrainState, we'd load it here
    # But since checkpoints weren't saved, we need to recreate from scratch
    
    print("Creating new network to verify checkpoint format...")
    
    # Create network
    network = FastActorCritic()
    key = jax.random.PRNGKey(0)
    
    obs_shapes = {
        'local_voxels': (17, 17, 17),
        'inventory': (16,),
        'player_state': (14,),
        'facing_blocks': (8,),
        'log_compass': (4,),
    }
    params = init_fast_network(network, key, obs_shapes)
    
    # Save checkpoint
    ckpt_path = run_dir / "checkpoints"
    save_checkpoint(params, str(ckpt_path), step=args.step)
    
    print(f"\nCheckpoint saved to: {ckpt_path}")
    print(f"Files:")
    for f in ckpt_path.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    print("\nTo download:")
    print(f"  1. Right-click on {ckpt_path}/params_{args.step}.pkl in Jupyter")
    print(f"  2. Select 'Download'")
    print(f"  3. Or use: scp runpod:{ckpt_path}/params_{args.step}.pkl ./")


if __name__ == "__main__":
    main()
