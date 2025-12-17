#!/usr/bin/env python3
"""
Main training entry point for MCRL experiments.

Usage:
    # Quick debug run
    python experiments/scripts/train.py --debug
    
    # Full training run
    python experiments/scripts/train.py --config experiments/configs/baseline.yaml
    
    # With WandB logging
    python experiments/scripts/train.py --wandb --run-name my_experiment
    
    # Custom settings
    python experiments/scripts/train.py --num-envs 4096 --total-steps 50000000
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcrl import MinecraftEnv, EnvConfig
from mcrl.training.config import (
    TrainConfig, 
    get_fast_debug_config, 
    get_baseline_config,
    get_high_explore_config,
    get_ultra_fast_config,
)
from mcrl.training.train import train, create_train_state
from mcrl.training.profiling import (
    Profiler, 
    MetricsTracker,
    benchmark_env_throughput,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MCRL agent")
    
    # Config
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    parser.add_argument("--ultra-fast", action="store_true", 
                        help="Use UltraFast config (no 3D conv, max throughput)")
    parser.add_argument("--preset", type=str, choices=["baseline", "high_explore"],
                        default="baseline", help="Config preset")
    
    # Training scale
    parser.add_argument("--num-envs", type=int, help="Number of parallel envs")
    parser.add_argument("--num-steps", type=int, help="Steps per rollout")
    parser.add_argument("--total-steps", type=int, help="Total training steps")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, help="Entropy coefficient")
    parser.add_argument("--clip-eps", type=float, help="PPO clip epsilon")
    
    # Environment
    parser.add_argument("--world-size", type=int, help="World size (cubic)")
    parser.add_argument("--max-ticks", type=int, help="Max episode ticks")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="mcrl", help="WandB project")
    parser.add_argument("--run-name", type=str, help="Run name")
    parser.add_argument("--output-dir", type=str, default="experiments/runs",
                        help="Output directory")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmarks before training")
    parser.add_argument("--profile", action="store_true",
                        help="Enable detailed profiling")
    parser.add_argument("--dashboard", action="store_true",
                        help="Enable dashboard metrics (requires dashboard server)")
    parser.add_argument("--start-dashboard", action="store_true",
                        help="Start dashboard server in background")
    
    return parser.parse_args()


def create_config(args) -> TrainConfig:
    """Create training config from arguments."""
    # Start with preset or config file
    if args.config:
        config = TrainConfig.from_yaml(args.config)
    elif args.debug:
        config = get_fast_debug_config()
    elif getattr(args, 'ultra_fast', False):
        config = get_ultra_fast_config()
    elif args.preset == "high_explore":
        config = get_high_explore_config()
    else:
        config = get_baseline_config()
    
    # Override with command line arguments
    if args.num_envs:
        config.num_envs = args.num_envs
    if args.num_steps:
        config.num_steps = args.num_steps
    if args.total_steps:
        config.total_timesteps = args.total_steps
    if args.lr:
        config.ppo.lr = args.lr
    if args.gamma:
        config.ppo.gamma = args.gamma
    if args.ent_coef:
        config.ppo.ent_coef = args.ent_coef
    if args.clip_eps:
        config.ppo.clip_eps = args.clip_eps
    if args.world_size:
        config.world_size = (args.world_size, args.world_size, args.world_size)
    if args.max_ticks:
        config.max_episode_ticks = args.max_ticks
    if args.seed:
        config.seed = args.seed
    
    # Logging config
    config.logging.use_wandb = args.wandb
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    if args.run_name:
        config.run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.run_name = f"ppo_{timestamp}"
    
    config.output_dir = args.output_dir
    
    # Recompute derived values
    config.__post_init__()
    
    return config


def setup_logging(config: TrainConfig) -> tuple:
    """Setup logging (WandB and local)."""
    # Create output directory
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.to_yaml(str(run_dir / "config.yaml"))
    
    # Setup WandB
    wandb_run = None
    if config.logging.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                name=config.run_name,
                config=config.to_dict(),
            )
        except ImportError:
            print("Warning: wandb not installed, skipping WandB logging")
    
    return run_dir, wandb_run


def run_benchmarks(config: TrainConfig):
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("Running Benchmarks")
    print("="*60)
    
    # Environment throughput
    print("\nEnvironment throughput...")
    env = MinecraftEnv(EnvConfig(world_size=config.world_size))
    
    results = benchmark_env_throughput(
        env,
        num_envs=min(config.num_envs, 512),  # Limit for benchmark
        num_steps=100,
    )
    
    print(f"  Steps/sec: {results['steps_per_sec']:,.0f}")
    print(f"  ms/batch: {results['ms_per_batch']:.2f}")
    
    # Network forward pass
    print("\nNetwork forward pass...")
    from mcrl.training.networks import create_network, init_network
    from mcrl.training.profiling import profile_network_forward
    
    network = create_network(config.network, num_actions=25)
    obs_shapes = {
        'local_voxels': (17, 17, 17),
        'inventory': (16,),
        'player_state': (14,),
        'facing_blocks': (8,),
    }
    key = jax.random.PRNGKey(0)
    params = init_network(network, key, obs_shapes)
    
    net_results = profile_network_forward(
        network, params,
        batch_size=config.num_envs,
        num_runs=50,
    )
    
    print(f"  Samples/sec: {net_results['samples_per_sec']:,.0f}")
    print(f"  ms/forward: {net_results['ms_per_forward']:.3f}")
    
    # Estimated training throughput
    total_forward_time = net_results['ms_per_forward'] * (1 + config.update_epochs * config.num_minibatches)
    estimated_sps = config.batch_size / (results['ms_per_batch'] + total_forward_time) * 1000
    
    print(f"\nEstimated training throughput: {estimated_sps:,.0f} steps/sec")
    print(f"Estimated time for {config.total_timesteps:,} steps: {config.total_timesteps / estimated_sps / 3600:.1f} hours")
    
    print("="*60 + "\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print JAX info
    print("="*60)
    print("MCRL Training")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Create config
    config = create_config(args)
    print(f"\nConfig: {config.run_name}")
    print(f"  Envs: {config.num_envs}")
    print(f"  Steps/rollout: {config.num_steps}")
    print(f"  Batch size: {config.batch_size:,}")
    print(f"  Total steps: {config.total_timesteps:,}")
    print(f"  Updates: {config.num_updates:,}")
    print(f"  Fast network: {getattr(config, 'use_fast_network', False)}")
    
    # Setup logging
    run_dir, wandb_run = setup_logging(config)
    print(f"\nOutput dir: {run_dir}")
    
    # Start dashboard server if requested
    dashboard_process = None
    if args.start_dashboard:
        import subprocess
        print("\nStarting dashboard server on port 3000...")
        dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "mcrl.dashboard.server", "--port", "3000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)  # Give it time to start
        print("Dashboard available at http://localhost:3000")
    
    # Run benchmarks if requested
    if args.benchmark:
        run_benchmarks(config)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    start_time = time.time()
    
    # Determine if dashboard metrics should be enabled
    use_dashboard = args.dashboard or args.start_dashboard
    metrics_history = []  # Initialize here in case of early interrupt
    
    try:
        final_state, metrics_history = train(config, verbose=True, dashboard=use_dashboard)
        
        # Save final checkpoint
        checkpoint_path = run_dir / "checkpoints" / "final"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics history
        metrics_file = run_dir / "metrics.jsonl"
        with open(metrics_file, "w") as f:
            for m in metrics_history:
                f.write(json.dumps(m) + "\n")
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"  Total time: {total_time / 3600:.2f} hours ({total_time:.0f}s)")
        print(f"  Average SPS: {config.total_timesteps / total_time:,.0f}")
        print(f"  Metrics saved to: {metrics_file}")
        print(f"  Checkpoints: {checkpoint_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        # Still save metrics
        if metrics_history:
            with open(run_dir / "metrics.jsonl", "w") as f:
                for m in metrics_history:
                    f.write(json.dumps(m) + "\n")
            print(f"Partial metrics saved to {run_dir / 'metrics.jsonl'}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        if wandb_run:
            wandb_run.finish()
        if dashboard_process:
            dashboard_process.terminate()
            print("Dashboard server stopped")


if __name__ == "__main__":
    main()
