#!/usr/bin/env python3
"""
Pre-training preflight checks and benchmarks for MCRL.

Runs before training to:
1. Verify CUDA/GPU availability
2. Benchmark environment throughput
3. Benchmark network forward/backward pass
4. Estimate training time and memory usage
5. Recommend configuration adjustments

Usage:
    python experiments/scripts/preflight.py
    python experiments/scripts/preflight.py --num-envs 4096 --world-size 128
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_jax_backend():
    """Check JAX backend and GPU availability."""
    import jax
    
    print("=" * 60)
    print("JAX Backend Check")
    print("=" * 60)
    
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {backend}")
    print(f"Devices: {devices}")
    
    if backend == "cpu":
        print("\n⚠️  WARNING: Running on CPU!")
        print("   For GPU, install JAX with CUDA:")
        print("   pip install -U 'jax[cuda12]'")
        print("   or with uv:")
        print("   uv pip install 'jax[cuda12]'")
        return False
    else:
        print(f"\n✓ GPU available: {devices[0]}")
        
        # Get memory info
        try:
            for device in devices:
                if hasattr(device, 'memory_stats'):
                    stats = device.memory_stats()
                    if stats:
                        total = stats.get('bytes_limit', 0) / (1024**3)
                        print(f"   Memory: {total:.1f} GB")
        except Exception:
            pass
        
        return True


def benchmark_world_generation(world_sizes=[64, 128, 256]):
    """Benchmark world generation at different sizes."""
    import jax
    from mcrl.systems.world_gen import generate_world
    
    print("\n" + "=" * 60)
    print("World Generation Benchmark")
    print("=" * 60)
    
    key = jax.random.PRNGKey(0)
    results = {}
    
    for size in world_sizes:
        # Warmup
        world = generate_world(key, size, size, size)
        jax.block_until_ready(world.blocks)
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            world = generate_world(key, size, size, size)
            jax.block_until_ready(world.blocks)
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        memory_mb = (size ** 3) / (1024 * 1024)
        
        results[size] = {'time_ms': avg_time * 1000, 'memory_mb': memory_mb}
        print(f"  {size}³: {avg_time*1000:.1f} ms, {memory_mb:.2f} MB")
    
    return results


def benchmark_env_throughput(num_envs_list=[64, 256, 1024, 2048], world_size=64):
    """Benchmark environment step throughput."""
    import jax
    import jax.numpy as jnp
    from mcrl import MinecraftEnv, EnvConfig
    
    print("\n" + "=" * 60)
    print(f"Environment Throughput (world_size={world_size}³)")
    print("=" * 60)
    
    results = {}
    
    for num_envs in num_envs_list:
        try:
            env = MinecraftEnv(EnvConfig(world_size=(world_size, world_size, world_size)))
            vec_reset, vec_step = env.make_vectorized()
            
            key = jax.random.PRNGKey(0)
            keys = jax.random.split(key, num_envs)
            
            # Initialize
            states, obs = vec_reset(keys)
            actions = jax.random.randint(key, (num_envs,), 0, 25)
            
            # Warmup
            for _ in range(10):
                states, obs, rewards, dones, infos = vec_step(states, actions)
            jax.block_until_ready(states.world.blocks)
            
            # Benchmark
            num_steps = 100
            start = time.perf_counter()
            for _ in range(num_steps):
                states, obs, rewards, dones, infos = vec_step(states, actions)
            jax.block_until_ready(states.world.blocks)
            elapsed = time.perf_counter() - start
            
            total_steps = num_envs * num_steps
            sps = total_steps / elapsed
            ms_per_batch = (elapsed / num_steps) * 1000
            
            results[num_envs] = {'sps': sps, 'ms_per_batch': ms_per_batch}
            print(f"  {num_envs:4d} envs: {sps:>10,.0f} steps/sec ({ms_per_batch:.2f} ms/batch)")
            
        except Exception as e:
            print(f"  {num_envs:4d} envs: FAILED ({e})")
            results[num_envs] = None
    
    return results


def benchmark_network(batch_sizes=[256, 512, 1024, 2048, 4096]):
    """Benchmark network forward and backward pass."""
    import jax
    import jax.numpy as jnp
    from mcrl.training.config import NetworkConfig
    from mcrl.training.networks import create_network, init_network
    
    print("\n" + "=" * 60)
    print("Network Benchmark")
    print("=" * 60)
    
    network = create_network(NetworkConfig(), num_actions=25)
    key = jax.random.PRNGKey(0)
    
    obs_shapes = {
        'local_voxels': (17, 17, 17),
        'inventory': (16,),
        'player_state': (14,),
        'facing_blocks': (8,),
    }
    
    params = init_network(network, key, obs_shapes)
    
    # Count parameters
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Parameters: {num_params:,}")
    
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # Create dummy batch
            dummy_obs = {
                'local_voxels': jnp.zeros((batch_size, 17, 17, 17), dtype=jnp.int32),
                'inventory': jnp.zeros((batch_size, 16), dtype=jnp.float32),
                'player_state': jnp.zeros((batch_size, 14), dtype=jnp.float32),
                'facing_blocks': jnp.zeros((batch_size, 8), dtype=jnp.int32),
            }
            
            # JIT compile forward
            forward_fn = jax.jit(lambda p, o: network.apply(p, o))
            
            # Warmup
            for _ in range(5):
                logits, value = forward_fn(params, dummy_obs)
            jax.block_until_ready(logits)
            
            # Benchmark forward
            num_runs = 50
            start = time.perf_counter()
            for _ in range(num_runs):
                logits, value = forward_fn(params, dummy_obs)
            jax.block_until_ready(logits)
            forward_time = (time.perf_counter() - start) / num_runs * 1000
            
            # Benchmark backward (grad)
            def loss_fn(params, obs):
                logits, value = network.apply(params, obs)
                return logits.mean() + value.mean()
            
            grad_fn = jax.jit(jax.grad(loss_fn))
            
            # Warmup
            for _ in range(5):
                grads = grad_fn(params, dummy_obs)
            jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
            
            # Benchmark backward
            start = time.perf_counter()
            for _ in range(num_runs):
                grads = grad_fn(params, dummy_obs)
            jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
            backward_time = (time.perf_counter() - start) / num_runs * 1000
            
            results[batch_size] = {
                'forward_ms': forward_time,
                'backward_ms': backward_time,
                'samples_per_sec': batch_size / (forward_time / 1000),
            }
            print(f"  batch={batch_size:4d}: forward={forward_time:.2f}ms, backward={backward_time:.2f}ms")
            
        except Exception as e:
            print(f"  batch={batch_size:4d}: FAILED ({e})")
            results[batch_size] = None
    
    return results


def estimate_training_time(env_results, net_results, total_steps=100_000_000, num_envs=2048, num_steps=256):
    """Estimate total training time."""
    print("\n" + "=" * 60)
    print("Training Time Estimate")
    print("=" * 60)
    
    if num_envs not in env_results or env_results[num_envs] is None:
        # Find closest available
        available = [k for k, v in env_results.items() if v is not None]
        if not available:
            print("  Cannot estimate - no successful benchmarks")
            return
        num_envs = max(available)
        print(f"  (Using {num_envs} envs for estimate)")
    
    batch_size = num_envs * num_steps
    
    env_sps = env_results[num_envs]['sps']
    
    # Network overhead (forward + backward per batch)
    if batch_size in net_results and net_results[batch_size]:
        net_time_ms = net_results[batch_size]['forward_ms'] + net_results[batch_size]['backward_ms']
    else:
        # Estimate from closest
        net_time_ms = 5.0  # Conservative estimate
    
    # Effective throughput
    env_time_per_batch = batch_size / env_sps
    total_time_per_batch = env_time_per_batch + (net_time_ms / 1000) * 5  # ~5 forward/backward per update
    effective_sps = batch_size / total_time_per_batch
    
    total_time_sec = total_steps / effective_sps
    total_time_hours = total_time_sec / 3600
    
    print(f"  Configuration:")
    print(f"    num_envs: {num_envs}")
    print(f"    num_steps: {num_steps}")
    print(f"    batch_size: {batch_size:,}")
    print(f"  Performance:")
    print(f"    env throughput: {env_sps:,.0f} steps/sec")
    print(f"    effective throughput: {effective_sps:,.0f} steps/sec")
    print(f"  Estimate for {total_steps:,} steps:")
    print(f"    Time: {total_time_hours:.1f} hours")
    
    return effective_sps, total_time_hours


def recommend_config(gpu_available, env_results, net_results):
    """Recommend configuration based on benchmarks."""
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    if not gpu_available:
        print("\n⚠️  CRITICAL: Install CUDA JAX for reasonable performance!")
        print("   CPU training is ~100x slower than GPU.")
        print("\n   Installation:")
        print("   pip install -U 'jax[cuda12]'")
        print("\n   After installing, re-run this preflight check.")
        return
    
    # Find optimal num_envs
    best_sps = 0
    best_num_envs = 1024
    for num_envs, result in env_results.items():
        if result and result['sps'] > best_sps:
            best_sps = result['sps']
            best_num_envs = num_envs
    
    print(f"\n  Recommended settings:")
    print(f"    num_envs: {best_num_envs}")
    print(f"    num_steps: 256")
    print(f"    world_size: 64 (for speed) or 128 (for realism)")
    print(f"    batch_size: {best_num_envs * 256:,}")
    
    # Memory estimate
    world_mem_mb = best_num_envs * (64 ** 3) / (1024 * 1024)
    net_mem_mb = 500  # Rough estimate for network + optimizer
    total_mem_gb = (world_mem_mb + net_mem_mb) / 1024
    
    print(f"\n  Estimated GPU memory: {total_mem_gb:.1f} GB")
    
    if total_mem_gb > 20:
        print(f"    ⚠️  May need to reduce num_envs on GPUs with <24GB VRAM")
    
    print(f"\n  GPU Recommendations:")
    print(f"    RTX 5090 (32GB): Excellent - can run 4096+ envs")
    print(f"    RTX 4090 (24GB): Good - 2048-3072 envs")
    print(f"    A40 (48GB): Best for large scale - 8192+ envs")


def main():
    parser = argparse.ArgumentParser(description="MCRL Pre-training Checks")
    parser.add_argument("--num-envs", type=int, nargs="+", 
                        default=[64, 256, 1024, 2048],
                        help="Environment counts to benchmark")
    parser.add_argument("--world-size", type=int, default=64,
                        help="World size for env benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[256, 512, 1024, 2048],
                        help="Batch sizes for network benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick check (fewer iterations)")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MCRL Pre-Training Preflight Check")
    print("=" * 60)
    
    # Check JAX backend
    gpu_available = check_jax_backend()
    
    # Run benchmarks
    world_results = benchmark_world_generation([64, 128] if args.quick else [64, 128, 256])
    env_results = benchmark_env_throughput(args.num_envs, args.world_size)
    net_results = benchmark_network(args.batch_sizes)
    
    # Estimate training time
    estimate_training_time(env_results, net_results)
    
    # Recommendations
    recommend_config(gpu_available, env_results, net_results)
    
    print("\n" + "=" * 60)
    print("Preflight Complete")
    print("=" * 60)
    
    if not gpu_available:
        print("\n❌ Not ready for training - GPU required")
        sys.exit(1)
    else:
        print("\n✓ Ready for training!")
        sys.exit(0)


if __name__ == "__main__":
    main()
