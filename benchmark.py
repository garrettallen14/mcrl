#!/usr/bin/env python3
"""
Benchmark script for MinecraftRL environment.

Tests:
1. Single environment step throughput
2. Vectorized environment throughput
3. World generation time
4. Memory usage

Run: python benchmark.py
"""

import time
import argparse
from typing import Callable

import jax
import jax.numpy as jnp

# Force CPU or GPU
# jax.config.update('jax_platform_name', 'cpu')


def benchmark_function(
    fn: Callable,
    name: str,
    warmup: int = 5,
    iterations: int = 100,
) -> float:
    """Benchmark a function and return iterations per second."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Time
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    # Ensure JAX computations complete
    jax.block_until_ready(fn())
    elapsed = time.perf_counter() - start
    
    per_second = iterations / elapsed
    print(f"{name}: {per_second:.1f} iterations/sec ({elapsed/iterations*1000:.3f} ms/iter)")
    return per_second


def main():
    parser = argparse.ArgumentParser(description="Benchmark MinecraftRL")
    parser.add_argument("--world-size", type=int, default=128, help="World size (cube)")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel envs")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MinecraftRL Benchmark")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"World size: {args.world_size}³")
    print(f"Parallel envs: {args.num_envs}")
    print()
    
    # Import after JAX config
    from mcrl import MinecraftEnv, Action
    from mcrl.systems.world_gen import generate_world
    
    # Create environment
    env = MinecraftEnv()
    key = jax.random.PRNGKey(42)
    
    print("-" * 60)
    print("World Generation")
    print("-" * 60)
    
    # Benchmark world generation
    def gen_world():
        return generate_world(key, args.world_size, args.world_size, args.world_size)
    
    # Compile
    world = gen_world()
    jax.block_until_ready(world.blocks)
    
    benchmark_function(gen_world, "World generation", iterations=10)
    print(f"  World shape: {world.shape}")
    print(f"  World memory: {world.blocks.nbytes / 1024 / 1024:.2f} MB")
    
    print()
    print("-" * 60)
    print("Single Environment")
    print("-" * 60)
    
    # Reset benchmark
    def reset_env():
        return env.reset(key)
    
    state, obs = reset_env()
    jax.block_until_ready(state.world.blocks)
    
    benchmark_function(reset_env, "Reset", iterations=20)
    
    # Step benchmark
    action = jnp.int32(Action.FORWARD)
    
    def step_env():
        nonlocal state
        state, obs, reward, done, info = env.step(state, action)
        return state
    
    step_env()  # Compile
    jax.block_until_ready(state.world.blocks)
    
    step_rate = benchmark_function(step_env, "Step", iterations=args.iterations)
    
    # Action sequence benchmark
    def action_sequence():
        nonlocal state
        for a in [Action.FORWARD, Action.ATTACK, Action.TURN_RIGHT, Action.JUMP]:
            state, _, _, _, _ = env.step(state, jnp.int32(a))
        return state
    
    action_sequence()
    jax.block_until_ready(state.world.blocks)
    
    seq_rate = benchmark_function(action_sequence, "4-action sequence", iterations=args.iterations)
    print(f"  Effective: {seq_rate * 4:.1f} steps/sec")
    
    print()
    print("-" * 60)
    print(f"Vectorized Environment ({args.num_envs} parallel)")
    print("-" * 60)
    
    # Create vectorized functions
    vec_reset, vec_step = env.make_vectorized()
    
    # Reset all envs
    keys = jax.random.split(key, args.num_envs)
    
    def vec_reset_all():
        return vec_reset(keys)
    
    states, obs = vec_reset_all()
    jax.block_until_ready(states.world.blocks)
    
    reset_rate = benchmark_function(vec_reset_all, "Vectorized reset", iterations=10)
    print(f"  Per-env: {reset_rate * args.num_envs:.1f} resets/sec")
    
    # Step all envs
    actions = jax.random.randint(key, (args.num_envs,), 0, 25)
    
    def vec_step_all():
        nonlocal states
        states, obs, rewards, dones, infos = vec_step(states, actions)
        return states
    
    vec_step_all()  # Compile
    jax.block_until_ready(states.world.blocks)
    
    vec_step_rate = benchmark_function(vec_step_all, "Vectorized step", iterations=args.iterations)
    total_steps = vec_step_rate * args.num_envs
    print(f"  Total throughput: {total_steps:.1f} steps/sec")
    print(f"  Per-env: {vec_step_rate:.1f} batches/sec")
    
    print()
    print("-" * 60)
    print("Memory Usage")
    print("-" * 60)
    
    # Estimate memory
    single_world_mb = args.world_size ** 3 / 1024 / 1024
    vec_world_mb = single_world_mb * args.num_envs
    print(f"  Single world: {single_world_mb:.2f} MB")
    print(f"  {args.num_envs} worlds: {vec_world_mb:.2f} MB")
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Single env step rate: {step_rate:.1f} steps/sec")
    print(f"Vectorized ({args.num_envs} envs): {total_steps:.1f} steps/sec")
    print(f"Speedup from vectorization: {total_steps / step_rate:.1f}x")
    
    # Estimate training time
    steps_for_diamond = 100_000_000  # 100M steps
    hours_single = steps_for_diamond / step_rate / 3600
    hours_vec = steps_for_diamond / total_steps / 3600
    print()
    print(f"Estimated time for 100M steps:")
    print(f"  Single env: {hours_single:.1f} hours")
    print(f"  Vectorized: {hours_vec:.1f} hours")


if __name__ == "__main__":
    main()
