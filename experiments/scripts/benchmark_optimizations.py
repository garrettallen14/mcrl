#!/usr/bin/env python3
"""
Benchmark optimizations for MCRL.

Compares original vs optimized implementations:
1. Observation extraction (get_local_voxels)
2. Network forward/backward pass
3. End-to-end environment step
"""

import sys
import time
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp


def benchmark_observation_extraction():
    """Compare original vs optimized voxel extraction."""
    from mcrl.core.state import WorldState
    from mcrl.utils.observations import get_local_voxels
    from mcrl.utils.observations_fast import (
        get_local_voxels_fast, 
        pad_world_blocks,
        encode_inventory_fast,
    )
    
    print("\n" + "=" * 60)
    print("Observation Extraction Benchmark")
    print("=" * 60)
    
    # Create test world
    W, H, D = 64, 64, 64
    blocks = jax.random.randint(jax.random.PRNGKey(0), (W, H, D), 0, 20, dtype=jnp.uint8)
    world = WorldState(blocks=blocks, tick=0, shape=(W, H, D))
    
    # Create a mock game state (just need world and player.pos)
    from mcrl.core.state import GameState, PlayerState, create_initial_player_state
    player = create_initial_player_state(jnp.array([32.0, 32.0, 32.0]), 0.0)
    
    # For original function, need full GameState
    class MockState:
        def __init__(self, world, player):
            self.world = world
            self.player = player
    
    state = MockState(world, player)
    
    # Prepare padded blocks for fast version
    padded = pad_world_blocks(blocks)
    
    # JIT compile both versions
    get_original = jax.jit(lambda s: get_local_voxels(s))
    get_fast = jax.jit(lambda pb, pp: get_local_voxels_fast(pb, pp))
    
    # Warmup
    _ = get_original(state)
    _ = get_fast(padded, player.pos)
    
    # Benchmark original
    num_runs = 1000
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = get_original(state)
    jax.block_until_ready(_)
    original_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark fast
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = get_fast(padded, player.pos)
    jax.block_until_ready(_)
    fast_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"  Original get_local_voxels: {original_time:.3f}ms")
    print(f"  Fast get_local_voxels:     {fast_time:.3f}ms")
    print(f"  Speedup: {original_time / fast_time:.1f}x")
    
    # Verify correctness
    orig_result = get_original(state)
    fast_result = get_fast(padded, player.pos)
    
    # Results should be identical
    match = jnp.allclose(orig_result, fast_result)
    print(f"  Results match: {match}")
    
    return original_time, fast_time


def benchmark_network():
    """Compare original vs optimized network."""
    from mcrl.training.networks import create_network, init_network
    from mcrl.training.networks_fast import create_fast_network, init_fast_network, count_params
    from mcrl.training.config import NetworkConfig
    
    print("\n" + "=" * 60)
    print("Network Benchmark")
    print("=" * 60)
    
    key = jax.random.PRNGKey(0)
    batch_size = 2048
    num_runs = 50
    
    # Create dummy observation batch
    dummy_obs = {
        'local_voxels': jnp.zeros((batch_size, 17, 17, 17), dtype=jnp.int32),
        'inventory': jnp.zeros((batch_size, 16), dtype=jnp.float32),
        'player_state': jnp.zeros((batch_size, 14), dtype=jnp.float32),
        'facing_blocks': jnp.zeros((batch_size, 8), dtype=jnp.int32),
    }
    
    # Original network
    obs_shapes = {
        'local_voxels': (17, 17, 17),
        'inventory': (16,),
        'player_state': (14,),
        'facing_blocks': (8,),
    }
    orig_network = create_network(NetworkConfig(), num_actions=25)
    orig_params = init_network(orig_network, key, obs_shapes)
    orig_param_count = count_params(orig_params)
    
    # Fast network
    fast_network = create_fast_network()
    fast_params = init_fast_network(fast_network, key)
    fast_param_count = count_params(fast_params)
    
    print(f"  Original params: {orig_param_count:,}")
    print(f"  Fast params:     {fast_param_count:,}")
    print(f"  Param reduction: {orig_param_count / fast_param_count:.1f}x")
    
    # JIT compile forward passes
    orig_forward = jax.jit(lambda p, o: orig_network.apply(p, o))
    fast_forward = jax.jit(lambda p, o: fast_network.apply(p, o))
    
    # Warmup
    for _ in range(5):
        _ = orig_forward(orig_params, dummy_obs)
        _ = fast_forward(fast_params, dummy_obs)
    jax.block_until_ready(_[0])
    
    # Benchmark original forward
    start = time.perf_counter()
    for _ in range(num_runs):
        logits, value = orig_forward(orig_params, dummy_obs)
    jax.block_until_ready(logits)
    orig_forward_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark fast forward
    start = time.perf_counter()
    for _ in range(num_runs):
        logits, value = fast_forward(fast_params, dummy_obs)
    jax.block_until_ready(logits)
    fast_forward_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"\n  Forward pass (batch={batch_size}):")
    print(f"    Original: {orig_forward_time:.2f}ms")
    print(f"    Fast:     {fast_forward_time:.2f}ms")
    print(f"    Speedup:  {orig_forward_time / fast_forward_time:.1f}x")
    
    # Benchmark backward passes
    def orig_loss(p, o):
        logits, value = orig_network.apply(p, o)
        return logits.mean() + value.mean()
    
    def fast_loss(p, o):
        logits, value = fast_network.apply(p, o)
        return logits.mean() + value.mean()
    
    orig_grad = jax.jit(jax.grad(orig_loss))
    fast_grad = jax.jit(jax.grad(fast_loss))
    
    # Warmup
    for _ in range(5):
        _ = orig_grad(orig_params, dummy_obs)
        _ = fast_grad(fast_params, dummy_obs)
    jax.block_until_ready(jax.tree_util.tree_leaves(_)[0])
    
    # Benchmark original backward
    start = time.perf_counter()
    for _ in range(num_runs):
        grads = orig_grad(orig_params, dummy_obs)
    jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
    orig_backward_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark fast backward
    start = time.perf_counter()
    for _ in range(num_runs):
        grads = fast_grad(fast_params, dummy_obs)
    jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
    fast_backward_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"\n  Backward pass (batch={batch_size}):")
    print(f"    Original: {orig_backward_time:.2f}ms")
    print(f"    Fast:     {fast_backward_time:.2f}ms")
    print(f"    Speedup:  {orig_backward_time / fast_backward_time:.1f}x")
    
    return {
        'orig_forward': orig_forward_time,
        'fast_forward': fast_forward_time,
        'orig_backward': orig_backward_time,
        'fast_backward': fast_backward_time,
    }


def benchmark_inventory_encoding():
    """Compare original vs optimized inventory encoding."""
    from mcrl.utils.observations import encode_inventory
    from mcrl.utils.observations_fast import encode_inventory_fast
    
    print("\n" + "=" * 60)
    print("Inventory Encoding Benchmark")
    print("=" * 60)
    
    # Create test inventory
    inventory = jnp.zeros((36, 2), dtype=jnp.int32)
    inventory = inventory.at[0, :].set(jnp.array([1, 10]))  # 10 oak logs
    inventory = inventory.at[1, :].set(jnp.array([5, 20]))  # 20 cobblestone
    
    # JIT compile
    orig_encode = jax.jit(encode_inventory)
    fast_encode = jax.jit(encode_inventory_fast)
    
    # Warmup
    _ = orig_encode(inventory)
    _ = fast_encode(inventory)
    
    # Benchmark
    num_runs = 10000
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = orig_encode(inventory)
    jax.block_until_ready(_)
    orig_time = (time.perf_counter() - start) / num_runs * 1000
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = fast_encode(inventory)
    jax.block_until_ready(_)
    fast_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"  Original: {orig_time:.4f}ms")
    print(f"  Fast:     {fast_time:.4f}ms")
    print(f"  Speedup:  {orig_time / fast_time:.1f}x")


def main():
    print("=" * 60)
    print("MCRL Optimization Benchmarks")
    print("=" * 60)
    
    # Check JAX backend
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Run benchmarks
    benchmark_observation_extraction()
    benchmark_inventory_encoding()
    net_results = benchmark_network()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Network forward speedup: {net_results['orig_forward'] / net_results['fast_forward']:.1f}x")
    print(f"  Network backward speedup: {net_results['orig_backward'] / net_results['fast_backward']:.1f}x")
    
    # Estimated impact on training
    orig_total = net_results['orig_forward'] + net_results['orig_backward']
    fast_total = net_results['fast_forward'] + net_results['fast_backward']
    
    print(f"\n  Estimated training impact:")
    print(f"    Original network time per update: {orig_total:.1f}ms")
    print(f"    Fast network time per update:     {fast_total:.1f}ms")
    print(f"    Potential speedup: {orig_total / fast_total:.1f}x")


if __name__ == "__main__":
    main()
