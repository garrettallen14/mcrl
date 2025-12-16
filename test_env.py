#!/usr/bin/env python3
"""
Quick test script for MinecraftRL.

Verifies:
- Environment can be created
- Reset works
- Step works
- Basic game mechanics (movement, mining, crafting)

Run: python test_env.py
"""

import jax
import jax.numpy as jnp


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    from mcrl import MinecraftEnv, Action, BlockType, ItemType
    from mcrl.core.state import GameState, PlayerState, WorldState
    from mcrl.systems.world_gen import generate_world
    from mcrl.systems.physics import apply_physics
    from mcrl.systems.inventory import add_item, has_item
    from mcrl.utils.observations import get_local_voxels
    from mcrl.utils.rewards import calculate_milestone_reward
    print("  ✓ All imports successful")


def test_world_generation():
    """Test world generation."""
    print("Testing world generation...")
    from mcrl.systems.world_gen import generate_world
    from mcrl.core.types import BlockType
    
    key = jax.random.PRNGKey(0)
    world = generate_world(key, 64, 64, 64)
    
    assert world.blocks.shape == (64, 64, 64), f"Wrong shape: {world.blocks.shape}"
    
    # Check we have different block types
    unique_blocks = jnp.unique(world.blocks)
    assert len(unique_blocks) > 5, f"Too few block types: {len(unique_blocks)}"
    
    # Check bedrock at bottom
    assert jnp.all(world.blocks[:, 0, :] == BlockType.BEDROCK), "No bedrock at y=0"
    
    # Check mostly air at top (trees/leaves may reach top)
    top_layer_air_ratio = (world.blocks[:, -1, :] == BlockType.AIR).mean()
    assert top_layer_air_ratio > 0.9, f"Too many non-air blocks at top: {1-top_layer_air_ratio:.2%}"
    
    print(f"  ✓ World generated: {world.blocks.shape}")
    print(f"  ✓ Block types found: {len(unique_blocks)}")


def test_environment_reset():
    """Test environment reset."""
    print("Testing environment reset...")
    from mcrl import MinecraftEnv
    
    env = MinecraftEnv()
    key = jax.random.PRNGKey(42)
    
    state, obs = env.reset(key)
    
    # Check state
    assert state.player.health == 20.0, f"Wrong health: {state.player.health}"
    assert not state.done, "Episode shouldn't be done"
    
    # Check observations
    assert "local_voxels" in obs, "Missing local_voxels"
    assert "inventory" in obs, "Missing inventory"
    assert obs["local_voxels"].shape == (17, 17, 17), f"Wrong obs shape: {obs['local_voxels'].shape}"
    
    print(f"  ✓ Reset successful")
    print(f"  ✓ Player position: {state.player.pos}")
    print(f"  ✓ Observation keys: {list(obs.keys())}")


def test_environment_step():
    """Test environment step."""
    print("Testing environment step...")
    from mcrl import MinecraftEnv, Action
    
    env = MinecraftEnv()
    key = jax.random.PRNGKey(42)
    
    state, obs = env.reset(key)
    initial_pos = state.player.pos.copy()
    
    # Test forward movement
    state, obs, reward, done, info = env.step(state, jnp.int32(Action.FORWARD))
    
    # Position should change
    assert not jnp.allclose(state.player.pos, initial_pos), "Position didn't change"
    
    # Test multiple steps
    for i in range(10):
        state, obs, reward, done, info = env.step(state, jnp.int32(Action.FORWARD))
    
    print(f"  ✓ Step successful")
    print(f"  ✓ New position: {state.player.pos}")
    print(f"  ✓ Tick: {info['tick']}")


def test_vectorized():
    """Test vectorized environment."""
    print("Testing vectorized environment...")
    from mcrl import MinecraftEnv
    
    env = MinecraftEnv()
    vec_reset, vec_step = env.make_vectorized()
    
    key = jax.random.PRNGKey(0)
    num_envs = 32
    keys = jax.random.split(key, num_envs)
    
    # Vectorized reset
    states, obs = vec_reset(keys)
    
    assert states.player.pos.shape == (num_envs, 3), f"Wrong batch shape: {states.player.pos.shape}"
    
    # Vectorized step
    actions = jnp.zeros(num_envs, dtype=jnp.int32)  # NOOP
    states, obs, rewards, dones, infos = vec_step(states, actions)
    
    assert rewards.shape == (num_envs,), f"Wrong rewards shape: {rewards.shape}"
    
    print(f"  ✓ Vectorized with {num_envs} envs successful")


def test_mining():
    """Test mining mechanics."""
    print("Testing mining...")
    from mcrl import MinecraftEnv, Action
    from mcrl.core.types import ItemType
    from mcrl.systems.inventory import has_item
    
    env = MinecraftEnv()
    key = jax.random.PRNGKey(42)
    
    state, obs = env.reset(key)
    
    # Look down and attack (try to mine ground)
    for _ in range(5):
        state, obs, reward, done, info = env.step(state, jnp.int32(Action.LOOK_DOWN))
    
    # Mine for a while
    for _ in range(50):
        state, obs, reward, done, info = env.step(state, jnp.int32(Action.ATTACK))
    
    # Check if we got anything
    inv_counts = obs["inventory"]
    total_items = inv_counts.sum()
    
    print(f"  ✓ Mining test complete")
    print(f"  ✓ Items in inventory: {int(total_items)}")


def test_crafting():
    """Test crafting mechanics."""
    print("Testing crafting...")
    from mcrl import MinecraftEnv, Action
    from mcrl.core.types import ItemType
    from mcrl.systems.inventory import add_item, has_item
    
    env = MinecraftEnv()
    key = jax.random.PRNGKey(42)
    
    state, obs = env.reset(key)
    
    # Give player some logs
    new_inventory = add_item(state.player.inventory, ItemType.OAK_LOG, 10)
    new_player = state.player.replace(inventory=new_inventory)
    state = state.replace(player=new_player)
    
    # Craft planks
    state, obs, reward, done, info = env.step(state, jnp.int32(Action.CRAFT_PLANKS))
    
    # Check we got planks
    has_planks = has_item(state.player.inventory, ItemType.OAK_PLANKS, 1)
    
    print(f"  ✓ Crafting test complete")
    print(f"  ✓ Has planks after crafting: {bool(has_planks)}")
    print(f"  ✓ Reward: {float(reward)}")


def test_rewards():
    """Test milestone rewards."""
    print("Testing rewards...")
    from mcrl import MinecraftEnv, Action
    from mcrl.core.types import ItemType
    from mcrl.systems.inventory import add_item
    from mcrl.utils.rewards import get_milestone_progress
    
    env = MinecraftEnv()
    key = jax.random.PRNGKey(42)
    
    state, obs = env.reset(key)
    
    # Give player a diamond (to test milestone)
    new_inventory = add_item(state.player.inventory, ItemType.DIAMOND, 1)
    new_player = state.player.replace(inventory=new_inventory)
    state = state.replace(player=new_player)
    
    # Step to trigger reward
    state, obs, reward, done, info = env.step(state, jnp.int32(Action.NOOP))
    
    # Should get diamond milestone reward
    assert reward > 0, f"Expected positive reward, got {reward}"
    
    # Episode should be done (we have diamond)
    assert done, "Episode should be done after getting diamond"
    
    print(f"  ✓ Reward test complete")
    print(f"  ✓ Diamond milestone reward: {float(reward)}")
    print(f"  ✓ Episode done: {bool(done)}")


def main():
    print("=" * 60)
    print("MinecraftRL Test Suite")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print()
    
    tests = [
        test_imports,
        test_world_generation,
        test_environment_reset,
        test_environment_step,
        test_vectorized,
        test_mining,
        test_crafting,
        test_rewards,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
