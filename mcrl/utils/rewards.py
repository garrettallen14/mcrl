"""
Reward calculation for MinecraftRL.

Implements sparse milestone rewards for the diamond task.
Each milestone is rewarded once per episode (tracked via bitmask).
"""

import jax.numpy as jnp

from mcrl.core.types import ItemType
from mcrl.core.state import GameState
from mcrl.systems.inventory import has_item


# Milestone definitions: (name, item_type, reward, bit_index)
MILESTONES = [
    ("log", ItemType.OAK_LOG, 1.0, 0),
    ("planks", ItemType.OAK_PLANKS, 2.0, 1),
    ("stick", ItemType.STICK, 4.0, 2),
    ("crafting_table", ItemType.CRAFTING_TABLE, 4.0, 3),
    ("wooden_pickaxe", ItemType.WOODEN_PICKAXE, 8.0, 4),
    ("cobblestone", ItemType.COBBLESTONE, 16.0, 5),
    ("stone_pickaxe", ItemType.STONE_PICKAXE, 32.0, 6),
    ("iron_ore", ItemType.IRON_ORE, 64.0, 7),
    ("furnace", ItemType.FURNACE, 128.0, 8),
    ("iron_ingot", ItemType.IRON_INGOT, 256.0, 9),
    ("iron_pickaxe", ItemType.IRON_PICKAXE, 512.0, 10),
    ("diamond", ItemType.DIAMOND, 1024.0, 11),
]

# Also check for any log type
LOG_TYPES = [ItemType.OAK_LOG, ItemType.BIRCH_LOG, ItemType.SPRUCE_LOG]


def calculate_milestone_reward(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate milestone-based reward.
    
    Returns:
        reward: Total reward for new milestones achieved this step
        new_flags: Updated milestone bitmask
    """
    inventory = state.player.inventory
    flags = state.reward_flags
    total_reward = jnp.float32(0.0)
    
    # Check log milestone (any log type)
    has_log = (
        has_item(inventory, ItemType.OAK_LOG, 1) |
        has_item(inventory, ItemType.BIRCH_LOG, 1) |
        has_item(inventory, ItemType.SPRUCE_LOG, 1)
    )
    already_got_log = (flags >> 0) & 1
    should_reward_log = has_log & ~already_got_log
    total_reward = total_reward + jnp.where(should_reward_log, 1.0, 0.0)
    flags = jnp.where(should_reward_log, flags | (1 << 0), flags)
    
    # Check other milestones
    for name, item_type, reward, bit in MILESTONES[1:]:  # Skip log, handled above
        has_it = has_item(inventory, item_type, 1)
        already_got = (flags >> bit) & 1
        should_reward = has_it & ~already_got
        total_reward = total_reward + jnp.where(should_reward, reward, 0.0)
        flags = jnp.where(should_reward, flags | (1 << bit), flags)
    
    return total_reward, flags


def get_milestone_progress(state: GameState) -> dict:
    """
    Get human-readable milestone progress.
    Returns dict of milestone_name -> achieved (bool)
    """
    flags = state.reward_flags
    progress = {}
    
    for name, _, _, bit in MILESTONES:
        achieved = bool((flags >> bit) & 1)
        progress[name] = achieved
    
    return progress


def count_milestones(state: GameState) -> jnp.ndarray:
    """Count number of milestones achieved."""
    flags = state.reward_flags
    # Count set bits
    count = jnp.int32(0)
    for i in range(12):
        count = count + ((flags >> i) & 1)
    return count


def check_episode_done(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Check if episode should end.
    
    Returns:
        done: Whether episode is done
        success: Whether agent achieved goal (diamond)
    """
    from mcrl.core.constants import MAX_EPISODE_TICKS
    
    # Success: got diamond
    has_diamond = has_item(state.player.inventory, ItemType.DIAMOND, 1)
    
    # Death: health <= 0
    is_dead = state.player.health <= 0
    
    # Timeout
    timed_out = state.world.tick >= MAX_EPISODE_TICKS
    
    done = has_diamond | is_dead | timed_out
    success = has_diamond
    
    return done, success


def get_shaped_reward(
    state: GameState, 
    prev_state: GameState,
    milestone_scale: float = 1.0,
    death_penalty: float = -100.0,
) -> jnp.ndarray:
    """
    Get shaped reward including milestones and death penalty.
    
    Args:
        state: Current state
        prev_state: Previous state (for delta calculations)
        milestone_scale: Multiplier for milestone rewards
        death_penalty: Penalty for dying
    
    Returns:
        Total reward for this step
    """
    # Milestone reward
    milestone_reward, _ = calculate_milestone_reward(state)
    
    # Death penalty
    just_died = (state.player.health <= 0) & (prev_state.player.health > 0)
    death_reward = jnp.where(just_died, death_penalty, 0.0)
    
    total = milestone_reward * milestone_scale + death_reward
    
    return total
