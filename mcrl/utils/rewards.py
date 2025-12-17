"""
Reward calculation for MinecraftRL.

Implements a principled dense + sparse reward system:

TIER 1: Milestone rewards (sparse, one-time, large)
  - Primary objective signal for diamond progression

TIER 2: First-time discovery bonuses (one-time per episode)
  - Breaking new block types
  - Picking up new item types
  - Using new crafting recipes

TIER 3: Depth exploration (one-time per Y level)
  - Encourages going underground where diamonds are

TIER 4: Per-block mining (tiny, capped)
  - Immediate feedback for taking action

Anti-exploitation:
  - All bonuses are one-time (tracked via bitmasks)
  - Block-break rewards capped at MAX_BLOCKS_REWARDED
  - No rewards for repeatable actions (place/break same block)

Performance notes:
  - All functions use vectorized JAX operations (no Python loops in hot paths)
  - Bitmask operations are fully vectorized using array indexing
"""

import jax
import jax.numpy as jnp

from mcrl.core.types import ItemType, BlockType
from mcrl.core.state import GameState
from mcrl.systems.inventory import has_item, get_item_count


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-computed arrays for vectorized milestone checking
# ═══════════════════════════════════════════════════════════════════════════════

# Milestone item types (excluding log which needs special handling)
_MILESTONE_ITEMS = jnp.array([
    ItemType.OAK_PLANKS,      # bit 1
    ItemType.STICK,           # bit 2
    ItemType.CRAFTING_TABLE,  # bit 3
    ItemType.WOODEN_PICKAXE,  # bit 4
    ItemType.COBBLESTONE,     # bit 5
    ItemType.STONE_PICKAXE,   # bit 6
    ItemType.IRON_ORE,        # bit 7
    ItemType.FURNACE,         # bit 8
    ItemType.IRON_INGOT,      # bit 9
    ItemType.IRON_PICKAXE,    # bit 10
    ItemType.DIAMOND,         # bit 11
], dtype=jnp.int32)

_MILESTONE_REWARDS = jnp.array([
    2.0, 4.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0
], dtype=jnp.float32)

_MILESTONE_BITS = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)

# Log types for first milestone
_LOG_TYPES = jnp.array([
    ItemType.OAK_LOG, ItemType.BIRCH_LOG, ItemType.SPRUCE_LOG
], dtype=jnp.int32)

# Vectorized first-break rewards (block_type, bit, reward)
_FIRST_BREAK_BLOCKS = jnp.array([
    BlockType.DIRT, BlockType.GRASS, BlockType.SAND, BlockType.GRAVEL,
    BlockType.STONE, BlockType.OAK_LOG, BlockType.BIRCH_LOG, BlockType.SPRUCE_LOG,
    BlockType.COAL_ORE, BlockType.IRON_ORE, BlockType.GOLD_ORE, BlockType.DIAMOND_ORE,
    BlockType.COBBLESTONE, BlockType.OAK_PLANKS,
], dtype=jnp.int32)
_FIRST_BREAK_BITS = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=jnp.int32)
_FIRST_BREAK_REWARDS_ARR = jnp.array([
    0.01, 0.01, 0.01, 0.02, 0.05, 0.1, 0.1, 0.1, 0.2, 0.5, 0.3, 1.0, 0.02, 0.02
], dtype=jnp.float32)

# Vectorized first-pickup rewards
_FIRST_PICKUP_ITEMS = jnp.array([
    ItemType.OAK_LOG, ItemType.BIRCH_LOG, ItemType.SPRUCE_LOG, ItemType.OAK_PLANKS,
    ItemType.COBBLESTONE, ItemType.COAL, ItemType.IRON_ORE, ItemType.GOLD_ORE,
    ItemType.DIAMOND, ItemType.STICK, ItemType.IRON_INGOT,
], dtype=jnp.int32)
_FIRST_PICKUP_BITS = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.int32)
_FIRST_PICKUP_REWARDS_ARR = jnp.array([
    0.1, 0.1, 0.1, 0.05, 0.1, 0.15, 0.3, 0.2, 0.5, 0.05, 0.4
], dtype=jnp.float32)

# Vectorized first-craft rewards
_FIRST_CRAFT_ITEMS = jnp.array([
    ItemType.OAK_PLANKS, ItemType.STICK, ItemType.CRAFTING_TABLE, ItemType.WOODEN_PICKAXE,
    ItemType.WOODEN_AXE, ItemType.WOODEN_SWORD, ItemType.FURNACE, ItemType.STONE_PICKAXE,
    ItemType.STONE_AXE, ItemType.IRON_PICKAXE, ItemType.IRON_AXE, ItemType.IRON_SWORD,
], dtype=jnp.int32)
_FIRST_CRAFT_BITS = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)
_FIRST_CRAFT_REWARDS_ARR = jnp.array([
    0.1, 0.1, 0.2, 0.3, 0.2, 0.15, 0.3, 0.3, 0.2, 0.5, 0.3, 0.3
], dtype=jnp.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: Milestone Rewards (sparse, one-time, large)
# ═══════════════════════════════════════════════════════════════════════════════
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

LOG_TYPES = [ItemType.OAK_LOG, ItemType.BIRCH_LOG, ItemType.SPRUCE_LOG]


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: First-Time Discovery Bonuses
# ═══════════════════════════════════════════════════════════════════════════════

# First time breaking each block type (bit_index -> (block_type, reward))
FIRST_BREAK_REWARDS = {
    BlockType.DIRT: (0, 0.01),
    BlockType.GRASS: (1, 0.01),
    BlockType.SAND: (2, 0.01),
    BlockType.GRAVEL: (3, 0.02),      # Can drop flint
    BlockType.STONE: (4, 0.05),       # Drops cobblestone - important!
    BlockType.OAK_LOG: (5, 0.1),      # Critical path
    BlockType.BIRCH_LOG: (6, 0.1),
    BlockType.SPRUCE_LOG: (7, 0.1),
    BlockType.COAL_ORE: (8, 0.2),     # Fuel source
    BlockType.IRON_ORE: (9, 0.5),     # Major milestone proximity
    BlockType.GOLD_ORE: (10, 0.3),
    BlockType.DIAMOND_ORE: (11, 1.0), # About to win!
    BlockType.COBBLESTONE: (12, 0.02),
    BlockType.OAK_PLANKS: (13, 0.02),
}

# First time picking up each item type (bit_index -> (item_type, reward))
FIRST_PICKUP_REWARDS = {
    ItemType.OAK_LOG: (0, 0.1),
    ItemType.BIRCH_LOG: (1, 0.1),
    ItemType.SPRUCE_LOG: (2, 0.1),
    ItemType.OAK_PLANKS: (3, 0.05),
    ItemType.COBBLESTONE: (4, 0.1),
    ItemType.COAL: (5, 0.15),
    ItemType.IRON_ORE: (6, 0.3),      # Iron ore drops itself
    ItemType.GOLD_ORE: (7, 0.2),      # Gold ore drops itself
    ItemType.DIAMOND: (8, 0.5),       # Additional to milestone
    ItemType.STICK: (9, 0.05),
    ItemType.IRON_INGOT: (10, 0.4),   # From smelting
}

# First time crafting each recipe (bit_index -> (recipe_name, output_item, reward))
FIRST_CRAFT_REWARDS = {
    ItemType.OAK_PLANKS: (0, 0.1),
    ItemType.STICK: (1, 0.1),
    ItemType.CRAFTING_TABLE: (2, 0.2),
    ItemType.WOODEN_PICKAXE: (3, 0.3),
    ItemType.WOODEN_AXE: (4, 0.2),
    ItemType.WOODEN_SWORD: (5, 0.15),
    ItemType.FURNACE: (6, 0.3),
    ItemType.STONE_PICKAXE: (7, 0.3),
    ItemType.STONE_AXE: (8, 0.2),
    ItemType.IRON_PICKAXE: (9, 0.5),
    ItemType.IRON_AXE: (10, 0.3),
    ItemType.IRON_SWORD: (11, 0.3),
}

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: Depth Exploration
# ═══════════════════════════════════════════════════════════════════════════════
DEPTH_REWARD_PER_LEVEL = 0.01  # +0.01 for each new Y level below surface
MAX_DEPTH_BONUS = 1.0          # Cap total depth bonus

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4: Per-Block Mining (tiny, capped)
# ═══════════════════════════════════════════════════════════════════════════════
BLOCK_BREAK_REWARD = 0.001     # Tiny per-block reward
MAX_BLOCKS_REWARDED = 1000     # Cap at 1000 blocks = +1.0 max


def calculate_milestone_reward(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate milestone-based reward (TIER 1).
    
    Fully vectorized - no Python loops for JIT efficiency.
    
    Returns:
        reward: Total reward for new milestones achieved this step
        new_flags: Updated milestone bitmask
    """
    inventory = state.player.inventory
    flags = state.reward_flags
    
    # === Check log milestone (any log type) - bit 0 ===
    # Vectorized check across all log types
    log_counts = jax.vmap(lambda item: get_item_count(inventory, item))(_LOG_TYPES)
    has_log = log_counts.sum() >= 1
    already_got_log = (flags >> 0) & 1
    should_reward_log = has_log & ~already_got_log
    log_reward = jnp.where(should_reward_log, 1.0, 0.0)
    flags = jnp.where(should_reward_log, flags | 1, flags)
    
    # === Check other milestones - vectorized ===
    # Get counts for all milestone items at once
    item_counts = jax.vmap(lambda item: get_item_count(inventory, item))(_MILESTONE_ITEMS)
    has_items = item_counts >= 1  # (11,) boolean array
    
    # Check which milestones are already achieved
    already_got = ((flags >> _MILESTONE_BITS) & 1).astype(jnp.bool_)
    
    # Determine which to reward
    should_reward = has_items & ~already_got  # (11,) boolean
    
    # Calculate total reward (masked sum)
    milestone_reward = jnp.where(should_reward, _MILESTONE_REWARDS, 0.0).sum()
    
    # Update flags (vectorized bit setting)
    new_flag_bits = jnp.where(should_reward, 1 << _MILESTONE_BITS, 0)
    flags = flags | new_flag_bits.sum().astype(jnp.uint32)
    
    total_reward = log_reward + milestone_reward
    
    return total_reward, flags


def calculate_dense_reward(
    state: GameState,
    prev_state: GameState,
    block_just_broken: jnp.ndarray,  # BlockType or -1 if none
) -> tuple[jnp.ndarray, GameState]:
    """
    Calculate dense rewards (TIER 2-4) and update tracking state.
    
    Fully vectorized - no Python loops for JIT efficiency.
    
    Args:
        state: Current game state
        prev_state: Previous game state
        block_just_broken: Block type that was just broken (-1 if none)
    
    Returns:
        reward: Total dense reward
        updated_state: State with updated tracking fields
    """
    total_reward = jnp.float32(0.0)
    
    # --- TIER 2a: First-time block break bonus (vectorized) ---
    broken_types = state.broken_block_types
    block_valid = block_just_broken >= 0
    
    # Check all block types at once
    is_this_block = (block_just_broken == _FIRST_BREAK_BLOCKS)  # (14,) bool
    already_broken = ((broken_types >> _FIRST_BREAK_BITS) & 1).astype(jnp.bool_)
    should_reward_break = block_valid & is_this_block & ~already_broken
    
    # Sum rewards and update flags
    break_reward = jnp.where(should_reward_break, _FIRST_BREAK_REWARDS_ARR, 0.0).sum()
    total_reward = total_reward + break_reward
    new_break_bits = jnp.where(should_reward_break, 1 << _FIRST_BREAK_BITS, 0)
    broken_types = broken_types | new_break_bits.sum().astype(jnp.uint32)
    
    # --- TIER 2b: First-time item pickup bonus (vectorized) ---
    picked_up = state.picked_up_items
    
    # Check all pickup items at once
    curr_counts = jax.vmap(lambda item: get_item_count(state.player.inventory, item))(_FIRST_PICKUP_ITEMS)
    prev_counts = jax.vmap(lambda item: get_item_count(prev_state.player.inventory, item))(_FIRST_PICKUP_ITEMS)
    just_got = (curr_counts >= 1) & (prev_counts < 1)
    already_counted = ((picked_up >> _FIRST_PICKUP_BITS) & 1).astype(jnp.bool_)
    should_reward_pickup = just_got & ~already_counted
    
    pickup_reward = jnp.where(should_reward_pickup, _FIRST_PICKUP_REWARDS_ARR, 0.0).sum()
    total_reward = total_reward + pickup_reward
    new_pickup_bits = jnp.where(should_reward_pickup, 1 << _FIRST_PICKUP_BITS, 0)
    picked_up = picked_up | new_pickup_bits.sum().astype(jnp.uint32)
    
    # --- TIER 2c: First-time craft bonus (vectorized) ---
    crafted = state.crafted_recipes
    
    # Check all craft items at once
    curr_craft = jax.vmap(lambda item: get_item_count(state.player.inventory, item))(_FIRST_CRAFT_ITEMS)
    prev_craft = jax.vmap(lambda item: get_item_count(prev_state.player.inventory, item))(_FIRST_CRAFT_ITEMS)
    just_crafted = (curr_craft >= 1) & (prev_craft < 1)
    already_crafted = ((crafted >> _FIRST_CRAFT_BITS) & 1).astype(jnp.bool_)
    should_reward_craft = just_crafted & ~already_crafted
    
    craft_reward = jnp.where(should_reward_craft, _FIRST_CRAFT_REWARDS_ARR, 0.0).sum()
    total_reward = total_reward + craft_reward
    new_craft_bits = jnp.where(should_reward_craft, 1 << _FIRST_CRAFT_BITS, 0)
    crafted = crafted | new_craft_bits.sum().astype(jnp.uint32)
    
    # --- TIER 3: Depth exploration bonus ---
    current_y = jnp.int32(state.player.pos[1])
    prev_min_y = state.min_y_reached
    new_min_y = jnp.minimum(current_y, prev_min_y)
    depth_gained = jnp.maximum(prev_min_y - new_min_y, 0)
    depth_reward = jnp.minimum(
        depth_gained * DEPTH_REWARD_PER_LEVEL,
        MAX_DEPTH_BONUS
    )
    total_reward = total_reward + depth_reward
    
    # --- TIER 4: Per-block mining bonus (capped) ---
    blocks_broken = state.blocks_broken_count
    new_blocks_broken = blocks_broken + jnp.where(block_valid, 1, 0)
    under_cap = blocks_broken < MAX_BLOCKS_REWARDED
    block_reward = jnp.where(block_valid & under_cap, BLOCK_BREAK_REWARD, 0.0)
    total_reward = total_reward + block_reward
    
    # Update state with new tracking values
    updated_state = state.replace(
        broken_block_types=broken_types,
        picked_up_items=picked_up,
        crafted_recipes=crafted,
        min_y_reached=new_min_y,
        blocks_broken_count=new_blocks_broken,
    )
    
    return total_reward, updated_state


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
    """Count number of milestones achieved (vectorized bit count)."""
    flags = state.reward_flags
    # Vectorized popcount using array operations
    bits = jnp.arange(12, dtype=jnp.int32)
    return ((flags >> bits) & 1).sum()


# Import at module level to avoid JIT issues
from mcrl.core.constants import MAX_EPISODE_TICKS as _MAX_EPISODE_TICKS


def check_episode_done(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Check if episode should end.
    
    Returns:
        done: Whether episode is done
        success: Whether agent achieved goal (diamond)
    """
    # Success: got diamond
    has_diamond = has_item(state.player.inventory, ItemType.DIAMOND, 1)
    
    # Death: health <= 0
    is_dead = state.player.health <= 0
    
    # Timeout
    timed_out = state.world.tick >= _MAX_EPISODE_TICKS
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE REWARD: Wood → Planks (for initial testing/debugging)
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-computed search offsets for finding nearby logs
# IMPORTANT: Must be large enough to cover spawn distance to nearest tree!
_LOG_SEARCH_RADIUS = 20  # Increased from 8 to cover typical spawn distances
_LOG_SEARCH_OFFSETS = jnp.stack(jnp.meshgrid(
    jnp.arange(-_LOG_SEARCH_RADIUS, _LOG_SEARCH_RADIUS + 1),
    jnp.arange(-2, 15),  # Check ground to tree canopy height
    jnp.arange(-_LOG_SEARCH_RADIUS, _LOG_SEARCH_RADIUS + 1),
    indexing='ij'
), axis=-1).reshape(-1, 3)


def _find_nearest_log_distance(world_blocks: jnp.ndarray, player_pos: jnp.ndarray) -> jnp.ndarray:
    """Find distance to nearest log block (vectorized)."""
    W, H, D = world_blocks.shape
    pos_int = player_pos.astype(jnp.int32)
    
    # Check positions around player
    check_positions = pos_int[None, :] + _LOG_SEARCH_OFFSETS
    
    # Bounds check
    in_bounds = (
        (check_positions[:, 0] >= 0) & (check_positions[:, 0] < W) &
        (check_positions[:, 1] >= 0) & (check_positions[:, 1] < H) &
        (check_positions[:, 2] >= 0) & (check_positions[:, 2] < D)
    )
    
    # Safe positions for gather
    safe_pos = jnp.clip(check_positions, 0, jnp.array([W-1, H-1, D-1]))
    
    # Get block types
    blocks = world_blocks[safe_pos[:, 0], safe_pos[:, 1], safe_pos[:, 2]]
    blocks = jnp.where(in_bounds, blocks, 0)
    
    # Check if log
    is_log = (
        (blocks == BlockType.OAK_LOG) |
        (blocks == BlockType.BIRCH_LOG) |
        (blocks == BlockType.SPRUCE_LOG)
    )
    
    # Compute distances (only for valid log positions)
    offsets_float = _LOG_SEARCH_OFFSETS.astype(jnp.float32)
    distances = jnp.sqrt(jnp.sum(offsets_float ** 2, axis=1))
    
    # Set non-log distances to large value
    distances = jnp.where(is_log, distances, 1000.0)
    
    return jnp.min(distances)


def _is_facing_log(world_blocks: jnp.ndarray, player_pos: jnp.ndarray, player_rot: jnp.ndarray) -> jnp.ndarray:
    """Check if player is looking at a log block within reach."""
    W, H, D = world_blocks.shape
    
    # Eye position
    eye = player_pos + jnp.array([0.0, 1.62, 0.0])
    
    # Look direction
    pitch_rad = player_rot[0] * (jnp.pi / 180.0)
    yaw_rad = player_rot[1] * (jnp.pi / 180.0)
    cos_pitch = jnp.cos(pitch_rad)
    direction = jnp.array([
        cos_pitch * jnp.sin(yaw_rad),
        -jnp.sin(pitch_rad),
        cos_pitch * jnp.cos(yaw_rad),
    ])
    
    # Check blocks along ray (reach distance ~4 blocks)
    distances = jnp.array([1.0, 2.0, 3.0, 4.0])
    positions = eye[None, :] + direction[None, :] * distances[:, None]
    block_pos = jnp.floor(positions).astype(jnp.int32)
    
    # Bounds check
    in_bounds = (
        (block_pos[:, 0] >= 0) & (block_pos[:, 0] < W) &
        (block_pos[:, 1] >= 0) & (block_pos[:, 1] < H) &
        (block_pos[:, 2] >= 0) & (block_pos[:, 2] < D)
    )
    
    safe_pos = jnp.clip(block_pos, 0, jnp.array([W-1, H-1, D-1]))
    blocks = world_blocks[safe_pos[:, 0], safe_pos[:, 1], safe_pos[:, 2]]
    blocks = jnp.where(in_bounds, blocks, 0)
    
    # Check if any is log
    is_log = (
        (blocks == BlockType.OAK_LOG) |
        (blocks == BlockType.BIRCH_LOG) |
        (blocks == BlockType.SPRUCE_LOG)
    )
    
    return is_log.any()


def calculate_simple_wood_reward(
    state: GameState,
    prev_state: GameState,
    block_just_broken: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simple reward function for testing: Mine wood and craft planks.
    
    Shaping Rewards (frequent, guide behavior):
        - Getting closer to wood: +0.1 per block closer (capped)
        - Looking at wood: +0.1 per step
        - Mining progress on wood: +0.3 per tick
        - Adjacent to wood (≤2 blocks): +0.05 per step
    
    Task Rewards (larger milestones):
        - Breaking any log block: +5.0
        - First log in inventory: +10.0 (one-time)
        - First planks crafted: +20.0 (one-time)
        - Each plank crafted: +2.0
    
    Uses reward_flags bits:
        - bit 0: got first log
        - bit 1: got first planks
    
    Returns:
        reward: Total reward this step
        new_flags: Updated reward flags
    """
    flags = state.reward_flags
    total_reward = jnp.float32(0.0)
    
    # Skip shaping rewards if already got log (focus on crafting)
    already_got_log = (flags >> 0) & 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHAPING: Distance to nearest log (potential-based)
    # ─────────────────────────────────────────────────────────────────────────
    curr_dist = _find_nearest_log_distance(state.world.blocks, state.player.pos)
    prev_dist = _find_nearest_log_distance(prev_state.world.blocks, prev_state.player.pos)
    
    # Reward for getting closer (only if we haven't got log yet)
    dist_improvement = prev_dist - curr_dist  # positive = got closer
    approach_reward = jnp.where(
        ~already_got_log.astype(jnp.bool_),
        jnp.clip(dist_improvement * 0.1, -0.2, 0.5),  # +0.1 per block closer (10x stronger)
        0.0
    )
    total_reward = total_reward + approach_reward
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHAPING: Looking at wood (stronger signal)
    # ─────────────────────────────────────────────────────────────────────────
    facing_log = _is_facing_log(state.world.blocks, state.player.pos, state.player.rot)
    look_reward = jnp.where(
        facing_log & ~already_got_log.astype(jnp.bool_),
        0.1,  # 5x stronger
        0.0
    )
    total_reward = total_reward + look_reward
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHAPING: Mining progress on wood
    # ─────────────────────────────────────────────────────────────────────────
    # Check if currently mining and target is log
    mining_block = state.player.mining_block
    W, H, D = state.world.blocks.shape
    mining_in_bounds = (
        (mining_block[0] >= 0) & (mining_block[0] < W) &
        (mining_block[1] >= 0) & (mining_block[1] < H) &
        (mining_block[2] >= 0) & (mining_block[2] < D)
    )
    
    mining_block_type = jnp.where(
        mining_in_bounds,
        state.world.blocks[
            jnp.clip(mining_block[0], 0, W-1),
            jnp.clip(mining_block[1], 0, H-1),
            jnp.clip(mining_block[2], 0, D-1)
        ],
        0
    )
    
    mining_log = (
        (mining_block_type == BlockType.OAK_LOG) |
        (mining_block_type == BlockType.BIRCH_LOG) |
        (mining_block_type == BlockType.SPRUCE_LOG)
    )
    
    # Reward for making mining progress on log (BIG reward - this is what we want!)
    progress_increased = state.player.mining_progress > prev_state.player.mining_progress
    mining_reward = jnp.where(
        mining_log & progress_increased & ~already_got_log.astype(jnp.bool_),
        0.3,  # 6x stronger - mining wood is the goal!
        0.0
    )
    total_reward = total_reward + mining_reward
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHAPING: Adjacent to wood (within 2 blocks)
    # ─────────────────────────────────────────────────────────────────────────
    adjacent_reward = jnp.where(
        (curr_dist <= 2.0) & ~already_got_log.astype(jnp.bool_),
        0.05,  # 5x stronger
        0.0
    )
    total_reward = total_reward + adjacent_reward
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Breaking a log (MAJOR milestone!)
    # ─────────────────────────────────────────────────────────────────────────
    is_log = (
        (block_just_broken == BlockType.OAK_LOG) |
        (block_just_broken == BlockType.BIRCH_LOG) |
        (block_just_broken == BlockType.SPRUCE_LOG)
    )
    log_break_reward = jnp.where(is_log, 5.0, 0.0)  # 10x stronger!
    total_reward = total_reward + log_break_reward
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK: First log pickup (HUGE milestone!)
    # ─────────────────────────────────────────────────────────────────────────
    has_log = (
        has_item(state.player.inventory, ItemType.OAK_LOG, 1) |
        has_item(state.player.inventory, ItemType.BIRCH_LOG, 1) |
        has_item(state.player.inventory, ItemType.SPRUCE_LOG, 1)
    )
    first_log = has_log & ~already_got_log
    total_reward = total_reward + jnp.where(first_log, 10.0, 0.0)  # 10x stronger!
    flags = jnp.where(first_log, flags | (1 << 0), flags)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK: First planks (ULTIMATE goal!)
    # ─────────────────────────────────────────────────────────────────────────
    has_planks = has_item(state.player.inventory, ItemType.OAK_PLANKS, 1)
    already_got_planks = (flags >> 1) & 1
    first_planks = has_planks & ~already_got_planks
    total_reward = total_reward + jnp.where(first_planks, 20.0, 0.0)  # 10x stronger!
    flags = jnp.where(first_planks, flags | (1 << 1), flags)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Crafting more planks
    # ─────────────────────────────────────────────────────────────────────────
    prev_planks = jnp.where(
        has_item(prev_state.player.inventory, ItemType.OAK_PLANKS, 1),
        prev_state.player.inventory[
            jnp.argmax(prev_state.player.inventory[:, 0] == ItemType.OAK_PLANKS), 1
        ],
        0
    )
    curr_planks = jnp.where(
        has_planks,
        state.player.inventory[
            jnp.argmax(state.player.inventory[:, 0] == ItemType.OAK_PLANKS), 1
        ],
        0
    )
    planks_gained = jnp.maximum(curr_planks - prev_planks, 0)
    craft_reward = planks_gained * 2.0  # +2.0 per plank crafted (8x stronger)
    total_reward = total_reward + craft_reward
    
    return total_reward, flags
