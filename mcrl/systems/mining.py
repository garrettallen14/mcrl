"""
Mining and block interaction system for MinecraftRL.

Handles:
- Raycasting to find targeted block
- Block breaking with tool effectiveness
- Block placement
- Block drops
"""

import jax
import jax.numpy as jnp

from mcrl.core.types import (
    BlockType, ItemType, ToolType, ToolCategory,
    BLOCK_DROPS, BLOCK_HARDNESS, TOOL_SPEED_MULTIPLIER,
    BLOCK_TOOL_REQUIREMENTS, TOOL_PROPERTIES,
)
from mcrl.core.state import GameState, WorldState
from mcrl.core.constants import PLAYER_EYE_HEIGHT, PLAYER_REACH
from mcrl.systems.inventory import add_item, get_equipped_item


def raycast_block(
    world: WorldState,
    pos: jnp.ndarray,
    rot: jnp.ndarray,
    max_dist: float = PLAYER_REACH,
) -> tuple[jnp.ndarray, jnp.bool_, jnp.ndarray]:
    """
    Cast ray from player eyes to find targeted block.
    
    Returns:
        hit_pos: Block position [x, y, z] (int32)
        hit: Whether a block was hit
        face: Normal of face hit [-1/0/1 for each axis]
    """
    # Eye position
    eye = pos + jnp.array([0.0, PLAYER_EYE_HEIGHT, 0.0])
    
    # Direction from pitch/yaw (degrees)
    pitch_rad = jnp.deg2rad(rot[0])
    yaw_rad = jnp.deg2rad(rot[1])
    
    direction = jnp.array([
        jnp.cos(pitch_rad) * jnp.sin(yaw_rad),
        -jnp.sin(pitch_rad),
        jnp.cos(pitch_rad) * jnp.cos(yaw_rad),
    ])
    
    # DDA-style raycast through voxel grid
    # Step along ray in small increments
    step_size = 0.1
    num_steps = int(max_dist / step_size)
    
    def ray_step(carry, step_idx):
        prev_pos, hit_pos, hit, face = carry
        
        t = step_idx * step_size
        current_pos = eye + direction * t
        block_pos = jnp.floor(current_pos).astype(jnp.int32)
        
        # Bounds check
        W, H, D = world.shape
        in_bounds = (
            (block_pos[0] >= 0) & (block_pos[0] < W) &
            (block_pos[1] >= 0) & (block_pos[1] < H) &
            (block_pos[2] >= 0) & (block_pos[2] < D)
        )
        
        # Get block type
        block_type = jnp.where(
            in_bounds,
            world.blocks[
                jnp.clip(block_pos[0], 0, W-1),
                jnp.clip(block_pos[1], 0, H-1),
                jnp.clip(block_pos[2], 0, D-1),
            ],
            BlockType.AIR
        )
        
        # Check if solid (not air, water, etc.)
        is_solid = (
            (block_type != BlockType.AIR) &
            (block_type != BlockType.WATER) &
            (block_type != BlockType.OAK_LEAVES)
        )
        
        # First hit
        new_hit = hit | (is_solid & in_bounds)
        new_hit_pos = jnp.where(
            is_solid & ~hit & in_bounds,
            block_pos,
            hit_pos
        )
        
        # Calculate face (which face we entered from)
        prev_block = jnp.floor(prev_pos).astype(jnp.int32)
        diff = block_pos - prev_block
        new_face = jnp.where(
            is_solid & ~hit & in_bounds,
            -jnp.sign(diff).astype(jnp.int32),
            face
        )
        
        return (current_pos, new_hit_pos, new_hit, new_face), None
    
    init = (
        eye,
        jnp.array([0, 0, 0], dtype=jnp.int32),
        jnp.bool_(False),
        jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    
    (_, hit_pos, hit, face), _ = jax.lax.scan(
        ray_step, init, jnp.arange(num_steps)
    )
    
    return hit_pos, hit, face


def get_tool_tier(item_type: jnp.ndarray) -> jnp.ndarray:
    """Get tool tier from item type."""
    tier = jnp.where(item_type == ItemType.WOODEN_PICKAXE, ToolType.WOODEN,
           jnp.where(item_type == ItemType.STONE_PICKAXE, ToolType.STONE,
           jnp.where(item_type == ItemType.IRON_PICKAXE, ToolType.IRON,
           jnp.where(item_type == ItemType.DIAMOND_PICKAXE, ToolType.DIAMOND,
           jnp.where(item_type == ItemType.WOODEN_AXE, ToolType.WOODEN,
           jnp.where(item_type == ItemType.STONE_AXE, ToolType.STONE,
           jnp.where(item_type == ItemType.IRON_AXE, ToolType.IRON,
           jnp.where(item_type == ItemType.WOODEN_SHOVEL, ToolType.WOODEN,
           jnp.where(item_type == ItemType.STONE_SHOVEL, ToolType.STONE,
           ToolType.HAND)))))))))
    return tier


def get_tool_category(item_type: jnp.ndarray) -> jnp.ndarray:
    """Get tool category from item type."""
    is_pickaxe = (
        (item_type == ItemType.WOODEN_PICKAXE) |
        (item_type == ItemType.STONE_PICKAXE) |
        (item_type == ItemType.IRON_PICKAXE) |
        (item_type == ItemType.DIAMOND_PICKAXE)
    )
    
    is_axe = (
        (item_type == ItemType.WOODEN_AXE) |
        (item_type == ItemType.STONE_AXE) |
        (item_type == ItemType.IRON_AXE)
    )
    
    is_shovel = (
        (item_type == ItemType.WOODEN_SHOVEL) |
        (item_type == ItemType.STONE_SHOVEL) |
        (item_type == ItemType.IRON_SHOVEL)
    )
    
    category = jnp.where(is_pickaxe, ToolCategory.PICKAXE,
               jnp.where(is_axe, ToolCategory.AXE,
               jnp.where(is_shovel, ToolCategory.SHOVEL,
               ToolCategory.NONE)))
    return category


def get_break_time(block_type: jnp.ndarray, tool_tier: jnp.ndarray, tool_cat: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate time to break a block in ticks.
    Returns -1 if block cannot be broken with current tool.
    """
    # Base hardness lookup (simplified - would be dict lookup in real code)
    base_time = jnp.where(block_type == BlockType.STONE, 30,
                jnp.where(block_type == BlockType.COBBLESTONE, 40,
                jnp.where(block_type == BlockType.DIRT, 10,
                jnp.where(block_type == BlockType.GRASS, 12,
                jnp.where(block_type == BlockType.SAND, 10,
                jnp.where(block_type == BlockType.OAK_LOG, 40,
                jnp.where(block_type == BlockType.OAK_PLANKS, 40,
                jnp.where(block_type == BlockType.COAL_ORE, 60,
                jnp.where(block_type == BlockType.IRON_ORE, 60,
                jnp.where(block_type == BlockType.GOLD_ORE, 60,
                jnp.where(block_type == BlockType.DIAMOND_ORE, 60,
                jnp.where(block_type == BlockType.OAK_LEAVES, 4,
                jnp.where(block_type == BlockType.BEDROCK, -1,
                20)))))))))))))  # Default
    
    # Tool speed multiplier
    speed_mult = jnp.where(tool_tier == ToolType.HAND, 1.0,
                 jnp.where(tool_tier == ToolType.WOODEN, 2.0,
                 jnp.where(tool_tier == ToolType.STONE, 4.0,
                 jnp.where(tool_tier == ToolType.IRON, 6.0,
                 jnp.where(tool_tier == ToolType.DIAMOND, 8.0,
                 1.0)))))
    
    # Check if correct tool category (pickaxe for stone/ore, etc.)
    needs_pickaxe = (
        (block_type == BlockType.STONE) |
        (block_type == BlockType.COBBLESTONE) |
        (block_type == BlockType.COAL_ORE) |
        (block_type == BlockType.IRON_ORE) |
        (block_type == BlockType.GOLD_ORE) |
        (block_type == BlockType.DIAMOND_ORE) |
        (block_type == BlockType.FURNACE)
    )
    
    has_pickaxe = tool_cat == ToolCategory.PICKAXE
    
    # Check minimum tier for ore
    min_tier_stone = (block_type == BlockType.IRON_ORE)  # Needs stone+
    min_tier_iron = (
        (block_type == BlockType.GOLD_ORE) |
        (block_type == BlockType.DIAMOND_ORE) |
        (block_type == BlockType.REDSTONE_ORE)
    )  # Needs iron+
    
    tier_ok = jnp.where(
        min_tier_iron,
        tool_tier >= ToolType.IRON,
        jnp.where(
            min_tier_stone,
            tool_tier >= ToolType.STONE,
            True
        )
    )
    
    # Apply tool effectiveness
    effective_speed = jnp.where(
        needs_pickaxe & has_pickaxe & tier_ok,
        speed_mult,
        jnp.where(needs_pickaxe & ~has_pickaxe, 0.3, 1.0)  # Wrong tool = slow
    )
    
    # Can't break without right tool tier
    can_break = jnp.where(needs_pickaxe, has_pickaxe & tier_ok, True)
    
    break_time = jnp.where(
        can_break & (base_time > 0),
        (base_time / effective_speed).astype(jnp.int32),
        jnp.where(base_time < 0, -1, 100)  # Long time if wrong tool
    )
    
    return break_time


def get_block_drop(block_type: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get item drop for a block type. Returns (item_type, count)."""
    item = jnp.where(block_type == BlockType.STONE, ItemType.COBBLESTONE,
           jnp.where(block_type == BlockType.DIRT, ItemType.DIRT,
           jnp.where(block_type == BlockType.GRASS, ItemType.DIRT,
           jnp.where(block_type == BlockType.SAND, ItemType.SAND,
           jnp.where(block_type == BlockType.GRAVEL, ItemType.GRAVEL,
           jnp.where(block_type == BlockType.OAK_LOG, ItemType.OAK_LOG,
           jnp.where(block_type == BlockType.BIRCH_LOG, ItemType.BIRCH_LOG,
           jnp.where(block_type == BlockType.SPRUCE_LOG, ItemType.SPRUCE_LOG,
           jnp.where(block_type == BlockType.COAL_ORE, ItemType.COAL,
           jnp.where(block_type == BlockType.IRON_ORE, ItemType.IRON_ORE,
           jnp.where(block_type == BlockType.GOLD_ORE, ItemType.GOLD_ORE,
           jnp.where(block_type == BlockType.DIAMOND_ORE, ItemType.DIAMOND,
           jnp.where(block_type == BlockType.COBBLESTONE, ItemType.COBBLESTONE,
           jnp.where(block_type == BlockType.OAK_PLANKS, ItemType.OAK_PLANKS,
           jnp.where(block_type == BlockType.CRAFTING_TABLE, ItemType.CRAFTING_TABLE,
           jnp.where(block_type == BlockType.FURNACE, ItemType.FURNACE,
           ItemType.EMPTY))))))))))))))))
    
    count = jnp.where(item != ItemType.EMPTY, 1, 0)
    
    return item, count


def process_mining(state: GameState) -> GameState:
    """
    Process one tick of mining/attacking.
    
    Mining is progressive: player must continue attacking the same block
    until it breaks. Switching targets resets progress.
    """
    player = state.player
    world = state.world
    
    # Raycast to find target block
    hit_pos, hit, face = raycast_block(world, player.pos, player.rot)
    
    # Get block type at hit position
    W, H, D = world.shape
    block_type = jnp.where(
        hit,
        world.blocks[
            jnp.clip(hit_pos[0], 0, W-1),
            jnp.clip(hit_pos[1], 0, H-1),
            jnp.clip(hit_pos[2], 0, D-1),
        ],
        BlockType.AIR
    )
    
    # Get equipped tool
    equipped_item = get_equipped_item(player.inventory, player.equipped_slot)
    tool_tier = get_tool_tier(equipped_item)
    tool_cat = get_tool_category(equipped_item)
    
    # Calculate break time
    break_time = get_break_time(block_type, tool_tier, tool_cat)
    
    # Check if same block as before
    same_block = jnp.all(hit_pos == player.mining_block)
    
    # Update mining progress
    new_progress = jnp.where(
        hit & same_block,
        player.mining_progress + 1,
        jnp.where(hit, 1, 0)  # Reset if different block
    )
    
    # Check if block breaks
    block_breaks = hit & (new_progress >= break_time) & (break_time > 0)
    
    # Get drop
    drop_item, drop_count = get_block_drop(block_type)
    
    # Update world (remove block)
    new_blocks = jnp.where(
        block_breaks,
        world.blocks.at[hit_pos[0], hit_pos[1], hit_pos[2]].set(BlockType.AIR),
        world.blocks
    )
    
    # Update inventory (add drop)
    new_inventory = jnp.where(
        block_breaks & (drop_item != ItemType.EMPTY),
        add_item(player.inventory, drop_item, drop_count),
        player.inventory
    )
    
    # Update mining state
    new_mining_block = jnp.where(
        block_breaks,
        jnp.array([-1, -1, -1], dtype=jnp.int32),  # Reset after break
        jnp.where(hit, hit_pos, player.mining_block)
    )
    new_mining_progress = jnp.where(block_breaks, 0, new_progress)
    
    # Create updated states
    new_world = world.replace(blocks=new_blocks)
    new_player = player.replace(
        inventory=new_inventory,
        mining_block=new_mining_block,
        mining_progress=new_mining_progress,
    )
    
    return state.replace(world=new_world, player=new_player)


def place_block(state: GameState, item_type: int) -> GameState:
    """Place a block from inventory."""
    player = state.player
    world = state.world
    
    # Raycast to find placement position
    hit_pos, hit, face = raycast_block(world, player.pos, player.rot)
    
    # Place position is hit position + face normal
    place_pos = hit_pos + face
    
    # Bounds check
    W, H, D = world.shape
    in_bounds = (
        (place_pos[0] >= 0) & (place_pos[0] < W) &
        (place_pos[1] >= 0) & (place_pos[1] < H) &
        (place_pos[2] >= 0) & (place_pos[2] < D)
    )
    
    # Check if position is empty
    current_block = jnp.where(
        in_bounds,
        world.blocks[
            jnp.clip(place_pos[0], 0, W-1),
            jnp.clip(place_pos[1], 0, H-1),
            jnp.clip(place_pos[2], 0, D-1),
        ],
        BlockType.STONE
    )
    is_empty = current_block == BlockType.AIR
    
    # Check if player has the item
    from mcrl.systems.inventory import has_item, remove_item
    has_block = has_item(player.inventory, item_type, 1)
    
    # Convert item to block type
    block_to_place = jnp.where(item_type == ItemType.COBBLESTONE, BlockType.COBBLESTONE,
                     jnp.where(item_type == ItemType.OAK_PLANKS, BlockType.OAK_PLANKS,
                     jnp.where(item_type == ItemType.CRAFTING_TABLE, BlockType.CRAFTING_TABLE,
                     jnp.where(item_type == ItemType.FURNACE, BlockType.FURNACE,
                     jnp.where(item_type == ItemType.DIRT, BlockType.DIRT,
                     jnp.where(item_type == ItemType.STONE, BlockType.STONE,
                     BlockType.AIR))))))
    
    can_place = hit & in_bounds & is_empty & has_block & (block_to_place != BlockType.AIR)
    
    # Place block
    new_blocks = jnp.where(
        can_place,
        world.blocks.at[place_pos[0], place_pos[1], place_pos[2]].set(block_to_place),
        world.blocks
    )
    
    # Remove from inventory
    new_inventory, _ = jax.lax.cond(
        can_place,
        lambda inv: remove_item(inv, item_type, 1),
        lambda inv: (inv, False),
        player.inventory
    )
    
    new_world = world.replace(blocks=new_blocks)
    new_player = player.replace(inventory=new_inventory)
    
    return state.replace(world=new_world, player=new_player)
