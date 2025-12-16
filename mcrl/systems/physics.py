"""
Physics engine for MinecraftRL.

Handles:
- Gravity and velocity
- AABB collision detection with voxel world
- Ground detection
- Fall damage

All functions are pure JAX for JIT compilation and vectorization.
"""

import jax
import jax.numpy as jnp
from functools import partial

from mcrl.core.types import BlockType
from mcrl.core.state import GameState, PlayerState, WorldState
from mcrl.core.constants import (
    GRAVITY,
    TERMINAL_VELOCITY,
    PLAYER_WIDTH,
    PLAYER_HEIGHT,
    TICK_DURATION,
    FALL_DAMAGE_THRESHOLD,
    FALL_DAMAGE_PER_BLOCK,
)


def is_solid_block(block_type: jnp.ndarray) -> jnp.ndarray:
    """Check if a block type is solid (collidable)."""
    return (
        (block_type != BlockType.AIR) &
        (block_type != BlockType.WATER) &
        (block_type != BlockType.OAK_LEAVES)  # Can walk through leaves
    )


def is_liquid_block(block_type: jnp.ndarray) -> jnp.ndarray:
    """Check if a block is liquid."""
    return (block_type == BlockType.WATER) | (block_type == BlockType.LAVA)


def get_block_at(world: WorldState, pos: jnp.ndarray) -> jnp.ndarray:
    """Get block type at integer position, with bounds checking."""
    W, H, D = world.shape
    x, y, z = pos[0], pos[1], pos[2]
    
    in_bounds = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H) &
        (z >= 0) & (z < D)
    )
    
    # Clamp to valid indices for the array access
    safe_x = jnp.clip(x, 0, W - 1)
    safe_y = jnp.clip(y, 0, H - 1)
    safe_z = jnp.clip(z, 0, D - 1)
    
    block = world.blocks[safe_x, safe_y, safe_z]
    
    # Out of bounds = solid (like bedrock walls)
    return jnp.where(in_bounds, block, BlockType.BEDROCK)


def check_collision_axis(
    world: WorldState,
    pos: jnp.ndarray,
    new_pos: jnp.ndarray,
    axis: int,
    half_width: float,
    height: float,
) -> tuple[jnp.ndarray, jnp.bool_]:
    """
    Check collision along a single axis and resolve.
    Returns (resolved_position, did_collide).
    """
    # Player bounding box at new position
    if axis == 1:  # Y axis (vertical)
        # Check blocks at feet and head level
        feet_y = jnp.floor(new_pos[1]).astype(jnp.int32)
        head_y = jnp.floor(new_pos[1] + height - 0.01).astype(jnp.int32)
        
        # Sample points around player
        px, pz = jnp.floor(new_pos[0]).astype(jnp.int32), jnp.floor(new_pos[2]).astype(jnp.int32)
        
        # Check if moving down (falling)
        moving_down = new_pos[1] < pos[1]
        
        # Ground collision
        ground_block = get_block_at(world, jnp.array([px, feet_y, pz]))
        ground_collision = is_solid_block(ground_block) & moving_down
        
        # Ceiling collision
        ceiling_block = get_block_at(world, jnp.array([px, head_y, pz]))
        ceiling_collision = is_solid_block(ceiling_block) & ~moving_down
        
        # Resolve
        resolved_y = jnp.where(
            ground_collision,
            (feet_y + 1).astype(jnp.float32),  # Snap to top of block
            jnp.where(
                ceiling_collision,
                (head_y - height).astype(jnp.float32),  # Snap below ceiling
                new_pos[1]
            )
        )
        
        resolved = new_pos.at[1].set(resolved_y)
        collided = ground_collision | ceiling_collision
        
    else:  # X or Z axis (horizontal)
        # Check blocks at player's horizontal extent
        check_y = jnp.floor(pos[1] + 0.5).astype(jnp.int32)  # Mid-height
        
        if axis == 0:  # X axis
            direction = jnp.sign(new_pos[0] - pos[0])
            edge_x = new_pos[0] + direction * half_width
            check_x = jnp.floor(edge_x).astype(jnp.int32)
            check_z = jnp.floor(pos[2]).astype(jnp.int32)
            
            block = get_block_at(world, jnp.array([check_x, check_y, check_z]))
            collision = is_solid_block(block)
            
            # Also check block above (for 2-block tall player)
            block_above = get_block_at(world, jnp.array([check_x, check_y + 1, check_z]))
            collision = collision | is_solid_block(block_above)
            
            resolved_x = jnp.where(
                collision,
                pos[0],  # Revert to old position
                new_pos[0]
            )
            resolved = new_pos.at[0].set(resolved_x)
            
        else:  # Z axis
            direction = jnp.sign(new_pos[2] - pos[2])
            edge_z = new_pos[2] + direction * half_width
            check_z = jnp.floor(edge_z).astype(jnp.int32)
            check_x = jnp.floor(pos[0]).astype(jnp.int32)
            
            block = get_block_at(world, jnp.array([check_x, check_y, check_z]))
            collision = is_solid_block(block)
            
            block_above = get_block_at(world, jnp.array([check_x, check_y + 1, check_z]))
            collision = collision | is_solid_block(block_above)
            
            resolved_z = jnp.where(
                collision,
                pos[2],
                new_pos[2]
            )
            resolved = new_pos.at[2].set(resolved_z)
        
        collided = collision
    
    return resolved, collided


@jax.jit
def apply_physics(state: GameState, dt: float = TICK_DURATION) -> GameState:
    """
    Apply physics for one tick.
    
    Steps:
    1. Apply gravity to velocity
    2. Compute proposed position
    3. Check and resolve collisions per axis
    4. Update ground state
    5. Apply fall damage if landed hard
    """
    player = state.player
    world = state.world
    
    # Store pre-physics Y for fall damage calculation
    prev_y = player.pos[1]
    prev_vel_y = player.vel[1]
    
    # Apply gravity
    new_vel_y = jnp.maximum(
        player.vel[1] + GRAVITY * dt,
        TERMINAL_VELOCITY
    )
    new_vel = player.vel.at[1].set(new_vel_y)
    
    # Proposed new position
    proposed_pos = player.pos + new_vel * dt
    
    # Collision resolution (Y first, then X, then Z)
    half_w = PLAYER_WIDTH / 2
    
    # Y-axis (vertical)
    resolved_pos, y_collided = check_collision_axis(
        world, player.pos, proposed_pos, axis=1, 
        half_width=half_w, height=PLAYER_HEIGHT
    )
    
    # X-axis
    resolved_pos, x_collided = check_collision_axis(
        world, player.pos, resolved_pos, axis=0,
        half_width=half_w, height=PLAYER_HEIGHT
    )
    
    # Z-axis
    resolved_pos, z_collided = check_collision_axis(
        world, player.pos, resolved_pos, axis=2,
        half_width=half_w, height=PLAYER_HEIGHT
    )
    
    # Ground detection
    feet_y = jnp.floor(resolved_pos[1] - 0.01).astype(jnp.int32)
    feet_x = jnp.floor(resolved_pos[0]).astype(jnp.int32)
    feet_z = jnp.floor(resolved_pos[2]).astype(jnp.int32)
    ground_block = get_block_at(world, jnp.array([feet_x, feet_y, feet_z]))
    on_ground = is_solid_block(ground_block)
    
    # Zero vertical velocity if on ground or hit ceiling
    final_vel_y = jnp.where(y_collided, 0.0, new_vel[1])
    final_vel = new_vel.at[1].set(final_vel_y)
    
    # Zero horizontal velocity if hit wall
    final_vel = jnp.where(x_collided, final_vel.at[0].set(0.0), final_vel)
    final_vel = jnp.where(z_collided, final_vel.at[2].set(0.0), final_vel)
    
    # Fall damage calculation
    # Damage when landing from a fall
    was_falling = prev_vel_y < -1.0
    just_landed = on_ground & was_falling & ~player.on_ground
    
    # Calculate fall distance (from velocity, approximate)
    fall_velocity = jnp.abs(prev_vel_y)
    # v² = 2gh -> h = v²/(2g)
    fall_distance = (fall_velocity ** 2) / (2 * jnp.abs(GRAVITY))
    
    damage = jnp.maximum(0.0, fall_distance - FALL_DAMAGE_THRESHOLD) * FALL_DAMAGE_PER_BLOCK
    damage = jnp.where(just_landed, damage, 0.0)
    
    new_health = jnp.maximum(0.0, player.health - damage)
    
    # Update player state
    new_player = player.replace(
        pos=resolved_pos,
        vel=final_vel,
        on_ground=on_ground,
        health=new_health,
    )
    
    return state.replace(player=new_player)
