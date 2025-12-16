"""
Procedural world generation for MinecraftRL.

Generates Minecraft-like terrain with:
- Layered terrain (bedrock, stone, dirt, grass)
- Ore distribution at correct depths
- Trees on surface
- Caves (optional, can be enabled)

All functions are pure JAX for JIT compilation.
"""

import jax
import jax.numpy as jnp
from functools import partial

from mcrl.core.types import BlockType
from mcrl.core.state import WorldState
from mcrl.core.constants import (
    BEDROCK_LAYERS,
    SURFACE_LEVEL,
    ORE_DISTRIBUTION,
    TREE_DENSITY,
    TREE_MIN_HEIGHT,
    TREE_MAX_HEIGHT,
)


def _simplex_noise_approx(x: jnp.ndarray, z: jnp.ndarray, 
                          freq: float, phase: jnp.ndarray) -> jnp.ndarray:
    """
    Approximate simplex noise using sin/cos combinations.
    Not true simplex but fast and good enough for terrain.
    """
    return (
        jnp.sin(x * freq + phase[0]) * jnp.cos(z * freq + phase[1]) +
        jnp.sin(x * freq * 1.3 + phase[0] * 0.7) * jnp.sin(z * freq * 0.9 + phase[1] * 1.1) * 0.5
    )


def _generate_height_map(
    key: jax.random.PRNGKey,
    width: int,
    depth: int,
    base_height: int = SURFACE_LEVEL,
) -> jnp.ndarray:
    """Generate 2D height map for terrain surface."""
    x = jnp.arange(width, dtype=jnp.float32)
    z = jnp.arange(depth, dtype=jnp.float32)
    xx, zz = jnp.meshgrid(x, z, indexing='ij')
    
    # Multi-octave noise for natural-looking terrain
    k1, k2, k3 = jax.random.split(key, 3)
    phase1 = jax.random.uniform(k1, (2,)) * 1000
    phase2 = jax.random.uniform(k2, (2,)) * 1000
    phase3 = jax.random.uniform(k3, (2,)) * 1000
    
    # Large features (hills/valleys)
    large = _simplex_noise_approx(xx, zz, 0.01, phase1) * 12
    # Medium features
    medium = _simplex_noise_approx(xx, zz, 0.03, phase2) * 6
    # Small details
    small = _simplex_noise_approx(xx, zz, 0.08, phase3) * 2
    
    height_map = base_height + large + medium + small
    return height_map.astype(jnp.int32)


def _place_ores(
    key: jax.random.PRNGKey,
    blocks: jnp.ndarray,
    height_map: jnp.ndarray,
) -> jnp.ndarray:
    """Place ore blocks in stone areas."""
    W, H, D = blocks.shape
    
    # Create coordinate grids
    y_coords = jnp.arange(H)[None, :, None]
    
    # Random values for ore placement
    key, k_ore = jax.random.split(key)
    ore_noise = jax.random.uniform(k_ore, (W, H, D))
    
    # Stone mask (only place ores in stone)
    is_stone = blocks == BlockType.STONE
    
    # Diamond: y < 16, very rare
    threshold = 0.0
    diamond_mask = is_stone & (y_coords < 16) & (ore_noise < threshold + 0.001)
    blocks = jnp.where(diamond_mask, BlockType.DIAMOND_ORE, blocks)
    threshold += 0.001
    
    # Gold: y < 32, rare
    gold_mask = is_stone & (y_coords < 32) & (ore_noise >= threshold) & (ore_noise < threshold + 0.002)
    blocks = jnp.where(gold_mask, BlockType.GOLD_ORE, blocks)
    threshold += 0.002
    
    # Iron: y < 64, medium
    iron_mask = is_stone & (y_coords < 64) & (ore_noise >= threshold) & (ore_noise < threshold + 0.008)
    blocks = jnp.where(iron_mask, BlockType.IRON_ORE, blocks)
    threshold += 0.008
    
    # Coal: everywhere, common
    coal_mask = is_stone & (ore_noise >= threshold) & (ore_noise < threshold + 0.015)
    blocks = jnp.where(coal_mask, BlockType.COAL_ORE, blocks)
    
    return blocks


def _place_trees_vectorized(
    key: jax.random.PRNGKey,
    blocks: jnp.ndarray,
    height_map: jnp.ndarray,
    density: float = TREE_DENSITY,
) -> jnp.ndarray:
    """
    Place trees on grass blocks.
    Simplified: just logs, no leaves for now (faster).
    """
    W, H, D = blocks.shape
    
    # Determine tree positions
    key, k_tree, k_height = jax.random.split(key, 3)
    tree_noise = jax.random.uniform(k_tree, (W, D))
    tree_heights = jax.random.randint(k_height, (W, D), TREE_MIN_HEIGHT, TREE_MAX_HEIGHT + 1)
    
    # Trees only on grass, with spacing
    valid_tree = tree_noise < density
    
    # For each potential tree position, place log column
    x_coords = jnp.arange(W)[:, None, None]
    z_coords = jnp.arange(D)[None, None, :]
    y_coords = jnp.arange(H)[None, :, None]
    
    # Surface height at each xz
    surface = height_map[:, None, :]  # [W, 1, D]
    
    # Tree trunk mask: above surface, below surface + tree_height, at tree position
    tree_mask = (
        valid_tree[:, None, :] &
        (y_coords > surface) &
        (y_coords <= surface + tree_heights[:, None, :])
    )
    
    # Only place in air
    is_air = blocks == BlockType.AIR
    blocks = jnp.where(tree_mask & is_air, BlockType.OAK_LOG, blocks)
    
    # Add leaves around top (simplified sphere)
    leaf_y = surface + tree_heights[:, None, :]
    leaf_mask = (
        valid_tree[:, None, :] &
        (jnp.abs(y_coords - leaf_y) <= 2) &
        (y_coords > surface + 2)
    )
    blocks = jnp.where(leaf_mask & is_air & (blocks != BlockType.OAK_LOG), 
                       BlockType.OAK_LEAVES, blocks)
    
    return blocks


@partial(jax.jit, static_argnums=(1, 2, 3))
def generate_world(
    key: jax.random.PRNGKey,
    width: int = 256,
    height: int = 128, 
    depth: int = 256,
) -> WorldState:
    """
    Generate a complete Minecraft-like world.
    
    Args:
        key: JAX random key
        width: World width (x-axis)
        height: World height (y-axis, vertical)
        depth: World depth (z-axis)
    
    Returns:
        WorldState with generated terrain
    """
    key, k_height, k_ore, k_tree = jax.random.split(key, 4)
    
    # Generate height map
    height_map = _generate_height_map(k_height, width, depth)
    
    # Initialize with air
    blocks = jnp.zeros((width, height, depth), dtype=jnp.uint8)
    
    # Create coordinate grids
    y_coords = jnp.arange(height)[None, :, None]  # [1, H, 1]
    
    # Layer 0-4: Bedrock
    bedrock_mask = y_coords < BEDROCK_LAYERS
    blocks = jnp.where(bedrock_mask, BlockType.BEDROCK, blocks)
    
    # Layer 5 to surface-4: Stone
    surface_y = height_map[:, None, :]  # [W, 1, D]
    stone_mask = (y_coords >= BEDROCK_LAYERS) & (y_coords < surface_y - 3)
    blocks = jnp.where(stone_mask, BlockType.STONE, blocks)
    
    # Surface-3 to surface-1: Dirt
    dirt_mask = (y_coords >= surface_y - 3) & (y_coords < surface_y)
    blocks = jnp.where(dirt_mask, BlockType.DIRT, blocks)
    
    # Surface: Grass
    grass_mask = y_coords == surface_y
    blocks = jnp.where(grass_mask, BlockType.GRASS, blocks)
    
    # Place ores in stone
    blocks = _place_ores(k_ore, blocks, height_map)
    
    # Place trees
    blocks = _place_trees_vectorized(k_tree, blocks, height_map)
    
    return WorldState(
        blocks=blocks,
        tick=jnp.int32(0),
        seed=key[0],
    )


def find_spawn_position(
    key: jax.random.PRNGKey,
    world: WorldState,
    margin: int = 32,
) -> jnp.ndarray:
    """Find a valid spawn position on the surface."""
    W, H, D = world.shape
    
    # Random x, z within margins
    key, k_x, k_z = jax.random.split(key, 3)
    spawn_x = jax.random.randint(k_x, (), margin, W - margin)
    spawn_z = jax.random.randint(k_z, (), margin, D - margin)
    
    # Find surface y (highest non-air block + 1)
    column = world.blocks[spawn_x, :, spawn_z]
    is_solid = column != BlockType.AIR
    # Find highest solid block
    y_indices = jnp.arange(H)
    highest_solid = jnp.where(is_solid, y_indices, 0).max()
    spawn_y = highest_solid + 1
    
    return jnp.array([spawn_x, spawn_y, spawn_z], dtype=jnp.float32)
