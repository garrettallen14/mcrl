"""
Agent analytics for training visualization.

Computes lightweight statistics from agent positions for dashboard display.
All functions are JAX-compatible for GPU acceleration.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any


def compute_position_heatmap(
    positions: jnp.ndarray,  # [num_agents, 3] float32
    world_size: tuple[int, int, int],
    grid_size: int = 16,
) -> jnp.ndarray:
    """
    Compute 2D heatmap of agent positions (top-down X-Z view).
    
    Args:
        positions: Agent positions [N, 3] as (x, y, z)
        world_size: World dimensions (W, H, D)
        grid_size: Number of cells per axis in heatmap
    
    Returns:
        heatmap: [grid_size, grid_size] int32 counts
    """
    W, H, D = world_size
    
    # Map positions to grid cells
    cell_x = jnp.clip((positions[:, 0] / W * grid_size).astype(jnp.int32), 0, grid_size - 1)
    cell_z = jnp.clip((positions[:, 2] / D * grid_size).astype(jnp.int32), 0, grid_size - 1)
    
    # Count agents per cell using histogram2d
    # Convert to 1D index for bincount
    cell_idx = cell_x * grid_size + cell_z
    counts = jnp.bincount(cell_idx, length=grid_size * grid_size)
    
    return counts.reshape(grid_size, grid_size)


def compute_depth_histogram(
    positions: jnp.ndarray,  # [num_agents, 3] float32
    world_size: tuple[int, int, int],
    num_bins: int = 20,
) -> jnp.ndarray:
    """
    Compute histogram of agent Y positions (depth distribution).
    
    Args:
        positions: Agent positions [N, 3] as (x, y, z)
        world_size: World dimensions (W, H, D)
        num_bins: Number of histogram bins
    
    Returns:
        histogram: [num_bins] int32 counts
    """
    W, H, D = world_size
    
    # Map Y to bin
    bin_idx = jnp.clip((positions[:, 1] / H * num_bins).astype(jnp.int32), 0, num_bins - 1)
    
    return jnp.bincount(bin_idx, length=num_bins)


def compute_position_stats(
    positions: jnp.ndarray,  # [num_agents, 3] float32
) -> Dict[str, float]:
    """
    Compute summary statistics for agent positions.
    
    Returns dict with:
        - mean_x, mean_y, mean_z
        - std_x, std_y, std_z
        - max_x, max_y, max_z (furthest agents)
    """
    mean = jnp.mean(positions, axis=0)
    std = jnp.std(positions, axis=0)
    max_pos = jnp.max(positions, axis=0)
    min_pos = jnp.min(positions, axis=0)
    
    return {
        'mean_x': float(mean[0]),
        'mean_y': float(mean[1]),
        'mean_z': float(mean[2]),
        'std_x': float(std[0]),
        'std_y': float(std[1]),
        'std_z': float(std[2]),
        'min_y': float(min_pos[1]),  # Deepest agent
        'max_y': float(max_pos[1]),  # Highest agent
        'spread': float(jnp.sqrt(std[0]**2 + std[2]**2)),  # Horizontal spread
    }


def compute_all_analytics(
    env_states,  # Batched environment states
    world_size: tuple[int, int, int],
    heatmap_size: int = 16,
    depth_bins: int = 20,
) -> Dict[str, Any]:
    """
    Compute all analytics for dashboard display.
    
    Args:
        env_states: Batched GameState from vectorized envs
        world_size: World dimensions
        heatmap_size: Grid resolution for heatmap
        depth_bins: Number of depth histogram bins
    
    Returns:
        Dictionary with:
            - heatmap: 2D array [grid, grid]
            - depth_hist: 1D array [bins]
            - stats: position statistics dict
    """
    # Extract positions from batched states
    # env_states.player.pos has shape [num_envs, 3]
    positions = env_states.player.pos
    
    heatmap = compute_position_heatmap(positions, world_size, heatmap_size)
    depth_hist = compute_depth_histogram(positions, world_size, depth_bins)
    stats = compute_position_stats(positions)
    
    return {
        'heatmap': heatmap.tolist(),  # Convert to Python list for JSON
        'depth_histogram': depth_hist.tolist(),
        'position_stats': stats,
        'num_agents': int(positions.shape[0]),
    }


# JIT-compiled version for performance
@jax.jit
def _compute_heatmap_jit(positions, world_w, world_d, grid_size):
    """JIT-compiled heatmap computation."""
    cell_x = jnp.clip((positions[:, 0] / world_w * grid_size).astype(jnp.int32), 0, grid_size - 1)
    cell_z = jnp.clip((positions[:, 2] / world_d * grid_size).astype(jnp.int32), 0, grid_size - 1)
    cell_idx = cell_x * grid_size + cell_z
    counts = jnp.bincount(cell_idx, length=grid_size * grid_size)
    return counts.reshape(grid_size, grid_size)


@jax.jit
def _compute_depth_hist_jit(positions, world_h, num_bins):
    """JIT-compiled depth histogram."""
    bin_idx = jnp.clip((positions[:, 1] / world_h * num_bins).astype(jnp.int32), 0, num_bins - 1)
    return jnp.bincount(bin_idx, length=num_bins)
