#!/usr/bin/env python3
"""
MCRL Pygame Visualization Suite

Renders the agent's view of the Minecraft world in real-time using pygame.
Shows top-down map, first-person view, inventory, and agent behavior.

Usage:
    python experiments/scripts/eval_pygame.py --checkpoint path/to/params.pkl
    python experiments/scripts/eval_pygame.py  # Uses latest checkpoint
"""

import argparse
import pickle
import time
import os
import sys
import math
from pathlib import Path

import pygame
import numpy as np

import jax
import jax.numpy as jnp

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcrl import MinecraftEnv, EnvConfig
from mcrl.core.types import BlockType, ItemType
from mcrl.training.networks_fast import create_fast_network


# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & RENDERING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Block colors (R, G, B)
BLOCK_COLORS = {
    BlockType.AIR: (135, 206, 235),         # Sky blue
    BlockType.STONE: (128, 128, 128),       # Gray
    BlockType.DIRT: (139, 90, 43),          # Brown
    BlockType.GRASS: (34, 139, 34),         # Forest green
    BlockType.COBBLESTONE: (105, 105, 105), # Dim gray
    BlockType.OAK_PLANKS: (222, 184, 135),  # Burlywood
    BlockType.OAK_LOG: (101, 67, 33),       # Dark brown
    BlockType.OAK_LEAVES: (50, 205, 50),    # Lime green
    BlockType.BEDROCK: (48, 48, 48),        # Dark gray
    BlockType.WATER: (30, 144, 255),        # Dodger blue
    BlockType.SAND: (238, 214, 175),        # Wheat
    BlockType.COAL_ORE: (54, 54, 54),       # Coal color
    BlockType.IRON_ORE: (210, 180, 140),    # Tan
    BlockType.GOLD_ORE: (255, 215, 0),      # Gold
    BlockType.DIAMOND_ORE: (0, 255, 255),   # Cyan
    BlockType.BIRCH_LOG: (245, 245, 220),   # Beige
    BlockType.SPRUCE_LOG: (62, 39, 35),     # Dark wood
}

# Action names for display
ACTION_NAMES = [
    "NOOP", "FORWARD", "BACK", "LEFT", "RIGHT",
    "JUMP", "SNEAK", "SPRINT",
    "LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP", "LOOK_DOWN",
    "ATTACK", "USE",
    "SLOT_1", "SLOT_2", "SLOT_3", "SLOT_4", "SLOT_5",
    "SLOT_6", "SLOT_7", "SLOT_8", "SLOT_9",
    "CRAFT_PLANKS", "CRAFT_STICKS"
]

# Item names
ITEM_NAMES = {
    0: "Empty",
    1: "Oak Log",
    2: "Oak Planks", 
    3: "Stick",
    4: "Wood Pick",
    5: "Stone Pick",
    6: "Iron Pick",
    7: "Cobblestone",
    8: "Coal",
    9: "Iron Ingot",
}

# Window dimensions
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# Panel sizes
MAP_SIZE = 400          # Top-down map
FPV_WIDTH = 600         # First-person view
FPV_HEIGHT = 400
SIDE_VIEW_HEIGHT = 200


def get_block_color(block_type: int) -> tuple:
    """Get RGB color for block type."""
    return BLOCK_COLORS.get(block_type, (255, 0, 255))  # Magenta for unknown


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_top_down_map(surface: pygame.Surface, voxels: np.ndarray, 
                        player_pos: np.ndarray, player_yaw: float,
                        rect: pygame.Rect):
    """Render top-down view of the local voxels with proper block representation."""
    size = voxels.shape[0]  # 17x17x17
    cell_size = rect.width // size
    
    # Fill background with dark
    pygame.draw.rect(surface, (20, 25, 30), rect)
    
    # First pass: draw terrain
    for x in range(size):
        for z in range(size):
            # Find highest non-air, non-leaf block for ground
            ground_block = BlockType.AIR
            tree_block = BlockType.AIR
            ground_y = 0
            
            for y in range(size - 1, -1, -1):
                block = voxels[x, y, z]
                if block != BlockType.AIR:
                    if block in [BlockType.OAK_LEAVES, BlockType.OAK_LOG, 
                                BlockType.BIRCH_LOG, BlockType.SPRUCE_LOG]:
                        tree_block = block
                    else:
                        ground_block = block
                        ground_y = y
                        break
            
            # Use ground block for base terrain
            if ground_block == BlockType.AIR and tree_block != BlockType.AIR:
                # Tree on air (shouldn't happen but handle it)
                color = get_block_color(tree_block)
            elif ground_block != BlockType.AIR:
                color = get_block_color(ground_block)
            else:
                color = (20, 25, 30)  # Empty/void
            
            # Draw cell with slight gradient based on height
            height_shade = min(1.0, 0.7 + ground_y / size * 0.3)
            shaded_color = tuple(int(c * height_shade) for c in color)
            
            cell_rect = pygame.Rect(
                rect.x + x * cell_size,
                rect.y + z * cell_size,
                cell_size,
                cell_size
            )
            pygame.draw.rect(surface, shaded_color, cell_rect)
    
    # Second pass: draw trees on top
    for x in range(size):
        for z in range(size):
            for y in range(size - 1, -1, -1):
                block = voxels[x, y, z]
                if block == BlockType.OAK_LOG or block == BlockType.BIRCH_LOG or block == BlockType.SPRUCE_LOG:
                    # Draw tree trunk
                    cx = rect.x + x * cell_size + cell_size // 2
                    cz = rect.y + z * cell_size + cell_size // 2
                    trunk_color = get_block_color(block)
                    pygame.draw.circle(surface, trunk_color, (cx, cz), cell_size // 3 + 1)
                    break
                elif block == BlockType.OAK_LEAVES:
                    # Draw leaves
                    cx = rect.x + x * cell_size + cell_size // 2
                    cz = rect.y + z * cell_size + cell_size // 2
                    pygame.draw.circle(surface, (34, 139, 34), (cx, cz), cell_size // 2)
                    break
    
    # Draw grid lines
    for i in range(size + 1):
        # Vertical lines
        x = rect.x + i * cell_size
        pygame.draw.line(surface, (40, 45, 50), (x, rect.y), (x, rect.y + rect.height), 1)
        # Horizontal lines
        y = rect.y + i * cell_size
        pygame.draw.line(surface, (40, 45, 50), (rect.x, y), (rect.x + rect.width, y), 1)
    
    # Draw player marker
    center = size // 2
    player_x = rect.x + center * cell_size + cell_size // 2
    player_z = rect.y + center * cell_size + cell_size // 2
    
    # Player triangle pointing in facing direction
    yaw_rad = math.radians(player_yaw)
    arrow_len = cell_size * 2.0
    
    # Draw FOV cone
    fov_half = math.radians(35)
    fov_len = cell_size * 5
    left_fov_x = player_x + math.sin(yaw_rad - fov_half) * fov_len
    left_fov_z = player_z + math.cos(yaw_rad - fov_half) * fov_len
    right_fov_x = player_x + math.sin(yaw_rad + fov_half) * fov_len
    right_fov_z = player_z + math.cos(yaw_rad + fov_half) * fov_len
    
    # Semi-transparent FOV (draw as lines since pygame doesn't do alpha easily)
    pygame.draw.polygon(surface, (60, 60, 100), [
        (player_x, player_z), (left_fov_x, left_fov_z), (right_fov_x, right_fov_z)
    ])
    
    # Arrow points
    tip_x = player_x + math.sin(yaw_rad) * arrow_len
    tip_z = player_z + math.cos(yaw_rad) * arrow_len
    
    left_x = player_x + math.sin(yaw_rad + 2.5) * arrow_len * 0.5
    left_z = player_z + math.cos(yaw_rad + 2.5) * arrow_len * 0.5
    
    right_x = player_x + math.sin(yaw_rad - 2.5) * arrow_len * 0.5
    right_z = player_z + math.cos(yaw_rad - 2.5) * arrow_len * 0.5
    
    # Player shadow
    pygame.draw.polygon(surface, (150, 50, 50), [
        (tip_x + 2, tip_z + 2), (left_x + 2, left_z + 2), (right_x + 2, right_z + 2)
    ])
    # Player arrow
    pygame.draw.polygon(surface, (255, 80, 80), [
        (tip_x, tip_z), (left_x, left_z), (right_x, right_z)
    ])
    pygame.draw.polygon(surface, (255, 200, 200), [
        (tip_x, tip_z), (left_x, left_z), (right_x, right_z)
    ], 2)
    
    # Player center dot
    pygame.draw.circle(surface, (255, 255, 255), (int(player_x), int(player_z)), 4)
    pygame.draw.circle(surface, (255, 80, 80), (int(player_x), int(player_z)), 3)
    
    # Border
    pygame.draw.rect(surface, (80, 80, 100), rect, 3)


def render_side_view(surface: pygame.Surface, voxels: np.ndarray, rect: pygame.Rect):
    """Render side cross-section view (Y vs X at center Z) with enhanced visuals."""
    size = voxels.shape[0]
    center = size // 2
    
    cell_w = rect.width // size
    cell_h = rect.height // size
    
    # Background gradient (sky to underground)
    for y in range(rect.height):
        t = y / rect.height
        if t < 0.3:
            # Sky
            r = int(135 - t * 100)
            g = int(206 - t * 100)
            b = int(235 - t * 50)
        else:
            # Underground darkness
            darkness = (t - 0.3) / 0.7
            r = int(40 * (1 - darkness * 0.5))
            g = int(35 * (1 - darkness * 0.5))
            b = int(30 * (1 - darkness * 0.5))
        pygame.draw.line(surface, (r, g, b), (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))
    
    for x in range(size):
        for y in range(size):
            block = voxels[x, y, center]
            
            if block == BlockType.AIR:
                continue  # Skip air, show background
            
            color = get_block_color(block)
            
            # Flip Y so ground is at bottom
            screen_y = size - 1 - y
            
            # Add depth shading
            depth = abs(x - center) / size
            shade = 1.0 - depth * 0.3
            shaded_color = tuple(int(c * shade) for c in color)
            
            cell_rect = pygame.Rect(
                rect.x + x * cell_w,
                rect.y + screen_y * cell_h,
                cell_w,
                cell_h
            )
            pygame.draw.rect(surface, shaded_color, cell_rect)
            
            # Add subtle border for definition
            if cell_w > 3:
                darker = tuple(int(c * 0.7) for c in shaded_color)
                pygame.draw.rect(surface, darker, cell_rect, 1)
    
    # Draw grid lines (subtle)
    for i in range(0, size + 1, 2):
        x = rect.x + i * cell_w
        pygame.draw.line(surface, (50, 50, 60), (x, rect.y), (x, rect.y + rect.height), 1)
        y = rect.y + i * cell_h
        pygame.draw.line(surface, (50, 50, 60), (rect.x, y), (rect.x + rect.width, y), 1)
    
    # Draw player position marker (eye level)
    player_x = rect.x + center * cell_w + cell_w // 2
    player_y = rect.y + (size - 1 - center - 1) * cell_h + cell_h // 2  # Eye level
    
    # Player glow
    for r in range(12, 4, -2):
        alpha = int(100 * (12 - r) / 8)
        pygame.draw.circle(surface, (255, alpha, alpha), (int(player_x), int(player_y)), r)
    
    # Player marker
    pygame.draw.circle(surface, (255, 100, 100), (int(player_x), int(player_y)), 5)
    pygame.draw.circle(surface, (255, 255, 255), (int(player_x), int(player_y)), 5, 2)
    
    # Border
    pygame.draw.rect(surface, (80, 80, 100), rect, 3)


def render_first_person_view(surface: pygame.Surface, voxels: np.ndarray,
                             facing_blocks: np.ndarray, player_rot: np.ndarray,
                             rect: pygame.Rect):
    """
    Render proper first-person view using raycasting through the voxel grid.
    Wolfenstein 3D style renderer adapted for Minecraft voxels.
    """
    # Constants
    FOV = 70  # Field of view in degrees
    MAX_DIST = 12  # Maximum ray distance
    NUM_RAYS = rect.width // 4  # One ray per 4 pixels for performance
    
    pitch = player_rot[0]  # Up/down angle
    yaw = player_rot[1]    # Left/right angle
    
    # Pre-calculate ray angles
    half_fov = FOV / 2
    ray_angles = np.linspace(-half_fov, half_fov, NUM_RAYS)
    
    # Voxel grid info
    size = voxels.shape[0]  # 17x17x17
    center = size // 2  # Player is at center of voxel grid
    
    # Sky gradient
    for y in range(rect.height // 2):
        # Gradient from light blue at horizon to darker blue at top
        t = y / (rect.height // 2)
        r = int(135 * (1 - t * 0.3))
        g = int(206 * (1 - t * 0.2))
        b = int(250 * (1 - t * 0.1))
        pygame.draw.line(surface, (r, g, b), 
                        (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))
    
    # Ground gradient
    for y in range(rect.height // 2, rect.height):
        t = (y - rect.height // 2) / (rect.height // 2)
        # Gradient from grass to darker at bottom
        r = int(34 + 20 * (1 - t))
        g = int(100 + 39 * (1 - t))
        b = int(34 + 20 * (1 - t))
        pygame.draw.line(surface, (r, g, b),
                        (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))
    
    # Column width for rendering
    col_width = rect.width / NUM_RAYS
    
    # Cast rays
    for ray_idx, angle_offset in enumerate(ray_angles):
        # Ray direction in world space
        ray_yaw = math.radians(yaw + angle_offset)
        ray_pitch = math.radians(pitch)
        
        # Direction vector
        cos_pitch = math.cos(ray_pitch)
        dir_x = cos_pitch * math.sin(ray_yaw)
        dir_y = -math.sin(ray_pitch)
        dir_z = cos_pitch * math.cos(ray_yaw)
        
        # Ray march through voxel grid
        hit_block = BlockType.AIR
        hit_dist = MAX_DIST
        hit_side = 0  # 0 = x face, 1 = z face, 2 = y face
        
        # Start at player eye position (center of voxel grid, slightly above ground)
        pos_x, pos_y, pos_z = center + 0.5, center + 1.6, center + 0.5
        
        # DDA algorithm for voxel traversal
        step_size = 0.1
        for step in range(int(MAX_DIST / step_size)):
            # Current voxel
            vx = int(pos_x)
            vy = int(pos_y)
            vz = int(pos_z)
            
            # Check bounds
            if 0 <= vx < size and 0 <= vy < size and 0 <= vz < size:
                block = voxels[vx, vy, vz]
                if block != BlockType.AIR:
                    hit_block = block
                    hit_dist = step * step_size
                    
                    # Determine which face was hit for shading
                    frac_x = pos_x - vx
                    frac_z = pos_z - vz
                    if min(frac_x, 1-frac_x) < min(frac_z, 1-frac_z):
                        hit_side = 0  # X face
                    else:
                        hit_side = 1  # Z face
                    break
            
            # Step ray forward
            pos_x += dir_x * step_size
            pos_y += dir_y * step_size
            pos_z += dir_z * step_size
        
        # Render column if we hit something
        if hit_block != BlockType.AIR and hit_dist > 0.1:
            # Calculate wall height on screen
            # Fix fisheye effect by using perpendicular distance
            perp_dist = hit_dist * math.cos(math.radians(angle_offset))
            wall_height = int(rect.height / (perp_dist * 0.8 + 0.3))
            wall_height = min(wall_height, rect.height * 2)
            
            # Get block color
            color = get_block_color(int(hit_block))
            
            # Apply shading based on distance and face
            shade = max(0.3, 1.0 - hit_dist / MAX_DIST * 0.7)
            if hit_side == 0:  # X face is darker
                shade *= 0.7
            elif hit_side == 1:  # Z face is medium
                shade *= 0.85
            
            shaded_color = tuple(int(c * shade) for c in color)
            
            # Draw column
            col_x = rect.x + int(ray_idx * col_width)
            col_top = rect.y + rect.height // 2 - wall_height // 2
            
            # Draw the wall strip
            wall_rect = pygame.Rect(col_x, col_top, max(1, int(col_width) + 1), wall_height)
            pygame.draw.rect(surface, shaded_color, wall_rect)
            
            # Add edge highlighting for closer blocks
            if hit_dist < 3 and col_width > 2:
                # Darker edge on right side
                edge_color = tuple(int(c * 0.7) for c in shaded_color)
                pygame.draw.line(surface, edge_color, 
                               (col_x + int(col_width), col_top),
                               (col_x + int(col_width), col_top + wall_height))
    
    # Draw crosshair
    center_x = rect.x + rect.width // 2
    center_y = rect.y + rect.height // 2
    
    # White crosshair with black outline for visibility
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        pygame.draw.line(surface, (0, 0, 0), 
                        (center_x - 12 + dx, center_y + dy), 
                        (center_x + 12 + dx, center_y + dy), 3)
        pygame.draw.line(surface, (0, 0, 0),
                        (center_x + dx, center_y - 12 + dy),
                        (center_x + dx, center_y + 12 + dy), 3)
    pygame.draw.line(surface, (255, 255, 255), (center_x - 10, center_y), (center_x + 10, center_y), 2)
    pygame.draw.line(surface, (255, 255, 255), (center_x, center_y - 10), (center_x, center_y + 10), 2)
    
    # Border
    pygame.draw.rect(surface, (100, 100, 100), rect, 2)


def render_compass(surface: pygame.Surface, log_compass: np.ndarray, 
                   player_yaw: float, rect: pygame.Rect, font: pygame.font.Font):
    """Render compass showing direction to nearest tree."""
    pygame.draw.rect(surface, (30, 30, 40), rect)
    
    center_x = rect.x + rect.width // 2
    center_y = rect.y + rect.height // 2 + 20
    radius = min(rect.width, rect.height) // 2 - 30
    
    # Draw compass circle
    pygame.draw.circle(surface, (60, 60, 70), (center_x, center_y), radius, 3)
    
    # Cardinal directions
    dirs = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
    for label, angle in dirs:
        rad = math.radians(angle)
        x = center_x + math.sin(rad) * (radius + 15)
        y = center_y - math.cos(rad) * (radius + 15)
        text = font.render(label, True, (150, 150, 150))
        surface.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))
    
    # Player direction arrow
    yaw_rad = math.radians(player_yaw)
    arrow_len = radius * 0.7
    px = center_x + math.sin(yaw_rad) * arrow_len
    py = center_y - math.cos(yaw_rad) * arrow_len
    pygame.draw.line(surface, (100, 100, 255), (center_x, center_y), (px, py), 3)
    pygame.draw.circle(surface, (100, 100, 255), (int(px), int(py)), 5)
    
    # Tree direction (from log_compass: forward, right, up, distance)
    forward, right, up, dist = log_compass
    
    if dist < 0.99:  # Tree found
        # Convert player-relative to world direction
        # forward = cos component, right = sin component relative to player facing
        tree_angle_rel = math.atan2(right, forward)
        tree_angle_world = yaw_rad + tree_angle_rel
        
        tree_x = center_x + math.sin(tree_angle_world) * radius * 0.8
        tree_y = center_y - math.cos(tree_angle_world) * radius * 0.8
        
        # Draw tree marker
        pygame.draw.circle(surface, (34, 139, 34), (int(tree_x), int(tree_y)), 10)
        pygame.draw.circle(surface, (139, 90, 43), (int(tree_x), int(tree_y + 5)), 4)
        
        # Distance text
        dist_blocks = dist * 30
        text = font.render(f"Tree: {dist_blocks:.0f}m", True, (100, 255, 100))
    else:
        text = font.render("No trees nearby", True, (150, 150, 150))
    
    surface.blit(text, (rect.x + 10, rect.y + 10))
    
    pygame.draw.rect(surface, (100, 100, 100), rect, 2)


def render_inventory(surface: pygame.Surface, inventory: np.ndarray,
                     rect: pygame.Rect, font: pygame.font.Font):
    """Render inventory panel."""
    pygame.draw.rect(surface, (40, 35, 30), rect)
    
    # Title
    title = font.render("INVENTORY", True, (200, 200, 200))
    surface.blit(title, (rect.x + 10, rect.y + 5))
    
    # Inventory slots
    slot_size = 40
    slots_per_row = 4
    start_y = rect.y + 35
    
    items = [
        ("Log", inventory[0], (101, 67, 33)),
        ("Plank", inventory[1], (222, 184, 135)),
        ("Stick", inventory[2], (139, 90, 43)),
        ("W.Pick", inventory[3], (180, 140, 100)),
        ("Cobble", inventory[4], (128, 128, 128)),
        ("S.Pick", inventory[5], (140, 140, 140)),
        ("Coal", inventory[6], (40, 40, 40)),
        ("Iron", inventory[7], (200, 200, 200)),
    ]
    
    for i, (name, count, color) in enumerate(items):
        row = i // slots_per_row
        col = i % slots_per_row
        
        x = rect.x + 10 + col * (slot_size + 5)
        y = start_y + row * (slot_size + 20)
        
        # Slot background
        slot_rect = pygame.Rect(x, y, slot_size, slot_size)
        pygame.draw.rect(surface, (60, 55, 50), slot_rect)
        pygame.draw.rect(surface, (80, 75, 70), slot_rect, 2)
        
        # Item color indicator
        if count > 0:
            inner_rect = pygame.Rect(x + 5, y + 5, slot_size - 10, slot_size - 10)
            pygame.draw.rect(surface, color, inner_rect)
            
            # Count
            count_text = font.render(str(int(count)), True, (255, 255, 255))
            surface.blit(count_text, (x + slot_size - count_text.get_width() - 2, 
                                      y + slot_size - count_text.get_height()))
        
        # Item name
        name_text = font.render(name, True, (150, 150, 150))
        surface.blit(name_text, (x, y + slot_size + 2))
    
    pygame.draw.rect(surface, (100, 100, 100), rect, 2)


def render_stats(surface: pygame.Surface, episode: int, step: int, 
                 total_reward: float, fps: float, player_state: dict,
                 rect: pygame.Rect, font: pygame.font.Font, big_font: pygame.font.Font):
    """Render statistics panel."""
    pygame.draw.rect(surface, (25, 25, 35), rect)
    
    y = rect.y + 10
    
    # Episode info
    text = big_font.render(f"Episode {episode}", True, (255, 200, 100))
    surface.blit(text, (rect.x + 10, y))
    y += 35
    
    # Stats
    stats = [
        f"Step: {step}",
        f"Reward: {total_reward:.2f}",
        f"FPS: {fps:.1f}",
        "",
        f"Pos: ({player_state['pos'][0]:.1f}, {player_state['pos'][1]:.1f}, {player_state['pos'][2]:.1f})",
        f"Yaw: {player_state['yaw']:.1f}°",
        f"Health: {player_state['health']:.0f}/20",
        f"Hunger: {player_state['hunger']:.0f}/20",
    ]
    
    for stat in stats:
        if stat:
            text = font.render(stat, True, (180, 180, 180))
            surface.blit(text, (rect.x + 10, y))
        y += 22
    
    pygame.draw.rect(surface, (100, 100, 100), rect, 2)


def render_actions(surface: pygame.Surface, actions: list, rewards: list,
                   rect: pygame.Rect, font: pygame.font.Font):
    """Render recent actions panel."""
    pygame.draw.rect(surface, (25, 30, 35), rect)
    
    # Title
    title = font.render("RECENT ACTIONS", True, (200, 200, 200))
    surface.blit(title, (rect.x + 10, rect.y + 5))
    
    y = rect.y + 30
    
    for action, reward in zip(actions[-10:], rewards[-10:]):
        action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"ACT_{action}"
        
        # Color based on reward
        if reward > 0.1:
            color = (100, 255, 100)  # Green
        elif reward < -0.1:
            color = (255, 100, 100)  # Red
        else:
            color = (150, 150, 150)  # Gray
        
        text = font.render(f"{action_name:12s} {reward:+.2f}", True, color)
        surface.blit(text, (rect.x + 10, y))
        y += 20
    
    pygame.draw.rect(surface, (100, 100, 100), rect, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualization(
    checkpoint_path: str,
    num_episodes: int = 10,
    max_steps: int = 2000,
    world_size: tuple = (64, 64, 64),
    target_fps: int = 30,
):
    """Run visualization with pygame rendering."""
    
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("MCRL - Minecraft Reinforcement Learning Visualizer")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    
    # Fonts
    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 24, bold=True)
    title_font = pygame.font.SysFont("monospace", 18, bold=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    if isinstance(ckpt, dict) and 'params' in ckpt:
        params = ckpt['params']
        if isinstance(params, dict) and 'params' in params:
            params = params['params']
    else:
        params = ckpt
    
    if 'params' not in params:
        params = {'params': params}
    
    # Create environment and network
    print("Creating environment...")
    env = MinecraftEnv(EnvConfig(world_size=world_size))
    network = create_fast_network(num_actions=25, ultra_fast=False)
    
    # JIT compile action selection
    @jax.jit
    def select_action(params, obs, key):
        batched_obs = {k: v[None, ...] for k, v in obs.items()}
        logits, value = network.apply(params, batched_obs)
        action = jax.random.categorical(key, logits[0])
        return action, logits[0]
    
    # Warm up
    print("JIT compiling...")
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)
    _ = select_action(params, obs, key)
    print("Ready!")
    
    # Main loop
    running = True
    episode = 0
    paused = False
    single_step = False
    speed_multiplier = 1
    
    while running and episode < num_episodes:
        # Reset for new episode
        key, reset_key = jax.random.split(key)
        state, obs = env.reset(reset_key)
        
        total_reward = 0.0
        actions_history = []
        rewards_history = []
        step = 0
        episode_start = time.time()
        
        episode += 1
        
        while step < max_steps and running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RIGHT:
                        single_step = True
                    elif event.key == pygame.K_UP:
                        speed_multiplier = min(10, speed_multiplier + 1)
                    elif event.key == pygame.K_DOWN:
                        speed_multiplier = max(1, speed_multiplier - 1)
                    elif event.key == pygame.K_r:
                        break  # Reset episode
            
            if not running:
                break
            
            # Step simulation (unless paused)
            if not paused or single_step:
                single_step = False
                
                for _ in range(speed_multiplier):
                    key, action_key = jax.random.split(key)
                    action, logits = select_action(params, obs, action_key)
                    action = int(action)
                    
                    state, obs, reward, done, info = env.step(state, jnp.int32(action))
                    reward = float(reward)
                    
                    total_reward += reward
                    actions_history.append(action)
                    rewards_history.append(reward)
                    step += 1
                    
                    if done or step >= max_steps:
                        break
            
            # Calculate FPS
            elapsed = time.time() - episode_start
            fps = step / elapsed if elapsed > 0 else 0
            
            # Get observation data
            voxels = np.array(obs['local_voxels'])
            facing = np.array(obs['facing_blocks'])
            inventory = np.array(obs['inventory'])
            log_compass = np.array(obs.get('log_compass', np.zeros(4)))
            player_state_arr = np.array(obs['player_state'])
            
            player_info = {
                'pos': np.array(state.player.pos),
                'yaw': float(state.player.rot[1]),
                'pitch': float(state.player.rot[0]),
                'health': float(state.player.health),
                'hunger': float(state.player.hunger),
            }
            
            # Clear screen
            screen.fill((20, 20, 25))
            
            # === RENDER PANELS ===
            
            # Title bar
            title = title_font.render("MCRL - Minecraft RL Agent Visualizer", True, (200, 200, 200))
            screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 10))
            
            # Controls help
            controls = font.render("SPACE: Pause | ←→: Step | ↑↓: Speed | R: Reset | ESC: Quit", True, (120, 120, 120))
            screen.blit(controls, (WINDOW_WIDTH // 2 - controls.get_width() // 2, 35))
            
            # Speed indicator
            speed_text = font.render(f"Speed: {speed_multiplier}x", True, (150, 150, 150))
            screen.blit(speed_text, (WINDOW_WIDTH - 100, 10))
            
            if paused:
                pause_text = big_font.render("PAUSED", True, (255, 200, 100))
                screen.blit(pause_text, (WINDOW_WIDTH - 120, 35))
            
            # Top-down map (left)
            map_rect = pygame.Rect(20, 60, MAP_SIZE, MAP_SIZE)
            render_top_down_map(screen, voxels, player_info['pos'], player_info['yaw'], map_rect)
            map_label = font.render("TOP-DOWN VIEW", True, (150, 150, 150))
            screen.blit(map_label, (map_rect.x + map_rect.width // 2 - map_label.get_width() // 2, map_rect.y - 18))
            
            # Side view (below map)
            side_rect = pygame.Rect(20, 480, MAP_SIZE, SIDE_VIEW_HEIGHT)
            render_side_view(screen, voxels, side_rect)
            side_label = font.render("SIDE VIEW (Cross-section)", True, (150, 150, 150))
            screen.blit(side_label, (side_rect.x + side_rect.width // 2 - side_label.get_width() // 2, side_rect.y - 18))
            
            # First-person view (center)
            fpv_rect = pygame.Rect(440, 60, FPV_WIDTH, FPV_HEIGHT)
            render_first_person_view(screen, voxels, facing, np.array([player_info['pitch'], player_info['yaw']]), fpv_rect)
            fpv_label = font.render("FIRST-PERSON VIEW", True, (150, 150, 150))
            screen.blit(fpv_label, (fpv_rect.x + fpv_rect.width // 2 - fpv_label.get_width() // 2, fpv_rect.y - 18))
            
            # Compass (right of FPV)
            compass_rect = pygame.Rect(1060, 60, 160, 180)
            render_compass(screen, log_compass, player_info['yaw'], compass_rect, font)
            
            # Stats (right)
            stats_rect = pygame.Rect(1060, 260, 320, 200)
            render_stats(screen, episode, step, total_reward, fps, player_info, stats_rect, font, big_font)
            
            # Inventory (below FPV)
            inv_rect = pygame.Rect(440, 480, 300, 180)
            render_inventory(screen, inventory, inv_rect, font)
            
            # Actions (right of inventory)
            actions_rect = pygame.Rect(760, 480, 280, 220)
            render_actions(screen, actions_history, rewards_history, actions_rect, font)
            
            # Action distribution (below compass)
            dist_rect = pygame.Rect(1060, 480, 320, 220)
            pygame.draw.rect(screen, (30, 30, 40), dist_rect)
            dist_title = font.render("ACTION PROBABILITIES", True, (200, 200, 200))
            screen.blit(dist_title, (dist_rect.x + 10, dist_rect.y + 5))
            
            # Show top 8 action probabilities
            probs = jax.nn.softmax(np.array(logits))
            top_actions = np.argsort(probs)[-8:][::-1]
            y = dist_rect.y + 30
            for idx in top_actions:
                prob = probs[idx]
                name = ACTION_NAMES[idx] if idx < len(ACTION_NAMES) else f"ACT_{idx}"
                
                # Bar
                bar_width = int(prob * 150)
                bar_color = (100, 200, 100) if prob > 0.2 else (100, 100, 150)
                pygame.draw.rect(screen, bar_color, (dist_rect.x + 100, y, bar_width, 16))
                
                # Text
                text = font.render(f"{name:12s} {prob:.1%}", True, (180, 180, 180))
                screen.blit(text, (dist_rect.x + 10, y))
                y += 22
            
            pygame.draw.rect(screen, (100, 100, 100), dist_rect, 2)
            
            # Update display
            pygame.display.flip()
            clock.tick(target_fps)
            
            if done:
                break
        
        # Episode complete - brief pause
        print(f"Episode {episode} complete: {step} steps, {total_reward:.2f} reward")
        pygame.time.wait(1000)
    
    pygame.quit()
    print("\n✓ Visualization complete!")


def find_latest_checkpoint(base_dir: str = "experiments/runs") -> str:
    """Find the most recent checkpoint."""
    checkpoints = []
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith('params_') and f.endswith('.pkl'):
                path = os.path.join(root, f)
                checkpoints.append((os.path.getmtime(path), path))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def main():
    parser = argparse.ArgumentParser(description="MCRL Pygame Visualizer")
    parser.add_argument("--checkpoint", "-c", type=str, help="Path to checkpoint file")
    parser.add_argument("--episodes", "-n", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", "-s", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--world-size", "-w", type=int, default=64, help="World size")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("No checkpoint specified, finding latest...")
        checkpoint_path = find_latest_checkpoint()
        print(f"Using: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    run_visualization(
        checkpoint_path=checkpoint_path,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        world_size=(args.world_size, args.world_size, args.world_size),
        target_fps=args.fps,
    )


if __name__ == "__main__":
    main()
