"""
Game constants for MinecraftRL.

All physics and gameplay constants in one place for easy tuning.
"""

import jax.numpy as jnp

# =============================================================================
# World Constants
# =============================================================================

# Default world size (can be overridden)
DEFAULT_WORLD_SIZE = (256, 128, 256)  # (width, height, depth) in blocks

# World generation
SEA_LEVEL = 62
SURFACE_LEVEL = 64  # Average surface height
BEDROCK_LAYERS = 5  # Bottom bedrock layers

# Ore distribution (y_min, y_max, chance per block)
ORE_DISTRIBUTION = {
    "coal": (0, 128, 0.015),      # Common, everywhere
    "iron": (0, 64, 0.008),       # Medium, lower half
    "gold": (0, 32, 0.002),       # Rare, deep
    "diamond": (0, 16, 0.001),    # Very rare, very deep
    "redstone": (0, 16, 0.004),   # Medium, very deep
    "lapis": (0, 32, 0.003),      # Medium-rare, deep
}

# Tree generation
TREE_DENSITY = 0.01  # Probability of tree at valid surface location
TREE_MIN_HEIGHT = 4
TREE_MAX_HEIGHT = 7

# =============================================================================
# Physics Constants
# =============================================================================

# Gravity (blocks per second squared)
GRAVITY = -32.0

# Terminal velocity (blocks per second) 
TERMINAL_VELOCITY = -78.4

# Player dimensions
PLAYER_WIDTH = 0.6   # blocks
PLAYER_HEIGHT = 1.8  # blocks
PLAYER_EYE_HEIGHT = 1.62  # blocks from feet

# Movement speeds (blocks per second)
WALK_SPEED = 4.317
SPRINT_SPEED = 5.612
SNEAK_SPEED = 1.31
SWIM_SPEED = 2.2
JUMP_VELOCITY = 9.0  # Initial upward velocity

# Player reach
PLAYER_REACH = 4.5  # blocks

# Fall damage
FALL_DAMAGE_THRESHOLD = 3.0  # blocks before taking damage
FALL_DAMAGE_PER_BLOCK = 1.0  # half-hearts per block over threshold

# =============================================================================
# Gameplay Constants
# =============================================================================

# Ticks
TICKS_PER_SECOND = 20
TICK_DURATION = 1.0 / TICKS_PER_SECOND  # 0.05 seconds

# Episode limits
MAX_EPISODE_TICKS = 36000  # 30 minutes at 20 ticks/sec
DEFAULT_TIMEOUT_TICKS = 36000

# Health and hunger (future use)
MAX_HEALTH = 20.0  # 10 hearts
MAX_HUNGER = 20.0  # 10 drumsticks
HUNGER_DEPLETION_RATE = 0.0  # Per tick when moving (disabled for now)
HEALTH_REGEN_THRESHOLD = 18.0  # Hunger level needed to regenerate

# =============================================================================
# Action Space Constants
# =============================================================================

# Camera rotation
CAMERA_TURN_SPEED = 15.0  # Degrees per action
CAMERA_PITCH_MIN = -90.0
CAMERA_PITCH_MAX = 90.0

# Number of discrete actions
NUM_ACTIONS = 25

# =============================================================================
# Observation Space Constants
# =============================================================================

# Local voxel observation size (cube around player)
LOCAL_OBS_RADIUS = 8  # Results in 17x17x17 cube
LOCAL_OBS_SIZE = LOCAL_OBS_RADIUS * 2 + 1

# Inventory size
INVENTORY_SIZE = 36  # Main inventory (not including armor/offhand)
HOTBAR_SIZE = 9

# =============================================================================
# Reward Constants
# =============================================================================

# Milestone rewards (exponential scaling)
MILESTONE_REWARDS = {
    "log": 1.0,
    "planks": 2.0,
    "stick": 4.0,
    "crafting_table": 4.0,
    "wooden_pickaxe": 8.0,
    "cobblestone": 16.0,
    "stone_pickaxe": 32.0,
    "iron_ore": 64.0,
    "furnace": 128.0,
    "iron_ingot": 256.0,
    "iron_pickaxe": 512.0,
    "diamond": 1024.0,
}

# Death penalty
DEATH_PENALTY = -100.0

# =============================================================================
# Smelting Constants
# =============================================================================

# Smelting time in ticks
SMELT_TIME = 200  # 10 seconds

# Fuel burn times in ticks
FUEL_BURN_TIME = {
    "coal": 1600,      # 80 seconds, smelts 8 items
    "charcoal": 1600,
    "planks": 300,     # 15 seconds, smelts 1.5 items
    "log": 300,
    "stick": 100,      # 5 seconds, smelts 0.5 items
}
