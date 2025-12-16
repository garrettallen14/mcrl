"""
Action processing for MinecraftRL.

Defines the 25 discrete actions matching diamond_env specification.
Actions are processed as pure functions for JAX compatibility.
"""

import jax
import jax.numpy as jnp
from enum import IntEnum

from mcrl.core.state import GameState
from mcrl.core.types import ItemType
from mcrl.core.constants import (
    WALK_SPEED,
    SPRINT_SPEED,
    SNEAK_SPEED,
    JUMP_VELOCITY,
    CAMERA_TURN_SPEED,
    CAMERA_PITCH_MIN,
    CAMERA_PITCH_MAX,
)


class Action(IntEnum):
    """
    25 discrete actions for Minecraft diamond task.
    
    Movement (6): Basic locomotion
    Camera (4): Looking around
    Interaction (2): Mining and placing
    Crafting (7): Macro-crafting actions
    Block placement (2): Place functional blocks
    Equipment (1): Equip best tool
    Smelting (1): Smelt iron ore
    Modifiers (2): Sprint and sneak
    """
    # Movement
    NOOP = 0
    FORWARD = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    JUMP = 5
    
    # Camera
    TURN_LEFT = 6
    TURN_RIGHT = 7
    LOOK_UP = 8
    LOOK_DOWN = 9
    
    # Interaction
    ATTACK = 10      # Mine/break blocks, attack entities
    USE = 11         # Place blocks, use items, interact
    
    # Crafting (macro actions)
    CRAFT_PLANKS = 12
    CRAFT_STICKS = 13
    CRAFT_TABLE = 14
    CRAFT_WOOD_PICK = 15
    CRAFT_STONE_PICK = 16
    CRAFT_FURNACE = 17
    CRAFT_IRON_PICK = 18
    
    # Block placement
    PLACE_TABLE = 19
    PLACE_FURNACE = 20
    
    # Equipment
    EQUIP_BEST_PICK = 21
    
    # Smelting
    SMELT_IRON = 22
    
    # Modifiers
    SPRINT = 23
    SNEAK = 24


def get_movement_direction(yaw: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get forward and right vectors from yaw angle."""
    yaw_rad = jnp.deg2rad(yaw)
    forward = jnp.array([jnp.sin(yaw_rad), 0.0, jnp.cos(yaw_rad)])
    right = jnp.array([jnp.cos(yaw_rad), 0.0, -jnp.sin(yaw_rad)])
    return forward, right


def process_movement(state: GameState, action: jnp.ndarray) -> GameState:
    """Process movement actions (forward/back/left/right/jump)."""
    player = state.player
    
    # Get movement vectors
    forward, right = get_movement_direction(player.rot[1])
    
    # Base speed (affected by sprint/sneak)
    speed = jnp.where(
        player.is_sprinting,
        SPRINT_SPEED,
        jnp.where(player.is_sneaking, SNEAK_SPEED, WALK_SPEED)
    )
    
    # Calculate horizontal velocity from action
    move_vel = jnp.zeros(3)
    move_vel = jnp.where(action == Action.FORWARD, forward * speed, move_vel)
    move_vel = jnp.where(action == Action.BACK, -forward * speed, move_vel)
    move_vel = jnp.where(action == Action.LEFT, -right * speed, move_vel)
    move_vel = jnp.where(action == Action.RIGHT, right * speed, move_vel)
    
    # Apply horizontal movement (keep vertical velocity)
    new_vel = player.vel.at[0].set(move_vel[0]).at[2].set(move_vel[2])
    
    # Handle jump
    can_jump = player.on_ground & (action == Action.JUMP)
    new_vel = jnp.where(
        can_jump,
        new_vel.at[1].set(JUMP_VELOCITY),
        new_vel
    )
    
    # Stop sprint if not moving forward
    is_moving_forward = (action == Action.FORWARD)
    new_sprinting = player.is_sprinting & is_moving_forward
    
    new_player = player.replace(
        vel=new_vel,
        is_sprinting=new_sprinting,
    )
    
    return state.replace(player=new_player)


def process_camera(state: GameState, action: jnp.ndarray) -> GameState:
    """Process camera rotation actions."""
    player = state.player
    
    # Calculate rotation changes
    pitch_delta = jnp.where(action == Action.LOOK_UP, -CAMERA_TURN_SPEED,
                  jnp.where(action == Action.LOOK_DOWN, CAMERA_TURN_SPEED, 0.0))
    
    yaw_delta = jnp.where(action == Action.TURN_LEFT, -CAMERA_TURN_SPEED,
                jnp.where(action == Action.TURN_RIGHT, CAMERA_TURN_SPEED, 0.0))
    
    # Apply rotation
    new_pitch = jnp.clip(
        player.rot[0] + pitch_delta,
        CAMERA_PITCH_MIN,
        CAMERA_PITCH_MAX
    )
    new_yaw = (player.rot[1] + yaw_delta) % 360.0
    
    new_rot = jnp.array([new_pitch, new_yaw])
    new_player = player.replace(rot=new_rot)
    
    return state.replace(player=new_player)


def process_modifiers(state: GameState, action: jnp.ndarray) -> GameState:
    """Process sprint/sneak toggle actions."""
    player = state.player
    
    # Toggle sprint
    new_sprinting = jnp.where(
        action == Action.SPRINT,
        ~player.is_sprinting,
        player.is_sprinting
    )
    
    # Toggle sneak
    new_sneaking = jnp.where(
        action == Action.SNEAK,
        ~player.is_sneaking,
        player.is_sneaking
    )
    
    # Can't sprint and sneak at same time
    new_sprinting = new_sprinting & ~new_sneaking
    
    new_player = player.replace(
        is_sprinting=new_sprinting,
        is_sneaking=new_sneaking,
    )
    
    return state.replace(player=new_player)


def equip_best_pickaxe(state: GameState) -> GameState:
    """Equip the best pickaxe in inventory."""
    from mcrl.systems.inventory import find_slot_with_item
    
    inventory = state.player.inventory
    
    # Check for pickaxes in order of quality
    diamond_slot = find_slot_with_item(inventory, ItemType.DIAMOND_PICKAXE)
    iron_slot = find_slot_with_item(inventory, ItemType.IRON_PICKAXE)
    stone_slot = find_slot_with_item(inventory, ItemType.STONE_PICKAXE)
    wood_slot = find_slot_with_item(inventory, ItemType.WOODEN_PICKAXE)
    
    # Select best available
    best_slot = jnp.where(diamond_slot >= 0, diamond_slot,
                jnp.where(iron_slot >= 0, iron_slot,
                jnp.where(stone_slot >= 0, stone_slot,
                jnp.where(wood_slot >= 0, wood_slot,
                state.player.equipped_slot))))
    
    # Move to hotbar slot 0 if found in main inventory
    # Simplified: just set equipped slot directly
    new_player = state.player.replace(
        equipped_slot=jnp.where(best_slot >= 0, best_slot, state.player.equipped_slot)
    )
    
    return state.replace(player=new_player)


def process_action(state: GameState, action: jnp.ndarray) -> GameState:
    """
    Process a single action and return updated game state.
    
    This is the main action dispatcher. It handles all 25 action types.
    """
    from mcrl.systems.mining import process_mining, place_block
    from mcrl.systems.crafting import process_craft
    
    # Movement actions (0-5)
    is_movement = (action >= Action.NOOP) & (action <= Action.JUMP)
    state = jax.lax.cond(
        is_movement,
        lambda s: process_movement(s, action),
        lambda s: s,
        state
    )
    
    # Camera actions (6-9)
    is_camera = (action >= Action.TURN_LEFT) & (action <= Action.LOOK_DOWN)
    state = jax.lax.cond(
        is_camera,
        lambda s: process_camera(s, action),
        lambda s: s,
        state
    )
    
    # Attack/mine (10)
    state = jax.lax.cond(
        action == Action.ATTACK,
        lambda s: process_mining(s),
        lambda s: s,
        state
    )
    
    # Use/place (11) - place equipped block
    state = jax.lax.cond(
        action == Action.USE,
        lambda s: place_block(s, s.player.inventory[s.player.equipped_slot, 0]),
        lambda s: s,
        state
    )
    
    # Crafting actions (12-18, 22)
    is_craft = (action >= Action.CRAFT_PLANKS) & (action <= Action.CRAFT_IRON_PICK)
    state = jax.lax.cond(
        is_craft | (action == Action.SMELT_IRON),
        lambda s: process_craft(s, action),
        lambda s: s,
        state
    )
    
    # Place crafting table (19)
    state = jax.lax.cond(
        action == Action.PLACE_TABLE,
        lambda s: place_block(s, ItemType.CRAFTING_TABLE),
        lambda s: s,
        state
    )
    
    # Place furnace (20)
    state = jax.lax.cond(
        action == Action.PLACE_FURNACE,
        lambda s: place_block(s, ItemType.FURNACE),
        lambda s: s,
        state
    )
    
    # Equip best pickaxe (21)
    state = jax.lax.cond(
        action == Action.EQUIP_BEST_PICK,
        equip_best_pickaxe,
        lambda s: s,
        state
    )
    
    # Sprint/sneak modifiers (23-24)
    is_modifier = (action == Action.SPRINT) | (action == Action.SNEAK)
    state = jax.lax.cond(
        is_modifier,
        lambda s: process_modifiers(s, action),
        lambda s: s,
        state
    )
    
    # Reset mining state if not attacking
    state = jax.lax.cond(
        action != Action.ATTACK,
        lambda s: s.replace(player=s.player.replace(
            mining_progress=jnp.int32(0),
            mining_block=jnp.array([-1, -1, -1], dtype=jnp.int32),
        )),
        lambda s: s,
        state
    )
    
    return state
