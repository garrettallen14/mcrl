"""
Crafting system for MinecraftRL.

Implements macro-crafting: single actions that execute full recipes.
This is key for sample efficiency - no need to learn inventory navigation.

Recipes are defined as:
    (output_item, output_count, [(input_item, input_count), ...])
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from mcrl.core.types import ItemType, BlockType
from mcrl.core.state import GameState
from mcrl.systems.inventory import has_item, add_item, remove_item, get_item_count


class Recipe(NamedTuple):
    """A crafting recipe."""
    output: int           # ItemType
    output_count: int     # How many items produced
    inputs: tuple         # ((ItemType, count), ...)
    requires_table: bool  # Needs crafting table nearby


# All recipes for diamond progression
RECIPES = {
    # Basic wood processing
    "planks_from_oak": Recipe(
        output=ItemType.OAK_PLANKS,
        output_count=4,
        inputs=((ItemType.OAK_LOG, 1),),
        requires_table=False,
    ),
    "planks_from_birch": Recipe(
        output=ItemType.OAK_PLANKS,  # Simplified: all planks same
        output_count=4,
        inputs=((ItemType.BIRCH_LOG, 1),),
        requires_table=False,
    ),
    "planks_from_spruce": Recipe(
        output=ItemType.OAK_PLANKS,
        output_count=4,
        inputs=((ItemType.SPRUCE_LOG, 1),),
        requires_table=False,
    ),
    "sticks": Recipe(
        output=ItemType.STICK,
        output_count=4,
        inputs=((ItemType.OAK_PLANKS, 2),),
        requires_table=False,
    ),
    
    # Crafting stations
    "crafting_table": Recipe(
        output=ItemType.CRAFTING_TABLE,
        output_count=1,
        inputs=((ItemType.OAK_PLANKS, 4),),
        requires_table=False,
    ),
    "furnace": Recipe(
        output=ItemType.FURNACE,
        output_count=1,
        inputs=((ItemType.COBBLESTONE, 8),),
        requires_table=True,
    ),
    
    # Tools - Pickaxes
    "wooden_pickaxe": Recipe(
        output=ItemType.WOODEN_PICKAXE,
        output_count=1,
        inputs=((ItemType.OAK_PLANKS, 3), (ItemType.STICK, 2)),
        requires_table=True,
    ),
    "stone_pickaxe": Recipe(
        output=ItemType.STONE_PICKAXE,
        output_count=1,
        inputs=((ItemType.COBBLESTONE, 3), (ItemType.STICK, 2)),
        requires_table=True,
    ),
    "iron_pickaxe": Recipe(
        output=ItemType.IRON_PICKAXE,
        output_count=1,
        inputs=((ItemType.IRON_INGOT, 3), (ItemType.STICK, 2)),
        requires_table=True,
    ),
    "diamond_pickaxe": Recipe(
        output=ItemType.DIAMOND_PICKAXE,
        output_count=1,
        inputs=((ItemType.DIAMOND, 3), (ItemType.STICK, 2)),
        requires_table=True,
    ),
    
    # Tools - Axes
    "wooden_axe": Recipe(
        output=ItemType.WOODEN_AXE,
        output_count=1,
        inputs=((ItemType.OAK_PLANKS, 3), (ItemType.STICK, 2)),
        requires_table=True,
    ),
    "stone_axe": Recipe(
        output=ItemType.STONE_AXE,
        output_count=1,
        inputs=((ItemType.COBBLESTONE, 3), (ItemType.STICK, 2)),
        requires_table=True,
    ),
    
    # Tools - Shovels
    "wooden_shovel": Recipe(
        output=ItemType.WOODEN_SHOVEL,
        output_count=1,
        inputs=((ItemType.OAK_PLANKS, 1), (ItemType.STICK, 2)),
        requires_table=True,
    ),
    
    # Tools - Swords
    "wooden_sword": Recipe(
        output=ItemType.WOODEN_SWORD,
        output_count=1,
        inputs=((ItemType.OAK_PLANKS, 2), (ItemType.STICK, 1)),
        requires_table=True,
    ),
    "stone_sword": Recipe(
        output=ItemType.STONE_SWORD,
        output_count=1,
        inputs=((ItemType.COBBLESTONE, 2), (ItemType.STICK, 1)),
        requires_table=True,
    ),
    
    # Torch
    "torch": Recipe(
        output=ItemType.TORCH,
        output_count=4,
        inputs=((ItemType.COAL, 1), (ItemType.STICK, 1)),
        requires_table=False,
    ),
    "torch_from_charcoal": Recipe(
        output=ItemType.TORCH,
        output_count=4,
        inputs=((ItemType.CHARCOAL, 1), (ItemType.STICK, 1)),
        requires_table=False,
    ),
}


# Smelting recipes: (input, output)
SMELTING_RECIPES = {
    ItemType.IRON_ORE: ItemType.IRON_INGOT,
    ItemType.GOLD_ORE: ItemType.GOLD_INGOT,
    ItemType.OAK_LOG: ItemType.CHARCOAL,
    ItemType.BIRCH_LOG: ItemType.CHARCOAL,
    ItemType.SPRUCE_LOG: ItemType.CHARCOAL,
    ItemType.COBBLESTONE: ItemType.STONE,  # Smooth stone
}

# Valid fuels and their burn time (in items smelted)
FUELS = {
    ItemType.COAL: 8,
    ItemType.CHARCOAL: 8,
    ItemType.OAK_PLANKS: 1,  # Actually 1.5, simplified
    ItemType.OAK_LOG: 1,
    ItemType.STICK: 0,  # 0.5, not enough for full smelt
}


def check_crafting_table_nearby(state: GameState, radius: int = 4) -> jnp.ndarray:
    """Check if there's a crafting table within reach."""
    player_pos = state.player.pos.astype(jnp.int32)
    world = state.world
    
    # Check nearby blocks
    found = jnp.bool_(False)
    
    # Simple check: iterate nearby positions
    for dx in range(-radius, radius + 1):
        for dy in range(-2, 3):  # Check 2 above and below
            for dz in range(-radius, radius + 1):
                check_pos = player_pos + jnp.array([dx, dy, dz])
                
                # Bounds check
                W, H, D = world.shape
                in_bounds = (
                    (check_pos[0] >= 0) & (check_pos[0] < W) &
                    (check_pos[1] >= 0) & (check_pos[1] < H) &
                    (check_pos[2] >= 0) & (check_pos[2] < D)
                )
                
                block = jnp.where(
                    in_bounds,
                    world.blocks[
                        jnp.clip(check_pos[0], 0, W-1),
                        jnp.clip(check_pos[1], 0, H-1),
                        jnp.clip(check_pos[2], 0, D-1)
                    ],
                    BlockType.AIR
                )
                
                found = found | (block == BlockType.CRAFTING_TABLE)
    
    return found


def check_furnace_nearby(state: GameState, radius: int = 4) -> jnp.ndarray:
    """Check if there's a furnace within reach."""
    player_pos = state.player.pos.astype(jnp.int32)
    world = state.world
    
    found = jnp.bool_(False)
    
    for dx in range(-radius, radius + 1):
        for dy in range(-2, 3):
            for dz in range(-radius, radius + 1):
                check_pos = player_pos + jnp.array([dx, dy, dz])
                
                W, H, D = world.shape
                in_bounds = (
                    (check_pos[0] >= 0) & (check_pos[0] < W) &
                    (check_pos[1] >= 0) & (check_pos[1] < H) &
                    (check_pos[2] >= 0) & (check_pos[2] < D)
                )
                
                block = jnp.where(
                    in_bounds,
                    world.blocks[
                        jnp.clip(check_pos[0], 0, W-1),
                        jnp.clip(check_pos[1], 0, H-1),
                        jnp.clip(check_pos[2], 0, D-1)
                    ],
                    BlockType.AIR
                )
                
                found = found | ((block == BlockType.FURNACE) | (block == BlockType.FURNACE_LIT))
    
    return found


def can_craft_recipe(state: GameState, recipe: Recipe) -> jnp.ndarray:
    """Check if a recipe can be crafted."""
    inventory = state.player.inventory
    
    # Check all inputs
    has_inputs = jnp.bool_(True)
    for item_type, count in recipe.inputs:
        has_inputs = has_inputs & has_item(inventory, item_type, count)
    
    # Check crafting table requirement
    needs_table = recipe.requires_table
    has_table = jnp.where(needs_table, check_crafting_table_nearby(state), True)
    
    return has_inputs & has_table


def craft_recipe(state: GameState, recipe: Recipe) -> GameState:
    """Execute a crafting recipe if possible."""
    can_craft = can_craft_recipe(state, recipe)
    
    # Remove inputs
    new_inventory = state.player.inventory
    for item_type, count in recipe.inputs:
        new_inventory, _ = jax.lax.cond(
            can_craft,
            lambda inv: remove_item(inv, item_type, count),
            lambda inv: (inv, False),
            new_inventory
        )
    
    # Add output
    new_inventory = jax.lax.cond(
        can_craft,
        lambda inv: add_item(inv, recipe.output, recipe.output_count),
        lambda inv: inv,
        new_inventory
    )
    
    new_player = state.player.replace(inventory=new_inventory)
    return state.replace(player=new_player)


def smelt_item(state: GameState, input_item: int) -> GameState:
    """
    Smelt an item (macro action: consumes input + fuel, produces output).
    Simplified: instant smelting if fuel available.
    """
    inventory = state.player.inventory
    
    # Check if input is smeltable
    output_item = SMELTING_RECIPES.get(int(input_item), None)
    if output_item is None:
        return state
    
    # Check for furnace nearby
    has_furnace = check_furnace_nearby(state)
    
    # Check for input item
    has_input = has_item(inventory, input_item, 1)
    
    # Check for fuel (try coal first, then others)
    has_coal = has_item(inventory, ItemType.COAL, 1)
    has_charcoal = has_item(inventory, ItemType.CHARCOAL, 1)
    has_planks = has_item(inventory, ItemType.OAK_PLANKS, 1)
    
    has_fuel = has_coal | has_charcoal | has_planks
    
    can_smelt = has_furnace & has_input & has_fuel
    
    # Consume input
    new_inventory, _ = jax.lax.cond(
        can_smelt,
        lambda inv: remove_item(inv, input_item, 1),
        lambda inv: (inv, False),
        inventory
    )
    
    # Consume fuel (priority: coal > charcoal > planks)
    # Simplified: always use 1 fuel per smelt
    def consume_fuel(inv):
        inv, used_coal = remove_item(inv, ItemType.COAL, 1)
        inv, used_charcoal = jax.lax.cond(
            ~used_coal,
            lambda i: remove_item(i, ItemType.CHARCOAL, 1),
            lambda i: (i, False),
            inv
        )
        inv, _ = jax.lax.cond(
            ~used_coal & ~used_charcoal,
            lambda i: remove_item(i, ItemType.OAK_PLANKS, 1),
            lambda i: (i, False),
            inv
        )
        return inv
    
    new_inventory = jax.lax.cond(
        can_smelt,
        consume_fuel,
        lambda inv: inv,
        new_inventory
    )
    
    # Add output
    new_inventory = jax.lax.cond(
        can_smelt,
        lambda inv: add_item(inv, output_item, 1),
        lambda inv: inv,
        new_inventory
    )
    
    new_player = state.player.replace(inventory=new_inventory)
    return state.replace(player=new_player)


# Recipe lookup by action (for action processing)
def process_craft(state: GameState, craft_action: int) -> GameState:
    """Process a crafting action by ID."""
    from mcrl.systems.actions import Action
    
    # Map action to recipe
    # Try multiple log types for planks
    state = jax.lax.cond(
        craft_action == Action.CRAFT_PLANKS,
        lambda s: _craft_planks(s),
        lambda s: s,
        state
    )
    
    state = jax.lax.cond(
        craft_action == Action.CRAFT_STICKS,
        lambda s: craft_recipe(s, RECIPES["sticks"]),
        lambda s: s,
        state
    )
    
    state = jax.lax.cond(
        craft_action == Action.CRAFT_TABLE,
        lambda s: craft_recipe(s, RECIPES["crafting_table"]),
        lambda s: s,
        state
    )
    
    state = jax.lax.cond(
        craft_action == Action.CRAFT_WOOD_PICK,
        lambda s: craft_recipe(s, RECIPES["wooden_pickaxe"]),
        lambda s: s,
        state
    )
    
    state = jax.lax.cond(
        craft_action == Action.CRAFT_STONE_PICK,
        lambda s: craft_recipe(s, RECIPES["stone_pickaxe"]),
        lambda s: s,
        state
    )
    
    state = jax.lax.cond(
        craft_action == Action.CRAFT_FURNACE,
        lambda s: craft_recipe(s, RECIPES["furnace"]),
        lambda s: s,
        state
    )
    
    state = jax.lax.cond(
        craft_action == Action.CRAFT_IRON_PICK,
        lambda s: craft_recipe(s, RECIPES["iron_pickaxe"]),
        lambda s: s,
        state
    )
    
    state = jax.lax.cond(
        craft_action == Action.SMELT_IRON,
        lambda s: smelt_item(s, ItemType.IRON_ORE),
        lambda s: s,
        state
    )
    
    return state


def _craft_planks(state: GameState) -> GameState:
    """Try to craft planks from any log type."""
    # Try oak first
    state = craft_recipe(state, RECIPES["planks_from_oak"])
    # If that didn't work (no oak), try birch
    state = jax.lax.cond(
        ~has_item(state.player.inventory, ItemType.OAK_PLANKS, 4),
        lambda s: craft_recipe(s, RECIPES["planks_from_birch"]),
        lambda s: s,
        state
    )
    # Try spruce
    state = jax.lax.cond(
        ~has_item(state.player.inventory, ItemType.OAK_PLANKS, 4),
        lambda s: craft_recipe(s, RECIPES["planks_from_spruce"]),
        lambda s: s,
        state
    )
    return state
