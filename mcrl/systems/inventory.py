"""
Inventory management for MinecraftRL.

Inventory format: [INVENTORY_SIZE, 2] where each row is (item_type, count)
- item_type: ItemType enum value (uint16)
- count: Stack count (uint16, max 64 for most items)

All functions are pure JAX for JIT compilation.
"""

import jax
import jax.numpy as jnp

from mcrl.core.types import ItemType
from mcrl.core.constants import INVENTORY_SIZE


def get_item_count(inventory: jnp.ndarray, item_type: int) -> jnp.ndarray:
    """Get total count of an item type across all slots."""
    matches = inventory[:, 0] == item_type
    counts = jnp.where(matches, inventory[:, 1], 0)
    return counts.sum()


def has_item(inventory: jnp.ndarray, item_type: int, count: int = 1) -> jnp.ndarray:
    """Check if inventory contains at least count of item_type."""
    return get_item_count(inventory, item_type) >= count


def find_slot_with_item(inventory: jnp.ndarray, item_type: int) -> jnp.ndarray:
    """Find first slot containing item_type, returns -1 if not found."""
    matches = inventory[:, 0] == item_type
    slot_indices = jnp.arange(INVENTORY_SIZE)
    # Return first matching slot or -1
    found_slot = jnp.where(matches, slot_indices, INVENTORY_SIZE).min()
    return jnp.where(found_slot < INVENTORY_SIZE, found_slot, -1)


def find_empty_slot(inventory: jnp.ndarray) -> jnp.ndarray:
    """Find first empty slot, returns -1 if inventory is full."""
    empty = (inventory[:, 0] == ItemType.EMPTY) | (inventory[:, 1] == 0)
    slot_indices = jnp.arange(INVENTORY_SIZE)
    found_slot = jnp.where(empty, slot_indices, INVENTORY_SIZE).min()
    return jnp.where(found_slot < INVENTORY_SIZE, found_slot, -1)


def find_slot_for_item(inventory: jnp.ndarray, item_type: int, max_stack: int = 64) -> jnp.ndarray:
    """
    Find best slot to add item: existing stack with space, or empty slot.
    Returns -1 if no valid slot.
    """
    # First try to find existing stack with space
    has_item_mask = inventory[:, 0] == item_type
    has_space_mask = inventory[:, 1] < max_stack
    stackable = has_item_mask & has_space_mask
    
    slot_indices = jnp.arange(INVENTORY_SIZE)
    existing_slot = jnp.where(stackable, slot_indices, INVENTORY_SIZE).min()
    
    # If no existing stack, find empty slot
    empty_slot = find_empty_slot(inventory)
    
    # Prefer existing stack
    return jnp.where(existing_slot < INVENTORY_SIZE, existing_slot, empty_slot)


def add_item(
    inventory: jnp.ndarray, 
    item_type: int, 
    count: int = 1,
    max_stack: int = 64,
) -> jnp.ndarray:
    """
    Add items to inventory.
    Returns updated inventory (items may be lost if inventory is full).
    """
    # Find slot to add to
    slot = find_slot_for_item(inventory, item_type, max_stack)
    
    # If no valid slot, return unchanged
    no_slot = slot < 0
    
    # Get current slot contents
    current_type = inventory[jnp.maximum(slot, 0), 0]
    current_count = inventory[jnp.maximum(slot, 0), 1]
    
    # Calculate new count (capped at max_stack)
    is_same_type = current_type == item_type
    is_empty = (current_type == ItemType.EMPTY) | (current_count == 0)
    
    new_count = jnp.where(
        is_same_type | is_empty,
        jnp.minimum(current_count + count, max_stack),
        current_count
    )
    
    new_type = jnp.where(is_empty, item_type, current_type)
    
    # Update inventory
    new_inventory = jnp.where(
        no_slot,
        inventory,
        inventory.at[slot, 0].set(new_type).at[slot, 1].set(new_count)
    )
    
    return new_inventory


def remove_item(
    inventory: jnp.ndarray,
    item_type: int,
    count: int = 1,
) -> tuple[jnp.ndarray, jnp.bool_]:
    """
    Remove items from inventory.
    Returns (updated_inventory, success).
    
    Removes from first slot containing the item.
    For removing from multiple slots, call repeatedly.
    """
    slot = find_slot_with_item(inventory, item_type)
    no_slot = slot < 0
    
    current_count = inventory[jnp.maximum(slot, 0), 1]
    has_enough = current_count >= count
    
    new_count = current_count - count
    
    # Clear slot if empty
    new_type = jnp.where(new_count <= 0, ItemType.EMPTY, item_type)
    new_count = jnp.maximum(0, new_count)
    
    success = ~no_slot & has_enough
    
    new_inventory = jnp.where(
        success,
        inventory.at[slot, 0].set(new_type).at[slot, 1].set(new_count),
        inventory
    )
    
    return new_inventory, success


def remove_items_multi(
    inventory: jnp.ndarray,
    requirements: list[tuple[int, int]],  # [(item_type, count), ...]
) -> tuple[jnp.ndarray, jnp.bool_]:
    """
    Remove multiple item types from inventory.
    All-or-nothing: only removes if ALL requirements are met.
    """
    # First check if we have everything
    has_all = jnp.bool_(True)
    for item_type, count in requirements:
        has_all = has_all & has_item(inventory, item_type, count)
    
    # If we have everything, remove items
    new_inventory = inventory
    for item_type, count in requirements:
        # Remove in a loop (simplified - could be optimized)
        new_inventory, _ = jax.lax.cond(
            has_all,
            lambda inv: remove_item(inv, item_type, count),
            lambda inv: (inv, False),
            new_inventory
        )
    
    return new_inventory, has_all


def get_equipped_item(inventory: jnp.ndarray, equipped_slot: int) -> jnp.ndarray:
    """Get item type in equipped slot."""
    return inventory[equipped_slot, 0]


def get_equipped_tool_tier(inventory: jnp.ndarray, equipped_slot: int) -> int:
    """
    Get tool tier of equipped item.
    Returns 0 (hand) if not a tool.
    """
    item = get_equipped_item(inventory, equipped_slot)
    
    # Pickaxe tiers
    is_wood_pick = item == ItemType.WOODEN_PICKAXE
    is_stone_pick = item == ItemType.STONE_PICKAXE
    is_iron_pick = item == ItemType.IRON_PICKAXE
    is_diamond_pick = item == ItemType.DIAMOND_PICKAXE
    
    tier = jnp.where(is_wood_pick, 1,
           jnp.where(is_stone_pick, 2,
           jnp.where(is_iron_pick, 3,
           jnp.where(is_diamond_pick, 4, 0))))
    
    return tier
