"""
Core type definitions for MinecraftRL.

Block, Item, and Tool types are defined as integer enums for JAX compatibility.
This design allows easy extension - just add new values to the IntEnum.

Future extensions:
- Add mob types (ZOMBIE, SKELETON, CREEPER, etc.)
- Add biome types
- Add dimension types (OVERWORLD, NETHER, END)
"""

from enum import IntEnum


class BlockType(IntEnum):
    """
    Block types in the world.
    
    Design notes:
    - Values 0-63: Natural blocks (terrain, ores)
    - Values 64-127: Crafted/placed blocks
    - Values 128-191: Reserved for future (liquids, redstone, etc.)
    - Values 192-255: Reserved for mods/extensions
    """
    # Natural terrain (0-31)
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    BEDROCK = 4
    SAND = 5
    GRAVEL = 6
    CLAY = 7
    
    # Wood types (8-15)
    OAK_LOG = 8
    OAK_LEAVES = 9
    BIRCH_LOG = 10
    BIRCH_LEAVES = 11
    SPRUCE_LOG = 12
    SPRUCE_LEAVES = 13
    # Reserved: 14-15 for more wood types
    
    # Ores (16-31)
    COAL_ORE = 16
    IRON_ORE = 17
    GOLD_ORE = 18
    DIAMOND_ORE = 19
    REDSTONE_ORE = 20
    LAPIS_ORE = 21
    EMERALD_ORE = 22
    COPPER_ORE = 23
    # Reserved: 24-31 for more ores
    
    # Liquids (32-39) - simplified for now
    WATER = 32
    LAVA = 33
    
    # Crafted blocks (64-127)
    OAK_PLANKS = 64
    BIRCH_PLANKS = 65
    SPRUCE_PLANKS = 66
    COBBLESTONE = 67
    STONE_BRICKS = 68
    BRICKS = 69
    GLASS = 70
    
    # Functional blocks (80-95)
    CRAFTING_TABLE = 80
    FURNACE = 81
    FURNACE_LIT = 82  # Furnace when active
    CHEST = 83
    # Reserved: 84-95 for more functional blocks
    
    # Decoration (96-127)
    TORCH = 96
    # Reserved: 97-127 for more decoration


class ItemType(IntEnum):
    """
    Item types for inventory.
    
    Design notes:
    - Values 0-63: Match block types where applicable (block items)
    - Values 64-127: Tools
    - Values 128-191: Resources/materials
    - Values 192-255: Special items (future: food, potions, etc.)
    
    Block items share IDs with their block counterparts for easy mapping.
    """
    # Block items (match BlockType)
    STONE = 1
    DIRT = 2
    SAND = 5
    GRAVEL = 6
    OAK_LOG = 8
    BIRCH_LOG = 10
    SPRUCE_LOG = 12
    COAL_ORE = 16  # Silk touch
    IRON_ORE = 17
    GOLD_ORE = 18
    DIAMOND_ORE = 19  # Silk touch
    COBBLESTONE = 67
    OAK_PLANKS = 64
    CRAFTING_TABLE = 80
    FURNACE = 81
    CHEST = 83
    TORCH = 96
    
    # Tools (64-95)
    WOODEN_PICKAXE = 100
    STONE_PICKAXE = 101
    IRON_PICKAXE = 102
    GOLD_PICKAXE = 103
    DIAMOND_PICKAXE = 104
    
    WOODEN_AXE = 105
    STONE_AXE = 106
    IRON_AXE = 107
    GOLD_AXE = 108
    DIAMOND_AXE = 109
    
    WOODEN_SHOVEL = 110
    STONE_SHOVEL = 111
    IRON_SHOVEL = 112
    GOLD_SHOVEL = 113
    DIAMOND_SHOVEL = 114
    
    WOODEN_SWORD = 115
    STONE_SWORD = 116
    IRON_SWORD = 117
    GOLD_SWORD = 118
    DIAMOND_SWORD = 119
    
    # Resources/materials (128-191)
    STICK = 128
    COAL = 129
    IRON_INGOT = 130
    GOLD_INGOT = 131
    DIAMOND = 132
    REDSTONE = 133
    LAPIS = 134
    EMERALD = 135
    COPPER_INGOT = 136
    CHARCOAL = 137
    
    # Future: Food items (192-223)
    # APPLE = 192
    # BREAD = 193
    # etc.
    
    # Empty slot marker
    EMPTY = 255


class ToolType(IntEnum):
    """
    Tool categories for mining speed calculation.
    
    Each tool type has different effectiveness against block types.
    """
    HAND = 0
    WOODEN = 1
    STONE = 2
    IRON = 3
    GOLD = 4  # Fast but low durability
    DIAMOND = 5


class ToolCategory(IntEnum):
    """Tool categories for determining which tool to use."""
    NONE = 0  # Hand
    PICKAXE = 1
    AXE = 2
    SHOVEL = 3
    SWORD = 4


# Mapping from ItemType to (ToolType, ToolCategory)
TOOL_PROPERTIES = {
    ItemType.WOODEN_PICKAXE: (ToolType.WOODEN, ToolCategory.PICKAXE),
    ItemType.STONE_PICKAXE: (ToolType.STONE, ToolCategory.PICKAXE),
    ItemType.IRON_PICKAXE: (ToolType.IRON, ToolCategory.PICKAXE),
    ItemType.GOLD_PICKAXE: (ToolType.GOLD, ToolCategory.PICKAXE),
    ItemType.DIAMOND_PICKAXE: (ToolType.DIAMOND, ToolCategory.PICKAXE),
    ItemType.WOODEN_AXE: (ToolType.WOODEN, ToolCategory.AXE),
    ItemType.STONE_AXE: (ToolType.STONE, ToolCategory.AXE),
    ItemType.IRON_AXE: (ToolType.IRON, ToolCategory.AXE),
    ItemType.GOLD_AXE: (ToolType.GOLD, ToolCategory.AXE),
    ItemType.DIAMOND_AXE: (ToolType.DIAMOND, ToolCategory.AXE),
    ItemType.WOODEN_SHOVEL: (ToolType.WOODEN, ToolCategory.SHOVEL),
    ItemType.STONE_SHOVEL: (ToolType.STONE, ToolCategory.SHOVEL),
    ItemType.IRON_SHOVEL: (ToolType.IRON, ToolCategory.SHOVEL),
    ItemType.GOLD_SHOVEL: (ToolType.GOLD, ToolCategory.SHOVEL),
    ItemType.DIAMOND_SHOVEL: (ToolType.DIAMOND, ToolCategory.SHOVEL),
    ItemType.WOODEN_SWORD: (ToolType.WOODEN, ToolCategory.SWORD),
    ItemType.STONE_SWORD: (ToolType.STONE, ToolCategory.SWORD),
    ItemType.IRON_SWORD: (ToolType.IRON, ToolCategory.SWORD),
    ItemType.GOLD_SWORD: (ToolType.GOLD, ToolCategory.SWORD),
    ItemType.DIAMOND_SWORD: (ToolType.DIAMOND, ToolCategory.SWORD),
}


# Block drops - what item drops when a block is broken
# Format: BlockType -> (ItemType, min_count, max_count)
# Future: Can add tool requirements, fortune effects, etc.
BLOCK_DROPS = {
    BlockType.STONE: (ItemType.COBBLESTONE, 1, 1),
    BlockType.DIRT: (ItemType.DIRT, 1, 1),
    BlockType.GRASS: (ItemType.DIRT, 1, 1),
    BlockType.SAND: (ItemType.SAND, 1, 1),
    BlockType.GRAVEL: (ItemType.GRAVEL, 1, 1),
    BlockType.OAK_LOG: (ItemType.OAK_LOG, 1, 1),
    BlockType.BIRCH_LOG: (ItemType.BIRCH_LOG, 1, 1),
    BlockType.SPRUCE_LOG: (ItemType.SPRUCE_LOG, 1, 1),
    BlockType.OAK_LEAVES: (ItemType.EMPTY, 0, 0),  # Could drop saplings
    BlockType.COAL_ORE: (ItemType.COAL, 1, 1),
    BlockType.IRON_ORE: (ItemType.IRON_ORE, 1, 1),  # Requires smelting
    BlockType.GOLD_ORE: (ItemType.GOLD_ORE, 1, 1),
    BlockType.DIAMOND_ORE: (ItemType.DIAMOND, 1, 1),
    BlockType.REDSTONE_ORE: (ItemType.REDSTONE, 4, 5),
    BlockType.LAPIS_ORE: (ItemType.LAPIS, 4, 8),
    BlockType.COBBLESTONE: (ItemType.COBBLESTONE, 1, 1),
    BlockType.OAK_PLANKS: (ItemType.OAK_PLANKS, 1, 1),
    BlockType.CRAFTING_TABLE: (ItemType.CRAFTING_TABLE, 1, 1),
    BlockType.FURNACE: (ItemType.FURNACE, 1, 1),
}


# Block hardness - base time to break in ticks (20 ticks = 1 second)
# -1 means unbreakable
BLOCK_HARDNESS = {
    BlockType.AIR: 0,
    BlockType.STONE: 30,
    BlockType.DIRT: 10,
    BlockType.GRASS: 12,
    BlockType.BEDROCK: -1,
    BlockType.SAND: 10,
    BlockType.GRAVEL: 12,
    BlockType.OAK_LOG: 40,
    BlockType.OAK_LEAVES: 4,
    BlockType.COAL_ORE: 60,
    BlockType.IRON_ORE: 60,
    BlockType.GOLD_ORE: 60,
    BlockType.DIAMOND_ORE: 60,
    BlockType.COBBLESTONE: 40,
    BlockType.OAK_PLANKS: 40,
    BlockType.CRAFTING_TABLE: 50,
    BlockType.FURNACE: 70,
    BlockType.WATER: -1,
    BlockType.LAVA: -1,
}


# Tool speed multipliers for each tool tier
TOOL_SPEED_MULTIPLIER = {
    ToolType.HAND: 1.0,
    ToolType.WOODEN: 2.0,
    ToolType.STONE: 4.0,
    ToolType.IRON: 6.0,
    ToolType.GOLD: 12.0,  # Gold is fast but fragile
    ToolType.DIAMOND: 8.0,
}


# Required tool category for blocks (None means any tool works)
# Also minimum tool tier required
BLOCK_TOOL_REQUIREMENTS = {
    # Stone variants need pickaxe
    BlockType.STONE: (ToolCategory.PICKAXE, ToolType.WOODEN),
    BlockType.COBBLESTONE: (ToolCategory.PICKAXE, ToolType.WOODEN),
    # Ores have tier requirements
    BlockType.COAL_ORE: (ToolCategory.PICKAXE, ToolType.WOODEN),
    BlockType.IRON_ORE: (ToolCategory.PICKAXE, ToolType.STONE),
    BlockType.GOLD_ORE: (ToolCategory.PICKAXE, ToolType.IRON),
    BlockType.DIAMOND_ORE: (ToolCategory.PICKAXE, ToolType.IRON),
    BlockType.REDSTONE_ORE: (ToolCategory.PICKAXE, ToolType.IRON),
    # Wood is faster with axe but doesn't require it
    BlockType.OAK_LOG: (ToolCategory.AXE, ToolType.HAND),
    BlockType.OAK_PLANKS: (ToolCategory.AXE, ToolType.HAND),
    # Dirt/sand faster with shovel
    BlockType.DIRT: (ToolCategory.SHOVEL, ToolType.HAND),
    BlockType.GRASS: (ToolCategory.SHOVEL, ToolType.HAND),
    BlockType.SAND: (ToolCategory.SHOVEL, ToolType.HAND),
    BlockType.GRAVEL: (ToolCategory.SHOVEL, ToolType.HAND),
}
