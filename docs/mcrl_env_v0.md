You're right. Let me actually spec this out properly.

The core insight: Minecraft's *mechanics* for diamond acquisition are shockingly simple. It's the rendering, networking, and Java bloat that make it slow. We can build the *game logic* from scratch in a way that's trivially parallelizable.

---

# MinecraftRL: A JAX-Native Voxel Environment

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MinecraftRL Engine                        │
├─────────────────────────────────────────────────────────────┤
│  World State (JAX Arrays)                                   │
│  ├── blocks: jnp.array[W, H, D] uint8 (block types)        │
│  ├── light: jnp.array[W, H, D] uint8 (optional)            │
│  └── metadata: jnp.array[W, H, D] uint8 (orientation etc)  │
├─────────────────────────────────────────────────────────────┤
│  Player State (Pytree)                                      │
│  ├── position: jnp.array[3] float32                        │
│  ├── velocity: jnp.array[3] float32                        │
│  ├── rotation: jnp.array[2] float32 (pitch, yaw)           │
│  ├── inventory: jnp.array[N_SLOTS] uint16                  │
│  ├── health: float32                                        │
│  └── equipped_slot: uint8                                   │
├─────────────────────────────────────────────────────────────┤
│  Step Function: (state, action) -> (state', obs, reward)   │
│  └── Pure JAX, vmap over batch dimension                   │
├─────────────────────────────────────────────────────────────┤
│  Observation Renderer                                       │
│  ├── Option A: Local voxel query (fast, ~1μs)              │
│  ├── Option B: Simple raycaster (medium, ~100μs)           │
│  └── Option C: Neural renderer (slow but pretty)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: World Representation

```python
import jax
import jax.numpy as jnp
from flax import struct

# Block types (uint8, 256 possible)
class BlockType:
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    WOOD_LOG = 4
    LEAVES = 5
    PLANKS = 6
    COBBLESTONE = 7
    COAL_ORE = 8
    IRON_ORE = 9
    DIAMOND_ORE = 10
    BEDROCK = 11
    WATER = 12
    LAVA = 13
    SAND = 14
    GRAVEL = 15
    CRAFTING_TABLE = 16
    FURNACE = 17
    # ... extend as needed

@struct.dataclass
class WorldState:
    """Immutable world state, JAX-compatible."""
    blocks: jnp.ndarray      # [W, H, D] uint8
    tick: jnp.int32          # Current game tick
    seed: jnp.uint32         # For procedural generation

@struct.dataclass  
class PlayerState:
    """Immutable player state."""
    pos: jnp.ndarray         # [3] float32 (x, y, z)
    vel: jnp.ndarray         # [3] float32
    rot: jnp.ndarray         # [2] float32 (pitch, yaw)
    inventory: jnp.ndarray   # [40] uint16 (item_id << 8 | count)
    health: jnp.float32
    hunger: jnp.float32
    equipped: jnp.uint8      # Inventory slot index
    on_ground: jnp.bool_
    
@struct.dataclass
class GameState:
    """Combined state for one environment instance."""
    world: WorldState
    player: PlayerState
    done: jnp.bool_
    reward_flags: jnp.uint32  # Bitmask of collected milestones
```

---

## Part 2: World Generation (Pure JAX)

```python
def generate_world(key: jax.random.PRNGKey, width: int, height: int, depth: int) -> jnp.ndarray:
    """Generate a Minecraft-like world using layered noise."""
    
    # Create coordinate grids
    x = jnp.arange(width)
    z = jnp.arange(depth)
    xx, zz = jnp.meshgrid(x, z, indexing='ij')
    
    # Height map using simplex-like noise (approximated with sin combinations)
    key, k1, k2, k3 = jax.random.split(key, 4)
    
    # Multi-octave noise for terrain height
    freq1, freq2, freq3 = 0.02, 0.05, 0.1
    phase1 = jax.random.uniform(k1, (2,)) * 1000
    phase2 = jax.random.uniform(k2, (2,)) * 1000
    phase3 = jax.random.uniform(k3, (2,)) * 1000
    
    height_map = (
        jnp.sin(xx * freq1 + phase1[0]) * jnp.cos(zz * freq1 + phase1[1]) * 10 +
        jnp.sin(xx * freq2 + phase2[0]) * jnp.cos(zz * freq2 + phase2[1]) * 5 +
        jnp.sin(xx * freq3 + phase3[0]) * jnp.cos(zz * freq3 + phase3[1]) * 2
    )
    surface_y = (64 + height_map).astype(jnp.int32)
    
    # Initialize world
    blocks = jnp.zeros((width, height, depth), dtype=jnp.uint8)
    y_coords = jnp.arange(height)[None, :, None]  # [1, H, 1]
    
    # Layer 0-4: Bedrock
    blocks = jnp.where(y_coords < 5, BlockType.BEDROCK, blocks)
    
    # Layer 5-surface: Stone with ores
    stone_mask = (y_coords >= 5) & (y_coords < surface_y[:, None, :])
    blocks = jnp.where(stone_mask, BlockType.STONE, blocks)
    
    # Ore distribution
    key, k_ore = jax.random.split(key)
    ore_noise = jax.random.uniform(k_ore, (width, height, depth))
    
    # Diamond: y < 16, 0.1% chance
    diamond_mask = stone_mask & (y_coords < 16) & (ore_noise < 0.001)
    blocks = jnp.where(diamond_mask, BlockType.DIAMOND_ORE, blocks)
    
    # Iron: y < 64, 0.8% chance  
    iron_mask = stone_mask & (y_coords < 64) & (ore_noise >= 0.001) & (ore_noise < 0.009)
    blocks = jnp.where(iron_mask, BlockType.IRON_ORE, blocks)
    
    # Coal: y < 80, 1.5% chance
    coal_mask = stone_mask & (y_coords < 80) & (ore_noise >= 0.009) & (ore_noise < 0.024)
    blocks = jnp.where(coal_mask, BlockType.COAL_ORE, blocks)
    
    # Surface layers
    grass_mask = (y_coords == surface_y[:, None, :])
    dirt_mask = (y_coords >= surface_y[:, None, :] - 3) & (y_coords < surface_y[:, None, :])
    blocks = jnp.where(dirt_mask, BlockType.DIRT, blocks)
    blocks = jnp.where(grass_mask, BlockType.GRASS, blocks)
    
    # Trees (simplified: columns of wood with leaf caps)
    key, k_tree = jax.random.split(key)
    tree_noise = jax.random.uniform(k_tree, (width, depth))
    tree_positions = tree_noise < 0.01  # 1% of surface blocks get trees
    
    # Place tree trunks (5 blocks tall) and leaves
    for dy in range(5):
        trunk_y = surface_y + 1 + dy
        trunk_mask = tree_positions[:, None, :] & (y_coords == trunk_y[:, None, :])
        blocks = jnp.where(trunk_mask & (dy < 4), BlockType.WOOD_LOG, blocks)
        
    # Simplified leaf sphere at top
    for dy in range(3, 6):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                if dx*dx + dz*dz <= 4:  # Radius 2 circle
                    # This would need proper indexing - simplified here
                    pass
    
    return blocks
```

---

## Part 3: Physics Engine (Pure JAX)

```python
GRAVITY = -32.0  # blocks/sec²
TERMINAL_VELOCITY = -78.4  # blocks/sec
PLAYER_SPEED = 4.317  # blocks/sec walking
JUMP_VELOCITY = 9.0  # blocks/sec

def apply_physics(state: GameState, dt: float = 0.05) -> GameState:
    """Update player physics. Pure function, vmappable."""
    player = state.player
    world = state.world
    
    # Apply gravity
    new_vel_y = jnp.maximum(
        player.vel[1] + GRAVITY * dt,
        TERMINAL_VELOCITY
    )
    new_vel = player.vel.at[1].set(new_vel_y)
    
    # Proposed new position
    proposed_pos = player.pos + new_vel * dt
    
    # Collision detection (AABB vs voxel grid)
    # Player hitbox: 0.6 wide, 1.8 tall, centered at pos
    player_min = proposed_pos - jnp.array([0.3, 0.0, 0.3])
    player_max = proposed_pos + jnp.array([0.3, 1.8, 0.3])
    
    # Check blocks player would occupy
    block_min = jnp.floor(player_min).astype(jnp.int32)
    block_max = jnp.ceil(player_max).astype(jnp.int32)
    
    # Simplified collision: check if any solid block in bounding box
    def is_solid(block_type):
        return block_type != BlockType.AIR
    
    # Ground check
    feet_y = jnp.floor(proposed_pos[1] - 0.01).astype(jnp.int32)
    feet_x = jnp.floor(proposed_pos[0]).astype(jnp.int32)
    feet_z = jnp.floor(proposed_pos[2]).astype(jnp.int32)
    
    # Bounds check
    in_bounds = (
        (feet_x >= 0) & (feet_x < world.blocks.shape[0]) &
        (feet_y >= 0) & (feet_y < world.blocks.shape[1]) &
        (feet_z >= 0) & (feet_z < world.blocks.shape[2])
    )
    
    block_below = jnp.where(
        in_bounds,
        world.blocks[feet_x, feet_y, feet_z],
        BlockType.AIR
    )
    
    on_ground = is_solid(block_below) & (new_vel[1] <= 0)
    
    # Resolve collision
    final_pos = jnp.where(
        on_ground,
        player.pos.at[1].set(feet_y + 1.0),  # Snap to ground
        proposed_pos
    )
    final_vel = jnp.where(
        on_ground,
        new_vel.at[1].set(0.0),
        new_vel
    )
    
    new_player = player.replace(
        pos=final_pos,
        vel=final_vel,
        on_ground=on_ground
    )
    
    return state.replace(player=new_player)
```

---

## Part 4: Action Processing

```python
# 25 discrete actions matching diamond_env
class Action:
    NOOP = 0
    FORWARD = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    JUMP = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    LOOK_UP = 8
    LOOK_DOWN = 9
    ATTACK = 10
    USE = 11
    CRAFT_PLANKS = 12
    CRAFT_STICKS = 13
    CRAFT_TABLE = 14
    CRAFT_WOOD_PICK = 15
    CRAFT_STONE_PICK = 16
    CRAFT_FURNACE = 17
    CRAFT_IRON_PICK = 18
    PLACE_TABLE = 19
    PLACE_FURNACE = 20
    EQUIP_BEST_PICK = 21
    SMELT_IRON = 22
    SPRINT = 23
    SNEAK = 24

TURN_SPEED = 15.0  # degrees per action

def process_action(state: GameState, action: jnp.int32) -> GameState:
    """Process one action. Pure function."""
    player = state.player
    
    # Movement direction from yaw
    yaw_rad = jnp.deg2rad(player.rot[1])
    forward = jnp.array([jnp.sin(yaw_rad), 0.0, jnp.cos(yaw_rad)])
    right = jnp.array([jnp.cos(yaw_rad), 0.0, -jnp.sin(yaw_rad)])
    
    # Compute velocity change based on action
    move_vel = jnp.zeros(3)
    move_vel = jnp.where(action == Action.FORWARD, forward * PLAYER_SPEED, move_vel)
    move_vel = jnp.where(action == Action.BACK, -forward * PLAYER_SPEED, move_vel)
    move_vel = jnp.where(action == Action.LEFT, -right * PLAYER_SPEED, move_vel)
    move_vel = jnp.where(action == Action.RIGHT, right * PLAYER_SPEED, move_vel)
    
    # Apply horizontal movement
    new_vel = player.vel.at[0].set(move_vel[0]).at[2].set(move_vel[2])
    
    # Jump
    new_vel = jnp.where(
        (action == Action.JUMP) & player.on_ground,
        new_vel.at[1].set(JUMP_VELOCITY),
        new_vel
    )
    
    # Rotation
    new_rot = player.rot
    new_rot = jnp.where(action == Action.TURN_LEFT, new_rot.at[1].add(-TURN_SPEED), new_rot)
    new_rot = jnp.where(action == Action.TURN_RIGHT, new_rot.at[1].add(TURN_SPEED), new_rot)
    new_rot = jnp.where(action == Action.LOOK_UP, new_rot.at[0].add(-TURN_SPEED), new_rot)
    new_rot = jnp.where(action == Action.LOOK_DOWN, new_rot.at[0].add(TURN_SPEED), new_rot)
    
    # Clamp pitch
    new_rot = new_rot.at[0].set(jnp.clip(new_rot[0], -90.0, 90.0))
    # Wrap yaw
    new_rot = new_rot.at[1].set(new_rot[1] % 360.0)
    
    new_player = player.replace(vel=new_vel, rot=new_rot)
    state = state.replace(player=new_player)
    
    # Handle interaction actions
    state = jax.lax.cond(
        action == Action.ATTACK,
        lambda s: process_attack(s),
        lambda s: s,
        state
    )
    
    # Handle crafting actions  
    state = jax.lax.cond(
        (action >= Action.CRAFT_PLANKS) & (action <= Action.SMELT_IRON),
        lambda s: process_craft(s, action),
        lambda s: s,
        state
    )
    
    return state
```

---

## Part 5: Block Breaking & Interaction

```python
# Tool effectiveness matrix [tool_type, block_type] -> ticks_to_break
# 0 = cannot break, positive = ticks needed
BREAK_TIME = jnp.array([
    # AIR  STONE DIRT GRASS LOG  LEAVES PLANKS COBBLE COAL IRON DIAMOND ...
    [0,    0,    4,   4,    12,  1,     8,     0,     0,   0,   0],      # Hand
    [0,    0,    4,   4,    8,   1,     6,     0,     0,   0,   0],      # Wood pick  
    [0,    6,    4,   4,    8,   1,     6,     4,     6,   0,   0],      # Stone pick
    [0,    4,    4,   4,    6,   1,     4,     3,     4,   6,   12],     # Iron pick
], dtype=jnp.int32)

def raycast_block(world: WorldState, pos: jnp.ndarray, rot: jnp.ndarray, max_dist: float = 4.0):
    """Cast ray from player eyes to find targeted block."""
    # Eye position (player pos + eye height)
    eye = pos + jnp.array([0.0, 1.62, 0.0])
    
    # Direction from pitch/yaw
    pitch_rad = jnp.deg2rad(rot[0])
    yaw_rad = jnp.deg2rad(rot[1])
    
    direction = jnp.array([
        jnp.cos(pitch_rad) * jnp.sin(yaw_rad),
        -jnp.sin(pitch_rad),
        jnp.cos(pitch_rad) * jnp.cos(yaw_rad)
    ])
    
    # DDA raycast through voxel grid
    # Step along ray until we hit a solid block or max distance
    def ray_step(carry, _):
        t, hit_pos, hit = carry
        current_pos = eye + direction * t
        block_pos = jnp.floor(current_pos).astype(jnp.int32)
        
        # Bounds check
        in_bounds = (
            (block_pos >= 0).all() & 
            (block_pos < jnp.array(world.blocks.shape)).all()
        )
        
        block_type = jnp.where(
            in_bounds,
            world.blocks[block_pos[0], block_pos[1], block_pos[2]],
            BlockType.AIR
        )
        
        is_solid = block_type != BlockType.AIR
        new_hit = hit | (is_solid & in_bounds)
        new_hit_pos = jnp.where(is_solid & ~hit, block_pos, hit_pos)
        
        return (t + 0.1, new_hit_pos, new_hit), None
    
    init = (0.0, jnp.zeros(3, dtype=jnp.int32), False)
    (_, hit_pos, hit), _ = jax.lax.scan(ray_step, init, None, length=int(max_dist / 0.1))
    
    return hit_pos, hit

def process_attack(state: GameState) -> GameState:
    """Handle attack action (mining)."""
    hit_pos, hit = raycast_block(state.world, state.player.pos, state.player.rot)
    
    # Get block type
    block_type = jnp.where(
        hit,
        state.world.blocks[hit_pos[0], hit_pos[1], hit_pos[2]],
        BlockType.AIR
    )
    
    # Get tool type from equipped item
    tool_type = get_tool_type(state.player.inventory, state.player.equipped)
    
    # Check if can break (simplified: instant break for now)
    can_break = BREAK_TIME[tool_type, block_type] > 0
    
    # Remove block and add to inventory
    new_blocks = jnp.where(
        hit & can_break,
        state.world.blocks.at[hit_pos[0], hit_pos[1], hit_pos[2]].set(BlockType.AIR),
        state.world.blocks
    )
    
    # Add drop to inventory
    drop_item = get_block_drop(block_type)
    new_inventory = jnp.where(
        hit & can_break,
        add_to_inventory(state.player.inventory, drop_item),
        state.player.inventory
    )
    
    new_world = state.world.replace(blocks=new_blocks)
    new_player = state.player.replace(inventory=new_inventory)
    
    return state.replace(world=new_world, player=new_player)
```

---

## Part 6: Crafting System

```python
# Crafting recipes: (output_item, output_count, [(input_item, input_count), ...])
RECIPES = {
    Action.CRAFT_PLANKS: (ItemType.PLANKS, 4, [(ItemType.LOG, 1)]),
    Action.CRAFT_STICKS: (ItemType.STICK, 4, [(ItemType.PLANKS, 2)]),
    Action.CRAFT_TABLE: (ItemType.CRAFTING_TABLE, 1, [(ItemType.PLANKS, 4)]),
    Action.CRAFT_WOOD_PICK: (ItemType.WOOD_PICKAXE, 1, [(ItemType.PLANKS, 3), (ItemType.STICK, 2)]),
    Action.CRAFT_STONE_PICK: (ItemType.STONE_PICKAXE, 1, [(ItemType.COBBLESTONE, 3), (ItemType.STICK, 2)]),
    Action.CRAFT_FURNACE: (ItemType.FURNACE, 1, [(ItemType.COBBLESTONE, 8)]),
    Action.CRAFT_IRON_PICK: (ItemType.IRON_PICKAXE, 1, [(ItemType.IRON_INGOT, 3), (ItemType.STICK, 2)]),
}

def process_craft(state: GameState, action: jnp.int32) -> GameState:
    """Process crafting action if resources available."""
    # Get recipe for action
    recipe = RECIPES.get(int(action))
    if recipe is None:
        return state
    
    output_item, output_count, inputs = recipe
    
    # Check if player has all inputs
    def has_inputs(inventory, inputs):
        has_all = True
        for item, count in inputs:
            item_count = get_item_count(inventory, item)
            has_all = has_all & (item_count >= count)
        return has_all
    
    can_craft = has_inputs(state.player.inventory, inputs)
    
    # Consume inputs and add output
    new_inventory = state.player.inventory
    for item, count in inputs:
        new_inventory = jnp.where(
            can_craft,
            remove_from_inventory(new_inventory, item, count),
            new_inventory
        )
    
    new_inventory = jnp.where(
        can_craft,
        add_to_inventory(new_inventory, output_item, output_count),
        new_inventory
    )
    
    new_player = state.player.replace(inventory=new_inventory)
    return state.replace(player=new_player)
```

---

## Part 7: Observation Rendering

```python
def render_local_voxels(state: GameState, radius: int = 8) -> jnp.ndarray:
    """
    Fast observation: local voxel grid around player.
    Returns [2*radius+1, 2*radius+1, 2*radius+1] uint8 array of block types.
    ~1μs per call, trivially vmappable.
    """
    center = jnp.floor(state.player.pos).astype(jnp.int32)
    
    # Create offset grid
    offsets = jnp.arange(-radius, radius + 1)
    ox, oy, oz = jnp.meshgrid(offsets, offsets, offsets, indexing='ij')
    
    # Sample positions
    sample_x = center[0] + ox
    sample_y = center[1] + oy
    sample_z = center[2] + oz
    
    # Bounds check and sample
    W, H, D = state.world.blocks.shape
    valid = (
        (sample_x >= 0) & (sample_x < W) &
        (sample_y >= 0) & (sample_y < H) &
        (sample_z >= 0) & (sample_z < D)
    )
    
    # Clamp to valid indices for gather
    safe_x = jnp.clip(sample_x, 0, W - 1)
    safe_y = jnp.clip(sample_y, 0, H - 1)
    safe_z = jnp.clip(sample_z, 0, D - 1)
    
    blocks = state.world.blocks[safe_x, safe_y, safe_z]
    blocks = jnp.where(valid, blocks, BlockType.AIR)
    
    return blocks

def render_simple_raycast(state: GameState, width: int = 64, height: int = 64) -> jnp.ndarray:
    """
    Simple raycasting renderer for RGB observation.
    Returns [H, W, 3] uint8 image.
    ~100μs per call with vmap over rays.
    """
    # Camera setup
    eye = state.player.pos + jnp.array([0.0, 1.62, 0.0])
    pitch = jnp.deg2rad(state.player.rot[0])
    yaw = jnp.deg2rad(state.player.rot[1])
    
    # Camera basis vectors
    forward = jnp.array([
        jnp.cos(pitch) * jnp.sin(yaw),
        -jnp.sin(pitch),
        jnp.cos(pitch) * jnp.cos(yaw)
    ])
    right = jnp.array([jnp.cos(yaw), 0.0, -jnp.sin(yaw)])
    up = jnp.cross(right, forward)
    
    # FOV
    fov = 70.0
    aspect = width / height
    
    # Generate ray directions for each pixel
    u = jnp.linspace(-1, 1, width) * jnp.tan(jnp.deg2rad(fov / 2)) * aspect
    v = jnp.linspace(-1, 1, height) * jnp.tan(jnp.deg2rad(fov / 2))
    uu, vv = jnp.meshgrid(u, v, indexing='xy')
    
    # Ray directions [H, W, 3]
    directions = (
        forward[None, None, :] + 
        uu[:, :, None] * right[None, None, :] + 
        vv[:, :, None] * up[None, None, :]
    )
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    
    # Raycast each pixel (vmapped)
    def cast_single_ray(direction):
        return raycast_to_color(state.world, eye, direction, max_dist=64.0)
    
    # Flatten, vmap, reshape
    flat_dirs = directions.reshape(-1, 3)
    colors = jax.vmap(cast_single_ray)(flat_dirs)
    image = colors.reshape(height, width, 3)
    
    return image.astype(jnp.uint8)

# Block colors for rendering
BLOCK_COLORS = jnp.array([
    [135, 206, 235],  # AIR -> sky blue
    [128, 128, 128],  # STONE
    [139, 90, 43],    # DIRT
    [34, 139, 34],    # GRASS
    [139, 90, 43],    # LOG
    [0, 100, 0],      # LEAVES
    [222, 184, 135],  # PLANKS
    [105, 105, 105],  # COBBLESTONE
    [50, 50, 50],     # COAL_ORE
    [210, 180, 140],  # IRON_ORE
    [0, 255, 255],    # DIAMOND_ORE (cyan)
    [20, 20, 20],     # BEDROCK
    [0, 0, 255],      # WATER
    [255, 100, 0],    # LAVA
], dtype=jnp.uint8)
```

---

## Part 8: Full Environment Step

```python
@struct.dataclass
class MinecraftEnv:
    """JAX-native Minecraft environment."""
    world_size: tuple = (256, 128, 256)
    max_steps: int = 36000  # 30 min at 20Hz
    
    def reset(self, key: jax.random.PRNGKey) -> tuple[GameState, jnp.ndarray]:
        """Initialize new episode."""
        key, k_world, k_spawn = jax.random.split(key, 3)
        
        # Generate world
        blocks = generate_world(k_world, *self.world_size)
        world = WorldState(blocks=blocks, tick=0, seed=k_world[0])
        
        # Find spawn point (random surface location)
        spawn_x = jax.random.randint(k_spawn, (), 32, self.world_size[0] - 32)
        spawn_z = jax.random.randint(k_spawn, (), 32, self.world_size[2] - 32)
        
        # Find surface y at spawn
        spawn_y = find_surface_y(blocks, spawn_x, spawn_z) + 1.0
        
        player = PlayerState(
            pos=jnp.array([spawn_x, spawn_y, spawn_z], dtype=jnp.float32),
            vel=jnp.zeros(3, dtype=jnp.float32),
            rot=jnp.array([0.0, jax.random.uniform(k_spawn, (), 0.0, 360.0)]),
            inventory=jnp.zeros(40, dtype=jnp.uint16),
            health=20.0,
            hunger=20.0,
            equipped=0,
            on_ground=True,
        )
        
        state = GameState(
            world=world,
            player=player,
            done=False,
            reward_flags=0,
        )
        
        obs = self.get_observation(state)
        return state, obs
    
    def step(self, state: GameState, action: jnp.int32) -> tuple[GameState, jnp.ndarray, jnp.float32, jnp.bool_, dict]:
        """Execute one environment step."""
        # Process action
        state = process_action(state, action)
        
        # Apply physics
        state = apply_physics(state, dt=0.05)
        
        # Increment tick
        new_world = state.world.replace(tick=state.world.tick + 1)
        state = state.replace(world=new_world)
        
        # Calculate reward (sparse milestones)
        reward, new_flags = self.calculate_reward(state)
        state = state.replace(reward_flags=new_flags)
        
        # Check termination
        done = (
            (state.player.health <= 0) |  # Death
            (state.world.tick >= self.max_steps) |  # Timeout
            has_diamond(state.player.inventory)  # Success
        )
        state = state.replace(done=done)
        
        obs = self.get_observation(state)
        info = {"tick": state.world.tick}
        
        return state, obs, reward, done, info
    
    def get_observation(self, state: GameState) -> dict:
        """Get observation dictionary."""
        return {
            "pov": render_simple_raycast(state),  # or render_local_voxels
            "inventory": encode_inventory(state.player.inventory),
            "equipped": state.player.equipped,
            "health": state.player.health,
            "position": state.player.pos,
        }
    
    def calculate_reward(self, state: GameState) -> tuple[jnp.float32, jnp.uint32]:
        """Sparse milestone rewards (each given once)."""
        flags = state.reward_flags
        reward = 0.0
        
        inv = state.player.inventory
        
        # Check each milestone
        milestones = [
            (has_item(inv, ItemType.LOG), 1, 0),
            (has_item(inv, ItemType.PLANKS), 2, 1),
            (has_item(inv, ItemType.STICK), 4, 2),
            (has_item(inv, ItemType.CRAFTING_TABLE), 4, 3),
            (has_item(inv, ItemType.WOOD_PICKAXE), 8, 4),
            (has_item(inv, ItemType.COBBLESTONE), 16, 5),
            (has_item(inv, ItemType.STONE_PICKAXE), 32, 6),
            (has_item(inv, ItemType.IRON_ORE), 64, 7),
            (has_item(inv, ItemType.FURNACE), 128, 8),
            (has_item(inv, ItemType.IRON_INGOT), 256, 9),
            (has_item(inv, ItemType.IRON_PICKAXE), 512, 10),
            (has_item(inv, ItemType.DIAMOND), 1024, 11),
        ]
        
        for condition, points, bit in milestones:
            already_got = (flags >> bit) & 1
            should_reward = condition & ~already_got
            reward = reward + jnp.where(should_reward, points, 0.0)
            flags = jnp.where(should_reward, flags | (1 << bit), flags)
        
        return reward, flags

# Vectorized environment for parallel training
def make_vec_env(num_envs: int):
    """Create vmapped environment for parallel rollouts."""
    env = MinecraftEnv()
    
    @jax.jit
    def vec_reset(keys):
        return jax.vmap(env.reset)(keys)
    
    @jax.jit
    def vec_step(states, actions):
        return jax.vmap(env.step)(states, actions)
    
    return vec_reset, vec_step
```

---

## Part 9: Performance Targets

```python
# Expected performance on single GPU (A100/H100)
"""
Operation               | Time per call | Batched (1024 envs)
------------------------|---------------|--------------------
reset()                 | ~1ms          | ~10ms (vmap)
step() w/ voxel obs     | ~10μs         | ~1ms (vmap)  
step() w/ raycast obs   | ~500μs        | ~50ms (vmap)
                        |               |
Target throughput:      |               |
- Voxel obs: 1M+ steps/sec              |
- Raycast obs: 20k steps/sec            |
"""

# Benchmark script
def benchmark():
    import time
    
    env = MinecraftEnv(world_size=(128, 64, 128))
    key = jax.random.PRNGKey(0)
    
    # Single env benchmark
    state, obs = env.reset(key)
    state, obs, reward, done, info = env.step(state, 1)  # Compile
    
    start = time.time()
    for _ in range(1000):
        state, obs, reward, done, info = env.step(state, 1)
    elapsed = time.time() - start
    print(f"Single env: {1000/elapsed:.0f} steps/sec")
    
    # Vectorized benchmark
    vec_reset, vec_step = make_vec_env(1024)
    keys = jax.random.split(key, 1024)
    states, obs = vec_reset(keys)
    actions = jax.random.randint(key, (1024,), 0, 25)
    states, obs, rewards, dones, infos = vec_step(states, actions)  # Compile
    
    start = time.time()
    for _ in range(100):
        states, obs, rewards, dones, infos = vec_step(states, actions)
    elapsed = time.time() - start
    print(f"Vectorized (1024 envs): {100*1024/elapsed:.0f} steps/sec")
```

---

## Summary: What We Just Specified

| Component | Implementation | Performance |
|-----------|----------------|-------------|
| **World** | JAX array [256, 128, 256] uint8 | ~8MB per world |
| **Physics** | Pure JAX, AABB collision | ~1μs per step |
| **Actions** | 25 discrete (matches diamond_env) | Trivial |
| **Crafting** | 7 recipes, instant execution | ~100ns |
| **Observation** | Voxel grid or raycasting | 1μs-500μs |
| **Vectorization** | `jax.vmap` over batch | 1024+ parallel |
| **Target** | 100k-1M steps/sec | 100-1000x MineRL |

This is buildable. Want me to create the actual project structure and start implementing?