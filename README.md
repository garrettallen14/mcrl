# MinecraftRL 🎮

A JAX-native Minecraft environment for high-performance reinforcement learning research.

## Features

- **Pure JAX Implementation**: Entire environment runs in JAX for JIT compilation and GPU acceleration
- **Vectorized Environments**: Trivially parallelize with `jax.vmap` for 100k+ steps/sec
- **Voxel Observations**: No slow pixel rendering - direct 3D block observations
- **Diamond Task**: Matches `diamond_env` specification with 25 discrete actions
- **Extensible**: Clean architecture for adding mobs, hunger, networking, etc.

## Quick Start

```python
import jax
from mcrl import MinecraftEnv, Action

# Create environment
env = MinecraftEnv()
key = jax.random.PRNGKey(0)

# Single environment
state, obs = env.reset(key)
state, obs, reward, done, info = env.step(state, Action.FORWARD)

# Vectorized (1024 parallel environments)
vec_reset, vec_step = env.make_vectorized()
keys = jax.random.split(key, 1024)
states, obs = vec_reset(keys)
```

## Installation

```bash
# CPU only
pip install -e .

# With CUDA support
pip install -e ".[cuda]"
```

## Benchmarks

Run the benchmark suite:

```bash
python benchmark.py --num-envs 256
```

Expected performance (varies by hardware):
- **Single env**: ~1,000-5,000 steps/sec
- **256 parallel envs**: ~100,000-500,000 steps/sec
- **World generation**: ~10-50 worlds/sec

## Architecture

```
mcrl/
├── core/           # Core types and state definitions
│   ├── types.py    # BlockType, ItemType, ToolType enums
│   ├── state.py    # GameState, PlayerState, WorldState
│   └── constants.py # Physics, gameplay constants
├── systems/        # Game systems
│   ├── world_gen.py   # Procedural world generation
│   ├── physics.py     # Gravity, collision detection
│   ├── actions.py     # Action processing
│   ├── crafting.py    # Recipe system
│   ├── mining.py      # Block breaking, raycasting
│   └── inventory.py   # Item management
├── utils/          # Utilities
│   ├── observations.py # Observation generation
│   └── rewards.py      # Milestone rewards
└── env.py          # Main environment class
```

## Action Space

25 discrete actions matching the diamond_env specification:

| Action | ID | Description |
|--------|-----|-------------|
| NOOP | 0 | Do nothing |
| FORWARD | 1 | Move forward |
| BACK | 2 | Move backward |
| LEFT | 3 | Strafe left |
| RIGHT | 4 | Strafe right |
| JUMP | 5 | Jump |
| TURN_LEFT | 6 | Turn camera left |
| TURN_RIGHT | 7 | Turn camera right |
| LOOK_UP | 8 | Look up |
| LOOK_DOWN | 9 | Look down |
| ATTACK | 10 | Mine/attack |
| USE | 11 | Place/use |
| CRAFT_PLANKS | 12 | Craft planks from logs |
| CRAFT_STICKS | 13 | Craft sticks |
| CRAFT_TABLE | 14 | Craft crafting table |
| CRAFT_WOOD_PICK | 15 | Craft wooden pickaxe |
| CRAFT_STONE_PICK | 16 | Craft stone pickaxe |
| CRAFT_FURNACE | 17 | Craft furnace |
| CRAFT_IRON_PICK | 18 | Craft iron pickaxe |
| PLACE_TABLE | 19 | Place crafting table |
| PLACE_FURNACE | 20 | Place furnace |
| EQUIP_BEST_PICK | 21 | Equip best pickaxe |
| SMELT_IRON | 22 | Smelt iron ore |
| SPRINT | 23 | Toggle sprint |
| SNEAK | 24 | Toggle sneak |

## Observations

```python
obs = {
    "local_voxels": (17, 17, 17),  # Block types around player
    "facing_blocks": (8,),         # Blocks along view ray
    "inventory": (16,),            # Key item counts
    "player_state": (14,),         # Position, health, etc.
    "tick": (),                    # Current game tick
}
```

## Rewards

Sparse milestone rewards for diamond progression:

| Milestone | Reward |
|-----------|--------|
| Log | +1 |
| Planks | +2 |
| Stick | +4 |
| Crafting Table | +4 |
| Wooden Pickaxe | +8 |
| Cobblestone | +16 |
| Stone Pickaxe | +32 |
| Iron Ore | +64 |
| Furnace | +128 |
| Iron Ingot | +256 |
| Iron Pickaxe | +512 |
| Diamond | +1024 |

## Roadmap

- [ ] GPU-accelerated raycasting for optional pixel rendering
- [ ] Mob system (hostile mobs, animals)
- [ ] Hunger and food
- [ ] Water/lava physics
- [ ] Nether/End dimensions
- [ ] Multiplayer support
- [ ] Integration with RL libraries (CleanRL, PureJaxRL)

## License

MIT
