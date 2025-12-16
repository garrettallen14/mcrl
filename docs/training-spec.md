# MCRL Training Specification

## Overview

This document specifies the complete training infrastructure for MCRL (MinecraftRL), a JAX-native voxel environment for reinforcement learning research.

**Goal**: Train an agent to obtain a diamond from scratch using PPO with voxel observations.

**Target**: 100M steps in <24 hours on a single A100/H100 GPU.

---

## 1. Algorithm: PPO with Vectorized Environments

### Why PPO
- **Research-validated**: DreamerV3, Plan4MC, and MineRL baselines all use PPO variants
- **JAX-native**: PureJaxRL patterns achieve 10M+ steps/sec
- **Stable**: Clipped objective prevents catastrophic updates
- **Memory-efficient**: On-policy, no replay buffer needed

### Core PPO Equations

```
L_clip(θ) = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]
L_value(φ) = E[(V_φ(s) - R)²]
L_entropy(θ) = E[H(π_θ(·|s))]

Total Loss = -L_clip + c_v * L_value - c_e * L_entropy
```

---

## 2. Hyperparameters

### Environment
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `world_size` | (128, 128, 128) | Smaller for faster iteration; scale up later |
| `max_episode_ticks` | 18000 | 15 min @ 20Hz; shorter than MC default for faster resets |
| `local_obs_radius` | 8 | 17³ = 4913 voxels (research-validated) |

### Training
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_envs` | 2048 | Good GPU utilization on A100 |
| `num_steps` | 256 | ~500k samples per update |
| `num_minibatches` | 8 | 64k samples per minibatch |
| `update_epochs` | 4 | Standard PPO |
| `total_timesteps` | 100_000_000 | Target 100M steps |

### PPO
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Standard; with linear decay |
| `gamma` | 0.999 | Long horizon (discount over 20k+ steps) |
| `gae_lambda` | 0.95 | Standard GAE |
| `clip_eps` | 0.2 | Standard PPO clip |
| `ent_coef` | 0.01 → 0.001 | Anneal for exploration→exploitation |
| `vf_coef` | 0.5 | Value loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |

### Network
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `embed_dim` | 64 | Block type embedding dimension |
| `cnn_channels` | (64, 128, 256) | 3D CNN channel progression |
| `hidden_dim` | 512 | Shared trunk width |
| `num_block_types` | 256 | uint8 block types |

---

## 3. Network Architecture

### Input Processing
```
Observation Dict:
├── local_voxels: (17, 17, 17) uint8 → 3D CNN → 256-d
├── facing_blocks: (8,) uint8 → Embed + MLP → 64-d
├── inventory: (16,) float32 → MLP → 128-d
└── player_state: (14,) float32 → MLP → 64-d

Concatenate → 512-d → Shared MLP (512) → Actor/Critic heads
```

### Voxel Encoder (3D CNN)
```
Input: (B, 17, 17, 17) uint8
├── Embed(256, 64) → (B, 17, 17, 17, 64)
├── Conv3D(64, k=3, s=2) + ReLU → (B, 8, 8, 8, 64)
├── Conv3D(128, k=3, s=2) + ReLU → (B, 4, 4, 4, 128)
├── Conv3D(256, k=3, s=1) + ReLU → (B, 2, 2, 2, 256)
├── GlobalAvgPool → (B, 256)
└── Dense(256) → (B, 256)
```

### Full Network
```
class ActorCritic:
    voxel_encoder: VoxelCNN
    inventory_mlp: MLP(128)
    state_mlp: MLP(64)
    trunk: MLP(512)
    actor: Dense(25)  # 25 actions
    critic: Dense(1)
```

---

## 4. Exploration Strategy

### Phase 1: Bootstrap (0-2M steps)
- **High entropy**: `ent_coef = 0.05`
- **Random actions**: Mix 20% random actions to find first trees
- **Goal**: Achieve >20% "got log" success rate

### Phase 2: Early Progression (2M-20M steps)
- **Entropy annealing**: 0.05 → 0.01
- **Intrinsic reward**: Count-based on inventory hash
  ```python
  bonus = 0.5 / sqrt(visit_count[hash(inventory)] + 1)
  ```
- **Goal**: Achieve wooden/stone pickaxe milestones

### Phase 3: Deep Progression (20M-100M steps)
- **Low entropy**: `ent_coef = 0.001`
- **Milestone focus**: Large sparse rewards dominate
- **Goal**: Iron pickaxe → diamond

### Intrinsic Reward Implementation
```python
def compute_intrinsic_reward(inventory, visit_counts):
    # Hash inventory to track unique states
    key_items = [LOG, PLANKS, STICK, TABLE, WOOD_PICK, COBBLE, 
                 STONE_PICK, IRON_ORE, FURNACE, IRON_INGOT, IRON_PICK, DIAMOND]
    inv_hash = tuple(min(get_count(inventory, item), 10) for item in key_items)
    
    count = visit_counts.get(inv_hash, 0)
    visit_counts[inv_hash] = count + 1
    
    return 0.5 / jnp.sqrt(count + 1)
```

---

## 5. Reward Structure

### Milestone Rewards (Sparse, Given Once)
| Milestone | Reward | Bit |
|-----------|--------|-----|
| log | 1 | 0 |
| planks | 2 | 1 |
| stick | 4 | 2 |
| crafting_table | 4 | 3 |
| wooden_pickaxe | 8 | 4 |
| cobblestone | 16 | 5 |
| stone_pickaxe | 32 | 6 |
| iron_ore | 64 | 7 |
| furnace | 128 | 8 |
| iron_ingot | 256 | 9 |
| iron_pickaxe | 512 | 10 |
| diamond | 1024 | 11 |

### Additional Signals
- **Death penalty**: -100
- **Intrinsic bonus**: ~0.1-1.0 per novel inventory state

---

## 6. Training Loop Structure

### PureJaxRL Pattern
```python
@struct.dataclass
class TrainState:
    params: FrozenDict
    opt_state: optax.OptState
    step: int

@struct.dataclass  
class Trajectory:
    obs: dict  # PyTree of observations
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray

def train_step(runner_state, _):
    # 1. Collect rollout (scan over num_steps)
    runner_state, trajectory = collect_rollout(runner_state)
    
    # 2. Compute advantages (reverse scan)
    advantages, returns = compute_gae(trajectory, gamma, gae_lambda)
    
    # 3. Flatten batch
    batch = flatten_trajectory(trajectory, advantages, returns)
    
    # 4. Multi-epoch updates (scan over epochs)
    train_state = update_epochs(runner_state.train_state, batch)
    
    # 5. Return updated state
    return runner_state.replace(train_state=train_state), metrics
```

### Full Training
```python
def train(config):
    # Initialize
    env = MinecraftEnv(config.env)
    network = ActorCritic(config.network)
    
    # JIT compile
    train_step_jit = jax.jit(train_step)
    
    # Run training (scan over updates)
    final_state, metrics = jax.lax.scan(
        train_step_jit,
        initial_runner_state,
        None,
        length=config.num_updates
    )
    
    return final_state, metrics
```

---

## 7. Logging & Diagnostics

### Metrics to Log (Every Update)
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `mean_reward` | Episode return | - |
| `milestone_rates` | % achieving each milestone | <10% log @ 2M = fail |
| `entropy` | Policy entropy | <0.5 early = collapse |
| `value_loss` | Critic MSE | - |
| `policy_loss` | Actor loss | - |
| `approx_kl` | KL(old, new) | >0.05 = unstable |
| `explained_var` | Value explained variance | <0.3 = poor value |
| `clip_frac` | Fraction of clipped ratios | >0.3 = too aggressive |
| `inventory_diversity` | Unique items seen | <3 @ 5M = stagnation |

### Profiling Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| `steps_per_sec` | Environment throughput | >500k |
| `updates_per_sec` | Training throughput | >10 |
| `gpu_util` | GPU utilization | >80% |
| `memory_gb` | GPU memory usage | <40GB |

### Checkpointing
- Save every 1M steps
- Keep last 5 + best (by milestone progress)
- Include: params, opt_state, step, visit_counts

---

## 8. Experiment Management

### Config System
```python
@dataclass
class Config:
    # Environment
    world_size: tuple = (128, 128, 128)
    max_episode_ticks: int = 18000
    
    # Training
    num_envs: int = 2048
    num_steps: int = 256
    total_timesteps: int = 100_000_000
    
    # PPO
    lr: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    
    # Network
    embed_dim: int = 64
    hidden_dim: int = 512
    
    # Exploration
    intrinsic_coef: float = 0.5
    ent_decay_steps: int = 50_000_000
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50

    # Experiment
    seed: int = 42
    wandb_project: str = "mcrl"
    run_name: str = "ppo_baseline"
```

### Directory Structure
```
experiments/
├── configs/
│   ├── base.yaml
│   ├── high_explore.yaml
│   └── fast_iter.yaml
├── runs/
│   └── {run_name}_{timestamp}/
│       ├── config.yaml
│       ├── checkpoints/
│       ├── logs/
│       └── metrics.jsonl
└── scripts/
    ├── train.py
    ├── eval.py
    └── analyze.py
```

---

## 9. Failure Detection & Early Stopping

### Automatic Abort Conditions
```python
def should_abort(metrics, step):
    # No progress on basic milestones
    if step > 2_000_000 and metrics['milestone_log_rate'] < 0.1:
        return True, "No log progress"
    
    # Policy collapse
    if metrics['entropy'] < 0.5 and step < 10_000_000:
        return True, "Entropy collapse"
    
    # Value function broken
    if metrics['explained_var'] < 0.2:
        return True, "Value function failed"
    
    return False, None
```

### Recovery Actions
| Condition | Action |
|-----------|--------|
| Entropy collapse | Increase ent_coef 5x, reduce LR 2x |
| No log progress | Add 50% random actions for 500k steps |
| Value instability | Reduce LR, increase GAE lambda |
| KL spike | Reduce clip_eps to 0.1 |

---

## 10. Implementation Checklist

### Phase 1: Core Training (Priority)
- [ ] `mcrl/training/networks.py` - VoxelCNN + ActorCritic
- [ ] `mcrl/training/ppo.py` - PPO loss functions
- [ ] `mcrl/training/rollout.py` - Trajectory collection
- [ ] `mcrl/training/train.py` - Main training loop

### Phase 2: Infrastructure
- [ ] `mcrl/training/config.py` - Configuration dataclass
- [ ] `mcrl/training/logging.py` - WandB + metrics
- [ ] `mcrl/training/checkpointing.py` - Save/load
- [ ] `mcrl/training/profiling.py` - Performance tracking

### Phase 3: Experiments
- [ ] `experiments/configs/` - Config files
- [ ] `experiments/scripts/train.py` - Entry point
- [ ] `experiments/scripts/eval.py` - Evaluation
- [ ] `experiments/scripts/sweep.py` - Hyperparameter search

### Phase 4: Optimization
- [ ] Profile and identify bottlenecks
- [ ] Optimize voxel encoding
- [ ] Tune batch sizes for GPU memory
- [ ] Add mixed precision if needed

---

## Summary

**Architecture**: PPO + 3D CNN voxel encoder + dict observations
**Scale**: 2048 parallel envs, 100M steps target
**Key innovations**: Inventory-hash intrinsic rewards, milestone-gated sparse rewards
**Diagnostics**: Milestone rates, entropy, explained variance, inventory diversity
**Target**: >50% stone pickaxe @ 20M steps, >10% iron pickaxe @ 50M steps
