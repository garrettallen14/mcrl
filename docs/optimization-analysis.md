# MCRL Performance Optimization Analysis

## Current Performance

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Env throughput (2048 envs) | 231k steps/sec | 1M+ steps/sec | **4x slower** |
| Network forward (2048 batch) | 15.5ms | <2ms | **8x slower** |
| Network backward (2048 batch) | 114ms | <10ms | **11x slower** |

---

## Environment Bottlenecks

### 1. `get_local_voxels()` - **CRITICAL**

**Current implementation:**
```python
# Creates 17³ = 4,913 index arrays via meshgrid
offsets = jnp.arange(-radius, radius + 1)
ox, oy, oz = jnp.meshgrid(offsets, offsets, offsets, indexing='ij')
sample_x = player_pos[0] + ox  # Creates new array
# ... fancy indexing
blocks = world.blocks[safe_x, safe_y, safe_z]
```

**Problems:**
- `meshgrid` creates 3 temporary arrays of shape (17, 17, 17)
- Fancy indexing with 3D arrays is slow on GPU (gather operations)
- Bounds checking creates boolean masks
- Called EVERY step

**Optimized approach:**
```python
# Use jax.lax.dynamic_slice - single contiguous memory read
def get_local_voxels_fast(world_blocks, player_pos, radius=8):
    # Pad world once at episode start (not every step)
    # Then use dynamic_slice for O(1) extraction
    start = player_pos - radius
    return jax.lax.dynamic_slice(
        padded_world,
        (start[0], start[1], start[2]),
        (17, 17, 17)
    )
```

**Expected speedup: 3-5x** for observation extraction.

---

### 2. `raycast_block()` - **HIGH IMPACT**

**Current implementation:**
```python
# jax.lax.scan over 45 iterations
step_size = 0.1
num_steps = int(4.5 / 0.1)  # 45 steps!

def ray_step(carry, step_idx):
    # Per-step: compute position, bounds check, block lookup
    ...
    
jax.lax.scan(ray_step, init, jnp.arange(45))
```

**Problems:**
- 45 sequential iterations per raycast
- Each iteration has branching (bounds check, solid check)
- Called for ATTACK action which is frequent

**Optimized approach: Vectorized DDA**
```python
def raycast_vectorized(world, eye, direction, max_dist=4.5):
    # Pre-compute all sample points at once
    t = jnp.linspace(0, max_dist, 20)  # Fewer samples
    positions = eye[None, :] + direction[None, :] * t[:, None]
    block_positions = jnp.floor(positions).astype(jnp.int32)
    
    # Vectorized bounds check and block lookup
    in_bounds = (block_positions >= 0).all(axis=1) & (block_positions < world_size).all(axis=1)
    
    # Single gather operation
    blocks = world.blocks[
        jnp.clip(block_positions[:, 0], 0, W-1),
        jnp.clip(block_positions[:, 1], 0, H-1),
        jnp.clip(block_positions[:, 2], 0, D-1),
    ]
    
    # Find first solid
    is_solid = (blocks != 0) & in_bounds
    first_hit = jnp.argmax(is_solid)
    
    return block_positions[first_hit], is_solid.any()
```

**Expected speedup: 2-3x** for mining actions.

---

### 3. `encode_inventory()` - **MEDIUM IMPACT**

**Current implementation:**
```python
# Python list comprehension - not JIT-friendly
tracked_items = [ItemType.OAK_LOG, ...]  # 16 items

def get_count(item_type):
    matches = inventory[:, 0] == item_type
    counts = jnp.where(matches, inventory[:, 1], 0)
    return counts.sum()

counts = jnp.array([get_count(item) for item in tracked_items])
```

**Problems:**
- List comprehension creates 16 separate operations
- Each `get_count` scans the full inventory

**Optimized approach:**
```python
def encode_inventory_fast(inventory, tracked_items_array):
    # tracked_items_array: pre-computed jnp.array of item IDs
    # Shape: (16,)
    
    # Vectorized comparison: (36, 2) vs (16,) -> (36, 16)
    matches = inventory[:, 0:1] == tracked_items_array[None, :]
    
    # Sum counts where matches
    counts = jnp.sum(inventory[:, 1:2] * matches, axis=0)
    
    return counts
```

**Expected speedup: 2x** for observation encoding.

---

### 4. Physics Collision - **MEDIUM IMPACT**

**Current implementation:**
- Multiple separate `get_block_at` calls
- Each call does bounds checking

**Optimized approach: Batched collision check**
```python
def check_collision_batched(world, positions):
    # positions: (N, 3) array of positions to check
    # Single vectorized bounds check and gather
    ...
```

---

## Network Bottlenecks

### 1. 3D CNN Embedding Layer - **CRITICAL**

**Current implementation:**
```python
# Creates massive intermediate tensor
x = nn.Embed(num_embeddings=256, features=64)(voxels)
# Input: (B, 17, 17, 17) -> Output: (B, 17, 17, 17, 64)
# For B=2048: 2048 × 17³ × 64 = 643M floats = 2.5 GB!
```

**Problems:**
- Embedding table lookup creates huge tensor
- Memory bandwidth bound
- Conv3D on (B, 17, 17, 17, 64) is expensive

**Optimized approaches:**

#### Option A: Reduce embedding dimension
```python
embed_dim = 16  # Instead of 64
# Reduces intermediate tensor by 4x
```

#### Option B: Skip embedding, use one-hot + 1x1x1 conv
```python
# One-hot: (B, 17, 17, 17, 256) - sparse
# 1x1x1 Conv: (B, 17, 17, 17, 64) - learned projection
# More cache-friendly for sparse inputs
```

#### Option C: Hash embedding (most efficient)
```python
def hash_embed(voxels, embed_dim=64, num_hashes=4):
    # Multiple hash functions for collision resistance
    # Much smaller embedding tables
    embeddings = []
    for i in range(num_hashes):
        hash_key = (voxels * primes[i]) % table_size
        embeddings.append(hash_tables[i][hash_key])
    return sum(embeddings)
```

**Expected speedup: 2-4x** for forward pass.

---

### 2. 3D Convolution Efficiency

**Current implementation:**
```python
nn.Conv(features=64, kernel_size=(3,3,3), strides=(2,2,2))
```

**Problem:** Standard Conv3D may not be well-optimized in JAX/XLA.

**Optimized approach: Depthwise-separable 3D conv**
```python
class DepthwiseSeparableConv3D(nn.Module):
    features: int
    kernel_size: tuple = (3, 3, 3)
    
    @nn.compact
    def __call__(self, x):
        # Depthwise: convolve each channel separately
        x = nn.Conv(
            features=x.shape[-1],  # Same as input channels
            kernel_size=self.kernel_size,
            feature_group_count=x.shape[-1],  # Depthwise
        )(x)
        # Pointwise: 1x1x1 conv to mix channels
        x = nn.Conv(features=self.features, kernel_size=(1,1,1))(x)
        return x
```

**Reduces parameters by ~9x** (3³ = 27 → 3 + 1 = 4 effective).

---

### 3. Custom Triton Kernels - **ADVANCED**

For maximum performance, we could write custom Triton kernels:

```python
import triton
import triton.language as tl

@triton.jit
def voxel_embed_kernel(
    voxels_ptr, embed_ptr, output_ptr,
    B, H, W, D, E,
    BLOCK_SIZE: tl.constexpr
):
    """Fused voxel lookup + embedding in single kernel."""
    pid = tl.program_id(0)
    
    # Each thread block handles one spatial location across batch
    b_idx = pid // (H * W * D)
    spatial_idx = pid % (H * W * D)
    
    # Load voxel type
    voxel_offset = b_idx * H * W * D + spatial_idx
    voxel_type = tl.load(voxels_ptr + voxel_offset)
    
    # Load embedding (coalesced)
    embed_offset = voxel_type * E
    for e in range(0, E, BLOCK_SIZE):
        embed_vals = tl.load(embed_ptr + embed_offset + e)
        tl.store(output_ptr + voxel_offset * E + e, embed_vals)
```

**Expected speedup: 2-3x** additional over optimized JAX.

---

## Implementation Priority

| Optimization | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| `get_local_voxels` with dynamic_slice | 3-5x | Low | **P0** |
| Reduce embedding dim (64→16) | 2-4x | Trivial | **P0** |
| Vectorized raycast | 2-3x | Medium | **P1** |
| Batched inventory encode | 2x | Low | **P1** |
| Depthwise-separable Conv3D | 1.5x | Medium | **P2** |
| Custom Triton kernels | 2-3x | High | **P3** |

---

## Quick Wins (Implement Now)

### 1. Pad world at reset, use dynamic_slice

```python
# In env reset:
PADDING = 8  # radius
padded_blocks = jnp.pad(
    world.blocks, 
    ((PADDING, PADDING), (PADDING, PADDING), (PADDING, PADDING)),
    constant_values=BlockType.BEDROCK
)

# In get_local_voxels:
def get_local_voxels_fast(padded_blocks, player_pos):
    start = player_pos.astype(jnp.int32)  # Already offset by padding
    return jax.lax.dynamic_slice(padded_blocks, start, (17, 17, 17))
```

### 2. Reduce embedding dimension

```python
# In config:
embed_dim: int = 16  # Was 64

# Network params drop from 1.6M to ~400K
# Memory bandwidth drops 4x
```

### 3. Pre-compute tracked items array

```python
# Global constant
TRACKED_ITEMS = jnp.array([
    ItemType.OAK_LOG, ItemType.BIRCH_LOG, ...
], dtype=jnp.int32)

def encode_inventory_fast(inventory):
    matches = inventory[:, 0:1] == TRACKED_ITEMS[None, :]
    return (inventory[:, 1:2] * matches).sum(axis=0)
```

---

## Expected Results After Optimization

| Metric | Current | After P0 | After P0+P1 |
|--------|---------|----------|-------------|
| Env throughput | 231k | 500-700k | 800k-1M |
| Network forward | 15.5ms | 4-6ms | 3-4ms |
| Network backward | 114ms | 30-40ms | 20-30ms |

**Total training time for 100M steps:**
- Current: ~7 minutes
- After P0: ~3 minutes  
- After P0+P1: ~2 minutes

---

## Verification

After each optimization:

```bash
python experiments/scripts/preflight.py --num-envs 2048 256 1024
```

Compare before/after throughput numbers.
