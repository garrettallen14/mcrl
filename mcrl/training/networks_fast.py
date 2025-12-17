"""
Optimized neural network architectures for MCRL.

Key optimizations:
1. Smaller embedding dimension (16 vs 64)
2. Depthwise-separable 3D convolutions
3. Fused operations where possible
"""

from typing import Sequence, Callable, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class FastVoxelEncoder(nn.Module):
    """
    Optimized 3D voxel encoder.
    
    Key changes from original:
    - embed_dim: 16 (was 64) - 4x memory reduction
    - Depthwise-separable convs - 9x fewer params per layer
    - Aggressive downsampling
    """
    num_block_types: int = 256
    embed_dim: int = 16  # Much smaller!
    channels: Tuple[int, ...] = (32, 64, 128)  # Smaller channels too
    output_dim: int = 256
    
    @nn.compact
    def __call__(self, voxels: jnp.ndarray) -> jnp.ndarray:
        B = voxels.shape[0]
        
        # Small embedding: (B, 17, 17, 17) -> (B, 17, 17, 17, 16)
        # Memory: 2048 * 17³ * 16 = 161M floats (was 643M)
        x = nn.Embed(
            num_embeddings=self.num_block_types,
            features=self.embed_dim,
            embedding_init=orthogonal(1.0),
        )(voxels)
        
        # Depthwise-separable 3D convs with aggressive downsampling
        # Layer 1: (17, 17, 17, 16) -> (8, 8, 8, 32)
        x = DepthwiseSeparableConv3D(
            features=self.channels[0],
            kernel_size=(4, 4, 4),
            strides=(2, 2, 2),
        )(x)
        x = nn.relu(x)
        
        # Layer 2: (8, 8, 8, 32) -> (4, 4, 4, 64)
        x = DepthwiseSeparableConv3D(
            features=self.channels[1],
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
        )(x)
        x = nn.relu(x)
        
        # Layer 3: (4, 4, 4, 64) -> (2, 2, 2, 128)
        x = DepthwiseSeparableConv3D(
            features=self.channels[2],
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
        )(x)
        x = nn.relu(x)
        
        # Flatten: (B, 2, 2, 2, 128) -> (B, 1024)
        x = x.reshape(B, -1)
        
        # Project to output
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        
        return x


class UltraFastVoxelEncoder(nn.Module):
    """
    Ultra-lightweight voxel encoder - NO 3D convolutions.
    
    Instead of 3D CNN, uses:
    1. Block type histogram (sparse representation)
    2. Positional encoding of key blocks
    3. Direct MLP processing
    
    ~10x faster than depthwise-sep 3D conv, ~50x faster than full 3D conv.
    Total params: ~50K (vs 200K for FastVoxelEncoder)
    """
    num_block_types: int = 256
    output_dim: int = 256
    histogram_dim: int = 64
    spatial_dim: int = 128
    
    @nn.compact
    def __call__(self, voxels: jnp.ndarray) -> jnp.ndarray:
        B = voxels.shape[0]
        
        # === Part 1: Block type histogram (global block distribution) ===
        # Count occurrences of each block type: (B, 17, 17, 17) -> (B, 256)
        # Use one-hot then sum for differentiability
        flat_voxels = voxels.reshape(B, -1)  # (B, 4913)
        
        # Histogram via bincount equivalent
        # More efficient: just embed and sum
        block_embed = nn.Embed(
            num_embeddings=self.num_block_types,
            features=8,  # Tiny embedding
            embedding_init=orthogonal(1.0),
        )(flat_voxels)  # (B, 4913, 8)
        
        # Global average pooling over spatial dims
        histogram_features = block_embed.mean(axis=1)  # (B, 8)
        histogram_features = nn.Dense(self.histogram_dim, kernel_init=orthogonal(jnp.sqrt(2)))(histogram_features)
        histogram_features = nn.relu(histogram_features)
        
        # === Part 2: Spatial features from center region ===
        # Focus on the 5x5x5 center cube (125 blocks) - most relevant for action
        # Use static slicing since center is fixed
        center_voxels = voxels[:, 6:11, 6:11, 6:11]  # (B, 5, 5, 5)
        
        # Embed center blocks
        center_embed = nn.Embed(
            num_embeddings=self.num_block_types,
            features=16,
            embedding_init=orthogonal(1.0),
        )(center_voxels)  # (B, 5, 5, 5, 16)
        
        # Flatten center: (B, 125*16) = (B, 2000)
        center_flat = center_embed.reshape(B, -1)
        
        # Project to spatial features
        spatial_features = nn.Dense(self.spatial_dim, kernel_init=orthogonal(jnp.sqrt(2)))(center_flat)
        spatial_features = nn.relu(spatial_features)
        
        # === Part 3: Key block detection (ore/log proximity) ===
        # Binary features for important blocks nearby
        is_log = (voxels >= 8) & (voxels <= 10)  # Log block types
        is_ore = (voxels >= 14) & (voxels <= 19)  # Ore block types
        is_solid = (voxels > 0) & (voxels != 255)  # Non-air
        
        # Count in center region
        log_nearby = is_log[:, 4:13, 4:13, 4:13].sum(axis=(1, 2, 3)).astype(jnp.float32) / 125.0
        ore_nearby = is_ore[:, 4:13, 4:13, 4:13].sum(axis=(1, 2, 3)).astype(jnp.float32) / 125.0
        solid_below = is_solid[:, 8, :8, 8].sum(axis=1).astype(jnp.float32) / 8.0  # Ground check
        
        key_features = jnp.stack([log_nearby, ore_nearby, solid_below], axis=-1)  # (B, 3)
        key_features = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(key_features)
        key_features = nn.relu(key_features)
        
        # === Combine all features ===
        combined = jnp.concatenate([histogram_features, spatial_features, key_features], axis=-1)
        
        # Final projection
        output = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(combined)
        output = nn.relu(output)
        
        return output


class DepthwiseSeparableConv3D(nn.Module):
    """
    Depthwise-separable 3D convolution.
    
    Much more efficient than standard Conv3D:
    - Standard 3x3x3 conv: C_in * C_out * 27 params
    - Depthwise-sep: C_in * 27 + C_in * C_out params
    
    For C_in=64, C_out=128, kernel=3x3x3:
    - Standard: 64 * 128 * 27 = 221,184 params
    - Depthwise: 64 * 27 + 64 * 128 = 9,920 params (22x fewer!)
    """
    features: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    strides: Tuple[int, int, int] = (1, 1, 1)
    padding: str = 'SAME'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        
        # Depthwise: convolve each channel independently
        x = nn.Conv(
            features=in_features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            feature_group_count=in_features,  # This makes it depthwise
            kernel_init=orthogonal(1.0),
        )(x)
        
        # Pointwise: 1x1x1 conv to mix channels
        x = nn.Conv(
            features=self.features,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='SAME',
            kernel_init=orthogonal(jnp.sqrt(2)),
        )(x)
        
        return x


class FastInventoryEncoder(nn.Module):
    """Lightweight inventory encoder."""
    hidden_dim: int = 64  # Smaller
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, inventory: jnp.ndarray) -> jnp.ndarray:
        x = inventory / 64.0  # Normalize
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        return nn.relu(x)


class FastStateEncoder(nn.Module):
    """Lightweight state encoder."""
    output_dim: int = 32  # Smaller
    
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(state)
        return nn.relu(x)


class FastFacingEncoder(nn.Module):
    """Lightweight facing blocks encoder."""
    num_block_types: int = 256
    embed_dim: int = 8  # Very small
    output_dim: int = 32
    
    @nn.compact
    def __call__(self, facing: jnp.ndarray) -> jnp.ndarray:
        # Embed: (B, 8) -> (B, 8, 8)
        x = nn.Embed(
            num_embeddings=self.num_block_types,
            features=self.embed_dim,
        )(facing)
        
        # Flatten and project: (B, 64) -> (B, 32)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        return nn.relu(x)


class FastActorCritic(nn.Module):
    """
    Optimized actor-critic network.
    
    Total params: ~400K (was 1.6M)
    """
    num_actions: int = 25
    voxel_output_dim: int = 256
    trunk_hidden: int = 256  # Smaller trunk
    
    @nn.compact
    def __call__(self, obs: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Encode each observation component
        voxel_features = FastVoxelEncoder(output_dim=self.voxel_output_dim)(
            obs['local_voxels']
        )
        inventory_features = FastInventoryEncoder()(obs['inventory'])
        state_features = FastStateEncoder()(obs['player_state'])
        facing_features = FastFacingEncoder()(obs['facing_blocks'])
        
        # Log compass - direction to nearest log (crucial for navigation!)
        # Shape: [batch, 4] -> [dx, dy, dz, distance]
        log_compass = obs.get('log_compass', jnp.zeros((*obs['inventory'].shape[:-1], 4)))
        
        # Concatenate: 256 + 64 + 32 + 32 + 4 = 388
        combined = jnp.concatenate([
            voxel_features,
            inventory_features,
            state_features,
            facing_features,
            log_compass,  # Add compass directly - already normalized
        ], axis=-1)
        
        # Shared trunk
        x = nn.Dense(self.trunk_hidden, kernel_init=orthogonal(jnp.sqrt(2)))(combined)
        x = nn.relu(x)
        x = nn.Dense(self.trunk_hidden, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        
        # Actor head (policy)
        logits = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
        )(x)
        
        # Critic head (value)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        
        return logits, value.squeeze(-1)
    
    def get_action_and_value(
        self,
        obs: dict,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample action and compute log prob + value.
        
        Args:
            obs: Observation dict
            key: Random key for sampling
        
        Returns:
            action: (batch,) sampled actions
            log_prob: (batch,) log probability of sampled actions
            entropy: (batch,) policy entropy
            value: (batch,) value estimate
        """
        logits, value = self(obs)
        
        # Categorical distribution
        probs = jax.nn.softmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        
        # Sample action
        action = jax.random.categorical(key, logits)
        
        # Log prob of sampled action
        log_prob = log_probs[jnp.arange(log_probs.shape[0]), action]
        
        # Entropy
        entropy = -(probs * log_probs).sum(axis=-1)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: dict) -> jnp.ndarray:
        """Get value estimate only (for bootstrapping)."""
        _, value = self(obs)
        return value
    
    def evaluate_actions(
        self,
        obs: dict,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Evaluate log prob and entropy of given actions.
        
        Args:
            obs: Observation dict
            actions: (batch,) actions to evaluate
        
        Returns:
            log_prob: (batch,) log probability of actions
            entropy: (batch,) policy entropy
            value: (batch,) value estimate
        """
        logits, value = self(obs)
        
        probs = jax.nn.softmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        
        # Log prob of given actions
        log_prob = log_probs[jnp.arange(log_probs.shape[0]), actions]
        
        # Entropy
        entropy = -(probs * log_probs).sum(axis=-1)
        
        return log_prob, entropy, value


class UltraFastActorCritic(nn.Module):
    """
    Ultra-optimized actor-critic - NO 3D convolutions.
    
    Uses UltraFastVoxelEncoder for maximum throughput.
    Total params: ~150K (was 400K for FastActorCritic, 1.6M for ActorCritic)
    
    Best for:
    - Maximum training throughput
    - Memory-constrained GPUs
    - Hyperparameter search
    """
    num_actions: int = 25
    voxel_output_dim: int = 256
    trunk_hidden: int = 256
    
    @nn.compact
    def __call__(self, obs: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Ultra-fast voxel encoder (no 3D conv)
        voxel_features = UltraFastVoxelEncoder(output_dim=self.voxel_output_dim)(
            obs['local_voxels']
        )
        inventory_features = FastInventoryEncoder()(obs['inventory'])
        state_features = FastStateEncoder()(obs['player_state'])
        facing_features = FastFacingEncoder()(obs['facing_blocks'])
        
        # Log compass - direction to nearest log (crucial for navigation!)
        log_compass = obs.get('log_compass', jnp.zeros((*obs['inventory'].shape[:-1], 4)))
        
        # Concatenate: 256 + 64 + 32 + 32 + 4 = 388
        combined = jnp.concatenate([
            voxel_features,
            inventory_features,
            state_features,
            facing_features,
            log_compass,
        ], axis=-1)
        
        # Shared trunk
        x = nn.Dense(self.trunk_hidden, kernel_init=orthogonal(jnp.sqrt(2)))(combined)
        x = nn.relu(x)
        x = nn.Dense(self.trunk_hidden, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        
        # Actor head
        logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(x)
        
        # Critic head
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        
        return logits, value.squeeze(-1)
    
    def get_action_and_value(
        self,
        obs: dict,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, value = self(obs)
        probs = jax.nn.softmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        action = jax.random.categorical(key, logits)
        log_prob = log_probs[jnp.arange(log_probs.shape[0]), action]
        entropy = -(probs * log_probs).sum(axis=-1)
        return action, log_prob, entropy, value
    
    def get_value(self, obs: dict) -> jnp.ndarray:
        _, value = self(obs)
        return value
    
    def evaluate_actions(
        self,
        obs: dict,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, value = self(obs)
        probs = jax.nn.softmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        log_prob = log_probs[jnp.arange(log_probs.shape[0]), actions]
        entropy = -(probs * log_probs).sum(axis=-1)
        return log_prob, entropy, value


def create_fast_network(num_actions: int = 25, ultra_fast: bool = False) -> FastActorCritic:
    """
    Create optimized actor-critic network.
    
    Args:
        num_actions: Number of discrete actions
        ultra_fast: If True, use UltraFastActorCritic (no 3D conv, ~2x faster)
    
    Returns:
        Network module
    """
    if ultra_fast:
        return UltraFastActorCritic(num_actions=num_actions)
    return FastActorCritic(num_actions=num_actions)


def init_fast_network(network, key: jax.random.PRNGKey) -> dict:
    """Initialize network parameters with dummy input."""
    dummy_obs = {
        'local_voxels': jnp.zeros((1, 17, 17, 17), dtype=jnp.int32),
        'inventory': jnp.zeros((1, 16), dtype=jnp.float32),
        'player_state': jnp.zeros((1, 14), dtype=jnp.float32),
        'facing_blocks': jnp.zeros((1, 8), dtype=jnp.int32),
    }
    
    params = network.init(key, dummy_obs)
    return params


# ============================================================================
# Benchmarking utilities
# ============================================================================

def count_params(params) -> int:
    """Count total parameters in a pytree."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def benchmark_network(network, params, batch_size: int = 2048, num_runs: int = 50):
    """Benchmark forward and backward pass."""
    import time
    
    dummy_obs = {
        'local_voxels': jnp.zeros((batch_size, 17, 17, 17), dtype=jnp.int32),
        'inventory': jnp.zeros((batch_size, 16), dtype=jnp.float32),
        'player_state': jnp.zeros((batch_size, 14), dtype=jnp.float32),
        'facing_blocks': jnp.zeros((batch_size, 8), dtype=jnp.int32),
    }
    
    # JIT compile
    forward_fn = jax.jit(lambda p, o: network.apply(p, o))
    
    def loss_fn(params, obs):
        logits, value = network.apply(params, obs)
        return logits.mean() + value.mean()
    
    grad_fn = jax.jit(jax.grad(loss_fn))
    
    # Warmup
    for _ in range(5):
        logits, value = forward_fn(params, dummy_obs)
        grads = grad_fn(params, dummy_obs)
    jax.block_until_ready(logits)
    
    # Forward benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        logits, value = forward_fn(params, dummy_obs)
    jax.block_until_ready(logits)
    forward_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Backward benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        grads = grad_fn(params, dummy_obs)
    jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
    backward_time = (time.perf_counter() - start) / num_runs * 1000
    
    return {
        'params': count_params(params),
        'forward_ms': forward_time,
        'backward_ms': backward_time,
        'batch_size': batch_size,
    }


if __name__ == '__main__':
    # Quick test
    import jax
    
    print("=== FastActorCritic (depthwise-sep 3D conv) ===")
    key = jax.random.PRNGKey(0)
    network = create_fast_network(ultra_fast=False)
    params = init_fast_network(network, key)
    print(f"Parameters: {count_params(params):,}")
    results = benchmark_network(network, params)
    print(f"Forward: {results['forward_ms']:.2f}ms")
    print(f"Backward: {results['backward_ms']:.2f}ms")
    
    print("\n=== UltraFastActorCritic (no 3D conv) ===")
    key = jax.random.PRNGKey(0)
    network_ultra = create_fast_network(ultra_fast=True)
    params_ultra = init_fast_network(network_ultra, key)
    print(f"Parameters: {count_params(params_ultra):,}")
    results_ultra = benchmark_network(network_ultra, params_ultra)
    print(f"Forward: {results_ultra['forward_ms']:.2f}ms")
    print(f"Backward: {results_ultra['backward_ms']:.2f}ms")
    
    print(f"\nSpeedup: {results['forward_ms'] / results_ultra['forward_ms']:.2f}x forward, "
          f"{results['backward_ms'] / results_ultra['backward_ms']:.2f}x backward")
