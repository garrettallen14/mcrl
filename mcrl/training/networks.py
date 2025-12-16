"""
Neural network architectures for MCRL.

Implements:
- VoxelEncoder: 3D CNN for local voxel observations
- ActorCritic: Combined policy and value network

All networks are Flax modules compatible with JAX transformations.
"""

from typing import Sequence, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant

from mcrl.training.config import NetworkConfig


class VoxelEncoder(nn.Module):
    """
    3D CNN encoder for voxel observations.
    
    Input: (batch, 17, 17, 17) uint8 block types
    Output: (batch, output_dim) feature vector
    
    Architecture:
    - Embed block types to dense vectors
    - Stack of 3D convolutions with downsampling
    - Global average pooling
    """
    num_block_types: int = 256
    embed_dim: int = 64
    channels: Sequence[int] = (64, 128, 256)
    output_dim: int = 256
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, voxels: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            voxels: (batch, H, W, D) uint8 or int32 block type indices
        
        Returns:
            features: (batch, output_dim) encoded features
        """
        # Embed block types: (B, 17, 17, 17) -> (B, 17, 17, 17, embed_dim)
        x = nn.Embed(
            num_embeddings=self.num_block_types,
            features=self.embed_dim,
            embedding_init=orthogonal(jnp.sqrt(2)),
        )(voxels)
        
        # 3D convolutions with downsampling
        # Input: (B, 17, 17, 17, embed_dim)
        for i, ch in enumerate(self.channels):
            x = nn.Conv(
                features=ch,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 2) if i < len(self.channels) - 1 else (1, 1, 1),
                padding='SAME',
                kernel_init=orthogonal(jnp.sqrt(2)),
            )(x)
            x = self.activation(x)
            # Optional: add residual or layer norm here
        
        # After convs: approximately (B, 4, 4, 4, 256) or (B, 2, 2, 2, 256)
        
        # Global average pooling: (B, H', W', D', C) -> (B, C)
        x = x.mean(axis=(1, 2, 3))
        
        # Project to output dimension
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
        )(x)
        x = self.activation(x)
        
        return x


class InventoryEncoder(nn.Module):
    """MLP encoder for inventory counts."""
    hidden_dim: int = 128
    output_dim: int = 128
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, inventory: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inventory: (batch, num_items) float32 item counts
        
        Returns:
            features: (batch, output_dim) encoded features
        """
        # Normalize counts (max stack is 64)
        x = inventory / 64.0
        
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = self.activation(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = self.activation(x)
        
        return x


class StateEncoder(nn.Module):
    """MLP encoder for player state vector."""
    hidden_dim: int = 64
    output_dim: int = 64
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            state: (batch, state_dim) float32 player state features
        
        Returns:
            features: (batch, output_dim) encoded features
        """
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)))(state)
        x = self.activation(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = self.activation(x)
        
        return x


class FacingEncoder(nn.Module):
    """Encoder for blocks along view ray."""
    num_block_types: int = 256
    embed_dim: int = 32
    output_dim: int = 64
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, facing: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            facing: (batch, num_blocks) uint8 block types along ray
        
        Returns:
            features: (batch, output_dim) encoded features
        """
        # Embed block types
        x = nn.Embed(
            num_embeddings=self.num_block_types,
            features=self.embed_dim,
        )(facing)  # (B, num_blocks, embed_dim)
        
        # Flatten and project
        x = x.reshape(x.shape[0], -1)  # (B, num_blocks * embed_dim)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = self.activation(x)
        
        return x


class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    
    Processes dict observations with separate encoders,
    combines features in shared trunk, outputs policy logits and value.
    """
    config: NetworkConfig
    num_actions: int = 25
    
    def setup(self):
        """Initialize sub-modules."""
        cfg = self.config
        
        # Encoders
        self.voxel_encoder = VoxelEncoder(
            num_block_types=cfg.num_block_types,
            embed_dim=cfg.embed_dim,
            channels=cfg.cnn_channels,
            output_dim=256,
        )
        
        self.inventory_encoder = InventoryEncoder(
            hidden_dim=cfg.inventory_hidden,
            output_dim=cfg.inventory_hidden,
        )
        
        self.state_encoder = StateEncoder(
            hidden_dim=cfg.state_hidden,
            output_dim=cfg.state_hidden,
        )
        
        self.facing_encoder = FacingEncoder(
            num_block_types=cfg.num_block_types,
            output_dim=cfg.facing_hidden,
        )
        
        # Shared trunk
        self.trunk = nn.Dense(
            cfg.trunk_hidden,
            kernel_init=orthogonal(jnp.sqrt(2)),
        )
        
        # Actor head
        self.actor = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),  # Small init for policy
        )
        
        # Critic head
        self.critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
        )
    
    def __call__(self, obs: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through network.
        
        Args:
            obs: Dictionary with keys:
                - local_voxels: (batch, 17, 17, 17) uint8
                - inventory: (batch, num_items) float32
                - player_state: (batch, state_dim) float32
                - facing_blocks: (batch, num_blocks) uint8
        
        Returns:
            logits: (batch, num_actions) policy logits
            value: (batch,) state value estimate
        """
        # Encode each observation component
        voxel_features = self.voxel_encoder(obs['local_voxels'])
        inventory_features = self.inventory_encoder(obs['inventory'])
        state_features = self.state_encoder(obs['player_state'])
        facing_features = self.facing_encoder(obs['facing_blocks'])
        
        # Concatenate all features
        combined = jnp.concatenate([
            voxel_features,
            inventory_features,
            state_features,
            facing_features,
        ], axis=-1)
        
        # Shared trunk
        activation = nn.relu if self.config.activation == 'relu' else nn.tanh
        hidden = activation(self.trunk(combined))
        
        # Actor and critic heads
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        
        return logits, value
    
    def get_action_and_value(
        self,
        obs: dict,
        key: jax.random.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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


def create_network(config: NetworkConfig, num_actions: int = 25) -> ActorCritic:
    """Factory function to create network from config."""
    return ActorCritic(config=config, num_actions=num_actions)


def init_network(
    network: ActorCritic,
    key: jax.random.PRNGKey,
    obs_shapes: dict,
) -> dict:
    """
    Initialize network parameters.
    
    Args:
        network: Network module
        key: Random key
        obs_shapes: Dict of observation shapes (without batch dim)
    
    Returns:
        params: Initialized parameters
    """
    # Create dummy observation batch
    dummy_obs = {
        'local_voxels': jnp.zeros((1,) + obs_shapes['local_voxels'], dtype=jnp.int32),
        'inventory': jnp.zeros((1,) + obs_shapes['inventory'], dtype=jnp.float32),
        'player_state': jnp.zeros((1,) + obs_shapes['player_state'], dtype=jnp.float32),
        'facing_blocks': jnp.zeros((1,) + obs_shapes['facing_blocks'], dtype=jnp.int32),
    }
    
    return network.init(key, dummy_obs)
