"""
PPO loss functions and advantage computation for MCRL.

Implements:
- Generalized Advantage Estimation (GAE)
- PPO clipped objective
- Value loss with optional clipping
- Entropy bonus
"""

from typing import Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
from functools import partial


class PPOBatch(NamedTuple):
    """Batch of data for PPO update."""
    obs: dict                  # Dict of observation arrays
    actions: jnp.ndarray       # (batch,) int32
    old_log_probs: jnp.ndarray # (batch,) float32
    advantages: jnp.ndarray    # (batch,) float32
    returns: jnp.ndarray       # (batch,) float32
    old_values: jnp.ndarray    # (batch,) float32


class PPOMetrics(NamedTuple):
    """Metrics from PPO update."""
    total_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_frac: jnp.ndarray
    explained_var: jnp.ndarray


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    next_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (num_steps, num_envs) rewards
        values: (num_steps, num_envs) value estimates
        dones: (num_steps, num_envs) episode done flags
        next_value: (num_envs,) bootstrap value for final state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: (num_steps, num_envs) advantage estimates
        returns: (num_steps, num_envs) return targets for value function
    """
    num_steps = rewards.shape[0]
    
    # Append next_value for easier indexing
    values_extended = jnp.concatenate([values, next_value[None, :]], axis=0)
    
    def gae_step(carry, t):
        """Single step of GAE computation (reverse scan)."""
        gae = carry
        
        # Index from the end (reverse)
        step = num_steps - 1 - t
        
        delta = (
            rewards[step] 
            + gamma * values_extended[step + 1] * (1 - dones[step])
            - values_extended[step]
        )
        
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        
        return gae, gae
    
    # Scan in reverse (from last step to first)
    _, advantages_reversed = jax.lax.scan(
        gae_step,
        jnp.zeros_like(next_value),  # Initial GAE = 0
        jnp.arange(num_steps),
    )
    
    # Reverse to get correct order
    advantages = advantages_reversed[::-1]
    
    # Returns = advantages + values
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(
    advantages: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Normalize advantages to zero mean and unit variance."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def clip_advantages(
    advantages: jnp.ndarray,
    clip_value: float = 10.0,
) -> jnp.ndarray:
    """Clip advantages to prevent extreme values."""
    return jnp.clip(advantages, -clip_value, clip_value)


def ppo_loss(
    params: dict,
    network_apply: callable,
    batch: PPOBatch,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    clip_value: bool = True,
    normalize_adv: bool = True,
    clip_adv: Optional[float] = None,
) -> Tuple[jnp.ndarray, PPOMetrics]:
    """
    Compute PPO loss.
    
    Args:
        params: Network parameters
        network_apply: Network apply function
        batch: PPO batch data
        clip_eps: PPO clip epsilon
        vf_coef: Value loss coefficient
        ent_coef: Entropy bonus coefficient
        clip_value: Whether to clip value loss
        normalize_adv: Whether to normalize advantages
        clip_adv: Optional advantage clipping value
    
    Returns:
        loss: Scalar total loss
        metrics: PPOMetrics with component losses
    """
    # Get new log probs, entropy, values
    new_log_probs, entropy, new_values = network_apply(
        params, batch.obs, batch.actions
    )
    
    # Process advantages
    advantages = batch.advantages
    if normalize_adv:
        advantages = normalize_advantages(advantages)
    if clip_adv is not None:
        advantages = clip_advantages(advantages, clip_adv)
    
    # Policy loss (clipped surrogate)
    ratio = jnp.exp(new_log_probs - batch.old_log_probs)
    
    policy_loss_1 = ratio * advantages
    policy_loss_2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    
    policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()
    
    # Value loss (optionally clipped)
    if clip_value:
        # Clipped value loss (from PPO paper)
        value_pred_clipped = batch.old_values + jnp.clip(
            new_values - batch.old_values,
            -clip_eps,
            clip_eps,
        )
        value_loss_1 = (new_values - batch.returns) ** 2
        value_loss_2 = (value_pred_clipped - batch.returns) ** 2
        value_loss = 0.5 * jnp.maximum(value_loss_1, value_loss_2).mean()
    else:
        value_loss = 0.5 * ((new_values - batch.returns) ** 2).mean()
    
    # Entropy loss (negative because we want to maximize entropy)
    entropy_loss = -entropy.mean()
    
    # Total loss
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    # Compute metrics
    with jax.numpy.errstate(invalid='ignore'):
        approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
    clip_frac = (jnp.abs(ratio - 1) > clip_eps).mean()
    
    # Explained variance
    explained_var = 1 - jnp.var(batch.returns - new_values) / (jnp.var(batch.returns) + 1e-8)
    
    metrics = PPOMetrics(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=-entropy_loss,  # Report as positive entropy
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        explained_var=explained_var,
    )
    
    return total_loss, metrics


def create_ppo_update_fn(
    network,
    optimizer,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    clip_value: bool = True,
    max_grad_norm: float = 0.5,
):
    """
    Create JIT-compiled PPO update function.
    
    Args:
        network: ActorCritic network module
        optimizer: Optax optimizer
        clip_eps: PPO clip epsilon
        vf_coef: Value loss coefficient
        ent_coef: Entropy coefficient
        clip_value: Whether to clip value loss
        max_grad_norm: Maximum gradient norm
    
    Returns:
        update_fn: (params, opt_state, batch) -> (params, opt_state, metrics)
    """
    def evaluate_fn(params, obs, actions):
        """Wrapper for network evaluation."""
        return network.apply(params, obs, actions, method=network.evaluate_actions)
    
    @jax.jit
    def update_fn(params, opt_state, batch, ent_coef_current):
        """Single PPO update step."""
        
        def loss_fn(params):
            return ppo_loss(
                params=params,
                network_apply=evaluate_fn,
                batch=batch,
                clip_eps=clip_eps,
                vf_coef=vf_coef,
                ent_coef=ent_coef_current,
                clip_value=clip_value,
            )
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Clip gradients
        grads, grad_norm = clip_grads(grads, max_grad_norm)
        
        # Apply optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, metrics
    
    return update_fn


def clip_grads(grads, max_norm: float):
    """Clip gradients by global norm."""
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads))
    )
    
    clip_coef = max_norm / (grad_norm + 1e-6)
    clip_coef = jnp.minimum(clip_coef, 1.0)
    
    grads = jax.tree_util.tree_map(lambda g: g * clip_coef, grads)
    
    return grads, grad_norm


def make_minibatches(
    batch: PPOBatch,
    key: jax.random.PRNGKey,
    num_minibatches: int,
) -> PPOBatch:
    """
    Shuffle and split batch into minibatches.
    
    Args:
        batch: Full batch of data
        key: Random key for shuffling
        num_minibatches: Number of minibatches to create
    
    Returns:
        minibatches: PPOBatch with additional leading dimension
    """
    batch_size = batch.actions.shape[0]
    minibatch_size = batch_size // num_minibatches
    
    # Shuffle indices
    indices = jax.random.permutation(key, batch_size)
    
    # Shuffle all arrays in batch
    def shuffle_array(x):
        return x[indices]
    
    shuffled = jax.tree_util.tree_map(shuffle_array, batch)
    
    # Reshape to (num_minibatches, minibatch_size, ...)
    def reshape_to_minibatches(x):
        shape = x.shape
        return x.reshape((num_minibatches, minibatch_size) + shape[1:])
    
    return jax.tree_util.tree_map(reshape_to_minibatches, shuffled)


# Import optax here to avoid circular imports in clip_grads
import optax
