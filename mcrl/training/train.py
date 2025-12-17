"""
Main training loop for MCRL.

PureJaxRL-style fully-JIT-compiled training with:
- Vectorized environments
- Scan-based rollout collection
- Multi-epoch minibatch updates
- Comprehensive logging and profiling
"""

from typing import Tuple, Dict, Any, Optional, NamedTuple
from functools import partial
import time
import os

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state

from mcrl import MinecraftEnv, EnvConfig
from mcrl.core.state import GameState
from mcrl.training.config import TrainConfig
from mcrl.training.networks import ActorCritic, create_network, init_network
from mcrl.training.networks_fast import (
    FastActorCritic, 
    create_fast_network, 
    init_fast_network,
    count_params,
)
from mcrl.training.ppo import (
    compute_gae, 
    ppo_loss, 
    PPOBatch, 
    PPOMetrics,
    make_minibatches,
    clip_grads,
)


@struct.dataclass
class TrainState:
    """Training state including parameters and optimizer."""
    params: dict
    opt_state: optax.OptState
    step: int
    
    @classmethod
    def create(cls, params, optimizer):
        return cls(
            params=params,
            opt_state=optimizer.init(params),
            step=0,
        )


@struct.dataclass
class Trajectory:
    """Collected trajectory data."""
    obs: dict                  # PyTree of observations
    actions: jnp.ndarray       # (num_steps, num_envs)
    log_probs: jnp.ndarray     # (num_steps, num_envs)
    values: jnp.ndarray        # (num_steps, num_envs)
    rewards: jnp.ndarray       # (num_steps, num_envs)
    dones: jnp.ndarray         # (num_steps, num_envs)


@struct.dataclass
class RunnerState:
    """State carried through training loop."""
    train_state: TrainState
    env_state: GameState       # Vectorized env states
    last_obs: dict             # Last observations
    key: jax.random.PRNGKey
    global_step: int


@struct.dataclass
class Metrics:
    """Training metrics for logging."""
    # PPO metrics
    total_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_frac: jnp.ndarray
    explained_var: jnp.ndarray
    
    # Episode metrics
    mean_reward: jnp.ndarray
    mean_episode_length: jnp.ndarray
    
    # Milestone metrics
    milestone_rates: jnp.ndarray  # (12,) rates for each milestone


def create_train_state(
    config: TrainConfig,
    key: jax.random.PRNGKey,
):
    """
    Create initial training state.
    
    Args:
        config: Training configuration
        key: Random key
    
    Returns:
        train_state: Initialized TrainState
        network: Network module (for apply calls)
    """
    key, init_key = jax.random.split(key)
    
    # Create network - use fast network if configured
    if getattr(config, 'use_fast_network', False):
        network = create_fast_network(num_actions=25)
        params = init_fast_network(network, init_key)
        print(f"Using FastActorCritic ({count_params(params):,} params)")
    else:
        network = create_network(config.network, num_actions=25)
        obs_shapes = {
            'local_voxels': (17, 17, 17),
            'inventory': (16,),
            'player_state': (14,),
            'facing_blocks': (8,),
        }
        params = init_network(network, init_key, obs_shapes)
        print(f"Using ActorCritic ({count_params(params):,} params)")
    
    # Create optimizer with learning rate schedule
    if config.ppo.lr_schedule == 'linear':
        lr_schedule = optax.linear_schedule(
            init_value=config.ppo.lr,
            end_value=0.0,
            transition_steps=config.num_updates,
        )
    else:
        lr_schedule = config.ppo.lr
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )
    
    ts = TrainState.create(params, optimizer)
    
    return ts, network, optimizer


def collect_rollout(
    runner_state: RunnerState,
    env: MinecraftEnv,
    network: ActorCritic,
    config: TrainConfig,
) -> Tuple[RunnerState, Trajectory]:
    """
    Collect rollout from vectorized environments.
    
    Args:
        runner_state: Current runner state
        env: Environment (for step function)
        network: Network module
        config: Training config
    
    Returns:
        runner_state: Updated runner state
        trajectory: Collected trajectory data
    """
    def step_fn(carry, _):
        """Single environment step."""
        train_state, env_state, last_obs, key = carry
        
        # Get action from policy
        key, action_key = jax.random.split(key)
        action, log_prob, entropy, value = network.apply(
            train_state.params,
            last_obs,
            action_key,
            method=network.get_action_and_value,
        )
        
        # Step environment (vectorized)
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, config.num_envs)
        
        # Manually vmap the step
        env_state, obs, reward, done, info = jax.vmap(
            lambda s, a: env._step(s, a)
        )(env_state, action)
        
        # Create transition
        transition = (last_obs, action, log_prob, value, reward, done)
        
        return (train_state, env_state, obs, key), transition
    
    # Scan over num_steps
    initial_carry = (
        runner_state.train_state,
        runner_state.env_state,
        runner_state.last_obs,
        runner_state.key,
    )
    
    final_carry, transitions = jax.lax.scan(
        step_fn,
        initial_carry,
        None,
        length=config.num_steps,
    )
    
    train_state, env_state, last_obs, key = final_carry
    
    # Unpack transitions - scan already stacks along axis 0 (time)
    obs, actions, log_probs, values, rewards, dones = transitions
    # obs is already a dict with shape (num_steps, num_envs, ...)
    
    trajectory = Trajectory(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
    )
    
    # Update runner state
    new_runner_state = runner_state.replace(
        train_state=train_state,
        env_state=env_state,
        last_obs=last_obs,
        key=key,
        global_step=runner_state.global_step + config.num_steps * config.num_envs,
    )
    
    return new_runner_state, trajectory


def update_ppo(
    train_state: TrainState,
    trajectory: Trajectory,
    last_obs: dict,
    network: ActorCritic,
    config: TrainConfig,
    key: jax.random.PRNGKey,
    ent_coef: float,
    optimizer: optax.GradientTransformation,
) -> Tuple[TrainState, PPOMetrics]:
    """
    Perform PPO update with multi-epoch minibatch training.
    
    Args:
        train_state: Current training state
        trajectory: Collected trajectory
        last_obs: Observation after last step (for bootstrapping)
        network: Network module
        config: Training config
        key: Random key for shuffling
        ent_coef: Current entropy coefficient
        optimizer: Optax optimizer for parameter updates
    
    Returns:
        train_state: Updated training state
        metrics: Aggregated PPO metrics
    """
    # Get bootstrap value
    next_value = network.apply(
        train_state.params,
        last_obs,
        method=network.get_value,
    )
    
    # Compute GAE
    advantages, returns = compute_gae(
        rewards=trajectory.rewards,
        values=trajectory.values,
        dones=trajectory.dones,
        next_value=next_value,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    
    # Flatten batch
    batch_size = config.num_steps * config.num_envs
    
    def flatten_fn(x):
        return x.reshape((batch_size,) + x.shape[2:])
    
    batch = PPOBatch(
        obs=jax.tree_util.tree_map(flatten_fn, trajectory.obs),
        actions=trajectory.actions.reshape(batch_size),
        old_log_probs=trajectory.log_probs.reshape(batch_size),
        advantages=advantages.reshape(batch_size),
        returns=returns.reshape(batch_size),
        old_values=trajectory.values.reshape(batch_size),
    )
    
    # Multi-epoch updates
    def epoch_fn(carry, _):
        """Single epoch of minibatch updates."""
        train_state, key, metrics_sum = carry
        
        key, shuffle_key = jax.random.split(key)
        minibatches = make_minibatches(batch, shuffle_key, config.num_minibatches)
        
        def minibatch_fn(train_state, minibatch):
            """Single minibatch update."""
            
            def loss_fn(params):
                new_log_probs, entropy, new_values = network.apply(
                    params,
                    minibatch.obs,
                    minibatch.actions,
                    method=network.evaluate_actions,
                )
                
                # Normalize advantages
                advantages = minibatch.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy loss
                ratio = jnp.exp(new_log_probs - minibatch.old_log_probs)
                policy_loss_1 = ratio * advantages
                policy_loss_2 = jnp.clip(ratio, 1 - config.ppo.clip_eps, 1 + config.ppo.clip_eps) * advantages
                policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = minibatch.old_values + jnp.clip(
                    new_values - minibatch.old_values,
                    -config.ppo.clip_eps,
                    config.ppo.clip_eps,
                )
                value_loss_1 = (new_values - minibatch.returns) ** 2
                value_loss_2 = (value_pred_clipped - minibatch.returns) ** 2
                value_loss = 0.5 * jnp.maximum(value_loss_1, value_loss_2).mean()
                
                # Entropy
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + config.ppo.vf_coef * value_loss + ent_coef * entropy_loss
                
                # Metrics
                approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                clip_frac = (jnp.abs(ratio - 1) > config.ppo.clip_eps).mean()
                explained_var = 1 - jnp.var(minibatch.returns - new_values) / (jnp.var(minibatch.returns) + 1e-8)
                
                metrics = PPOMetrics(
                    total_loss=total_loss,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy_loss=-entropy_loss,
                    approx_kl=approx_kl,
                    clip_frac=clip_frac,
                    explained_var=explained_var,
                )
                
                return total_loss, metrics
            
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            
            # Update parameters using the optimizer from train_state
            # Note: We use the same optimizer chain created in create_train_state
            updates, opt_state = optimizer.update(
                grads, train_state.opt_state, train_state.params
            )
            params = optax.apply_updates(train_state.params, updates)
            
            train_state = train_state.replace(
                params=params,
                opt_state=opt_state,
                step=train_state.step + 1,
            )
            
            return train_state, metrics
        
        # Scan over minibatches
        train_state, epoch_metrics = jax.lax.scan(
            minibatch_fn,
            train_state,
            minibatches,
        )
        
        # Sum metrics
        metrics_sum = jax.tree_util.tree_map(
            lambda a, b: a + b.mean(axis=0),
            metrics_sum,
            epoch_metrics,
        )
        
        return (train_state, key, metrics_sum), None
    
    # Initialize metrics accumulator
    zero_metrics = PPOMetrics(
        total_loss=jnp.float32(0),
        policy_loss=jnp.float32(0),
        value_loss=jnp.float32(0),
        entropy_loss=jnp.float32(0),
        approx_kl=jnp.float32(0),
        clip_frac=jnp.float32(0),
        explained_var=jnp.float32(0),
    )
    
    # Scan over epochs
    (train_state, _, metrics_sum), _ = jax.lax.scan(
        epoch_fn,
        (train_state, key, zero_metrics),
        None,
        length=config.update_epochs,
    )
    
    # Average metrics over epochs
    num_updates = config.update_epochs
    metrics = jax.tree_util.tree_map(lambda x: x / num_updates, metrics_sum)
    
    return train_state, metrics


def train(config: TrainConfig, verbose: bool = True, dashboard: bool = False):
    """
    Main training function.
    
    Args:
        config: Training configuration
        verbose: Whether to print progress
        dashboard: Whether to send metrics to dashboard server
    
    Returns:
        final_state: Final training state
        metrics_history: List of metrics dicts
    """
    # Setup dashboard metrics collector if enabled
    metrics_collector = None
    if dashboard:
        try:
            from mcrl.dashboard.metrics import get_collector
            metrics_collector = get_collector()
            print("Dashboard metrics collector enabled")
        except ImportError:
            print("Dashboard not available, continuing without it")
    
    # Initialize
    key = jax.random.PRNGKey(config.seed)
    key, init_key, env_key = jax.random.split(key, 3)
    
    # Create environment
    env_config = EnvConfig(
        world_size=config.world_size,
        max_episode_ticks=config.max_episode_ticks,
    )
    env = MinecraftEnv(env_config)
    
    # Create network and training state
    train_state, network, optimizer = create_train_state(config, init_key)
    
    # Initialize environments
    print(f"Initializing {config.num_envs} environments...")
    env_keys = jax.random.split(env_key, config.num_envs)
    env_states, initial_obs = jax.vmap(env._reset)(env_keys)
    print("Environments initialized")
    
    # Create runner state
    runner_state = RunnerState(
        train_state=train_state,
        env_state=env_states,
        last_obs=initial_obs,
        key=key,
        global_step=0,
    )
    
    # Training metrics
    metrics_history = []
    start_time = time.time()
    last_log_time = start_time
    
    if verbose:
        print(f"Starting training: {config.total_timesteps:,} steps")
        print(f"  Envs: {config.num_envs}, Steps/rollout: {config.num_steps}")
        print(f"  Batch size: {config.batch_size:,}, Updates: {config.num_updates:,}")
    
    # JIT compile the training step (critical for performance!)
    @jax.jit
    def _collect_rollout_jit(runner_state):
        """JIT-compiled rollout collection."""
        def step_fn(carry, _):
            train_state, env_state, last_obs, rng = carry
            rng, action_key = jax.random.split(rng)
            
            action, log_prob, entropy, value = network.apply(
                train_state.params,
                last_obs,
                action_key,
                method=network.get_action_and_value,
            )
            
            env_state, obs, reward, done, info = jax.vmap(
                lambda s, a: env._step(s, a)
            )(env_state, action)
            
            transition = (last_obs, action, log_prob, value, reward, done)
            return (train_state, env_state, obs, rng), transition
        
        initial_carry = (
            runner_state.train_state,
            runner_state.env_state,
            runner_state.last_obs,
            runner_state.key,
        )
        
        final_carry, transitions = jax.lax.scan(
            step_fn, initial_carry, None, length=config.num_steps
        )
        
        train_state, env_state, last_obs, rng = final_carry
        obs, actions, log_probs, values, rewards, dones = transitions
        
        trajectory = Trajectory(
            obs=obs, actions=actions, log_probs=log_probs,
            values=values, rewards=rewards, dones=dones,
        )
        
        new_runner_state = runner_state.replace(
            train_state=train_state,
            env_state=env_state,
            last_obs=last_obs,
            key=rng,
            global_step=runner_state.global_step + config.num_steps * config.num_envs,
        )
        return new_runner_state, trajectory
    
    @jax.jit
    def _update_ppo_jit(train_state, trajectory, last_obs, update_key, ent_coef):
        """JIT-compiled PPO update."""
        return update_ppo(
            train_state, trajectory, last_obs, network, config,
            update_key, ent_coef, optimizer
        )
    
    print("JIT compiling training step (this takes a few minutes)...")
    
    # Training loop
    for update in range(config.num_updates):
        update_start = time.time()
        
        # Compute current entropy coefficient (annealing) - use jnp to avoid retracing
        progress = runner_state.global_step / config.total_timesteps
        ent_coef = jnp.float32(config.ppo.ent_coef + (config.ppo.ent_coef_final - config.ppo.ent_coef) * progress)
        
        # Collect rollout (JIT compiled)
        runner_state, trajectory = _collect_rollout_jit(runner_state)
        
        # PPO update (JIT compiled)
        key, update_key = jax.random.split(runner_state.key)
        runner_state = runner_state.replace(key=key)
        
        train_state, ppo_metrics = _update_ppo_jit(
            runner_state.train_state,
            trajectory,
            runner_state.last_obs,
            update_key,
            ent_coef,
        )
        runner_state = runner_state.replace(train_state=train_state)
        
        # Block until computation is done (for accurate timing)
        jax.block_until_ready(train_state.params)
        
        # Compute episode metrics
        episode_rewards = float(trajectory.rewards.sum(axis=0).mean())
        
        # Log metrics
        if update % config.logging.log_interval == 0:
            current_time = time.time()
            update_time = current_time - update_start
            elapsed = current_time - start_time
            steps_per_sec = config.batch_size / update_time
            total_steps = runner_state.global_step
            
            metrics = {
                'step': int(total_steps),
                'update': update,
                'mean_reward': float(episode_rewards),
                'entropy': float(ppo_metrics.entropy_loss),
                'policy_loss': float(ppo_metrics.policy_loss),
                'value_loss': float(ppo_metrics.value_loss),
                'approx_kl': float(ppo_metrics.approx_kl),
                'clip_frac': float(ppo_metrics.clip_frac),
                'explained_var': float(ppo_metrics.explained_var),
                'ent_coef': float(ent_coef),
                'steps_per_sec': float(steps_per_sec),
                'elapsed': elapsed,
                '_elapsed': elapsed,  # For dashboard compatibility
            }
            
            metrics_history.append(metrics)
            
            # Send to dashboard if enabled
            if metrics_collector is not None:
                metrics_collector.record(metrics)
            
            if verbose:
                # Calculate ETA
                progress = total_steps / config.total_timesteps
                if progress > 0:
                    eta_sec = elapsed / progress - elapsed
                    eta_str = f"{eta_sec/60:.0f}m" if eta_sec < 3600 else f"{eta_sec/3600:.1f}h"
                else:
                    eta_str = "..."
                
                print(f"[{elapsed:6.0f}s] Update {update:5d} | Step {total_steps:10,} ({100*progress:5.1f}%) | "
                      f"Reward {episode_rewards:7.2f} | "
                      f"Entropy {ppo_metrics.entropy_loss:.3f} | "
                      f"KL {ppo_metrics.approx_kl:.4f} | "
                      f"SPS {steps_per_sec:>8,.0f} | "
                      f"ETA {eta_str}")
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\nTraining complete: {config.total_timesteps:,} steps in {total_time:.1f}s")
        print(f"Average throughput: {config.total_timesteps / total_time:,.0f} steps/sec")
    
    return runner_state, metrics_history


if __name__ == "__main__":
    # Quick test
    from mcrl.training.config import get_fast_debug_config
    
    config = get_fast_debug_config()
    final_state, metrics = train(config)
    print(f"Final step: {final_state.global_step}")
