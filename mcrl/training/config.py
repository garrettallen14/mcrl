"""
Training configuration for MCRL.

All hyperparameters in one place for easy experimentation.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import yaml


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    # Voxel encoder
    num_block_types: int = 256
    embed_dim: int = 64
    cnn_channels: Tuple[int, ...] = (64, 128, 256)
    
    # Other encoders
    inventory_hidden: int = 128
    state_hidden: int = 64
    facing_hidden: int = 64
    
    # Shared trunk
    trunk_hidden: int = 512
    
    # Activation
    activation: str = "relu"


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    # Learning rate
    lr: float = 3e-4
    lr_schedule: str = "linear"  # "linear" or "constant"
    
    # GAE
    gamma: float = 0.999  # High for long horizon
    gae_lambda: float = 0.95
    
    # PPO clipping
    clip_eps: float = 0.2
    clip_value: bool = True
    
    # Loss coefficients
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001
    ent_decay_steps: int = 50_000_000
    
    # Optimization
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # Early stopping if KL exceeds
    
    # Normalization
    normalize_advantages: bool = True
    clip_advantages: Optional[float] = 10.0


@dataclass 
class ExplorationConfig:
    """Exploration and intrinsic reward settings."""
    # Intrinsic reward
    use_intrinsic: bool = True
    intrinsic_coef: float = 0.5
    intrinsic_decay: float = 0.9999  # Per-step decay
    
    # Random action mixing (bootstrap phase)
    random_action_prob: float = 0.0
    random_action_decay_steps: int = 2_000_000
    
    # Visit counts
    max_visit_count: int = 100_000  # Cap counts for numerical stability


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    # Intervals (in updates)
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    profile_interval: int = 100
    
    # Checkpointing
    max_checkpoints: int = 5
    save_best: bool = True
    
    # WandB
    use_wandb: bool = True
    wandb_project: str = "mcrl"
    wandb_entity: Optional[str] = None
    
    # Console
    verbose: bool = True


@dataclass
class TrainConfig:
    """Complete training configuration."""
    # Sub-configs
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment
    # 64³ is 8x faster than 128³ for world gen; use 64 for iteration, 128 for final
    world_size: Tuple[int, int, int] = (64, 64, 64)
    max_episode_ticks: int = 18000  # 15 min @ 20Hz
    
    # Training scale - Conservative defaults that work on most GPUs
    # Scale up based on your GPU memory (see presets below)
    num_envs: int = 1024  # Safe for 24GB+ GPUs
    num_steps: int = 128  # Shorter rollouts use less memory
    num_minibatches: int = 8  # 1024*128/8 = 16k minibatch size
    update_epochs: int = 4
    total_timesteps: int = 100_000_000  # 100M steps target
    
    # Use optimized (fast) network
    use_fast_network: bool = True
    
    # Derived values (computed in __post_init__)
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)
    num_updates: int = field(init=False)
    
    # Experiment
    seed: int = 42
    run_name: str = "ppo_baseline"
    output_dir: str = "experiments/runs"
    
    def __post_init__(self):
        """Compute derived values."""
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = self.total_timesteps // self.batch_size
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse nested configs
        network = NetworkConfig(**data.pop('network', {}))
        ppo = PPOConfig(**data.pop('ppo', {}))
        exploration = ExplorationConfig(**data.pop('exploration', {}))
        logging = LoggingConfig(**data.pop('logging', {}))
        
        return cls(
            network=network,
            ppo=ppo,
            exploration=exploration,
            logging=logging,
            **data
        )
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        data = {
            'network': self.network.__dict__,
            'ppo': self.ppo.__dict__,
            'exploration': self.exploration.__dict__,
            'logging': self.logging.__dict__,
            'world_size': self.world_size,
            'max_episode_ticks': self.max_episode_ticks,
            'num_envs': self.num_envs,
            'num_steps': self.num_steps,
            'num_minibatches': self.num_minibatches,
            'update_epochs': self.update_epochs,
            'total_timesteps': self.total_timesteps,
            'seed': self.seed,
            'run_name': self.run_name,
            'output_dir': self.output_dir,
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def to_dict(self) -> dict:
        """Convert to flat dict for logging."""
        return {
            # Network
            'network/embed_dim': self.network.embed_dim,
            'network/trunk_hidden': self.network.trunk_hidden,
            # PPO
            'ppo/lr': self.ppo.lr,
            'ppo/gamma': self.ppo.gamma,
            'ppo/clip_eps': self.ppo.clip_eps,
            'ppo/ent_coef': self.ppo.ent_coef,
            'ppo/vf_coef': self.ppo.vf_coef,
            # Training
            'train/num_envs': self.num_envs,
            'train/num_steps': self.num_steps,
            'train/batch_size': self.batch_size,
            'train/total_timesteps': self.total_timesteps,
            # Exploration
            'explore/intrinsic_coef': self.exploration.intrinsic_coef,
            # Meta
            'seed': self.seed,
        }


# Preset configurations
def get_fast_debug_config() -> TrainConfig:
    """Small config for fast debugging (works on CPU or GPU)."""
    return TrainConfig(
        world_size=(32, 32, 32),  # Tiny world
        num_envs=32,
        num_steps=64,
        num_minibatches=2,
        total_timesteps=100_000,
        use_fast_network=True,
        logging=LoggingConfig(
            log_interval=1,
            save_interval=10,
            use_wandb=False,
        ),
    )


def get_baseline_config() -> TrainConfig:
    """Standard baseline for 24-32GB GPUs."""
    return TrainConfig(
        world_size=(64, 64, 64),
        num_envs=1024,
        num_steps=128,
        num_minibatches=8,
        total_timesteps=100_000_000,
        use_fast_network=True,
    )


def get_high_explore_config() -> TrainConfig:
    """High exploration for bootstrap phase."""
    return TrainConfig(
        world_size=(64, 64, 64),
        num_envs=1024,
        num_steps=128,
        num_minibatches=8,
        use_fast_network=True,
        ppo=PPOConfig(
            ent_coef=0.05,
            ent_coef_final=0.01,
            ent_decay_steps=30_000_000,
        ),
        exploration=ExplorationConfig(
            intrinsic_coef=1.0,
            random_action_prob=0.1,
            random_action_decay_steps=10_000_000,
        ),
        total_timesteps=100_000_000,
    )


def get_large_scale_config() -> TrainConfig:
    """Large scale config for 48GB+ GPUs (A40, A100)."""
    return TrainConfig(
        world_size=(64, 64, 64),
        num_envs=2048,
        num_steps=256,
        num_minibatches=16,
        total_timesteps=200_000_000,
        use_fast_network=True,
        ppo=PPOConfig(
            lr=2e-4,
            gamma=0.999,
            ent_coef=0.02,
            ent_coef_final=0.005,
        ),
    )


def get_rtx4090_config() -> TrainConfig:
    """Optimized config for RTX 4090 (24GB)."""
    return TrainConfig(
        world_size=(64, 64, 64),
        num_envs=512,
        num_steps=128,
        num_minibatches=4,
        total_timesteps=100_000_000,
        use_fast_network=True,
    )
