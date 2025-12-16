"""
Profiling and performance monitoring for MCRL training.

Tracks:
- Steps per second (environment throughput)
- Updates per second (training throughput)
- GPU memory usage
- Per-component timing breakdown
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
import time
from collections import deque

import jax
import jax.numpy as jnp


@dataclass
class Timer:
    """Simple timer for profiling code blocks."""
    name: str
    start_time: float = 0.0
    total_time: float = 0.0
    count: int = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        self.total_time += elapsed
        self.count += 1
    
    @property
    def avg_time(self) -> float:
        return self.total_time / max(1, self.count)
    
    @property
    def avg_time_ms(self) -> float:
        return self.avg_time * 1000


@dataclass
class Profiler:
    """
    Training profiler for tracking performance metrics.
    
    Usage:
        profiler = Profiler()
        
        with profiler.timer("rollout"):
            # Collect rollout
            
        with profiler.timer("update"):
            # PPO update
            
        metrics = profiler.get_metrics()
    """
    window_size: int = 100
    
    # Timers
    timers: Dict[str, Timer] = field(default_factory=dict)
    
    # Step tracking
    steps_history: deque = field(default_factory=lambda: deque(maxlen=100))
    step_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_step_time: float = 0.0
    
    # Memory tracking
    peak_memory_mb: float = 0.0
    
    def timer(self, name: str) -> Timer:
        """Get or create a named timer."""
        if name not in self.timers:
            self.timers[name] = Timer(name=name)
        return self.timers[name]
    
    def record_steps(self, num_steps: int):
        """Record steps completed."""
        current_time = time.perf_counter()
        
        if self.last_step_time > 0:
            elapsed = current_time - self.last_step_time
            self.steps_history.append(num_steps)
            self.step_times.append(elapsed)
        
        self.last_step_time = current_time
    
    def get_steps_per_second(self) -> float:
        """Get average steps per second over recent history."""
        if len(self.step_times) == 0:
            return 0.0
        
        total_steps = sum(self.steps_history)
        total_time = sum(self.step_times)
        
        return total_steps / max(total_time, 1e-6)
    
    def update_memory(self):
        """Update peak memory usage (requires JAX device memory stats)."""
        try:
            # Get memory info from JAX
            devices = jax.devices()
            if devices:
                device = devices[0]
                # This may not work on all backends
                memory_stats = device.memory_stats()
                if memory_stats:
                    peak_bytes = memory_stats.get('peak_bytes_in_use', 0)
                    self.peak_memory_mb = peak_bytes / (1024 * 1024)
        except Exception:
            pass  # Memory stats not available
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all profiling metrics."""
        metrics = {
            'profile/steps_per_sec': self.get_steps_per_second(),
            'profile/peak_memory_mb': self.peak_memory_mb,
        }
        
        # Add timer metrics
        for name, timer in self.timers.items():
            metrics[f'profile/{name}_ms'] = timer.avg_time_ms
            metrics[f'profile/{name}_count'] = timer.count
        
        return metrics
    
    def reset_timers(self):
        """Reset all timers (keep names)."""
        for timer in self.timers.values():
            timer.total_time = 0.0
            timer.count = 0
    
    def summary(self) -> str:
        """Get formatted summary string."""
        lines = ["=== Profiling Summary ==="]
        lines.append(f"Steps/sec: {self.get_steps_per_second():,.0f}")
        lines.append(f"Peak memory: {self.peak_memory_mb:.1f} MB")
        lines.append("")
        lines.append("Timers:")
        
        for name, timer in sorted(self.timers.items()):
            lines.append(f"  {name}: {timer.avg_time_ms:.2f} ms (n={timer.count})")
        
        return "\n".join(lines)


@dataclass
class MetricsTracker:
    """
    Track and aggregate training metrics over time.
    
    Handles:
    - Running averages
    - Min/max tracking
    - Milestone success rates
    """
    window_size: int = 100
    
    # Metric histories
    histories: Dict[str, deque] = field(default_factory=dict)
    
    # Milestone tracking
    milestone_counts: Dict[str, int] = field(default_factory=dict)
    episode_count: int = 0
    
    def record(self, metrics: Dict[str, float]):
        """Record a batch of metrics."""
        for key, value in metrics.items():
            if key not in self.histories:
                self.histories[key] = deque(maxlen=self.window_size)
            self.histories[key].append(float(value))
    
    def record_episode(self, milestones: int):
        """Record episode completion with milestone bitmask."""
        self.episode_count += 1
        
        # Decode milestone bits
        milestone_names = [
            "log", "planks", "stick", "crafting_table", "wooden_pickaxe",
            "cobblestone", "stone_pickaxe", "iron_ore", "furnace",
            "iron_ingot", "iron_pickaxe", "diamond"
        ]
        
        for i, name in enumerate(milestone_names):
            if milestones & (1 << i):
                self.milestone_counts[name] = self.milestone_counts.get(name, 0) + 1
    
    def get_average(self, key: str) -> float:
        """Get running average for a metric."""
        if key not in self.histories or len(self.histories[key]) == 0:
            return 0.0
        return sum(self.histories[key]) / len(self.histories[key])
    
    def get_milestone_rates(self) -> Dict[str, float]:
        """Get success rate for each milestone."""
        if self.episode_count == 0:
            return {}
        
        rates = {}
        for name, count in self.milestone_counts.items():
            rates[f"milestone/{name}"] = count / self.episode_count
        
        return rates
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all tracked metrics."""
        summary = {}
        
        # Averages
        for key in self.histories:
            summary[f"{key}_avg"] = self.get_average(key)
        
        # Milestone rates
        summary.update(self.get_milestone_rates())
        
        return summary
    
    def check_failure_conditions(self, step: int) -> tuple[bool, Optional[str]]:
        """
        Check for early stopping conditions.
        
        Returns:
            (should_stop, reason)
        """
        # No progress on basic milestones
        if step > 2_000_000:
            log_rate = self.milestone_counts.get("log", 0) / max(1, self.episode_count)
            if log_rate < 0.1:
                return True, "No log progress after 2M steps"
        
        # Entropy collapse
        entropy = self.get_average("entropy")
        if step > 1_000_000 and entropy < 0.5:
            return True, f"Entropy collapse: {entropy:.3f}"
        
        # Value function failure
        explained_var = self.get_average("explained_var")
        if step > 1_000_000 and explained_var < 0.2:
            return True, f"Value function failed: explained_var={explained_var:.3f}"
        
        return False, None


def benchmark_env_throughput(
    env,
    num_envs: int = 1024,
    num_steps: int = 1000,
    warmup_steps: int = 100,
) -> Dict[str, float]:
    """
    Benchmark environment throughput.
    
    Args:
        env: MinecraftEnv instance
        num_envs: Number of parallel environments
        num_steps: Steps to benchmark
        warmup_steps: Warmup steps (not timed)
    
    Returns:
        Dict with throughput metrics
    """
    import jax
    import jax.numpy as jnp
    
    key = jax.random.PRNGKey(0)
    
    # Create vectorized functions
    vec_reset, vec_step = env.make_vectorized()
    
    # Initialize
    keys = jax.random.split(key, num_envs)
    states, obs = vec_reset(keys)
    
    # Random actions
    actions = jax.random.randint(key, (num_envs,), 0, 25)
    
    # Warmup
    for _ in range(warmup_steps):
        states, obs, rewards, dones, infos = vec_step(states, actions)
    jax.block_until_ready(states.world.blocks)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        states, obs, rewards, dones, infos = vec_step(states, actions)
    jax.block_until_ready(states.world.blocks)
    elapsed = time.perf_counter() - start
    
    total_steps = num_envs * num_steps
    steps_per_sec = total_steps / elapsed
    
    return {
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "ms_per_batch": (elapsed / num_steps) * 1000,
    }


def profile_network_forward(
    network,
    params,
    batch_size: int = 2048,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Profile network forward pass.
    
    Args:
        network: ActorCritic network
        params: Network parameters
        batch_size: Batch size to test
        num_runs: Number of runs to average
    
    Returns:
        Dict with timing metrics
    """
    # Create dummy batch
    dummy_obs = {
        'local_voxels': jnp.zeros((batch_size, 17, 17, 17), dtype=jnp.int32),
        'inventory': jnp.zeros((batch_size, 16), dtype=jnp.float32),
        'player_state': jnp.zeros((batch_size, 14), dtype=jnp.float32),
        'facing_blocks': jnp.zeros((batch_size, 8), dtype=jnp.int32),
    }
    
    # JIT compile
    forward_fn = jax.jit(lambda p, o: network.apply(p, o))
    
    # Warmup
    for _ in range(10):
        logits, value = forward_fn(params, dummy_obs)
    jax.block_until_ready(logits)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        logits, value = forward_fn(params, dummy_obs)
    jax.block_until_ready(logits)
    elapsed = time.perf_counter() - start
    
    return {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "total_sec": elapsed,
        "ms_per_forward": (elapsed / num_runs) * 1000,
        "samples_per_sec": (batch_size * num_runs) / elapsed,
    }
