"""
Metrics collector for training dashboard.

Provides thread-safe metric storage and retrieval for real-time dashboard updates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import deque
import threading
import time
import json


@dataclass
class MetricsCollector:
    """
    Thread-safe metrics collector for training monitoring.
    
    Usage:
        collector = MetricsCollector()
        
        # During training
        collector.record({
            'step': 1000,
            'reward': 10.5,
            'entropy': 2.1,
        })
        
        # In dashboard
        recent = collector.get_recent(100)
        summary = collector.get_summary()
    """
    max_history: int = 10000
    
    # Internal storage
    _history: deque = field(default_factory=lambda: deque(maxlen=10000))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _start_time: float = field(default_factory=time.time)
    
    # Current state
    _current: Dict[str, Any] = field(default_factory=dict)
    _milestone_counts: Dict[str, int] = field(default_factory=dict)
    _episode_count: int = 0
    
    # Listeners for SSE
    _listeners: List = field(default_factory=list)
    
    def __post_init__(self):
        self._history = deque(maxlen=self.max_history)
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def record(self, metrics: Dict[str, Any]):
        """Record a batch of metrics."""
        with self._lock:
            # Add timestamp
            metrics['_timestamp'] = time.time()
            metrics['_elapsed'] = time.time() - self._start_time
            
            # Store in history
            self._history.append(metrics)
            
            # Update current
            self._current.update(metrics)
            
            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(metrics)
                except Exception:
                    pass
    
    def record_episode(self, milestones: int, reward: float, length: int):
        """Record episode completion."""
        with self._lock:
            self._episode_count += 1
            
            # Decode milestones
            milestone_names = [
                "log", "planks", "stick", "crafting_table", "wooden_pickaxe",
                "cobblestone", "stone_pickaxe", "iron_ore", "furnace",
                "iron_ingot", "iron_pickaxe", "diamond"
            ]
            
            for i, name in enumerate(milestone_names):
                if milestones & (1 << i):
                    self._milestone_counts[name] = self._milestone_counts.get(name, 0) + 1
    
    def get_recent(self, n: int = 100) -> List[Dict]:
        """Get n most recent metric records."""
        with self._lock:
            return list(self._history)[-n:]
    
    def get_current(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self._lock:
            return dict(self._current)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            recent = list(self._history)[-100:]
            
            if not recent:
                return {}
            
            summary = {
                'elapsed_sec': time.time() - self._start_time,
                'total_records': len(self._history),
                'episode_count': self._episode_count,
            }
            
            # Compute averages for numeric fields
            numeric_fields = ['reward', 'entropy', 'policy_loss', 'value_loss', 
                             'approx_kl', 'clip_frac', 'explained_var', 'steps_per_sec']
            
            for field in numeric_fields:
                values = [r.get(field) for r in recent if field in r and r[field] is not None]
                if values:
                    summary[f'{field}_avg'] = sum(values) / len(values)
                    summary[f'{field}_min'] = min(values)
                    summary[f'{field}_max'] = max(values)
            
            # Milestone rates
            if self._episode_count > 0:
                for name, count in self._milestone_counts.items():
                    summary[f'milestone_{name}_rate'] = count / self._episode_count
            
            return summary
    
    def get_milestone_rates(self) -> Dict[str, float]:
        """Get milestone success rates."""
        with self._lock:
            if self._episode_count == 0:
                return {}
            return {
                name: count / self._episode_count 
                for name, count in self._milestone_counts.items()
            }
    
    def get_timeseries(self, field: str, n: int = 1000) -> List[tuple]:
        """Get timeseries data for a specific field."""
        with self._lock:
            recent = list(self._history)[-n:]
            return [
                (r.get('step', i), r.get(field))
                for i, r in enumerate(recent)
                if field in r
            ]
    
    def add_listener(self, callback):
        """Add a callback for real-time updates."""
        with self._lock:
            self._listeners.append(callback)
    
    def remove_listener(self, callback):
        """Remove a listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)
    
    def to_json(self) -> str:
        """Export all data as JSON."""
        with self._lock:
            return json.dumps({
                'history': list(self._history),
                'summary': self.get_summary(),
                'milestones': self._milestone_counts,
            })
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._history.clear()
            self._current.clear()
            self._milestone_counts.clear()
            self._episode_count = 0
            self._start_time = time.time()


# Global collector instance
_global_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
