"""
Simple checkpoint save/load utilities for MCRL.

Uses msgpack for fast, portable serialization.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import numpy as np


def save_checkpoint(
    params: Any,
    path: str,
    step: int = 0,
    metadata: Optional[Dict] = None,
    max_checkpoints: int = 5,
) -> str:
    """
    Save model parameters to a checkpoint file.
    
    Args:
        params: JAX pytree of parameters
        path: Directory to save checkpoint
        step: Training step number
        metadata: Optional metadata dict
        max_checkpoints: Max number of checkpoints to keep (0 = unlimited)
    
    Returns:
        Path to saved checkpoint
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Convert JAX arrays to numpy for serialization
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), params)
    
    # Save params with pickle (simple and fast)
    params_file = path / f"params_{step}.pkl"
    with open(params_file, 'wb') as f:
        pickle.dump(params_np, f)
    
    # Save metadata
    meta = {
        'step': step,
        'param_count': sum(p.size for p in jax.tree_util.tree_leaves(params_np)),
        **(metadata or {}),
    }
    meta_file = path / f"metadata_{step}.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Cleanup old checkpoints to prevent disk overflow
    if max_checkpoints > 0:
        _cleanup_old_checkpoints(path, max_checkpoints)
    
    size_mb = params_file.stat().st_size / (1024 * 1024)
    print(f"Saved checkpoint to {params_file} ({size_mb:.1f} MB)")
    return str(params_file)


def _cleanup_old_checkpoints(path: Path, max_checkpoints: int):
    """Remove old checkpoints, keeping only the most recent ones."""
    param_files = sorted(path.glob("params_*.pkl"))
    
    if len(param_files) <= max_checkpoints:
        return
    
    # Remove oldest checkpoints
    files_to_remove = param_files[:-max_checkpoints]
    for pf in files_to_remove:
        # Get step number from filename
        step = pf.stem.split('_')[1]
        meta_file = path / f"metadata_{step}.json"
        
        # Remove both files
        pf.unlink(missing_ok=True)
        if meta_file.exists():
            meta_file.unlink(missing_ok=True)
        
        print(f"  Removed old checkpoint: {pf.name}")


def load_checkpoint(
    path: str,
    step: Optional[int] = None,
) -> tuple[Any, Dict]:
    """
    Load model parameters from checkpoint.
    
    Args:
        path: Directory containing checkpoint
        step: Specific step to load (None = latest)
    
    Returns:
        (params, metadata) tuple
    """
    path = Path(path)
    
    # Find checkpoint files
    if step is not None:
        params_file = path / f"params_{step}.pkl"
        meta_file = path / f"metadata_{step}.json"
    else:
        # Find latest
        param_files = sorted(path.glob("params_*.pkl"))
        if not param_files:
            raise FileNotFoundError(f"No checkpoints found in {path}")
        params_file = param_files[-1]
        step = int(params_file.stem.split('_')[1])
        meta_file = path / f"metadata_{step}.json"
    
    # Load params
    with open(params_file, 'rb') as f:
        params_np = pickle.load(f)
    
    # Convert to JAX arrays
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params_np)
    
    # Load metadata
    metadata = {}
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
    
    print(f"Loaded checkpoint from {params_file}")
    return params, metadata


def list_checkpoints(path: str) -> list[int]:
    """List available checkpoint steps."""
    path = Path(path)
    param_files = sorted(path.glob("params_*.pkl"))
    steps = [int(f.stem.split('_')[1]) for f in param_files]
    return steps
