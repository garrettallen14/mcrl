#!/usr/bin/env python3
"""
Evaluation script for trained MCRL agents.

Features:
- Load trained checkpoints
- Run episodes with visualization
- ASCII rendering of what the agent sees
- Statistics collection
"""

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcrl import MinecraftEnv, EnvConfig
from mcrl.core.types import BlockType, ItemType
from mcrl.training.networks_fast import FastActorCritic, UltraFastActorCritic
from mcrl.utils.checkpoint import load_checkpoint, list_checkpoints


# Block type display characters
BLOCK_CHARS = {
    BlockType.AIR: ' ',
    BlockType.STONE: '█',
    BlockType.DIRT: '▓',
    BlockType.GRASS: '░',
    BlockType.SAND: '·',
    BlockType.WATER: '~',
    BlockType.OAK_LOG: 'O',
    BlockType.BIRCH_LOG: 'B',
    BlockType.SPRUCE_LOG: 'S',
    BlockType.OAK_LEAVES: '*',
    BlockType.OAK_PLANKS: '#',
    BlockType.COBBLESTONE: '%',
    BlockType.COAL_ORE: 'c',
    BlockType.IRON_ORE: 'i',
    BlockType.DIAMOND_ORE: 'D',
    BlockType.CRAFTING_TABLE: 'T',
    BlockType.FURNACE: 'F',
    BlockType.BEDROCK: '▄',
}

# Action names for display
ACTION_NAMES = [
    'NOOP', 'FORWARD', 'BACK', 'LEFT', 'RIGHT',
    'JUMP', 'TURN_R', 'TURN_L', 'LOOK_UP', 'LOOK_DOWN',
    'ATTACK', 'USE', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_TABLE',
    'CRAFT_W_PICK', 'CRAFT_S_PICK', 'CRAFT_I_PICK', 'CRAFT_FURNACE',
    'PLACE', 'SLOT_0', 'SLOT_1', 'SLOT_2', 'SLOT_3', 'SLOT_4',
]


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def render_voxels_top_down(voxels: np.ndarray, player_y: int = 8) -> str:
    """
    Render top-down view of voxels at player height.
    
    Args:
        voxels: (17, 17, 17) block types
        player_y: Y level to render (0-16, player at center = 8)
    
    Returns:
        ASCII string representation
    """
    lines = []
    lines.append("┌─────────────────┐")
    
    # Render XZ slice at player Y level
    for z in range(16, -1, -1):  # Z from far to near
        row = "│"
        for x in range(17):
            block = int(voxels[x, player_y, z])
            char = BLOCK_CHARS.get(block, '?')
            if x == 8 and z == 8:
                char = '@'  # Player position
            row += char
        row += "│"
        lines.append(row)
    
    lines.append("└─────────────────┘")
    return "\n".join(lines)


def render_voxels_side(voxels: np.ndarray) -> str:
    """
    Render side view (XY slice at player Z).
    """
    lines = []
    lines.append("Side View (XY):")
    lines.append("┌─────────────────┐")
    
    for y in range(16, -1, -1):  # Y from top to bottom
        row = "│"
        for x in range(17):
            block = int(voxels[x, y, 8])  # Z=8 (player Z)
            char = BLOCK_CHARS.get(block, '?')
            if x == 8 and y == 8:
                char = '@'
            row += char
        row += "│"
        lines.append(row)
    
    lines.append("└─────────────────┘")
    return "\n".join(lines)


def render_facing_blocks(facing: np.ndarray) -> str:
    """Render blocks along view ray."""
    chars = []
    for block in facing:
        char = BLOCK_CHARS.get(int(block), '?')
        chars.append(char)
    return "View ray: [" + "".join(chars) + "]"


def render_inventory(inventory: np.ndarray) -> str:
    """Render inventory counts."""
    items = [
        ('Log', int(inventory[0] + inventory[1] + inventory[2])),
        ('Plank', int(inventory[3])),
        ('Stick', int(inventory[4])),
        ('Cobble', int(inventory[5])),
        ('Coal', int(inventory[6])),
        ('Iron', int(inventory[7] + inventory[8])),
        ('Diamond', int(inventory[9])),
    ]
    parts = [f"{name}:{count}" for name, count in items if count > 0]
    return "Inventory: " + (", ".join(parts) if parts else "(empty)")


def render_compass(compass: np.ndarray) -> str:
    """Render log compass direction."""
    fwd, right, up, dist = compass
    
    # Direction arrow
    if abs(fwd) > abs(right):
        arrow = "↑" if fwd > 0 else "↓"
    else:
        arrow = "→" if right > 0 else "←"
    
    dist_blocks = dist * 30  # Denormalize
    return f"Log compass: {arrow} ({dist_blocks:.0f} blocks)"


def render_frame(
    obs: dict,
    action: int,
    reward: float,
    step: int,
    total_reward: float,
    player_pos: np.ndarray,
    player_rot: np.ndarray,
) -> str:
    """Render full frame."""
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append(f"Step {step:4d} | Action: {ACTION_NAMES[action]:12s} | Reward: {reward:+.2f} | Total: {total_reward:.1f}")
    lines.append(f"Pos: ({player_pos[0]:.1f}, {player_pos[1]:.1f}, {player_pos[2]:.1f}) | Rot: (pitch={player_rot[0]:.0f}, yaw={player_rot[1]:.0f})")
    lines.append("=" * 60)
    
    # Voxel views side by side
    voxels = np.array(obs['local_voxels'])
    top_view = render_voxels_top_down(voxels)
    
    lines.append("")
    lines.append("Top-down view (you are @):")
    lines.append(top_view)
    
    # Facing blocks
    lines.append("")
    lines.append(render_facing_blocks(np.array(obs['facing_blocks'])))
    
    # Compass
    if 'log_compass' in obs:
        lines.append(render_compass(np.array(obs['log_compass'])))
    
    # Inventory
    lines.append(render_inventory(np.array(obs['inventory'])))
    
    lines.append("")
    
    return "\n".join(lines)


def run_episode(
    env: MinecraftEnv,
    network: FastActorCritic,
    params: dict,
    key: jax.random.PRNGKey,
    max_steps: int = 500,
    render: bool = True,
    delay: float = 0.1,
) -> dict:
    """
    Run a single episode.
    
    Returns:
        Episode statistics dict
    """
    key, reset_key = jax.random.split(key)
    state, obs = env.reset(reset_key)
    
    total_reward = 0.0
    logs_collected = 0
    planks_crafted = 0
    
    for step in range(max_steps):
        # Get action from policy
        key, action_key = jax.random.split(key)
        action, log_prob, entropy, value = network.apply(
            params,
            jax.tree_util.tree_map(lambda x: x[None], obs),  # Add batch dim
            action_key,
            method=network.get_action_and_value,
        )
        action = int(action[0])
        
        # Track inventory before step
        prev_logs = float(obs['inventory'][0] + obs['inventory'][1] + obs['inventory'][2])
        prev_planks = float(obs['inventory'][3])
        
        # Step environment
        state, obs, reward, done, info = env.step(state, jnp.int32(action))
        reward = float(reward)
        total_reward += reward
        
        # Track items
        curr_logs = float(obs['inventory'][0] + obs['inventory'][1] + obs['inventory'][2])
        curr_planks = float(obs['inventory'][3])
        
        if curr_logs > prev_logs:
            logs_collected += int(curr_logs - prev_logs)
        if curr_planks > prev_planks:
            planks_crafted += int(curr_planks - prev_planks)
        
        # Render
        if render:
            clear_screen()
            frame = render_frame(
                obs, action, reward, step, total_reward,
                np.array(state.player.pos),
                np.array(state.player.rot),
            )
            print(frame)
            time.sleep(delay)
        
        if done:
            break
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'logs_collected': logs_collected,
        'planks_crafted': planks_crafted,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MCRL agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between frames (seconds)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (just collect stats)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Check available checkpoints
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint path not found: {ckpt_path}")
        sys.exit(1)
    
    steps = list_checkpoints(args.checkpoint)
    if not steps:
        print(f"No checkpoints found in {args.checkpoint}")
        print("Looking for params_*.pkl files...")
        sys.exit(1)
    
    print(f"Available checkpoints: {steps}")
    
    # Load checkpoint
    params, metadata = load_checkpoint(args.checkpoint, args.step)
    print(f"Loaded checkpoint: {metadata}")
    
    # Create network and environment
    network = FastActorCritic()
    env = MinecraftEnv(EnvConfig(world_size=(64, 64, 64)))
    
    # Run episodes
    key = jax.random.PRNGKey(args.seed)
    all_stats = []
    
    print(f"\nRunning {args.episodes} episodes...")
    print("Press Ctrl+C to stop\n")
    
    try:
        for ep in range(args.episodes):
            key, ep_key = jax.random.split(key)
            
            if not args.no_render:
                print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            
            stats = run_episode(
                env, network, params, ep_key,
                max_steps=args.max_steps,
                render=not args.no_render,
                delay=args.delay,
            )
            all_stats.append(stats)
            
            print(f"Episode {ep + 1}: Reward={stats['total_reward']:.1f}, "
                  f"Steps={stats['steps']}, Logs={stats['logs_collected']}, "
                  f"Planks={stats['planks_crafted']}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    # Summary
    if all_stats:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        avg_reward = np.mean([s['total_reward'] for s in all_stats])
        avg_steps = np.mean([s['steps'] for s in all_stats])
        total_logs = sum(s['logs_collected'] for s in all_stats)
        total_planks = sum(s['planks_crafted'] for s in all_stats)
        print(f"Episodes: {len(all_stats)}")
        print(f"Avg Reward: {avg_reward:.1f}")
        print(f"Avg Steps: {avg_steps:.0f}")
        print(f"Total Logs: {total_logs}")
        print(f"Total Planks: {total_planks}")


if __name__ == "__main__":
    main()
