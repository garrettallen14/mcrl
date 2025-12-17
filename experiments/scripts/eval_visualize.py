#!/usr/bin/env python3
"""
MCRL Evaluation & Visualization Suite

Loads a trained checkpoint and visualizes the agent's behavior in real-time
with rich terminal rendering of the Minecraft environment.

Usage:
    python experiments/scripts/eval_visualize.py --checkpoint path/to/params.pkl
    python experiments/scripts/eval_visualize.py  # Uses latest checkpoint
"""

import argparse
import pickle
import time
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcrl import MinecraftEnv, EnvConfig
from mcrl.core.types import BlockType, ItemType
from mcrl.training.networks_fast import create_fast_network, init_fast_network


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

# Block type to visual character mapping (with colors)
BLOCK_CHARS = {
    BlockType.AIR: (' ', '\033[0m'),           # Empty
    BlockType.STONE: ('█', '\033[90m'),        # Dark gray
    BlockType.DIRT: ('▓', '\033[33m'),         # Brown
    BlockType.GRASS: ('▓', '\033[32m'),        # Green
    BlockType.COBBLESTONE: ('▒', '\033[90m'),  # Gray
    BlockType.OAK_PLANKS: ('▒', '\033[93m'),   # Yellow
    BlockType.OAK_LOG: ('║', '\033[33m'),      # Brown vertical
    BlockType.OAK_LEAVES: ('♣', '\033[92m'),   # Bright green
    BlockType.BEDROCK: ('▀', '\033[35m'),      # Magenta
    BlockType.WATER: ('~', '\033[94m'),        # Blue
    BlockType.SAND: ('░', '\033[93m'),         # Yellow
    BlockType.COAL_ORE: ('●', '\033[90m'),     # Dark with dot
    BlockType.IRON_ORE: ('●', '\033[97m'),     # White dot
    BlockType.GOLD_ORE: ('●', '\033[93m'),     # Gold dot
    BlockType.DIAMOND_ORE: ('◆', '\033[96m'),  # Cyan diamond
}

# Action names
ACTION_NAMES = [
    "NOOP", "FORWARD", "BACK", "LEFT", "RIGHT",
    "JUMP", "SNEAK", "SPRINT",
    "LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP", "LOOK_DOWN",
    "ATTACK", "USE",
    "SLOT_1", "SLOT_2", "SLOT_3", "SLOT_4", "SLOT_5",
    "SLOT_6", "SLOT_7", "SLOT_8", "SLOT_9",
    "CRAFT_PLANKS", "CRAFT_STICKS"
]

# Item names
ITEM_NAMES = {
    ItemType.EMPTY: "Empty",
    ItemType.OAK_LOG: "Oak Log",
    ItemType.OAK_PLANKS: "Oak Planks",
    ItemType.STICK: "Stick",
    ItemType.WOODEN_PICKAXE: "Wood Pick",
    ItemType.STONE_PICKAXE: "Stone Pick",
    ItemType.IRON_PICKAXE: "Iron Pick",
    ItemType.COBBLESTONE: "Cobblestone",
    ItemType.COAL: "Coal",
    ItemType.IRON_INGOT: "Iron Ingot",
}

RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')


def move_cursor(row, col):
    """Move cursor to position."""
    print(f'\033[{row};{col}H', end='')


def get_block_char(block_type: int) -> tuple[str, str]:
    """Get character and color for block type."""
    return BLOCK_CHARS.get(block_type, ('?', '\033[91m'))


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_top_down_view(voxels: np.ndarray, size: int = 17) -> list[str]:
    """
    Render top-down view of voxels (looking down from above).
    Shows highest non-air block at each x,z position.
    """
    lines = []
    half = size // 2
    
    # Header
    lines.append(f"{BOLD}═══ TOP-DOWN VIEW (↑=North) ═══{RESET}")
    
    # Find player center
    center = size // 2
    
    for z in range(size):
        row = ""
        for x in range(size):
            # Find highest non-air block
            highest_block = BlockType.AIR
            for y in range(size - 1, -1, -1):
                if voxels[x, y, z] != BlockType.AIR:
                    highest_block = voxels[x, y, z]
                    break
            
            char, color = get_block_char(highest_block)
            
            # Mark player position
            if x == center and z == center:
                row += f"\033[91m@{RESET}"  # Red @ for player
            else:
                row += f"{color}{char}{RESET}"
        
        lines.append(row)
    
    return lines


def render_side_view(voxels: np.ndarray, size: int = 17) -> list[str]:
    """
    Render side view (looking from behind player, seeing in front).
    X is horizontal, Y is vertical, at player's Z.
    """
    lines = []
    center = size // 2
    
    lines.append(f"{BOLD}═══ SIDE VIEW (Y vs X at Z=center) ═══{RESET}")
    
    # Show Y from top to bottom
    for y in range(size - 1, -1, -1):
        row = ""
        for x in range(size):
            block = voxels[x, y, center]
            char, color = get_block_char(block)
            
            # Mark player position (at center x, player eye level ~y=8-9)
            if x == center and y in [center, center + 1]:
                row += f"\033[91m@{RESET}"
            else:
                row += f"{color}{char}{RESET}"
        
        lines.append(row)
    
    return lines


def render_front_view(voxels: np.ndarray, player_yaw: float, size: int = 17) -> list[str]:
    """
    Render what's in front of the player based on yaw.
    This shows a slice in the direction the player is facing.
    """
    lines = []
    center = size // 2
    
    # Determine facing direction based on yaw
    # yaw: 0=+Z, 90=-X, 180=-Z, 270=+X
    yaw_norm = player_yaw % 360
    if 45 <= yaw_norm < 135:
        direction = "West (-X)"
        # Looking -X, so show Y vs Z at X=front
        slice_dim = 0
    elif 135 <= yaw_norm < 225:
        direction = "South (-Z)"
        slice_dim = 2
    elif 225 <= yaw_norm < 315:
        direction = "East (+X)"
        slice_dim = 0
    else:
        direction = "North (+Z)"
        slice_dim = 2
    
    lines.append(f"{BOLD}═══ FRONT VIEW (Facing {direction}) ═══{RESET}")
    
    # Show cross-section in facing direction
    for y in range(size - 1, -1, -1):
        row = ""
        for i in range(size):
            if slice_dim == 0:  # X is the slice dim, show Y vs Z
                block = voxels[center + 2, y, i]  # +2 blocks in front
            else:  # Z is the slice dim, show Y vs X
                block = voxels[i, y, center + 2]
            
            char, color = get_block_char(block)
            row += f"{color}{char}{RESET}"
        
        lines.append(row)
    
    return lines


def render_facing_blocks(obs: dict) -> list[str]:
    """Render the blocks the player is looking at (ray cast results)."""
    lines = []
    lines.append(f"{BOLD}═══ FACING BLOCKS (0.5→4.0m) ═══{RESET}")
    
    facing = obs.get('facing_blocks', np.zeros(8, dtype=np.uint8))
    
    row = "  "
    for i, block_type in enumerate(facing):
        char, color = get_block_char(int(block_type))
        dist = 0.5 + i * 0.5
        row += f"{color}[{char}]{RESET}"
    
    lines.append(row)
    lines.append(f"  {''.join([f'{0.5+i*0.5:.1f}' + ' ' for i in range(8)])}")
    
    return lines


def render_inventory(obs: dict) -> list[str]:
    """Render player inventory."""
    lines = []
    lines.append(f"{BOLD}═══ INVENTORY ═══{RESET}")
    
    # The inventory observation is encoded item counts
    inv = obs.get('inventory', np.zeros(16))
    
    items = [
        ("Oak Log", inv[0] if len(inv) > 0 else 0),
        ("Planks", inv[1] if len(inv) > 1 else 0),
        ("Sticks", inv[2] if len(inv) > 2 else 0),
        ("Wood Pick", inv[3] if len(inv) > 3 else 0),
        ("Cobble", inv[4] if len(inv) > 4 else 0),
        ("Stone Pick", inv[5] if len(inv) > 5 else 0),
        ("Coal", inv[6] if len(inv) > 6 else 0),
        ("Iron", inv[7] if len(inv) > 7 else 0),
    ]
    
    for name, count in items:
        if count > 0:
            lines.append(f"  {name}: {int(count)}")
    
    if all(c == 0 for _, c in items):
        lines.append(f"  {DIM}(empty){RESET}")
    
    return lines


def render_log_compass(obs: dict) -> list[str]:
    """Render the log compass showing direction to nearest tree."""
    lines = []
    lines.append(f"{BOLD}═══ LOG COMPASS ═══{RESET}")
    
    compass = obs.get('log_compass', np.zeros(4))
    forward, right, up, dist = compass
    
    # Visual compass
    lines.append("       ↑N        ")
    
    # Arrow pointing to tree
    if dist < 0.99:
        # Calculate arrow direction
        if abs(forward) > abs(right):
            if forward > 0.3:
                arrow = "   ↑   "
            elif forward < -0.3:
                arrow = "   ↓   "
            else:
                arrow = "   ●   "
        else:
            if right > 0.3:
                arrow = "     → "
            elif right < -0.3:
                arrow = " ←     "
            else:
                arrow = "   ●   "
        
        lines.append(f"  W ←  {arrow}  → E")
        lines.append("       ↓S        ")
        lines.append(f"  Distance: {dist * 30:.1f} blocks")
        lines.append(f"  Dir: F={forward:+.2f} R={right:+.2f} U={up:+.2f}")
    else:
        lines.append("  W ←  [?]  → E")
        lines.append("       ↓S        ")
        lines.append(f"  {DIM}No trees nearby{RESET}")
    
    return lines


def render_player_state(state, obs: dict) -> list[str]:
    """Render player state information."""
    lines = []
    lines.append(f"{BOLD}═══ PLAYER STATE ═══{RESET}")
    
    player = state.player
    pos = player.pos
    rot = player.rot
    
    lines.append(f"  Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    lines.append(f"  Rotation: pitch={rot[0]:.1f}° yaw={rot[1]:.1f}°")
    lines.append(f"  Health: {player.health:.1f}/20  Hunger: {player.hunger:.1f}/20")
    lines.append(f"  On Ground: {'Yes' if player.on_ground else 'No'}")
    
    if hasattr(player, 'mining_progress') and player.mining_progress > 0:
        lines.append(f"  Mining: {player.mining_progress} ticks")
    
    return lines


def render_action_history(actions: list, rewards: list) -> list[str]:
    """Render recent action history."""
    lines = []
    lines.append(f"{BOLD}═══ RECENT ACTIONS ═══{RESET}")
    
    for i, (action, reward) in enumerate(zip(actions[-8:], rewards[-8:])):
        action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"ACT_{action}"
        reward_str = f"{reward:+.2f}" if reward != 0 else "0.00"
        color = '\033[92m' if reward > 0 else ('\033[91m' if reward < 0 else DIM)
        lines.append(f"  {action_name:12s} → {color}{reward_str}{RESET}")
    
    return lines


def render_stats(episode: int, step: int, total_reward: float, fps: float) -> list[str]:
    """Render episode statistics."""
    lines = []
    lines.append(f"{BOLD}═══ EPISODE {episode} ═══{RESET}")
    lines.append(f"  Step: {step}")
    lines.append(f"  Total Reward: {total_reward:.2f}")
    lines.append(f"  FPS: {fps:.1f}")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualization(
    checkpoint_path: str,
    num_episodes: int = 5,
    max_steps: int = 1000,
    delay: float = 0.05,
    world_size: tuple = (64, 64, 64),
):
    """Run evaluation with live visualization."""
    
    print(f"\n{'='*60}")
    print("MCRL Evaluation & Visualization Suite")
    print(f"{'='*60}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    # Handle various checkpoint formats
    if isinstance(ckpt, dict):
        if 'params' in ckpt:
            params = ckpt['params']
            # params might be nested further
            if isinstance(params, dict) and 'params' in params:
                params = params['params']
            metadata = ckpt.get('metadata', {})
            print(f"  Step: {metadata.get('step', 'unknown')}")
            print(f"  Reward: {metadata.get('reward', 'unknown')}")
        else:
            # Direct params dict
            params = ckpt
            print("  (Direct params dict)")
    else:
        params = ckpt
        print("  (Raw params)")
    
    # Wrap in 'params' for network.apply if not already
    if 'params' not in params:
        params = {'params': params}
    
    print(f"  Param keys: {list(params.get('params', params).keys())[:5]}...")
    
    # Create environment
    print(f"\nCreating environment (world size: {world_size})...")
    env = MinecraftEnv(EnvConfig(world_size=world_size))
    
    # Create network
    print("Creating network...")
    network = create_fast_network(num_actions=25, ultra_fast=False)
    
    # Verify params work
    print("Verifying network...")
    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key)
    
    # Create dummy observation for network shape
    dummy_obs = {
        'local_voxels': jnp.zeros((17, 17, 17), dtype=jnp.uint8),
        'facing_blocks': jnp.zeros(8, dtype=jnp.uint8),
        'inventory': jnp.zeros(16, dtype=jnp.float32),
        'player_state': jnp.zeros(14, dtype=jnp.float32),
        'log_compass': jnp.zeros(4, dtype=jnp.float32),
    }
    
    # JIT compile the action selection
    @jax.jit
    def select_action(params, obs, key):
        # Format observation for network - add batch dimension to all components
        batched_obs = {
            'local_voxels': obs['local_voxels'][None, ...],
            'facing_blocks': obs['facing_blocks'][None, ...],
            'inventory': obs['inventory'][None, ...],
            'player_state': obs['player_state'][None, ...],
            'log_compass': obs.get('log_compass', jnp.zeros(4))[None, ...],
        }
        
        # Get action distribution
        logits, value = network.apply(params, batched_obs)
        
        # Sample action
        action = jax.random.categorical(key, logits[0])
        # Value might be (B,) or (B, 1)
        v = value[0] if value.ndim == 1 else value[0, 0]
        return action, v
    
    # Warm up JIT
    print("JIT compiling...")
    _ = select_action(params, obs, key)
    print("Ready!\n")
    
    time.sleep(1)
    
    # Run episodes
    for episode in range(num_episodes):
        # Reset environment
        key, reset_key = jax.random.split(key)
        state, obs = env.reset(reset_key)
        
        total_reward = 0.0
        actions_history = []
        rewards_history = []
        step = 0
        
        episode_start = time.time()
        last_render = time.time()
        
        while step < max_steps:
            step_start = time.time()
            
            # Select action
            key, action_key = jax.random.split(key)
            action, value = select_action(params, obs, action_key)
            action = int(action)
            
            # Step environment
            state, obs, reward, done, info = env.step(state, jnp.int32(action))
            reward = float(reward)
            
            total_reward += reward
            actions_history.append(action)
            rewards_history.append(reward)
            
            # Render visualization
            if time.time() - last_render > delay:
                clear_screen()
                
                # Get voxels as numpy for rendering
                voxels = np.array(obs['local_voxels'])
                
                # Calculate FPS
                elapsed = time.time() - episode_start
                fps = step / elapsed if elapsed > 0 else 0
                
                # Build visualization
                lines = []
                
                # Header
                lines.extend(render_stats(episode + 1, step, total_reward, fps))
                lines.append("")
                
                # Main views (side by side)
                top_down = render_top_down_view(voxels)
                side_view = render_side_view(voxels)
                
                # Combine views side by side
                lines.append(f"{BOLD}{'─'*40} WORLD VIEWS {'─'*40}{RESET}")
                max_lines = max(len(top_down), len(side_view))
                for i in range(max_lines):
                    left = top_down[i] if i < len(top_down) else " " * 20
                    right = side_view[i] if i < len(side_view) else ""
                    lines.append(f"{left}    │    {right}")
                
                lines.append("")
                
                # Info panels (side by side)
                compass = render_log_compass(obs)
                inventory = render_inventory(obs)
                player = render_player_state(state, obs)
                actions = render_action_history(actions_history, rewards_history)
                
                # Left column: compass + inventory
                # Right column: player + actions
                lines.append(f"{BOLD}{'─'*40} STATUS {'─'*45}{RESET}")
                
                left_col = compass + [""] + inventory
                right_col = player + [""] + actions
                
                max_info = max(len(left_col), len(right_col))
                for i in range(max_info):
                    left = left_col[i] if i < len(left_col) else ""
                    right = right_col[i] if i < len(right_col) else ""
                    # Pad left column
                    left_padded = left + " " * (35 - len(left.replace('\033[0m', '').replace('\033[1m', '').replace('\033[2m', '').replace('\033[32m', '').replace('\033[33m', '').replace('\033[91m', '').replace('\033[92m', '').replace('\033[93m', '')))
                    lines.append(f"  {left}{'':30s}│  {right}")
                
                lines.append("")
                lines.append(f"{DIM}Press Ctrl+C to stop{RESET}")
                
                # Print all lines
                print('\n'.join(lines))
                
                last_render = time.time()
            
            step += 1
            
            if done:
                break
            
            # Small delay for visibility
            time.sleep(max(0, delay - (time.time() - step_start)))
        
        # Episode summary
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1} Complete!")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Time: {time.time() - episode_start:.1f}s")
        print(f"{'='*60}")
        
        if episode < num_episodes - 1:
            print("\nStarting next episode in 3 seconds...")
            time.sleep(3)
    
    print("\n✓ Evaluation complete!")


def find_latest_checkpoint(base_dir: str = "experiments/runs") -> str:
    """Find the most recent checkpoint."""
    checkpoints = []
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith('params_') and f.endswith('.pkl'):
                path = os.path.join(root, f)
                checkpoints.append((os.path.getmtime(path), path))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def main():
    parser = argparse.ArgumentParser(description="MCRL Evaluation & Visualization")
    parser.add_argument("--checkpoint", "-c", type=str, help="Path to checkpoint file")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", "-s", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--delay", "-d", type=float, default=0.1, help="Delay between frames (seconds)")
    parser.add_argument("--world-size", "-w", type=int, default=64, help="World size (cubic)")
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("No checkpoint specified, finding latest...")
        checkpoint_path = find_latest_checkpoint()
        print(f"Using: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    try:
        run_visualization(
            checkpoint_path=checkpoint_path,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            delay=args.delay,
            world_size=(args.world_size, args.world_size, args.world_size),
        )
    except KeyboardInterrupt:
        print("\n\nStopped by user.")


if __name__ == "__main__":
    main()
