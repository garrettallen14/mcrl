Proposed Round 3 Research Queries
Query 27: PureJaxRL-Style PPO for Voxel Environments
"What are the exact implementation details for PPO in fully JAX-native environments like Craftax, Gymnax, or JaxMARL? Specifically: How is rollout storage handled on GPU (pytrees vs arrays)? What are optimal hyperparameters (num_envs, num_steps, minibatch_size, update_epochs) for 1024+ parallel envs? How does GAE interact with vmapped value functions? Include code patterns or papers from 2024-2025."

Query 28: 3D Convolution Architectures for Voxel RL
"What neural network architectures work best for 3D voxel grid observations in RL? Compare: (1) 3D CNNs, (2) sparse convolutions, (3) voxel transformers (like VoxFormer), (4) point cloud networks applied to non-air voxels. Which balance feature extraction quality vs. inference speed for real-time RL training? Any Minecraft or voxel game examples?"

Query 29: Intrinsic Motivation for Tech-Tree Progression
"What intrinsic reward mechanisms best accelerate discovery of Minecraft's tech tree without dense supervision? Compare: RND (random network distillation), ICM (curiosity), novelty-based bonuses on inventory changes, count-based exploration on block types mined, and goal-conditioned HER-style hindsight. Which avoids reward hacking (e.g., repeatedly breaking/placing blocks) while encouraging actual progression?"

Query 30: Hierarchical RL for Long-Horizon Crafting
"For a 12-milestone crafting progression (wood→planks→sticks→table→wood_pick→cobble→stone_pick→iron_ore→furnace→iron_ingot→iron_pick→diamond), what hierarchical RL approaches work best? Compare: (1) Options framework with learned termination, (2) Feudal networks (manager/worker), (3) Goal-conditioned policies with automatic subgoal discovery, (4) Fixed skill decomposition with PPO over options. Which achieves best sample efficiency in procedural environments?"

### Rollout Storage on GPU: PyTrees vs. Arrays

In fully JAX-native PPO implementations (e.g., **PureJaxRL**, used in **Craftax**, **Gymnax**, and extended in **JaxMARL**), rollout storage is handled using **JAX PyTrees** rather than flat arrays. 

- Observations, actions, log probabilities, values, rewards, and dones are collected in a `TrainState` or custom `Trajectory` dataclass (often using `flax.struct.dataclass`).
- The rollout buffer is a nested PyTree with leading dimensions `(num_steps, num_envs)` for time and parallel environments.
- This structure stays entirely on the GPU/TPU, enabling end-to-end JIT compilation of the entire training loop (rollout + GAE + updates).
- PyTrees are preferred because JAX transformations (e.g., `jax.jit`, `jax.vmap`, `jax.scan`) naturally operate on them, avoiding manual flattening/reshaping that would break differentiability or compilation.
- Example pattern from PureJaxRL-style code:
  ```python
  @flax.struct.dataclass
  class Trajectory:
      obs: jnp.ndarray  # shape (num_steps + 1, num_envs, obs_dim)
      actions: jnp.ndarray
      logprobs: jnp.ndarray
      values: jnp.ndarray
      rewards: jnp.ndarray
      dones: jnp.ndarray
  ```
- This approach enables massive vectorization (e.g., 2048+ envs on a single GPU) and speedups of 10x–4000x compared to CPU-vectorized PyTorch baselines.

Flat arrays are rarely used for storage in modern JAX RL, as they complicate scanning over trajectories and interacting with neural networks (which output PyTrees via Flax/Optax).

### Optimal Hyperparameters for 1024+ Parallel Envs

JAX-native environments like Gymnax, Craftax, and JaxMARL exploit massive parallelism (vmap over envs), so hyperparameters shift toward larger batch sizes while keeping total steps per update reasonable for throughput.

Common patterns from 2024 implementations (PureJaxRL, Craftax baselines, JaxMARL IPPO/MAPPO):

| Hyperparameter       | Typical Range for 1024–4096+ Envs                  | Notes/Examples |
|----------------------|----------------------------------------------------|---------------|
| `num_envs`          | 1024–8192 (or higher, limited by GPU memory)      | Craftax/PureJaxRL often uses 2048–4096; Brax/Jumanji up to 8192+. Higher = better GPU utilization. |
| `num_steps`         | 128–512                                           | Shorter horizons common due to large parallelism (e.g., 256 in recurrent PPO for Craftax; 128–256 in Gymnax baselines). Total samples per update = num_envs × num_steps ≈ 1M–4M. |
| `minibatch_size`    | 32k–256k (or full batch if memory allows)         | Often computed as (num_envs × num_steps) // num_minibatches. 4–16 minibatches typical. |
| `update_epochs`     | 3–8                                               | 4 common (PureJaxRL default); higher (8–10) for harder tasks like Craftax-Classic to stabilize. |

- Total batch size (num_envs × num_steps) is often 1M–8M timesteps per PPO update for high-throughput training on A100/H100 GPUs.
- Other stable hypers: γ=0.99, GAE λ=0.95, clip=0.2, ent_coef=0.01–0.001, LR=1e-4–3e-4 (scheduled).
- These enable training Craftax-Classic to ~90% optimal reward in <1 hour on a single GPU (1B steps).

Tune via random search; larger parallelism reduces need for many epochs per update.

### GAE Interaction with Vmapped Value Functions

In JAX PPO, environments and networks are vmapped over the environment dimension for parallel rollouts.

- The value function is vmapped: `value = vmap(network.apply)(params, obs)` → shape `(num_envs, 1)`.
- During rollout, values are collected with shape `(num_steps + 1, num_envs)`.
- GAE is computed via a reverse `jax.lax.scan` over the time dimension (independent across envs due to episodic resets):
  ```python
  def compute_gae(carry, step):
      last_gae_lam = carry
      delta = step['reward'] + gamma * step['next_value'] * (1 - step['done']) - step['value']
      gae = delta + gamma * gae_lambda * (1 - step['done']) * last_gae_lam
      return gae, gae
  
  _, advantages = jax.lax.scan(compute_gae, jnp.zeros(num_envs), traj_dict, reverse=True)
  returns = advantages + traj_dict['values'][:-1]
  ```
- Vmapping ensures parallel value estimates; scan handles temporal dependencies efficiently without Python loops.
- Advantages are normalized across the full batch (or per-minibatch) for stability.
- This pattern (from PureJaxRL) is used directly in Craftax, Gymnax, and JaxMARL PPO baselines.

### Code Patterns and Recent Papers (2024–2025)

**Key code pattern** (PureJaxRL-style, single-file JITted loop):
- Full training loop JITted: rollout via scanned/vmapped step → GAE via reverse scan → shuffled minibatches via `jax.random.permutation` → multiple epochs of value_and_grad updates.
- Example repos:
  - PureJaxRL (luchris429/purejaxrl): Core template for Gymnax/Craftax PPO.
  - Craftax Baselines (MichaelTMatthews/Craftax_Baselines): PPO with OptimisticReset for non-auto-reset envs.
  - JaxMARL (FLAIROx/JaxMARL): IPPO/MAPPO extending PureJaxRL for multi-agent.
  - TransformerXL PPO (Reytuag/transformerXL_PPO_JAX): Recurrent extensions beating Craftax baselines.

**Relevant papers (2024–2025)**:
- "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning" (Matthews et al., 2024): Introduces Craftax (JAX-native), uses PureJaxRL PPO; solves Craftax-Classic in ~1 hour.
- "JaxMARL: Multi-Agent RL Environments and Algorithms in JAX" (Rutherford et al., 2024 update): IPPO/MAPPO in pure JAX, 100x+ speedups.
- Various 2024–2025 works on open-ended RL (e.g., XLand-MiniGrid, Kinetix) build on these for trillion-step training via JAX parallelism.

These patterns prioritize full JIT compilation and massive vmap for GPU efficiency. For hands-on, start with PureJaxRL notebooks.

### Comparison of Architectures for 3D Voxel Grid Observations in RL

3D voxel grids are common in environments like robotics simulations or voxel-based games (e.g., Minecraft-like worlds), where observations represent occupied space (e.g., blocks or objects) in a 3D volume. These grids are often **sparse** (most voxels are empty/air), making efficient processing crucial for RL, especially during real-time training where the policy network must run at high frequency (e.g., thousands of steps per second).

Here's a comparison of the requested architectures:

| Architecture                  | Description                                                                 | Feature Extraction Quality                                                                 | Inference Speed & Efficiency                                                                 | Suitability for Real-Time RL Training |
|-------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------|
| **(1) Dense 3D CNNs**        | Standard 3D convolutions on a full voxel grid (e.g., VoxNet-style).        | Good local feature learning via convolutions, but limited by resolution due to cubic memory/compute growth (O(n³)). Struggles with fine details in large grids. | Very slow and memory-intensive; impractical for grids >32³-64³ without downsampling. High FLOPs even on empty space. | Poor – too slow for frequent policy evaluations in RL loops. Rarely used directly for large/sparse voxels. |
| **(2) Sparse Convolutions**  | Convolutions only on occupied voxels (e.g., Minkowski Engine, SECOND, VoxelNeXt). Skips empty space using sparse tensors. | Excellent local and contextual features; widely used in 3D detection/segmentation. Preserves sparsity, enabling higher resolutions. | Fast: 5-10x speedup over dense CNNs; low memory as it ignores empty voxels (common in games, >90% empty). Real-time capable (e.g., 20-50+ Hz on modern GPUs). | **Best balance** – high quality + fast inference. Ideal for real-time RL (e.g., actor-critic updates). |
| **(3) Voxel Transformers**   | Transformer-based on voxels/queries (e.g., VoxFormer, VoTr). Often sparse queries + self-attention for global context. | Superior global reasoning and long-range dependencies; SOTA in scene completion/detection. Better at complex spatial relationships. | Slower due to attention (O(n²)); mitigated in sparse versions but still heavier than convolutions. Higher memory for attention matrices. | Moderate – good quality but inference slower; less ideal for high-throughput RL training unless heavily optimized/sparsified. |
| **(4) Point Cloud Networks** | Treat non-air voxels as points; use PointNet++, set abstraction, or graph convs. | Strong on irregular data; permutation-invariant. Good for fine-grained local geometry without grid quantization loss. | Variable: Fast with MLPs (e.g., PointNet), but neighbor search (kNN) can bottleneck. Hybrids (point-voxel) often fastest overall. | Good – efficient for sparse data; comparable or faster than sparse convs in some cases. Suitable for RL if sparsity is high. |

#### Key Trade-offs for Feature Quality vs. Inference Speed in Real-Time RL
- **Quality**: Voxel transformers excel at global context (e.g., understanding distant occlusions or structures). Sparse convs provide strong local+mid-range features. Point cloud nets preserve exact positions. Dense 3D CNNs lag due to resolution limits.
- **Speed**: Sparse convolutions and point cloud approaches (especially hybrids like PV-CNN or PVCNN++) offer the best balance – they handle sparsity natively, achieving real-time speeds (e.g., 10-50 ms per forward pass on GPU) while scaling to large grids (e.g., 200³+ resolutions).
- **Real-time RL training**: RL requires millions of environment steps, with the observation encoder running per step. Sparse convs or point-voxel hybrids minimize latency, enabling faster training walls-clock time. Transformers add overhead, risking bottlenecks in actor rollout/collection phases.

Evidence from benchmarks (primarily 3D detection, analogous to voxel feature extraction):
- Sparse convs outperform dense by 5-10x in speed/memory while matching or exceeding accuracy.
- Point-voxel fusions (e.g., PV-RCNN++) achieve higher accuracy than pure voxel/sparse with similar or better speed.
- Sparse voxel transformers (e.g., VoxFormer) are SOTA in quality but ~2-3x slower than conv-based.

#### Minecraft/Voxel Game Examples in RL
Minecraft RL benchmarks (e.g., MineRL, MineDojo) typically use **pixel-based observations** (rendered 2D RGB frames) rather than raw 3D voxel grids:
- **VPT (Video PreTraining, 2022)**: Uses a residual CNN + transformer on 2D frames; fine-tuned with RL.
- **DreamerV3 (2023)**: World model-based RL; first to obtain diamonds from scratch (sparse rewards, no human data). Uses RSSM (Recurrent State-Space Model) on compressed image features – effectively 2D CNN encoder.
- Recent works (2024-2025, e.g., CLIP4MC for vision-language rewards): Still frame-based VLMs or CNNs.

No major published works use direct 3D voxel observations for Minecraft RL (as of late 2025). Reasons:
- Minecraft provides easy RGB rendering; voxel access would require custom env mods.
- Agent POV is first-person; full 3D grid might give unfair global view (cheating-like).
- Some research envs (e.g., ManiSkill, DMC with depth) use point clouds or voxels, where sparse convs/point nets shine for manipulation tasks.

In custom voxel RL (e.g., robotics sims with voxel occupancy), **sparse convolutions** or **point cloud networks** are preferred for the quality-speed balance in real-time training.

If your env provides sparse voxels, start with sparse convs (e.g., via MinkowskiEngine) for robust performance. For ultimate speed with good quality, consider point-voxel hybrids.

**Novelty-based bonuses on inventory changes and count-based exploration on block types mined** are the intrinsic reward mechanisms that best accelerate discovery of Minecraft's tech tree (e.g., progressing from punching trees to crafting iron tools via resource gathering, mining rarer ores, and crafting intermediates) without dense supervision.

These directly incentivize **hierarchical progression** tied to the game's core mechanics—inventory accumulation of novel items (e.g., wood → planks → sticks → tools) and mining new block types (e.g., wood → cobblestone → iron ore → diamonds)—while robustly avoiding reward hacking like repeated break/place cycles on the same block (which yield no net inventory gain or new block discoveries).

### Detailed Comparison
The table below compares all methods based on:
- **Tech Tree Acceleration**: Sample efficiency and progression depth in Minecraft-like environments (e.g., MineRL, Crafter/Craftax proxies with tech trees via crafting achievements).
- **Hack Resistance**: Avoidance of local loops (e.g., break/place same block for transient state changes).
- **Mechanism & Rationale**: How it works intrinsically (no external goals/rewards).

| Mechanism | Tech Tree Acceleration | Hack Resistance | Mechanism & Rationale |
|-----------|-------------------------|-----------------|-----------------------|
| **RND (Random Network Distillation)** | Moderate (good initial exploration; plateaus on long-horizon tech) | Poor-Moderate | Fixed random network predicts next-state features; reward = prediction error on novel states. Explores unseen pixels/states but prone to local novelty (e.g., spinning for texture noise) or repetitive actions if they generate unpredictable visuals. In Craftax (Minecraft proxy), lags behind RNN baselines on intermediate tech (tools/enemies). |
| **ICM (Curiosity Module)** | Moderate (better than random but limited depth) | Moderate | Inverse/forward models predict actions/dynamics; reward = forward-model error. Rewards controllable novelty, reducing "noisy TV" exploits vs. RND. In Malmo/Minecraft, encourages leaving spawn but fails consistent tasks (e.g., tree-finding) due to replay buffer delays; no tech tree gains. |
| **Novelty-based on Inventory Changes** | **High** (directly tracks tech milestones) | **Excellent** | Density model (e.g., kNN/VAE) on inventory vectors; reward bonus for rare Δinventory (e.g., +1 log → +64 planks). Tech tree = inventory novelty sequence; no net change from break/place hacks. Ideal for open-ended RL; complements finding skills. |
| **Count-based on Block Types Mined** | **High** (proven SOTA for resource discovery) | **Excellent** | Pseudo-counts (e.g., 1/√N_type_visits) on unique blocks mined; bonus decays per type. Directly rewards tech prerequisites (e.g., mine cobblestone → iron); hacks on same block yield diminishing returns. Used in Plan4MC's Finding-skill (state-grid counts proxy block discovery); enables iron pickaxe in 7M steps (beats baselines). |
| **Goal-Conditioned HER-Style Hindsight** | Moderate (needs subgoal curriculum) | Good | Relabel failed trajectories to achieved subgoals (e.g., hindsight "got wood" as goal). Goal-conditioned policy learns multi-goal skills; hindsight densifies sparse rewards. Useful for defined tech (e.g., craft X), but open-ended needs auto-goals (e.g., next inventory item); less pure intrinsic, risks goal-irrelevant loops if poor relabeling. |

### Why These Two Excel for Progression + Anti-Hacking
- **Tied to Tech Tree Structure**: Minecraft progression is a directed acyclic graph (DAG) of inventory/block unlocks (e.g., need 3 cobble + 2 sticks → stone pick → iron). Inventory novelty/counts reward exactly this DAG traversal, creating automatic curriculum.
- **Hack-Proof**: Break/place same block → 0 net inventory change, 0 new type → 0 bonus. RND/ICM reward transient pixel/state changes from loops.
- **Sample Efficiency**: Plan4MC (count-based Finding) solves full tech tree (16+ skills) in 7M steps vs. baselines' failure; inventory novelty similarly sparse but high-signal.
- **Implementation Notes**:
  - **Inventory Novelty**: Embed inventory (e.g., 100-slot vector) via autoencoder; r_int = -log(density(new_inv)).
  - **Block Counts**: Track mined types (200+ in MC); r_int = β / √N_type (β tunes exploration).

**Hybrid Recommendation**: Use **count-based on blocks** for early exploration (finding rares) + **inventory novelty** for mid-game crafting/chaining. This mirrors Plan4MC's success (SOTA demo-free tech tree solver) while being fully intrinsic.

The 12-milestone progression (wood → planks → sticks → table → wood_pick → cobble → stone_pick → iron_ore → furnace → iron_ingot → iron_pick → diamond) is a classic **long-horizon**, **hierarchical**, **sparse-reward** crafting task, similar to the ObtainDiamond benchmark in MineRL (a Minecraft-based RL environment). This requires temporal abstraction to manage credit assignment over thousands of steps and efficient exploration in procedural worlds.

Hierarchical RL (HRL) approaches generally outperform flat RL here due to decomposition into sub-tasks (e.g., "gather wood," "craft pickaxe," "mine iron"). Below is a comparison of the four approaches, based on performance in similar Minecraft/MineRL tasks and general HRL literature.

### (1) Options Framework with Learned Termination (e.g., Option-Critic)
Learns temporally extended actions ("options") end-to-end, including initiation, intra-option policy, and termination function. Options emerge automatically via gradients on termination.

**Pros**:
- Reduces manual engineering.
- Allows discovery of useful milestones.

**Cons**:
- Learning terminations and options jointly is unstable in sparse, long-horizon settings.
- Often requires dense intrinsic rewards or pretraining.

In Minecraft-like tasks: Used in early works (e.g., H-DRLN for lifelong learning) and modern agents, but rarely tops leaderboards alone. In MineRL competitions, it appears in hybrids but struggles with full diamond progression without demonstrations.

Sample efficiency: Moderate; better than flat RL but often outperformed by goal-conditioned methods.

### (2) Feudal Networks (FuN: Manager/Worker)
A manager sets abstract goals (directions in latent space) at low frequency; a worker executes primitive actions to achieve them, with intrinsic rewards for progress.

**Pros**:
- Decouples high-level planning from low-level control.
- Emergent sub-policies for milestones.
- Strong on credit assignment via dilated transitions.

**Cons**:
- Latent goals can be hard to interpret/train in highly procedural worlds.

In Minecraft-like tasks: Explicitly used in recent hierarchical agents (e.g., with Option-Critic/HIRO/HAC). FuN-inspired decoupling helps in long chains, but less dominant than goal-conditioned variants in MineRL.

Sample efficiency: High in some benchmarks (e.g., Atari/Montezuma's Revenge analogs), but in full procedural crafting, hybrids perform better.

### (3) Goal-Conditioned Policies with Automatic Subgoal Discovery (e.g., HIRO, HAC, or with HER/landmarks)
High-level policy selects subgoals (e.g., "obtain planks"); low-level goal-conditioned policy achieves them. Automatic discovery via hindsight experience replay (HER), landmarks, or causality.

**Pros**:
- Flexible subgoal space (states/items).
- HER relabeling provides dense signals.
- Excellent for hierarchical sparse tasks.

**Cons**:
- Subgoal feasibility/reachability can be tricky without off-policy corrections.

In Minecraft-like tasks: Dominant in MineRL competitions and recent works. Winners (e.g., 2019 HDQfD, 2020 SEIHAI) used hierarchical goal-conditioned setups, often with demonstrations for bootstrapping. Methods like HIRO/HAC excel at subgoal chains; automatic discovery (e.g., via landmarks or reachability) handles procedural variability well.

Sample efficiency: **Highest** among pure HRL; enables solving ObtainDiamond with limited samples (e.g., <8M in competitions).

### (4) Fixed Skill Decomposition with PPO over Options
Pre-define or hand-craft skills/options (e.g., one per milestone: "craft table," "mine cobble"); train low-level skills separately, then use PPO for high-level selection over fixed options.

**Pros**:
- Stable training (PPO is robust).
- Reusable skills.

**Cons**:
- Requires domain knowledge for decomposition.
- Less adaptable to procedural changes or new tasks.
- Fixed hierarchy limits discovery.

In Minecraft-like tasks: Used in some MineRL entries (e.g., aggregating actions into options) and skill-based planners. Effective with good decomposition but lags behind discovered hierarchies in open-ended settings.

Sample efficiency: Good if decomposition is expert-designed, but lower than automatic methods in unexplored procedural environments.

### Which Achieves Best Sample Efficiency in Procedural Environments?
**(3) Goal-conditioned policies with automatic subgoal discovery** perform best overall for sample efficiency in procedural, long-horizon crafting like this progression/MineRL ObtainDiamond.

Evidence:
- MineRL competitions (2019–2021) → Top solutions (e.g., HDQfD, SEIHAI) were hierarchical goal-conditioned, often with data-efficient tweaks like off-policy learning or demonstrations.
- Recent works (e.g., skill graphs with finding/manipulation/crafting subgoals, or HER-enhanced hierarchies) solve full tech trees efficiently without heavy reliance on fixed structures.
- Goal-conditioned HRL + automatic discovery (e.g., via HER, landmarks, or intrinsic rewards) handles sparse rewards and procedural generation better than pure options (unstable terminations) or feudal (latent abstraction issues).
- Fixed decomposition works but requires manual insight, reducing generality/efficiency in fully procedural setups.

Hybrids (e.g., goal-conditioned + learned options/terminations) are common in state-of-the-art, but pure automatic goal-conditioned edges out for efficiency in these environments. If demonstrations are available (common in MineRL), all improve dramatically via pretraining.