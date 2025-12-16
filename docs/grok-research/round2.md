Looking at the Round 1 findings, I'll now craft Round 2 queries that are sharper, more opinionated, and directly target implementation decisions for building your custom Minecraft RL environment and selecting the right agent architecture.

---

# Round 2: Implementation-Focused Research Queries

## Environment Engineering (Custom Simulator)

**Query 12: Craftium vs MineRL Deep Dive**
> "Compare Craftium (Luanti/Minetest-based, 2024-2025) vs MineRL for training Minecraft RL agents. What are the exact API differences, action space implementations, and observation formats? Has anyone published transfer results from Craftium-trained agents to real Minecraft? What modifications are needed to match MineRL's task definitions (e.g., ObtainDiamond)? Include any December 2024-December 2025 papers or experiments."

**Query 13: Building a Minimal Viable Minecraft Simulator**
> "If I wanted to build a custom Minecraft-like voxel simulator from scratch optimized purely for RL training (not gameplay), what are the minimum required game mechanics to preserve for diamond-obtaining policies to transfer? Specifically: which physics (gravity, collision), which crafting subset, which world generation features, and which visual fidelity level are essential vs. can be simplified? Reference any ablation studies or simulator design papers from 2024-2025."

**Query 14: GPU-Accelerated Voxel Environments**
> "Are there any GPU-accelerated voxel world simulators (JAX, CUDA, or similar) for RL as of late 2025? How do projects like MineWorld (neural simulator) or any Isaac Gym-style voxel environments compare to CPU-based Craftium/MineRL in steps-per-second? What would it take to port Minecraft-like mechanics to a Brax/MJX-style differentiable simulator?"

**Query 15: Action Space Engineering Details**
> "What is the exact action space specification used by LARM (ICML 2025), RL-GPT (NeurIPS 2024), and Dreamer 4 for Minecraft? Include the discrete action dimensions, camera discretization bins, and any macro-actions (crafting, navigation). How do these differ from VPT's original action space? Which specific abstractions (e.g., auto-craft, auto-attack-until-break) contribute most to sample efficiency?"

**Query 16: Observation Stack Design**
> "What is the optimal observation stack for a Minecraft RL agent targeting diamond acquisition in 2025? Compare: (1) raw pixels only at various resolutions, (2) pixels + inventory vector, (3) pixels + inventory + compass/goal direction, (4) pixels + semantic segmentation, (5) text state description. Which combination achieves best sample efficiency vs. compute tradeoff? Reference LARM, Optimus-2, or any 2025 ablations."

---

## Agent Architecture Research

**Query 17: LARM Architecture Deep Dive**
> "Explain the LARM (ICML 2025) architecture in detail: How does TinyLLaVA-3.1B integrate with the CLIP vision encoder? What is the exact structure of the action and value heads? How does Referee RL differ from standard PPO? What is the training loop (online vs offline, batch sizes, learning rates)? How does it achieve 93% diamond sword success with only 42 hours on a single RTX 4090?"

**Query 18: Autoregressive vs. Diffusion Policies for Games**
> "Compare autoregressive action prediction (like LARM, VPT) vs. diffusion-based policies (like DIAMOND's approach) for game-playing agents as of 2025. Which is more sample efficient? Which handles the hybrid discrete+continuous action space of Minecraft better? Are there any head-to-head comparisons on similar benchmarks?"

**Query 19: World Model Architecture Choices**
> "For building a world model agent for Minecraft in late 2025, should I use: (1) RSSM-style latent dynamics (DreamerV3), (2) block-causal transformers (Dreamer 4), (3) diffusion-based frame prediction (DIAMOND/GameNGen), or (4) discrete tokenized world models (Genie 3)? Compare on: training stability, rollout fidelity over long horizons, compute requirements, and ability to handle Minecraft's partial observability and stochasticity."

**Query 20: Scaling Laws for Game-Playing Agents**
> "Are there scaling laws for Minecraft or similar game-playing agents as of 2025? How does performance scale with: model parameters, training compute, environment steps, and dataset size (for imitation)? Reference any papers that study these relationships explicitly, especially comparing small models (<1B) vs. large models (>10B) on the same tasks."

**Query 21: LLM Integration Patterns**
> "What are the proven architectural patterns for integrating LLMs into Minecraft agents? Compare: (1) LLM as high-level planner with separate low-level policy (Voyager, RL-GPT), (2) LLM backbone fine-tuned end-to-end for actions (LARM), (3) LLM for reward design only (Auto MC-Reward), (4) LLM for curriculum/goal generation. Which pattern achieves best performance-to-compute ratio for long-horizon tasks like diamond acquisition?"

---

## Transfer & Deployment

**Query 22: Sim-to-Sim Transfer Validation**
> "Has anyone validated that policies trained in Craftium, MineWorld, or other lightweight Minecraft simulators successfully transfer to full Minecraft (MineRL/MineDojo) without fine-tuning? What is the performance drop? Which simulator features (visual style, physics, action timing) cause the largest transfer gaps?"

**Query 23: Real-Time Inference Requirements**
> "What are the inference latency requirements for a Minecraft agent to play in real-time (20 tick/second)? Which architectures (VPT, LARM, Dreamer) meet this constraint on consumer hardware (RTX 3090/4090)? What optimizations (quantization, distillation, action chunking) are used to achieve real-time play?"

---

## Opinionated Synthesis Queries

**Query 24: If You Were Building This Today**
> "If a researcher in December 2025 wanted to build a Minecraft RL agent that can obtain diamonds with <10M environment steps and deploy to real Minecraft, what would be the recommended: (1) training environment (MineRL vs Craftium vs custom), (2) agent architecture (world model vs model-free vs LLM-hybrid), (3) action space design, (4) observation format? Synthesize opinions from top labs (DeepMind, OpenAI, NVIDIA, academic groups) based on 2024-2025 publications."

**Query 25: Open Problems & Low-Hanging Fruit**
> "What are the remaining open problems in Minecraft RL as of late 2025 that are tractable for a small research team? Which subproblems (e.g., exploration, credit assignment, environment speed, action abstraction) have clear solutions now vs. remain unsolved? Where would a 6-month research effort have the highest impact?"

**Query 26: Environment Engineering Roadmap**
> "Outline a concrete engineering roadmap for building a high-performance Minecraft RL training environment in 2025. Include: choice of base engine (Craftium/Luanti vs custom voxel engine vs neural world model), required modifications for RL compatibility, target throughput benchmarks, action/observation API design, and integration with modern RL libraries (CleanRL, RLlib, custom JAX). Reference any open-source implementations or blueprints from recent papers."

---

## Summary: Round 2 Focus Areas

| Query | Goal |
|-------|------|
| 12-14 | **Environment selection/design** — Pick or build the right simulator |
| 15-16 | **Interface specification** — Exact action/observation formats |
| 17-21 | **Agent architecture** — Which model structure to use |
| 22-23 | **Transfer & deployment** — Validate real-world viability |
| 24-26 | **Opinionated synthesis** — Expert recommendations & roadmap |

---

Shall I save this as `round2.md` in your docs folder, or would you like me to refine any of these queries first?

### Overview Comparison: Craftium vs. MineRL

**Craftium** (released in 2024, with ongoing development into 2025) is an open-source framework for creating customizable, Minecraft-like 3D environments for reinforcement learning (RL) research. It is built on **Luanti** (formerly Minetest, renamed in late 2024), a free voxel game engine written in C++. Craftium wraps Luanti in a Python library compatible with the modern **Gymnasium** (and PettingZoo for multi-agent) API. It emphasizes extensibility via Lua mods, high performance (C++ backend, no Java overhead), and full open-source nature, allowing deep modifications.

**MineRL** is an established benchmark suite (2019 onward) for sample-efficient RL in actual **Minecraft** (Java Edition). It provides pre-defined tasks like ObtainDiamond, with human demonstration datasets, but limited customization due to Minecraft's closed-source nature and Java performance constraints.

Key high-level differences:
- **Performance**: Craftium achieves significantly higher steps per second (>2,000 more than Minecraft-based setups like MineRL/MineDojo) due to Luanti's efficiency.
- **Customization**: Craftium is designed for easy creation of new environments/tasks via Lua scripting and mods; MineRL tasks are fixed or require complex hacks.
- **Openness**: Craftium/Luanti are fully open-source; MineRL requires a licensed Minecraft copy.
- **Focus**: Craftium supports open-ended, procedural, multi-agent, and continual learning scenarios; MineRL focuses on hierarchical long-horizon tasks with baselines from competitions.

### Exact API Differences

- **API Standard**:
  - Craftium: Modern **Gymnasium** (successor to Gym), with PettingZoo for multi-agent.
  - MineRL: Older **Gym** (though compatible with Gymnasium wrappers).

- **Environment Creation**:
  - Craftium: Highly flexible—define worlds via Lua mods, procedural generation, or existing Luanti games (e.g., VoxeLibre for closer Minecraft fidelity).
  - MineRL: Fixed environments; customization is limited.

### Action Space Implementations

Both use complex, human-like actions to mimic player controls (keyboard/mouse simulation).

- **MineRL**:
  - Dictionary-based action space (gym.spaces.Dict).
  - Includes: camera (continuous pitch/yaw), movement (forward/back/left/right/jump/sneak/sprint/attack), crafting/smelt/equip/place, inventory management.
  - Human-like: Continuous camera movements, discrete buttons; supports GUI interactions (e.g., open inventory).

- **Craftium**:
  - Also dictionary-based (Gymnasium-compatible Dict).
  - Core actions: Keyboard commands (e.g., movement, jump, attack) and mouse movements (camera).
  - Explicitly inspired by MineRL—default action space is very similar (key/mouse controls).
  - Provides **ActionWrappers** to simplify/subset the space for specific tasks (e.g., remove unnecessary actions).

Overall, action spaces are near-identical in structure and intent for comparability, with Craftium allowing easier wrapping/restriction.

### Observation Formats

- **MineRL**:
  - Primary: RGB POV image (typically 640x360x3, but resizable to 64x64 in some baselines).
  - Some tasks add vector observations (e.g., inventory counts, compass angle).

- **Craftium**:
  - Explicitly inspired by MineRL: Only RGB frames by default (640x360x3 pixels).
  - Pure pixel-based for vision-focused agents; additional observations (e.g., inventory) can be added via custom Lua mods.

Observations are deliberately aligned for potential sim-to-real transfer experiments.

### Transfer Results from Craftium-Trained Agents to Real Minecraft

No published results found (as of December 2025) on transferring policies from Craftium/Luanti to actual Minecraft (e.g., via MineRL or Malmo).

Reasons:
- Visual and physics differences: Luanti games (even Minecraft clones like VoxeLibre) have distinct textures, block behaviors, crafting recipes, and mob AI compared to vanilla Minecraft.
- No experiments or papers mention sim-to-real transfer in this direction.
- Craftium positions itself as a faster, more customizable alternative rather than a direct simulator for Minecraft transfer.

### Modifications Needed to Match MineRL's Task Definitions (e.g., ObtainDiamond)

Craftium does not include a pre-built **ObtainDiamond** equivalent, as it focuses on framework extensibility rather than fixed benchmarks.

To replicate MineRL's ObtainDiamond (obtain a diamond through mining, crafting tools, etc.):

1. Use or create a Luanti "game" mod closely mimicking Minecraft mechanics (e.g., **VoxeLibre** or **Mineclonia**—community Minecraft clones with similar crafting, ores, tools).
2. Define the task in Lua:
   - Procedural world generation with diamond ore at appropriate depths.
   - Reward function: Sparse rewards for milestones (e.g., obtain wood → craft table → tools → iron → diamond), or dense shaping.
   - Termination: On obtaining diamond or timeout/death.
3. Align action/observation spaces (already similar by default).
4. Optional: Add inventory observations if needed (via Lua extensions).

This is straightforward due to Craftium's design—examples in docs show similar resource-gathering tasks.

### Recent Papers/Experiments (December 2024–December 2025)

- **Craftium core paper** (updated version): "Craftium: Bridging Flexibility and Efficiency for Rich 3D Single- and Multi-Agent Environments" (arXiv:2407.03969v2, June 2025). Accepted at **ICML 2025**. Focuses on performance benchmarks and new environments (procedural tasks, multi-agent).
- **ARM-FM paper** (arXiv:2510.14176, October 2025): Uses Craftium for evaluating automated reward machines in a complex, procedurally generated 3D resource-gathering task (Minecraft-inspired, sparse rewards). Demonstrates scalability in long-horizon settings.

No other major papers in this exact timeframe; Craftium is still emerging post-2024 launch.

For diamond-obtaining policies to transfer effectively from a custom voxel simulator to vanilla Minecraft (e.g., via MineRL/MineDojo wrappers used in benchmarks like DreamerV3/V4), preserve core mechanics that enforce the exact tech tree progression: wood → stone → iron tools → deep mining for diamonds. This requires ~20k-30M steps of RL training, sparse rewards on milestones, and pixel-based observations/actions mimicking human play (20 FPS, mouse/keyboard). Policies fail without fidelity in block interactions, tool efficacy, and spatial navigation. 

arxiv.org +1

Physics (Gravity, Collision)Aspect
Essential (Preserve)
Can Simplify
Gravity
Yes: Standard fall damage, jump height (~1.25 blocks), affects digging down safely to Y=5-12 for diamonds. Policies learn pathfinding around falls. 

arxiv.org

No fluid dynamics (lava as instant-kill zones; water optional). Ignore falling blocks (sand/gravel rare in strip-mining).
Collision
Yes: Voxel grid (1x1x1 blocks), no-phasing, player bounding box. Critical for mining tunnels, avoiding suffocation. 

arxiv.org

Deterministic (no jitter); simplify mob collisions (turn off mobs initially, add as ablation). Tool-dependent break times (e.g., iron pick ~0.3s/stone vs. fist 7s).
Other
Momentum from jumps/sprinting for efficient movement.
No entity physics (e.g., mob knockback); fixed player speed.

Rationale: DreamerV3/4 policies succeed in full Minecraft physics (MineRL 1.11.2) without simplifications, but Craftium (Minetest-based, 38x faster) ablates via discrete actions (10: move/jump/attack/look) and proves 3D gravity/collision suffice for diamond-like collection in procedural dungeons. 

arxiv.org

 No 2024-2025 ablations show transfer drop from physics sim; failures stem from hierarchy/exploration, not physics. 

ijitee.org

Crafting SubsetEssential subset (7 recipes, exact 3x3/2x2 grids):Planks (1 log → 4 planks)
Sticks (2 planks → 4 sticks)
Wooden pickaxe (3 planks top, 2 sticks middle)
Crafting table (4 planks 2x2)
Stone pickaxe (3 cobble top, 2 sticks)
Furnace (8 cobble ring)
Iron pickaxe (3 ingots top, 2 sticks); smelt ore (iron ore + fuel → ingot)

Fuel: Charcoal (log in furnace) or coal.Can simplify: No advanced recipes (e.g., no iron door/boat). UI as abstract state (vector for inventory) + pixel cues. Deterministic crafting (no shapeless variants). Offhand auto-equip.Rationale: Hierarchical RL ablations confirm these subtasks (gather wood/cobble/iron ore, craft table/pickaxe, smelt) as minimal composable primitives; policies BC from MineRL data then RL-fine-tune. 

ijitee.org

 DreamerV4 world models accurately predict these UI/block interactions for offline diamond success (0.7%). 

arxiv.org

World Generation FeaturesAspect
Essential (Preserve)
Can Simplify
Structure
Layered: Surface grass/trees (wood), stone (Y=0-64), ores (iron Y=0-64, diamond Y=5-12, coal surface-deep). Caves/lava pools (Y<10) for risk/reward. Infinite procedural (Perlin noise).
Flat world (256x256 spawn), uniform ore density (e.g., 1/1000 blocks diamond). No biomes/mobs/structures (add post-iron).
Scale
Deep mining (strip ~2x1x2 tunnels at target Y).
Fixed seed for reproducibility; soft-reset inventories/world chunks.

Rationale: MineRL worlds are procedural infinite 3D; policies explore ~17 in-game days for diamonds. Craftium generates procedural dungeons with diamond goals, ablates difficulty (room/monster count) showing transfer via fine-tuning—implies layered voxels + ores transfer without full biomes. 

arxiv.org

Visual Fidelity LevelLevel
Essential?
Details
Low (Recommended)
Yes
64x64 RGB/grayscale stacked (4 frames), distinct block textures/colors (wood brown, stone gray, ores shiny). CNN policies recognize via patches. 

arxiv.org +1

Medium
Optional
128x128, basic lighting/shadows.
High
No
Full MC textures/RT—wasteful for RL (slow sim).

Rationale: MineRL/Dreamer use 64x64 pixels; Craftium 64x64 grayscale succeeds on diamond nav. World models (DreamerV4) compress to 256 tokens accurately. 

arxiv.org

Overall Design Recommendation: Build in fast engine (e.g., JAX/C++/Lua like Craftium/Minetest: 2k+ steps/sec). Start minimal (flat layered world, no mobs), ablate adding caves/mobs. Test transfer: Train policy to 10-20% diamond rate, eval in MineRL (expect 50-80% drop if visuals/physics mismatch). No 2024-2025 papers ablate full transfer, but Craftium validates minimal 3D voxels for analogous tasks. 

arxiv.org

As of late 2025, there are no widely available **GPU-accelerated voxel world simulators** specifically designed for reinforcement learning (RL) using JAX, CUDA, or similar frameworks that fully replicate Minecraft-like open-world voxel environments (e.g., infinite terrain, complex crafting, fluid dynamics, and entity interactions). Existing GPU-accelerated voxel tools, such as **Voxcraft-sim** (CUDA-based voxel physics) or NVIDIA's **GVDB** (sparse voxel compute and rendering), focus on specialized physics or rendering but lack RL integration and Minecraft-scale world mechanics.

Neural world models like **MineWorld** (an open-source Microsoft project from 2025) offer fast, controllable simulation of Minecraft-like environments via learned neural networks rather than explicit physics. These are GPU-accelerated (often via JAX or similar) and enable real-time interaction, but they are **neural simulators** (imagined rollouts from a trained model) rather than traditional physics-based ones. They excel in speed for RL training in imagined spaces but may lack perfect fidelity to real Minecraft physics.

No direct equivalents to **Isaac Gym** (NVIDIA's GPU-accelerated rigid-body simulator for RL) exist for voxels. Isaac Gym/Isaac Lab environments are rigid-body focused (e.g., robots, manipulation) with no built-in voxel worlds or Minecraft ports.

### Performance Comparisons
Performance comparisons (steps-per-second, or SPS) vary by hardware, environment complexity, and parallelism:

- **CPU-based Minecraft-like setups**:
  - **MineRL** (original Java-based Minecraft): Typically low, often **<100 SPS** single-environment due to Java overhead and single-threaded bottlenecks.
  - **Craftium** (2025 framework wrapping the open-source Luanti/Minetest voxel engine): Significantly faster than Minecraft-based alternatives, achieving **>2,000 additional SPS** (e.g., thousands SPS total on modern CPUs) via optimized integration and modding.

- **Neural/GPU alternatives**:
  - **MineWorld** and similar neural simulators (e.g., compared to closed-source Oasis): Much faster for rollouts (real-time or better on GPU), but noted as slower in some human-in-the-loop tests compared to top models like Dreamer 4's internal simulator. They enable massive parallel RL training on GPUs (millions of steps/sec in batched imagination).
  - No direct Isaac Gym-style voxel envs exist for comparison, but Isaac Gym rigid-body envs reach **hundreds of thousands to millions SPS** on a single GPU via massive parallelism.

Overall, CPU voxel sims like Craftium/MineRL lag far behind GPU rigid-body sims (e.g., Brax/MJX or Isaac Gym) in throughput, often by 1-3 orders of magnitude when scaling to thousands of parallel environments.

### Porting Minecraft-Like Mechanics to a Brax/MJX-Style Simulator
**Brax** (Google's JAX-based physics engine) and **MJX** (MuJoCo's JAX reimplementation, integrated into newer Brax versions) are fully differentiable, GPU/TPU-accelerated rigid-body simulators ideal for RL. Porting full Minecraft mechanics would be highly challenging and likely incomplete due to fundamental differences:

- **Core limitations** — Minecraft is voxel-based (discrete grid of blocks with destruction/placement, gravity-affected falling sand, fluids, redstone logic, crafting trees). Brax/MJX excel at continuous rigid-body dynamics (joints, actuators, contacts) but lack native support for:
  - Discrete voxel grids.
  - Massive destruction/construction.
  - Fluid simulation.
  - Procedural infinite terrain.
  - Complex state machines (e.g., crafting recipes).

- **What would it take?**
  1. **Model the world as composed rigid bodies**: Represent each block as a small rigid body or compound object in MJX (using MJCF/XML definitions). This could handle basic placement/removal and physics (e.g., stacking, falling).
  2. **Custom collision and dynamics**: Implement voxel-like rules via custom forces, constraints, or contact callbacks in JAX (differentiable for gradient-based RL).
  3. **Grid management**: Use sparse data structures (e.g., hash maps or octrees in JAX arrays) for efficient storage/updates of large worlds; dynamically add/remove bodies.
  4. **Parallelism**: Batch thousands of environments on GPU, leveraging Brax's strengths for fast RL training.
  5. **Observations/actions**: Define discrete actions (e.g., mine/place block in direction) and voxel-based observations (e.g., local grid scans).
  6. **Effort level**: Significant custom engineering (months to years for a research prototype). Partial ports (e.g., simple block-stacking worlds) are feasible, similar to existing Brax grasping envs but voxel-discretized. Full fidelity (infinite worlds, fluids, entities) may require hybrid approaches or extensions beyond current Brax/MJX capabilities.

In summary, while GPU-accelerated differentiable sims like Brax/MJX dominate rigid-body RL, voxel worlds remain mostly CPU-bound or neural-approximated. Advances in neural simulators (e.g., MineWorld) provide the closest high-speed alternative for Minecraft-like RL as of late 2025.

### VPT (OpenAI Video PreTraining, 2022) Action Space
VPT uses the **native human-like action space** from the MineRL environment (specifically compatible with MineRL v1.0+), designed to mimic real human controls in Minecraft:
- **Keyboard actions**: Discrete buttons including forward, back, left, right, jump, sprint, sneak, attack, use, craft, equip, place, etc. (binary or categorical, e.g., attack is on/off).
- **Mouse actions (camera)**: Continuous delta pitch and yaw (mouse movement), not discretized into bins — full analog control for precise looking.
- **Overall structure**: Combination of discrete keyboard actions and continuous camera movements, executed at ~20 Hz.
- No built-in macro-actions or high-level abstractions like auto-craft or auto-attack-until-break; all actions are low-level and atomic (e.g., crafting requires manually opening the inventory and selecting recipes step-by-step).
This is the most challenging space due to its high dimensionality and precision requirements, making pure RL from scratch nearly impossible without massive data.

### LARM (ICML 2025) Action Space
LARM (Large Auto-Regressive Model for Long-Horizon Embodied Intelligence) uses a **tokenized low-level action space**, treating actions autoregressively like language tokens:
- The model directly predicts discrete action tokens (one per step) using a lightweight LLM backbone (<5B parameters) with separate actor and critic heads.
- Actions are atomic/low-level, similar to standard Minecraft RL setups (keyboard + mouse controls).
- **Camera**: Likely discretized or tokenized (common in autoregressive models for efficiency), but exact bins not specified in available sources.
- No explicit macro-actions mentioned (e.g., no auto-craft or navigation macros); focus is on end-to-end prediction of primitive actions.
- Differs from VPT: More efficient inference (single-token action prediction) but still operates in a large primitive action space, relying on a giant LLM referee for sparse reward shaping rather than action abstraction.

### RL-GPT (NeurIPS 2024) Action Space
RL-GPT integrates LLM-generated **code-as-policy** with a low-level RL policy, creating a hybrid hierarchical action space:
- **Base space**: Low-level primitive actions (similar to VPT/MineRL: discrete keyboard + continuous/discretized camera).
- **High-level abstractions**: GPT-4 iteratively generates Python code for macro-actions (e.g., navigation routines, precise movement, or subtask controllers like "harvest tree" or "mine block").
  - These coded macros are inserted into the RL action space as additional options or executed sequentially when conditions match.
  - Examples include coded motion planners for low-level control (e.g., breaking trees efficiently).
- **Camera**: Handled within low-level RL or coded macros (likely continuous or finely controlled via code).
- Explicit macros for crafting/navigation via code generation; includes "auto-craft" variants in baselines (e.g., MINEAGENT with AUTOCRAFT).
- Differs from VPT: Significantly augments the primitive space with dynamic, LLM-optimized high-level coded actions, enabling faster diamond acquisition (within a single day on one GPU).

### Dreamer 4 (2025, Google DeepMind) Action Space
Dreamer 4 uses the **standard low-level Minecraft action space** from the OpenAI VPT dataset/MineRL:
- Full keyboard and mouse controls: Discrete keyboard inputs + continuous mouse delta (pitch/yaw).
- No discretization of camera mentioned; continuous for precise control.
- No macro-actions or high-level abstractions; purely primitive actions (over 20,000 steps needed for diamond mining).
- Training occurs entirely in imagination (world model), using offline data from human gameplay videos.
- Differs from VPT: Same space, but Dreamer 4 achieves diamond mining purely offline (no online interaction), while VPT relies on behavioral cloning from labeled videos.

### Key Differences from VPT's Original Action Space
- VPT pioneered the full human-like primitive space (continuous camera + discrete keys), making it harder than earlier MineRL competitions' simplified/discretized spaces.
- LARM and Dreamer 4 stick closely to this primitive space (low-level, no macros).
- RL-GPT deviates most by dynamically extending it with LLM-coded macros, reducing effective dimensionality for long-horizon tasks.

### Abstractions Contributing Most to Sample Efficiency
Sample efficiency in Minecraft RL is critically limited by the vast primitive action space, long horizons (~20k+ steps for diamonds), and sparse rewards. High-level abstractions address this by reducing exploration complexity and enabling temporal extension:

| Abstraction Type                  | Examples in Papers/Methods                  | Impact on Sample Efficiency                                                                 |
|-----------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------|
| **Coded Macros / Code-as-Policy** | RL-GPT (GPT-generated Python for navigation, harvesting, precise control) | Highest impact: Automates repetitive subtasks, allows RL to focus on high-level decisions. Enables diamond mining in dramatically fewer environment steps than primitive-only methods (e.g., vs. DreamerV3/VPT finetuning). |
| **Auto-Craft / Recipe Macros**    | Mentioned in RL-GPT baselines (e.g., MINEAGENT + AUTOCRAFT); hierarchical crafting skills in Plan4MC | Bypasses manual inventory management; critical for tech-tree progression without millions of failed attempts. |
| **Auto-Attack-Until-Break / Mining Loops** | Common in coded macros (e.g., "mine block until destroyed") | Reduces redundant low-level attack repetitions; hugely improves efficiency in resource gathering (trees, stone, ores). |
| **Hierarchical Skills / Options** | Plan4MC (RL-learned basic skills + LLM-planned sequences) | Good, but less automatic than code generation; still far better than primitives. |

**Most contributory**: LLM-generated coded macros (as in RL-GPT) and auto-attack/auto-craft loops, as they provide flexible, composable temporal abstractions without hand-coding, drastically cutting required samples compared to pure primitive RL (e.g., DreamerV3 needed massive online interaction; RL-GPT achieves similar milestones orders of magnitude faster). These turn intractable exploration into manageable hierarchical planning.

**For Minecraft RL agents targeting diamond acquisition (e.g., ObtainDiamond or crafting diamond tools/equipment), the optimal observation stack in 2025 balances rich visual context with structured state information to handle long-horizon planning, sparse rewards, and partial observability.**

### Key Benchmarks and References
- **MineRL/MineDojo Diamond Tasks**: Standard for evaluation—requires ~20k+ actions from scratch (punch tree → wood → planks → sticks → crafting table → stone tools → iron → furnace → iron tools → diamond mining/crafting).
- **LARM (arXiv 2405.17424v2, Feb 2025)**: First to craft *enchanted diamond sword* (16/30 success; diamond sword 28/30). 42 GPU-hours on RTX 4090.
- **DreamerV4 (Nature, Apr 2025)**: 100% diamond success in 100M steps (10 seeds); first from-scratch pixels + vectors (9 days A100).
- **Optimus-2/3 (CVPR/ICLR 2025)**: Diamond group SR 13-15%; video-pretrained on 30M+ GOA pairs (2-3 days, 8x L40).

No 2025 paper directly ablates *all* your options head-to-head, but SOTA designs and partial ablations (e.g., vector vs. pixel-only, history/state omission) converge on **pixels + structured state** (inventory vectors or text-encoded equivalents). Raw pixels alone fail long-horizon due to aliasing/inventory blindness; semantic seg adds compute without proportional gains in these works.

### Comparison Table
| Stack | Description | Pros | Cons | Sample Efficiency | Compute Tradeoff | Diamond SR (SOTA Ex.) | Ref |
|-------|-------------|------|------|-------------------|------------------|-----------------------|-----|
| **1. Raw Pixels Only** | 64-640×360 RGB POV (e.g., 64×64 in Dreamer). | Simple; end-to-end vision. | No inventory/health awareness; poor long-horizon (forgets tools). | Low (100M+ steps needed). | Low (but scales poorly). | <1% without vectors. | DreamerV4 abl. implies drop. |
| **2. Pixels + Inventory Vector** | Pixels + counts vector (e.g., 9-slot inv, equipped one-hot). | Tracks progress (e.g., "have iron?"); boosts hierarchy. | No direction/health; aliasing in caves. | High (10-100x fewer steps vs. 1). | Good (vectors cheap). | 100% diamonds (100M steps). | DreamerV4 (vectors critical). |
| **3. Pixels + Inv + Compass/Goal Dir.** | + yaw/pitch vector or goal embedding (text/one-hot). | Navigation/planning (e.g., "face ore"); goal-cond. | Minor overhead. | Highest (fastest convergence). | Excellent (vectors << pixels). | 13-15% diamond tools. | Optimus GOAP (goal-text + hist.); MineRL std. |
| **4. Pixels + Semantic Seg** | Pixels + per-pixel block IDs/masks (e.g., MineCLIP seg). | Precise object detection (ores/mobs). | High preprocess compute; overkill for diamonds. | Medium (helps exploration). | Poor (seg model + memory). | Not SOTA; older MineRL wins used lightly. | No 2025 adoption (ViT/CLIP better). |
| **5. Text State Desc.** | LLM-text (inv, health, blocks, history); ± pixels. | Interpretable; multimodal reasoning. | Token limits; encoding lossy. | Medium-high (if pretrained). | Medium (LLM inference). | Ench. diamond sword (16/30; 42h). | LARM (text + pixels). |

### Best Sample Efficiency vs. Compute Tradeoff: **(3) Pixels + Inventory + Compass/Goal Direction**
- **Why?** 
  - **Efficiency**: Vectors (inv/health/dir) provide "free" grounding—DreamerV4 hits 100% diamonds in 100M steps from scratch; Optimus adds goal for 15% on crafts (vs. 1% baselines). LARM's text-equiv. (inv/history) achieves *enchanted* diamonds in 42h single-GPU online RL.
  - **Compute**: ~1-10 GPU-days; vectors add <1% FLOPs vs. pixels. Seg (4) bloats; text (5) needs heavy pretrain/encoding (LARM: CLIP+3B LLM).
  - **Ablations Confirm**: Optimus-2: omitting obs-action history drops 36-42%. Dreamer: vectors enable from-scratch success.
- **Implementation Tip**: 320×240 pixels (balance res/speed); inv as 20D vector (9 slots + equipped + extras); goal as one-hot or CLIP-embedded dir/text. PPO/Dreamer/AR policies excel.

**Runner-up**: (2) if no goals (pure RL); upgrade to (5) + pixels for open-world (LARM-style). Avoid 1/4 for 2025 SOTA.

**LARM** (Large Auto-Regressive Model for Long-Horizon Embodied Intelligence) is a reinforcement learning agent architecture designed for open-world tasks in Minecraft, presented in a paper submitted to **ICML 2025**. It uses a lightweight multimodal LLM backbone to enable efficient, generalizable decision-making in long-horizon, sparse-reward environments.

### Integration of TinyLLaVA-3.1B with CLIP Vision Encoder
LARM's core is the decoder of **TinyLLaVA-3.1B**, a 3.1B-parameter multimodal LLM. The vision encoder comes from **CLIP** (specifically a variant like CLIP-Large). Multi-view images (from the agent's perspective) and text inputs (task description, agent state like position/inventory, and environment feedback) are encoded into tokens using CLIP. These visual and text tokens, plus a learnable "skill token," are fed into the frozen TinyLLaVA decoders (with trainable LoRA adapters) for cross-modal feature interaction. The processed skill token then feeds into separate action and value heads.

### Structure of Action and Value Heads
A single shared LARM model parametrizes both the actor (policy) and critic (value function) via two distinct trainable prediction heads attached to the skill token output:
- **Action Head** — Predicts discrete high-level skills (e.g., "chop tree," "craft stick") by matching the skill token's features to predefined skill descriptions/embeddings.
- **Value Head (Critic)** — Estimates the state value V(s_t), i.e., the expected cumulative discounted return from the current state.

The decoders remain frozen during RL fine-tuning, with only LoRA modules and heads updated for efficiency.

### How Referee RL Differs from Standard PPO
Referee RL builds on **PPO** but addresses reward vanishing in ultra-sparse, long-horizon settings (where environment rewards are delayed/extremely rare, leading to near-zero advantages and ineffective updates). The key innovation is an external "referee" (a large LLM like GPT-4) that provides dense auxiliary rewards (b_r_t) for each action, evaluating its immediate contribution to the goal (categorized as positive/negative impact or correct/incorrect). The total reward becomes r_t + λ * b_r_t (environment + referee). This supplies shaped, informative feedback early in training, preventing gradient vanishing while standard PPO relies solely on sparse environment rewards.

### Training Loop
Training is **online** (real-time exploration in the Minecraft environment):
1. The agent explores for a fixed horizon T, collecting trajectories with states, observations, actions, environment rewards r_t, and referee rewards b_r_t.
2. Batches are sampled from the rollout buffer.
3. Multiple PPO update iterations (N_π) optimize the shared policy and value via clipped surrogate objectives, GAE advantages, and value loss.

No specific batch sizes or learning rates are detailed in the paper (it follows standard PPO implementations). Pre-training uses Minecraft webpage data for domain adaptation. Fine-tuning for the hardest task (enchanted diamond tool) takes ~**42 hours** on a **single RTX 4090** GPU.

### Performance: 93% Diamond Sword Success with 42 Hours on RTX 4090
LARM achieves strong results across benchmarks:
- In MineDojo tasks → High success rates, e.g., **93%** for harvesting a stick (a subtask in the tech tree).
- In Mineflayer (harder evaluation) → **93.3%** (28/30) success for harvesting a regular diamond sword; **53.3%** (16/30) for an **enchanted** diamond sword/tool—the first method to reliably obtain enchanted diamond equipment, far surpassing priors like Voyager or STEVE.

Efficiency stems from the lightweight 3.1B model (deployable/inference-friendly), Referee RL's dense guidance enabling faster convergence in sparse rewards, and online exploration on a single consumer GPU (42 hours for the enchanted diamond task, vs. much longer/more resource-intensive training in prior works).

### Overview of Approaches

**Autoregressive action prediction** models, such as **Video PreTraining (VPT)** and **Large Auto-Regressive Model (LARM)**, generate actions sequentially by predicting the next action (or skill) conditioned on previous observations and actions. VPT (2022) uses behavioral cloning on large-scale human Minecraft gameplay videos to learn a foundation model, enabling tasks like tree chopping and tool crafting. LARM (2024) extends this to long-horizon planning, achieving feats like crafting enchanted diamond equipment by modeling high-level skills autoregressively with visual and textual inputs.

**Diffusion-based policies**, exemplified by approaches like **DIAMOND** (DIffusion As a Model Of eNvironment Dreams, 2024), treat action generation (or world modeling) as a denoising process. DIAMOND uses diffusion for world modeling in environments like Atari and CS:GO, training RL agents in imagined rollouts for improved visual fidelity and stability. While not directly a policy in Minecraft, diffusion has been applied to Minecraft world models (e.g., OASIS, a 2024 diffusion-based simulator).

### Sample Efficiency

Sample efficiency refers to how effectively a method learns from limited data or environment interactions.

- Autoregressive models (like VPT and LARM) generally excel in data-constrained settings. Studies show autoregressive approaches are more sample-efficient on smaller datasets, requiring less capacity and compute to reach strong performance.
- Diffusion models perform better in highly data-limited or overparameterized regimes but often need more training epochs and larger models to escape underfitting. In RL contexts (e.g., DIAMOND), diffusion improves sample efficiency via better world modeling for imagination-based training, but direct policy comparisons favor autoregressive for low-data scenarios.

Overall, **autoregressive methods are typically more sample-efficient** in Minecraft-like imitation learning from demonstrations, where data is often scarce or noisy.

### Handling Hybrid Discrete + Continuous Action Spaces in Minecraft

Minecraft actions are hybrid: discrete (e.g., jump, attack, craft) and continuous (e.g., camera mouse movements).

- Autoregressive models handle this by discretizing continuous actions (e.g., binning camera angles into tokens, as in VPT and MineWorld, a 2025 autoregressive world model). This allows seamless sequential prediction but introduces quantization error.
- Diffusion models naturally excel at continuous actions (via Gaussian noise processes) and can extend to discrete via masking or hybrid formulations. Recent works (e.g., CANDI, hybrid discrete-continuous diffusion) address mixed spaces effectively. In Minecraft world models, diffusion (OASIS) struggles with action precision compared to autoregressive (MineWorld achieves higher F1 scores on discrete actions and real-time FPS).

**Diffusion-based approaches have a theoretical edge for hybrid spaces** due to better modeling of continuous distributions and multimodality, but in practice for Minecraft (2025), autoregressive with discretization performs strongly and enables real-time interaction.

### Head-to-Head Comparisons on Similar Benchmarks

No direct head-to-head comparisons exist between pure autoregressive action policies (VPT/LARM) and diffusion policies on identical Minecraft benchmarks as of late 2025.

- Indirect evidence from world modeling: MineWorld (autoregressive) outperforms OASIS (diffusion) in generation speed, action accuracy, and controllability on Minecraft datasets.
- Broader RL/robotics surveys show diffusion policies shining in continuous control with multimodality, while autoregressive dominates long-horizon discrete planning (e.g., LARM's enchanted diamond success vs. earlier methods).
- Minecraft-specific agents (e.g., Voyager, STEVE) lean autoregressive or hierarchical, with diffusion more common in simulation/world generation.

In summary, autoregressive remains dominant for Minecraft agents due to proven long-horizon success and efficiency, while diffusion shows promise for richer action modeling but lags in direct applications and comparisons. Future hybrids may combine strengths.

### Comparison of World Model Approaches for Minecraft Agents (Late 2025)

Minecraft poses unique challenges for world models: high-dimensional visual inputs, **partial observability** (first-person view, limited FOV), **stochasticity** (random mob behavior, procedural generation, resource drops), vast open worlds, and extremely **long horizons** (e.g., obtaining diamonds requires 10,000–20,000+ actions chaining subtasks like mining trees, crafting tools, and digging deep).

Below is a comparison of the four options based on the requested criteria. This draws from established methods (DreamerV3) and recent advancements (2024–2025 papers on Dreamer 4, diffusion models like DIAMOND/GameNGen, and tokenized approaches like Genie series/IRIS/MineWorld).

| Criterion                  | (1) RSSM-style Latent Dynamics (DreamerV3) | (2) Block-Causal Transformers (Dreamer 4) | (3) Diffusion-Based Frame Prediction (DIAMOND/GameNGen) | (4) Discrete Tokenized World Models (Genie 3 / similar) |
|----------------------------|--------------------------------------------|----------------------------------------------------|----------------------------------------------------------|---------------------------------------------------------|
| **Training Stability**    | High: Proven robust with symlog rewards, KL balancing, and transformations. Stable across domains without much tuning. | High: Uses "shortcut forcing" (advanced objective) for faster, more stable training than standard transformers or diffusion. Handles large-scale offline data well. | Medium: Diffusion models can be unstable (mode collapse, noise schedule sensitivity); DIAMOND requires careful design (e.g., adaptive noise) for RL use. GameNGen needed two-phase training (RL data collection first). | High: Autoregressive transformers on discrete tokens (VQ-VAE + Transformer) are stable and scalable, similar to LLMs. Genie series builds on video tokenizers with strong consistency. |
| **Rollout Fidelity Over Long Horizons** | Medium: Good for thousands of steps in DreamerV3 (diamond task online), but compounding errors in latent space limit very long horizons in complex 3D. | High: Excellent; accurate object interactions and mechanics in Minecraft. Supports 20,000+ action sequences for diamond mining purely offline, with real-time consistency. | Low-Medium: High visual fidelity short-term (GameNGen real-time DOOM/Minecraft-like), but autoregressive diffusion suffers compounding errors/blurring over long rollouts without distillation or fixes. | High: Strong long-term consistency (minutes of coherent simulation in Genie 3); discrete tokens reduce drift, enabling reliable long-horizon imagination. |
| **Compute Requirements**  | Low-Medium: Efficient RSSM (RNN-based); DreamerV3 trains on single GPUs for many tasks, including Minecraft diamonds. | Medium: Block-causal transformer with shortcut forcing enables real-time inference on single GPU (e.g., H100 for 20 FPS Minecraft). Scalable but larger models than RSSM. | High: Iterative denoising is slow/expensive for rollouts; GameNGen augmented Stable Diffusion but needed TPUs for training; real-time requires optimizations. | Medium-High: Autoregressive token prediction is efficient with KV caching; Genie 3 real-time at 24 FPS/720p, but large tokenizers/transformers need substantial training compute. |
| **Handling Partial Observability & Stochasticity** | High: Recurrent stochastic latents explicitly model uncertainty/partial obs; handles Minecraft's POMDP nature well (e.g., exploring unknown worlds). | High: Causal tokenizer + dynamics model stochastic latents; excels in partial obs from pixels, models Minecraft stochasticity (e.g., drops, mobs) accurately. | Medium: Pixel-level generation captures details but struggles with latent uncertainty; stochastic via noise, but less explicit POMDP handling than latent state-space models. | High: Discrete tokens + autoregressive prediction model stochastic transitions; Genie 3 handles diverse procedural worlds and interactions, with emergent consistency under partial views. |

### Recommendation
For building a **Minecraft world model agent in late 2025**, the strongest option is **(2) block-causal transformers (Dreamer 4)**. It directly achieves state-of-the-art in Minecraft (first to obtain diamonds purely offline, outperforming priors like VPT with 100× less data) while addressing all criteria effectively: stable training via shortcut forcing, high-fidelity long-horizon rollouts with accurate physics/objects, efficient real-time inference on one GPU, and strong handling of partial observability/stochasticity through scalable latent dynamics.

- If compute is very limited or you prefer proven online training: Go with (1) DreamerV3 (still the baseline for from-scratch diamonds).
- Diffusion (3) excels at visual detail but lags in long-horizon reliability and efficiency for agent planning.
- Discrete tokenized (4) is promising for open-ended generation (e.g., Genie 3's diverse worlds) but less proven specifically for Minecraft agent control compared to Dreamer 4.

Dreamer 4 represents the current frontier for practical, high-performance Minecraft agents.

**Yes, scaling laws have been empirically identified for Minecraft agents and similar game-playing environments (e.g., procedurally generated, open-ended games like NetHack, Procgen, and Crafter-inspired benchmarks), particularly in imitation learning (IL/behavior cloning from expert data), reinforcement learning (RL), world modeling, and multi-agent settings.**

These laws generally follow **power-law relationships** (similar to language modeling), where performance metrics like mean return \(R\), cross-entropy loss \(L\), or task success rate improve predictably with increases in model parameters \(N\), training compute \(C\) (often \(\propto N \times D\) or \(N \times E\)), dataset size \(D\) (for IL), or environment steps/interactions \(E\) (for RL). Optimal allocations (e.g., larger \(N\) at higher \(C\)) shift predictably, enabling extrapolations.

Key papers explicitly studying these (with Minecraft or analogs):

| Paper | Environment(s) | Method | Key Scaling Relationships | Model Sizes & Comparisons |
|-------|----------------|--------|---------------------------|---------------------------|
| **TeamCraft** (arXiv:2412.05255, ICLR 2025 submission) | Minecraft (multi-agent collaboration: build structures, coordinate via VL prompts) | IL from 55k expert demos | **Data scaling**: Subgoal/task success ↑ with training data (diminishing returns; decentralized coordination plateaus sooner). No formal power law, but clear trend: more data enables complex coordination/generalization to novel goals/scenes/agents. | 7B vs. 13B: 7B approaches 13B perf. with more data (esp. generalization); model scaling alone insufficient. |
| **JARVIS-VLA** (arXiv:2503.16365) | Minecraft (1k+ atomic tasks: craft/smelt/cook/mine/kill) | VLA (post-train VLMs on non-trajectory VL data + IL) | **Dataset scaling (post-training)**: Task success ↑ linearly with VL data scale (knowledge/grounding/alignment); lower loss correlates with +success (e.g., spatial grounding boosts crafting). Trajectories: success ↑ but needs eval loss <0.3. | 7B (Qwen2-VL/LLaVA) >> 248M baselines (VPT/STEVE); 7B hits 80-97% success vs. <50% small. |
| **Scaling Laws for IL in Single-Agent Games** (arXiv:2307.09423) | NetHack (roguelike: exploration/combat/items; Minecraft-like open-ended) | BC (IL); IMPALA (RL) | **Compute**: \(R_{opt} \approx (a C^\gamma + b)^{-1}\) (\(\gamma \approx 0.49\) BC NetHack); \(L_{opt} \approx a C^{-\gamma} + b\). **Optimal**: \(N_{opt} \approx a C^{0.6}\), \(D_{opt} \approx a C^{0.4}\) (BC); verified predictions. RL similar (\(\alpha=0.43\), \(\beta=0.56\)). | 200k-200M params: Large >> small at iso-\(C\) (1.7x SOTA return: 7784 vs. 4504); optima shift to larger \(N\). |
| **Scaling Laws for Single-Agent RL** (arXiv:2301.13442) | Procgen (16 procedural 2D games: generalization/exploration; Minecraft-analog) | PPO (RL) | **Intrinsic perf.**: \(I^{-\beta} = (N_c/N)^{\alpha_N} + (E_c/E)^{\alpha_E}\) (\(\alpha_N, \alpha_E \approx 0.3-0.8\)); **Optimal**: \(N \propto C^{0.4-0.8}\). Horizon ↑ shifts optima (affine in samples). | Small CNNs (1k-10M?): Large more \(C\)-efficient; smaller optima than LM scaling (long horizons). |
| **Scaling Laws for Pre-Training Agents/World Models** (OpenReview D0XpSucS3l) | Video games + robotics | IL (behavior); world modeling | Power laws in params/\(D\)/\(C\) (like LMs); coeffs. vary by tokenizer/task/arch. (optimal sizing implications). | Not specified; focuses trends. |
| **Towards a Science of Scaling Agent Systems** (arXiv:2512.08296) | PlanCraft (Minecraft-like crafting/planning); others | Multi-agent (LLM teams) | **Model capability**: Perf. ↑ with intelligence index (accelerating returns). Coord. scaling: logistic (not power); task-contingent (e.g., +81% parallel, -70% sequential). Predictive \(R^2=0.51\). | "Nano" (~<1B?) to GPT-5/Gemini-2.5 (>10B equiv.); large >> small, but saturation >45% single-agent baseline. |
| **JOWA** (arXiv:2410.00564) | Atari (55 games) | Offline MBRL | Perf. ↑ reliably with params (steepest curve vs. baselines). | 40M/70M/150M: 150M = 79% human (IQM); scales TD-learning stably. |

**Performance Scaling Summary**:
- **Model Parameters (\(N\))**: ↑\(N\) yields power-law gains (e.g., \(\alpha \approx 0.5-0.6\)); large models (100M-13B) > small (<200M) at iso-\(C\) (1.7x return NetHack; 7B ≈13B w/ data TeamCraft). Optima: \(N_{opt} \propto C^{0.4-0.6}\); long-horizon tasks favor larger \(N\).
- **Training Compute (\(C\))**: Inverse power law for return (\(R \propto C^\gamma\), \(\gamma \approx 0.3-0.5\)); predictable optima/extrapolations (e.g., >10^{17} FLOPs for human-level NetHack).
- **Environment Steps (\(E\))**: Power law (\(\alpha_E \approx 0.3-0.8\)); sample-efficient transfer in large models (e.g., 5k steps/game JOWA).
- **Dataset Size (\(D\), IL)**: Power law (\(D_{opt} \propto C^{0.4}\)); linear ↑success (JARVIS); diminishing in multi-agent (TeamCraft). Plateau ~300B tokens synth. data analogs.

**Small (<1B) vs. Large (>10B) on Same Tasks**: No direct Minecraft comparison (most Minecraft uses 7B+ LLMs; classics <200M). Trends: Large excel in coord./generalization (PlanCraft: GPT-5 >> nano); small viable/scale well in IL/RL classics (NetHack/Procgen: 200M >> 200k). Data often bridges gap (TeamCraft 7B→13B).

Overall, scaling holds but is **task/horizon-contingent** (diminishing in multi-agent/sequential; data-critical for VLAs). More work needed for >10B on Minecraft (e.g., extrapolate NetHack: ~67B for expert).

**Proven Architectural Patterns for Integrating LLMs into Minecraft Agents**

Minecraft serves as a challenging benchmark for LLM-integrated agents due to its open-world nature, partial observability, sparse rewards, and long-horizon tasks like **diamond acquisition** (ObtainDiamond: gather wood → craft tools → mine progressively harder ores → obtain diamond ore/tool, requiring 10k–50k+ steps).

The four specified patterns are well-established in recent works (2023–2025), often evaluated on **MineDojo** (multi-task benchmark) or **Mineflayer** (achievement-based). Here's a breakdown:

1. **LLM as High-Level Planner with Separate Low-Level Policy** (e.g., Voyager, RL-GPT):  
   LLM (e.g., GPT-4) decomposes tasks into subgoals, generates code/plans, or selects modes; low-level is code execution (JS/Python via APIs like Mineflayer) or a fine-tuned RL policy (e.g., PPO). Includes skill libraries for reuse and feedback loops (errors, verification). Voyager adds auto-curriculum (LLM-proposed tasks via novelty/unvisited items).  
   *Strengths*: Zero/few-shot via prompting; compositional skills mitigate hallucination/long context. *Weaknesses*: Slow inference (multiple LLM calls/iter); code errors.

2. **LLM Backbone Fine-Tuned End-to-End for Actions** (e.g., LARM):  
   Small LLM (<5B params, e.g., Qwen) fine-tuned autoregressively to output discrete actions directly (no text/code). Trained via RL (PPO) with LLM "referee" (GPT-4) for dense auxiliary rewards, solving sparse reward vanishing in long horizons.  
   *Strengths*: Fast inference (~0.58s/step on RTX4090); handles multimodal obs (RGB/status). *Weaknesses*: Requires fine-tuning data/compute.

3. **LLM for Reward Design Only** (e.g., Auto MC-Reward):  
   LLM (GPT-4) iteratively designs/verifies dense reward functions (Reward Designer/Critic) and analyzes failures (Trajectory Analyzer) for a separate RL agent (PPO on pre-trained IL model). Rewards code as Python fns using obs (RGB, blocks, inventory).  
   *Strengths*: Boosts any RL baseline w/o architecture changes. *Weaknesses*: Dependent on base RL quality; not standalone agent.

4. **LLM for Curriculum/Goal Generation**:  
   LLM generates task sequences, subgoals, or adaptive curricula (e.g., Voyager/DEPS/Optimus-2: MLLM decomposes "get diamond" into "chop tree → craft table → ..."; auto-advances via progress). Often combined w/ #1 (hierarchical). Pure examples scarce; e.g., Plan4MC uses LLM planner for RL. *Strengths*: Handles open-ended exploration. *Weaknesses*: Suboptimal without low-level execution.

**Comparison Table** (focus: diamond acquisition/long-horizon perf; data from papers; ~30 trials unless noted)

| Pattern | Example | Diamond Success Rate | Key Metrics (Samples/Steps/Iter) | Compute (Train + Inf) | Notes vs Baselines |
|---------|---------|-----------------------|---------------------------------|-----------------------|-----------------------------------------------------|
| **1. High-Level LLM + Low-Level** | Voyager | ~33% (diamond tools; 1/3 trials) | 102 iters (diamond); 160 total iters | ~640 GPT-4 calls (~$20–50 API; low env compute) | 15x faster wooden/8x stone/6x iron vs AutoGPT/ReAct; only to diamond. |
| | RL-GPT | 8% (ObtainDiamond); 58% w/ iters | 3M samples | Single RTX3090 (~day for diamonds) | SOTA MineDojo (e.g., 67% wooden pick); >Plan4MC (0%). |
| **2. Fine-Tuned E2E LLM** | LARM | 93% (diamond sword); 53% (enchanted) | N/A (lifelong) | 42h RTX4090 (enchanted train) | >RL-GPT/Plan4MC (e.g., 27% iron sword vs 0%); 1st enchanted diamond. |
| **3. LLM Rewards** | Auto MC-Reward + PPO/IL | 36.5% (diamond ore, forest) | 256k RL frames/task | Pretrain 11M frames (32 A800s); LLM negligible | Sparse: 0.5%; IL: 29%/0%; +lava avoidance. |
| **4. LLM Curriculum/Goals** | Voyager/Optimus-2 (combo) | Varies (e.g., Optimus-2: 13–28% diamond group) | Subgoal seq. | As #1 | Boosts #1; e.g., DEPS-Oracle 60% (but oracle). |

**Best Performance-to-Compute Ratio for Long-Horizon Tasks (e.g., Diamond)**:  
**Pattern 2 (E2E fine-tuned LLM, LARM)** achieves the best ratio. It delivers top success (enchanted diamond: 53%, beyond plain diamond in others) w/ modest compute (42 GPUh train → fast 0.58s/step inf) and single-model generalization across tasks. Hierarchical (#1) excels in sample efficiency/no fine-tune (RL-GPT: 8% @3M; Voyager: API-only), but slower inf/lower peak perf. Rewards (#3) amplify RL cheaply but cap at ~36%. Curriculum (#4) enhances others but not standalone.  
For deploy: #2 (efficient post-train); research/open-ended: #1 (promptable).

**No, there is no publicly available validation showing that RL policies trained in Craftium, MineWorld, or similar lightweight Minecraft-inspired simulators successfully transfer zero-shot (without fine-tuning) to full Minecraft environments like MineRL or MineDojo.**

Craftium (arXiv:2407.03969 and ICLR submission) is a Gymnasium/PettingZoo wrapper around the open-source Luanti/Minetest voxel engine, designed as a **standalone alternative** to Minecraft-based RL platforms like MineRL and MineDojo. It emphasizes flexibility (Lua modding for custom worlds, biomes, mobs, multi-agent support) and speed (~2,700 steps/sec vs. MineDojo's ~70 steps/sec in benchmarks). Papers position it for rapid prototyping of rich 3D tasks (e.g., procedural open worlds up to 64K³ blocks), continual RL, and MARL, but explicitly note Minecraft's closed-source Java limitations as a motivation to avoid it altogether—no transfer experiments, domain adaptation, or evaluations on MineRL/MineDojo benchmarks are reported.

MineWorld (arXiv:2504.08388) is **not an RL simulator** but a real-time autoregressive Transformer world model for Minecraft video prediction and action generation. It simulates gameplay by predicting tokenized states/actions but lacks RL policy training interfaces or transfer tests to full Minecraft.

Other lightweight options (e.g., wrappers like MineStudio or MineRL itself) accelerate full Minecraft but do not create "sim-to-real" gaps—they run the actual game, so policies transfer perfectly (zero drop) by design. No papers or discussions validate cross-engine transfers from Minetest-like sims.

**Performance drop: Unknown quantitatively, as no zero-shot experiments exist.** Policies would likely fail or drop to near-zero success due to domain gaps—Craftium policies are not evaluated outside its ecosystem.

**Largest transfer gaps (inferred from engine differences):**
| Feature | Craftium/Minetest | Full Minecraft (MineRL/MineDojo) | Expected Gap Impact |
|---------|-------------------|----------------------------------|---------------------|
| **Visual Style** | Voxel-based (Luanti community textures/mods); customizable rendering; lower-res by default (e.g., 64x64 grayscale in expts). | Proprietary Java textures, lighting, water/shader effects; higher fidelity. | **High**: CNN/RNN policies overfit to textures, lighting, block appearances—major distribution shift. |
| **Physics** | Lua-modifiable (gravity, block breaking, mob AI, collisions); engine tweaks for RL (sync updates, resets). | Fixed proprietary physics; precise timings for mining/crafting/mobs. | **High**: Mismatched block hardness, fall damage, entity behaviors break low-level skills (e.g., harvesting). |
| **Action Timing** | ~2,700 steps/sec; customizable wrappers (21 keys + mouse, discrete/continuous). | ~20-70 steps/sec; MineRL-normalized (20 actions/sec). | **Medium**: Timing mismatches resolvable via replay buffers, but compounds with physics. |

Visual and physics mismatches are primary culprits, as Minetest is "inspired by" but not a Minecraft clone—assets/mods differ, preventing plug-and-play. Fine-tuning or adaptation (e.g., via DRED-style UED) would be needed, but untested.

**For a Minecraft agent to play in real-time at 20 ticks per second (TPS), inference latency must be under 50 ms per tick (1/20 second) to match the game's native pace without perceptible lag.** This assumes per-action decisions; **action chunking** (predicting multiple actions per inference) amortizes latency (e.g., 10 actions in 400 ms yields ~40 ms effective), but chunks are typically 4–20 actions for smooth play.

| Architecture | Meets 20 TPS on RTX 3090/4090? | Key Latency Details | Minecraft Achievements |
|--------------|--------------------------------|---------------------|-------------------------|
| **VPT** (Video PreTraining, ~0.5B params) | **No** | No reported real-time latencies; operates at 20 Hz interface but requires cluster-scale training (720 V100s). Inference likely >50 ms/action on consumer GPUs without heavy optimization. | Crafts diamond tools (2.5% success); human-level on subtasks like iron pickaxe. |
| **LARM** (Large Auto-Regressive Model, 3.1B params) | **No** | 580 ms/inference on RTX 4090 (~1.7 actions/sec); "meets online high-level action" needs but not low-level 20 TPS. 850 ms on RTX 3090. | First to harvest enchanted diamond sword; 90%+ on subtasks (e.g., iron ingot, furnace). |
| **DreamerV4** (2B params) | **Marginally (RTX 4090 possible with tweaks)** | 21 FPS (~48 ms/frame) on H100; real-time interactive at >20 FPS. RTX 4090/3090 may hit 10–15 FPS due to lower compute/VRAM but feasible with quantization (fits 24 GB VRAM). | First offline agent to mine diamonds (0.7% success, 100x less data than VPT); stone/iron tools 90%+. |

**DreamerV4 is the only one achieving real-time constraints**, via world-model imagination for offline training and fast rollout. VPT/LARM are too slow for per-tick decisions on consumer hardware.

**Optimizations for real-time play** (used/across these):
- **Quantization**: Not core in papers but essential for 3090/4090 (e.g., FP16/INT8 reduces DreamerV4 VRAM/accelerates ~1.5–2x).
- **Distillation**: LARM distills GPT-4 knowledge via "referee RL"; VPT uses IDM for pseudo-labels (semi-distilled BC).
- **Action chunking**: DreamerV4's multi-token prediction (MTP, L=8 actions); VPT discretizes mouse/actions into bins (foveated, effective chunking).
- **DreamerV4-specific**: **Shortcut forcing** (4-step sampling vs. 64, 16x faster), 2D transformer (GQA, axial attention), register tokens for context (192 frames/9.6s).
- **LARM**: LoRA for efficient fine-tuning (frozen LLM + trainable adapter).
- General: Parallel decoding (e.g., spatial tokens), reduced resolution (128x128), efficient heads (policy/value).

### Current State of Minecraft RL (Late 2025)

Minecraft remains one of the most challenging benchmarks for reinforcement learning (RL) due to its open-world nature: procedural generation, long-horizon tasks (e.g., tech tree progression to diamond tools), sparse rewards, partial observability, and a massive action space (keyboard + mouse). Progress has shifted heavily toward **imitation learning from human data** and **LLM-guided agents**, with pure RL struggling on complex tasks without demonstrations.

Key milestones:
- **VPT (Video PreTraining, OpenAI 2022)**: Seminal work using behavioral cloning on labeled contractor data + inverse dynamics modeling to pseudo-label massive unlabeled YouTube videos. Zero-shot agents perform early-game tasks; fine-tuned agents craft diamond pickaxes ~2.5% of the time.
- **Voyager (2023)**: LLM-based agent (GPT-4) with skill library, automatic curriculum, and code execution for exploration. Unlocks full tech tree, discovers 3.3× more items than priors, generalizes to new worlds.
- **Plan4MC (2023)**: Hierarchical skill learning via RL (no demos) + LLM-planned skill graphs. Solves 24–40 diverse tasks without demonstrations, sample-efficient for tech tree progression.
- Recent works (2024–2025): Hierarchical deep RL models improve sample efficiency in sparse-reward settings; vision-language models tailored for RL friendliness; exploratory interpretability on VPT revealing misgeneralizations (e.g., confusing villagers for trees).

The MineRL competition (last active ~2022) is inactive, with no 2025 events. Pure RL from scratch rarely reaches diamond-level without heavy engineering.

### Solved vs. Unsolved Subproblems

| Subproblem              | Status (Late 2025)                                                                 | Key Solutions/Advances                                                                 | Remaining Challenges |
|-------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|----------------------|
| **Environment Speed**   | Largely solved                                                                    | MineDojo (fast simulator based on Minecraft); parallel envs (thousands of workers); GPU-accelerated alternatives like Craftax/Crafter for prototyping. | Full-fidelity Minecraft still slow for massive scaling; real-time interaction limits. |
| **Action Abstraction**  | Mostly solved                                                                     | Hierarchical policies (e.g., high-level skills over primitives); VPT's behavioral cloning on human-like mouse/keyboard; LLM-generated subgoals/actions (Voyager, Plan4MC). | Fine-grained continuous mouse control in novel situations; robust recovery from failures. |
| **Credit Assignment**   | Partially solved (improved but hard in long horizons)                             | Hierarchical RL (rewards at subgoal completion); dense reward shaping via LLMs or intrinsic curiosity; temporal credit probes in transformers. | Extreme delays (20k+ steps to meaningful rewards); sparse natural rewards lead to hacking if shaped poorly. |
| **Exploration**         | Unsolved in pure RL; mitigated with priors                                         | Curiosity/intrinsic rewards for skills; LLM-driven curricula (Voyager auto-generates quests); procedural generation forces diversity but exacerbates credit issues. | Systematic open-world discovery without priors; avoiding local optima in infinite-like worlds; balancing breadth vs. depth. |

### Remaining Open Problems Tractable for a Small Team

Pure RL agents still fail at diamond obtainment reliably without demos or LLMs. Tractable open problems (feasible with 2–6 people, modest compute like a few GPUs):

1. **Zero-shot generalization from foundation models**: Fine-tune/extend VPT or similar on new tasks (e.g., Nether/End progression, redstone engineering). Interpretability fixes for misgeneralizations observed in VPT.

2. **Hybrid RL + LLM methods without proprietary models**: Use open LLMs (e.g., Llama-3) for planning/curricula over RL-learned skills (building on Plan4MC). Focus on demonstration-free long-horizon tasks.

3. **Better hierarchical decomposition**: Automatic skill discovery in procedural worlds; improving subgoal proposal for sparse rewards.

4. **Multi-task / continual learning**: Agents that accumulate skills across diverse procedurally generated worlds without catastrophic forgetting.

5. **Robustness and safety**: Goal misgeneralization (e.g., aggressive behaviors toward villagers); reward hacking in open-ended settings.

6. **Sample-efficient pure RL baselines**: Novel intrinsic rewards or exploration bonuses tailored to Minecraft's crafting/exploration loop.

These are tractable because open-source foundations exist (VPT code/data, MineDojo envs, Voyager-inspired frameworks).

### Highest-Impact 6-Month Effort

For a small team:
- **Extend Voyager-style agents with open-source LLMs and light RL fine-tuning**. Voyager's skill library + curriculum is a breakthrough for open-ended play. Replacing GPT-4 with a fine-tuned open model (e.g., via LoRA) while adding RL for low-level robustness could yield a fully open agent unlocking the full tech tree reliably. Impact: Democratizes SOTA Minecraft agents; advances lifelong learning benchmarks.
- Alternative high-impact: **Interpretability + steering of VPT-like models**. Build tools to probe/edit internal representations, fixing issues like misidentification. This could enable safer deployment in interactive envs and transfer insights to broader vision-language-action models.

Either direction could produce a strong paper (e.g., NeurIPS/ICLR) and open-source artifact, pushing the field toward more general, demonstration-free agents in procedural environments.

### Choice of Base Engine

In 2025, the optimal base for a high-performance Minecraft-like RL training environment prioritizes **scalability**, **parallelization**, and **low computational overhead** to enable massive sample collection (billions of steps).

- **Recommended: Craftium on Luanti (formerly Minetest)**  
  Craftium (2024-2025, accepted at ICML 2025) is the leading open-source framework for this purpose. It builds on Luanti, a lightweight, fully open-source C++ voxel engine with extensive modding via Lua. Unlike proprietary Minecraft-based setups (e.g., MineDojo/Voyager), Luanti/Craftium is highly efficient, headless-capable, and designed for RL with Gymnasium/PettingZoo APIs. It supports rich 3D visuals, procedural worlds, and multi-agent setups while running hundreds of instances in parallel on modest hardware.  
  Reference: Craftium GitHub (mikelma/craftium) and arXiv:2407.03969.

- **Alternative: Proprietary Minecraft (via mods like Fabric/Forge)**  
  Used in classics like MineDojo (2022, basis for Voyager), VPT (OpenAI), and recent works (e.g., RL-GPT, Plan4MC). Offers exact Minecraft fidelity but is slower, harder to parallelize (often requires virtual displays or heavy mods), and has maintenance issues (MineDojo is reportedly unmaintained as of late 2024). Recent 2025 papers (e.g., CraftGround) improve on this with updated Minecraft versions, but performance lags behind open alternatives.

- **Custom Voxel Engine**  
  Feasible for ultimate control (e.g., JAX-accelerated like Craftax, a 2D fast benchmark), but building a full 3D Minecraft-like engine from scratch (OpenGL/Vulkan + procedural gen) is time-intensive and rarely justifies the effort when Luanti provides near-equivalent features with better ecosystem support. Only pursue if needing extreme optimizations beyond Luanti's capabilities.

- **Neural World Model (e.g., DreamerV4-style simulation)**  
  Emerging in 2025 (e.g., MineWorld arXiv:2504.08388; DreamerV4 achieves diamond collection from offline data). Train a generative model (transformer-based) on Minecraft videos/datasets (e.g., VPT dataset) for fast, GPU-native imagination rollouts. Excellent for sample efficiency but lacks perfect fidelity for long-horizon tasks; best as a hybrid (real env for collection + neural for planning). Not yet a full replacement for training throughput.

**Verdict**: Start with **Craftium/Luanti** for high-performance pure RL training. Fall back to Minecraft mods for exact replication of prior papers (Voyager, VPT).

### Required Modifications for RL Compatibility

For Craftium/Luanti (minimal changes needed):

- Fork Luanti and add remote control interface (keyboard/mouse emulation over socket/TCP).
- Extend Lua API for reward shaping, termination conditions, and structured observations (e.g., inventory, biome data).
- Implement headless mode (no rendering during training; render only for eval).
- Add vectorized parallel env support (e.g., batch actions across instances).
- Mod for Minecraft-like mechanics if using base Luanti (use VoxeLibre/MineClone2 mod for near-identical blocks/crafting).

For Minecraft-based:

- Use Fabric/Forge mods (as in Voyager/MineDojo) for API exposure.
- Malmo-like integrations or custom mods for discrete actions and observations.
- Headless via virtual frames or offscreen rendering.

General RL mods:

- Discrete action space wrapper.
- Frame stacking, POV + inventory/equipment observations.
- Reward wrappers for sparse tasks (e.g., tech tree progression).

### Target Throughput Benchmarks

High-performance goals focus on **environment steps per second (SPS)** across parallel instances:

- Legacy Minecraft (MineRL/MineDojo): ~100-500 SPS total on multi-core CPU (limited by Java overhead; often 10-50 instances).
- Craftium/Luanti: 1,000-10,000+ SPS possible (C++ efficiency; reports of 100+ parallel instances on single machine; far superior to Minecraft).
- JAX/GPU-accelerated (e.g., Craftax-inspired 3D): 10k-100k+ SPS (full pipeline on GPU).
- Neural models (DreamerV4/MineWorld): Millions of imagined steps/sec on single GPU, but real interaction limited.

**2025 Targets**:
- Baseline: 5,000+ SPS for 256+ parallel envs (achievable with Luanti).
- Advanced: 50,000+ SPS with headless batching on multi-GPU cluster.
- For diamond-level tasks: Aim for 1B+ steps in days (vs. weeks on Minecraft).

### Action/Observation API Design

Standardize on **Gymnasium** (single-agent) or **PettingZoo** (multi-agent) for compatibility.

- **Observations**:
  - Primary: RGB POV (64x64 or 128x128 stacked frames for temporal info).
  - Auxiliary: Structured dict (inventory slots, equipped item, health/hunger, biome, nearby entities, compass).
  - Optional: Egocentric depth map or voxel grid for advanced perception.

- **Actions**:
  - Discrete space (preferred for PPO/CleanRL): ~20-30 actions (move forward/back/strafe/jump/sneak, camera delta (e.g., 8 directions x pitch), attack/use/place, craft/equip hotbar slots).
  - Continuous alternative: Mouse delta + buttons (for JAX/Dreamer).
  - Hierarchical option: High-level skills (e.g., "mine tree") via Lua mods.

Design for low-latency: Send batched actions, receive batched obs/rewards/dones.

### Integration with Modern RL Libraries

- **CleanRL**: Ideal for single-agent PPO baselines; lightweight, reproducible. Direct Gymnasium support.
- **RLlib (Ray)**: Scalable distributed training; handles multi-agent natively; good for large clusters.
- **Custom JAX (e.g., PureJaxRL, DreamerV3/V4 impls)**: Best for max throughput (GPU-native envs via vmap); enables world-model integration. Use for Dreamer-style agents.

Hybrid: Train with JAX for speed, eval in CleanRL.

### Open-Source Implementations and Blueprints from Recent Papers

- **Craftium** (2024-2025): Primary blueprint; full Gymnasium wrapper for Luanti. GitHub: mikelma/craftium.
- **MineDojo/Voyager** (2022-2023, active issues 2025): Minecraft-based; Voyager for LLM+RL agents.
- **DreamerV3/V4** (2023-2025): World model mastery (diamonds from scratch/offline). GitHub: danijar/dreamerv3.
- **MineWorld** (2025): Real-time neural Minecraft simulator. arXiv:2504.08388.
- **CraftGround** (2025): Updated flexible Minecraft env.
- **CLIP4MC** (ECCV 2024): RL-friendly VLM for rewards.

**Roadmap Phases**:
1. Prototype with Craftium + CleanRL PPO.
2. Scale parallel instances + JAX integration.
3. Add hierarchical actions/world models for open-ended tasks.
4. Benchmark on tech tree (e.g., obtain diamond). 

This setup enables state-of-the-art RL research in 2025 with feasible compute.

**1. Training Environment: Custom (e.g., diamond_env based on MineRL)**

DeepMind's DreamerV3/V4 sets the 2024-2025 SOTA for sample-efficient diamond acquisition (first from scratch, no human data; ~30M steps to first diamond, reliable at 100M). It uses a **custom standardized Minecraft Diamond Environment** (diamond_env): MineRL v0.4.4 (MC 1.11.2 modded), with bug fixes, 36k-step episodes, 12 sparse milestone rewards (+1 each, diamond last), infinite procedural worlds.

| Environment | Pros for Goal | Cons | DeepMind/OpenAI/NVIDIA/Acad View (2024-2025) |
|-------------|---------------|------|---------------------------------------------|
| **Custom (diamond_env/MineRL variant)** | Proven <100M diamonds; multimodal obs/actions match MC; modded Java Edition closest to "real" deploy (port policy via bot/mod). | Older MC version (1.11); ~100-500 steps/sec. | **DeepMind SOTA**: First no-data diamonds; scalable. |
| **MineRL** | Benchmark for diamonds; large human datasets (if hybrid). | Slow (~50 steps/sec); outdated comps (2021: no full solves w/o data). | Legacy; DeepMind builds on/fixes it. |
| **Craftium** | 2k+ steps/sec (C++/Lua, Minetest voxel); fully customizable/open-source; Gymnasium; multi-agent/open-world. | Not actual MC (voxel approx.); no vanilla deploy. | Acad (ICML 2025): Faster MineRL alt for iter/open-ended RL; no diamond SOTA yet. |

**Rec**: Start with **diamond_env** (GitHub: danijar/diamond_env) for proven efficiency/deploy fidelity; prototype in Craftium for speed.

**2. Agent Architecture: World Model (e.g., DreamerV3+)**

World models excel for sample efficiency in sparse/open-world (Minecraft: long-horizon, 20k+ actions/seq). DeepMind's DreamerV3 (RSSM world model + actor/critic; single A100) first solves diamonds from pixels/sparse rewards (~30M steps first; scales to reliable). DreamerV4 adds offline video pretrain (100x less data than OpenAI VPT).

| Arch | Sample Eff (<100M diamonds) | DeepMind/OpenAI/NVIDIA/Acad (2024-2025) |
|------|-----------------------------|-----------------------------------------|
| **World Model (Dreamer)** | Best: ~30M first (online); offline diamonds (V4). Imagines trajectories for exploration. | **DeepMind SOTA** (Nature 2025): Generalizes 150+ tasks; robotics-applicable. |
| **Model-Free (PPO/Rainbow)** | Poor: 100M+ w/o data; needs shaping/curricula. | Baselines; lags Dreamer 3-10x. |
| **LLM-Hybrid (Voyager)** | Fast tech unlocks (15x prior); code-gen skills. | NVIDIA/MineDojo (2023, cited 2025): Open-ended; but API-heavy, not pure RL steps. |

**Rec**: **World model** (DreamerV3 impl: danijar.com/dreamerv3); <10M challenging w/o data (closest ~30M), but scales predictably (bigger model/data eff).

**3. Action Space Design: Discrete Hierarchical (25 actions)**

Diamond_env: Categorical 25 actions (movement/turn/jump/attack + task hotkeys: craft planks/stick/table/pickaxes/furnace, place/equip/smelt). Abstracts crafting (no GUI nav); discrete camera (turn up/down/left/right). Enables long-horizon w/o explosion.

**Rec**: **Match diamond_env**: Low-level locomotion (forward/back/left/right/jump/attack/noop/turn×4) + high-level crafts/equips/smelt (12). Discrete avoids continuous camera instability; hierarchical boosts efficiency (DeepMind SOTA).

**4. Observation Format: Multimodal (Low-res RGB + Inventory State)**

64×64×3 first-person RGB + vectors: inventory counts/max (391), equipped (393 one-hot), health/hunger/breath scalars. Enables world model to predict physics/inventory.

**Rec**: **64×64 RGB PoV + inventory dict + vitals**. Pixels for vision/physics; state for exact counts (critical for crafting); low-res efficient.

**Overall for <10M + Real Deploy**: Pure RL <10M diamonds w/o data unachieved (DeepMind ~30M closest); use Dreamer on diamond_env (train fast/customize), fine-tune policy for vanilla MC via bot (e.g., Mineflayer/keyboard emu). Craftium accelerates dev but approx MC.