# Research Query Set for Minecraft RL Agent Development

Here are standalone, well-scoped queries designed to extract maximum signal from recent research. Each is crafted to be self-contained and focused on actionable insights.

---

## Core SOTA & Architecture Queries

**Query 1: Diamond Benchmark Performance**
> "What is the current state-of-the-art for AI agents obtaining diamonds in Minecraft from scratch as of 2024-2025? Include the model architectures used (e.g., VPT, GROOT, STEVE-1), training paradigms (imitation learning, RL, hybrid), compute requirements, and success rates. Focus on papers from NeurIPS, ICML, ICLR, and arXiv within the last 12 months."

**Query 2: Foundation Model Approaches**
> "How are vision-language models and large language models being used as the backbone for Minecraft agents in 2024? Compare approaches like GROOT, STEVE-1, and Voyager in terms of how they leverage pretrained representations, what modalities they consume, and how they handle long-horizon planning. What architectural choices lead to better sample efficiency?"

---

## Bottleneck & Constraint Analysis

**Query 3: Training Bottlenecks**
> "What are the primary computational and algorithmic bottlenecks for training reinforcement learning agents in Minecraft? Specifically: environment step throughput, reward sparsity, observation dimensionality, action space complexity, and credit assignment over long horizons. What do researchers identify as the limiting factors for scaling these systems?"

**Query 4: Environment Throughput Requirements**
> "What are the minimum environment steps-per-second and frames-per-second throughput requirements for training competitive RL agents in complex 3D games? Reference benchmarks from MineRL, Atari, and procedurally generated environments. What is the relationship between environment speed and wall-clock training time for achieving human-level performance?"

---

## Lightweight Simulation & World Models

**Query 5: Minecraft World Simulators**
> "What lightweight alternatives to running full Minecraft exist for training AI agents? Include MineDojo, MineRL, and any recent work on Minecraft world models, neural simulators, or simplified voxel environments. Compare their fidelity, speed (steps/second), and whether agents trained on them transfer to real Minecraft."

**Query 6: World Model Architectures for Games**
> "What is the state-of-the-art in learned world models for training game-playing agents without running the actual game engine? Focus on recent work like DIAMOND, Genie, GameNGen, and similar approaches. What resolution and temporal fidelity do these models achieve, and can agents trained purely in dreamed rollouts transfer to real environments?"

---

## Action Space & Observation Design

**Query 7: Action Space Design**
> "How should action spaces be designed for Minecraft RL agents? Compare continuous vs. discrete, hierarchical vs. flat, and primitive actions vs. learned skills/options. What action abstractions (e.g., craft-item macros, navigation primitives) have proven most effective for sample efficiency and transfer?"

**Query 8: Observation Resolution & Modality**
> "What observation resolution and modality combinations work best for Minecraft agents? Compare raw pixels at various resolutions, segmentation masks, inventory state, and text descriptions. Is there a minimum visual resolution below which agents fail to learn, and what is the compute-performance tradeoff?"

---

## RL Environment Engineering

**Query 9: High-Performance RL Environment Design**
> "What are the design principles for building high-throughput RL training environments in 2024? Reference vectorized environments, GPU-accelerated simulators (IsaacGym, Brax, EnvPool), and async actor-learner architectures. What steps-per-second benchmarks should a custom environment target to be viable for modern RL algorithms?"

**Query 10: Reward Shaping & Curriculum**
> "What reward shaping and curriculum learning strategies are most effective for training Minecraft agents on long-horizon tasks like obtaining diamonds? How do researchers handle the sparse reward problem, and what intermediate objectives or intrinsic motivation signals have proven useful?"

---

## Transfer & Generalization

**Query 11: Sim-to-Real & Fidelity Requirements**
> "What level of simulation fidelity is required for Minecraft agent policies to transfer from lightweight simulators to the full game? Are there studies on the minimal world model accuracy needed, and how do distributional shifts between training and deployment environments affect agent performance?"

---

## Summary Usage

| Query | Primary Focus |
|-------|---------------|
| 1-2 | Current SOTA architectures & results |
| 3-4 | Bottleneck analysis & throughput requirements |
| 5-6 | Lightweight simulation alternatives |
| 7-8 | Action/observation space engineering |
| 9-10 | RL environment design best practices |
| 11 | Transfer & fidelity requirements |

These queries are designed to be run independently. I'd recommend prioritizing **5, 6, 9** for environment engineering insights, and **1, 2, 7** for architecture decisions.

**The state-of-the-art (SOTA) for AI agents obtaining diamonds in Minecraft from scratch—starting in a random world with an empty inventory and no prior items—has advanced significantly in 2024-2025, particularly through world-model-based RL, hybrid LLM-RL systems, and autoregressive policies.**

Key papers from NeurIPS 2024, ICML 2025, ICLR 2025 (where applicable), and recent arXiv focus on long-horizon tasks requiring ~20,000+ actions (e.g., wood→tools→mining→smelting→diamonds). Baselines like VPT (2022: imitation/RL on human videos, ~20% with massive data), GROOT (ICLR 2024: 16% via video imitation), and STEVE-1 (2023: 0% on diamonds via text-to-behavior) are outperformed.

| Agent/Paper | Conference/arXiv | Architecture | Training Paradigm | Compute | Diamond Success Rate | Notes |
|-------------|------------------|--------------|-------------------|---------|----------------------|-------|
| **LARM** | ICML 2025 | <5B autoregressive (TinyLLaVA-3.1B + LoRA); CLIP vision + action/value heads | Referee RL (PPO variant w/ GPT-4 dense rewards for long-horizon) | 42h on 1x RTX 4090 (online) | 53% (16/30) enchanted diamond sword; 93% (28/30) plain diamond sword (Mineflayer env) | **SOTA for advanced tasks**: First to harvest *enchanted* diamond gear (Nether travel, mob combat, enchanting). Generalizes across open-world tasks. |
| **RL-GPT** | NeurIPS 2024 (arXiv Feb 2024) | Hierarchical: GPT-4 slow/fast planners + critic; PPO policy | Hybrid: LLM code-as-policy + online PPO RL (3M steps) | 1 day on 1x RTX 3090 | 8% ObtainDiamond | **SOTA sample efficiency** (3M steps vs. VPT's 16.8B). Auto-decomposes tasks; beats DreamerV3 (2%), VPT (20% w/ expert data). |
| **Dreamer 4** | arXiv Sep 2025 | 2B transformer world model (block-causal attn, causal tokenizer) | Offline imagination RL (PMPO in world model) + BC finetune | 256-1024 TPUs; real-time inf. on 1x H100 | 0.7% (1000 episodes) | **First purely offline** diamonds (VPT dataset, 100x less data than VPT). No env interaction; simulates crafting/mining. |
| **DreamerV3** | Nature Apr 2025 (arXiv 2023) | RSSM world model + MLP actor/critic | Online imagination RL | ~9 GPU-days on 1x A100 (100M steps) | 0.4% discovery rate | First no-human-data from scratch; 100% runs find ≥1 diamond. |

**Trends**:
- **Paradigms**: Shift to hybrids (RL + LLMs/world models) for long-horizon planning; offline/imagination RL (Dreamer series) for safety/robotics.
- **Challenges**: Sparse rewards, exploration (diamonds Y=5-12), ~24k actions. LARM pushes to enchanted gear (SOTA milestone).
- **Eval**: MineDojo/ObtainDiamond standard; MCU (ICML 2025) benchmarks show VPT/STEVE-1/GROOT struggle on diversity.
- Withdrawn (Odyssey, ICLR 2025) claimed 92.5% but relied on pre-built skills (not pure scratch).

LARM holds SOTA for hardest variants; RL-GPT for efficiency. No NeurIPS 2025 papers yet (conference ongoing).

Voyager (2023), **STEVE-1** (2023), and **GROOT** (2023-2024) represent key approaches to building Minecraft agents that leverage large language models (LLMs) and/or vision-language models (VLMs) as backbones. These agents operate in the open-ended MineDojo/Minecraft environment, focusing on exploration, skill acquisition, and task completion.

### Comparison Overview

| Aspect                  | Voyager (2023)                                                                 | STEVE-1 (2023)                                                                 | GROOT (2023, ICLR 2024 Spotlight)                                              |
|-------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **Primary Backbone**   | LLM (GPT-4) for planning, curriculum, and code generation                     | Vision model (fine-tuned VPT) + VLM (MineCLIP) for goal embeddings; generative policy inspired by diffusion/unCLIP | Video encoder-decoder transformer (built on VPT foundation); causal transformers for policy |
| **Pretrained Representations Leveraged** | LLM's world knowledge (Minecraft recipes, strategies from training data); no visual pretraining for planning | VPT (behavior cloning from unlabeled Minecraft videos) for low-level control; MineCLIP (contrastive video-text) for reward/goal space | VPT (imitation from videos) as base; structured goal space induced from video instruction encoder |
| **Modalities Consumed** | Text (prompts, feedback); structured state/info (inventory, biome via API); no raw vision for planning | Raw pixels (vision) + text/visual instructions                          | Video frames (vision) as instructions; raw pixels for control                 |
| **Long-Horizon Planning** | Hierarchical: LLM generates high-level plans, automatic curriculum for exploration, skill library (code snippets) retrieval/composition, iterative prompting with self-verification/error feedback | Short-horizon focus; generative sampling in latent goal space for instruction-following; struggles with very long sequences due to limited memory | Goal-conditioned policy with structured latent space for composition; supports emergent long behaviors via video goal synthesis, but primarily short-to-medium horizon |
| **Sample Efficiency / Architectural Strengths** | Extremely high: few environment interactions (uses code execution via Mineflayer API); lifelong learning via growing library alleviates forgetting | Moderate: builds on VPT pretraining; efficient for instruction-tuned tasks but requires fine-tuning on latents | High: imitation from gameplay videos; KL-regularized encoder-decoder avoids mode collapse, enables goal composition; outperforms baselines on benchmarks with fewer samples |

### Detailed Analysis

- **Leveraging Pretrained Representations**:
  - **Voyager** relies heavily on the LLM's pretrained knowledge of Minecraft (e.g., tech tree, crafting recipes) to propose feasible subgoals and generate executable code. This zero-shot transfer from text data enables rapid progress without visual pretraining.
  - **STEVE-1** adapts VPT's behavioral priors (learned from millions of hours of human gameplay videos) and aligns them with MineCLIP's semantic embeddings, turning text instructions into visual goal latents for policy conditioning.
  - **GROOT** extends VPT-like imitation learning by training on paired gameplay videos and instructions, creating a rich, structured goal representation that supports composition (e.g., combining skills for novel tasks).

- **Modalities**:
  - Voyager is largely text/state-based, avoiding raw vision to simplify control (uses high-level APIs).
  - STEVE-1 and GROOT are fully vision-based, processing raw pixels like humans, with STEVE-1 adding text and GROOT focusing on video demonstrations as multimodal instructions.

- **Long-Horizon Planning**:
  - Voyager excels here through decomposition into reusable code skills, retrieval-augmented planning, and curriculum-driven exploration—unlocking tech tree milestones 15x faster than priors in evaluations.
  - STEVE-1 is designed for shorter horizons (e.g., early-game tasks), with success on ~12/13 benchmarks but limitations in memory for extended sequences.
  - GROOT handles medium-to-long horizons better via goal composition in latent space, achieving high Elo ratings against humans and agents on diverse tasks (collect, build, craft).

- **Architectural Choices for Sample Efficiency**:
  - **High-level abstractions** (e.g., Voyager's code-as-actions) drastically reduce interaction needs—agents explore via simulation-like code execution rather than trial-and-error in pixels.
  - **Pretrained behavioral priors** (VPT in STEVE-1/GROOT) provide strong initialization, allowing fine-tuning with far fewer samples than pure RL.
  - **Structured/memory-augmented components** (skill libraries in Voyager, latent goal spaces in GROOT, episodic recall in extensions) mitigate forgetting and enable composition, compounding efficiency over time.
  - Vision-based approaches (STEVE-1, GROOT) trade some efficiency for realism but gain robustness via video pretraining; hierarchical hybrids (seen in later works like JARVIS-1) combine both for best results.

In 2024-2025, trends shift toward hybrid VLM-LLM agents (e.g., JARVIS-1, STEVE series evolutions) that integrate multimodal memory and direct pixel control, improving on these foundations for even longer horizons and open-ended creativity. Pure LLM-driven agents like Voyager remain highly efficient for planning, while VLM-backed ones close the gap to human-like perception.

**Primary Computational and Algorithmic Bottlenecks**

Minecraft serves as a challenging benchmark for reinforcement learning (RL) due to its open-world nature, procedural generation, and human-like interface (20 Hz tick rate with pixel observations and mouse/keyboard actions). Key works like MineRL, VPT (Video PreTraining), Voyager, and Plan4MC highlight persistent issues in training agents for tasks like ObtainDiamond (crafting tools to mine a rare diamond).

| Bottleneck                  | Description and Impact                                                                 | Evidence from Research |
|-----------------------------|----------------------------------------------------------------------------------------|-------------------------|
| **Environment Step Throughput** | Minecraft simulators (e.g., MineRL, MineDojo) run slowly: ~20 steps/sec nominally, max ~300 steps/sec per parallel env (multi-core). This limits total samples (e.g., MineRL caps at 8M steps), as RL requires billions for convergence. Faster sims like Procgen hit thousands/sec but lack Minecraft's complexity. | MineRL: 5-10 parallel envs feasible; VPT RL fine-tuning needs ~1.4e10 frames (~144 hrs on 80 GPUs + 56k CPUs). Slow sims divert 99%+ compute to agents, not envs. |
| **Reward Sparsity**        | Rewards only for rare milestones (e.g., diamonds 2-10x rarer than other ores; hierarchical: wood → planks → tools → diamond). Naive exploration fails; agents stall on early subtasks. | ObtainDiamond: Exponential milestone rewards (e.g., +1024 for diamond); humans solve in <15 min, RL needs demos/IL to bootstrap. |
| **Observation Dimensionality** | High-dim pixel obs (64x64x3 POV) + inventory vector; perceptual aliasing (same visuals, different utilities). Noisy web videos add artifacts. | CNNs/LSTMs needed; VPT filters ~270k hrs video to 70k clean hrs for training. |
| **Action Space Complexity** | Factored multi-discrete (e.g., forward/back/left/right/sneak/sprint/jump/attack/use/craft: ~2-8 options each) + continuous camera (pitch/yaw) + equip slot. ~Thousands effective combos; GUI drags/crafting explode space. | Action shaping (discretize camera, remap) boosts performance; AR policies handle params sequentially. |
| **Credit Assignment over Long Horizons** | Episodes up to 18k steps (15 min); deep hierarchies (10+ subtasks) delay rewards thousands of steps. RL struggles with variance/forgetting. | Plan4MC: 25-121 planning steps/task; VPT RL: fragile progress (e.g., die after smelting). |

**Limiting Factors for Scaling Systems**

Researchers identify **sample inefficiency** as primary: RL needs 10-1000x more data than imitation (e.g., MineRL: 8M steps vs. Atari's 20B; VPT: IL on 70k video hrs enables 2.5% diamond success, RL alone ~0%).

- **Compute Walls**: Slow sims + agent-heavy workloads (99% compute on policy nets); parallel envs limited (~300/sec/env). VPT RL: 1.4e10 frames on massive clusters.
- **Exploration/Forgetting**: Sparse/long-horizon → poor baselines; RLHF/KL losses mitigate but scale poorly.
- **Data/Task Design**: Manual tasks bottleneck scaling; auto-synthesis proposed (100k+ tasks). Demos/LLMs (Voyager) bootstrap but don't fully solve RL scaling.

**Mitigations and Progress**: Imitation (VPT on YouTube videos), hierarchies (Plan4MC: 40 tasks, 10+ skills), LLMs (Voyager: open-ended), faster sims (MineStudio). Pure RL scales poorly without priors; hybrids dominate.
Goodhearts law is only part of the problem. 

If we use environments that are simple enough that they can run at thousands of FPS on a single CPU (e.g., Atari games), then our agents have to be even simpler to satisfy the paradigm posed by the big world hypothesis (which is the right setting imo). 

Currently if we sum the total compute used in a typical deep-RL experiment, 99+% of it goes to the agent. This is problematic because we are benchmarking algorithms in an unrealistic situation. 

The two solutions to this problem are to either make the agents very simple (e.g., we did that in Javed, Shah, Sutton, and White 2023) or make the environments more complex. 

I am all for keeping our environments fast and making the agents very simple, but the community generally frowns upon that. People love their big agents running on GPUs. The solution for those people is to use even bigger and more complex environments. 

Spending 99+% of the compute/memory of an experiment on the agent is the bigger issue I am pointing to.

### Benchmarks for Environment Throughput in RL Training

Training **competitive** (near or super-human) RL agents in complex environments requires generating billions of environment interactions (steps/frames). The key metric for feasibility is **throughput**: environment steps-per-second (or frames-per-second, FPS) across parallel instances, as modern RL uses distributed actors to collect experience.

Higher throughput directly reduces **wall-clock training time** for a fixed number of steps (sample complexity). Sample efficiency (performance per step) is separate but complementary—better algorithms reduce required steps, while faster environments reduce time per step.

#### Atari (2D Arcade Games)
- Classic benchmark for human-level performance (e.g., DQN, Rainbow achieve super-human scores).
- Standard ALE emulator: ~several thousand FPS total with parallel environments (e.g., 10k–20k FPS across 100+ instances on CPU).
- Accelerated versions (e.g., CuLE GPU emulator): up to millions of FPS in batched mode.
- Typical training: 200 million frames (~50 million steps, with frame-skip=4).
- Throughput requirement for practical training: **At least 10,000–100,000 steps/second total** (via parallelism) to reach human-level in days on a single machine; original DQN took weeks.

#### Procgen (Procedurally Generated 2D/3D-like Games)
- Designed for generalization and sample efficiency testing (16 diverse games, e.g., CoinRun).
- Engineered for high speed: **Thousands of steps/second per core** (e.g., 5,000–10,000+ SPS on single CPU; vectorized batches reach millions total).
- Training to competitive levels: 200–500 million steps.
- Minimum for competitive agents: **High single-core speed (thousands SPS)** enables efficient benchmarking without heavy parallelism.

#### MineRL (Minecraft-based 3D Environments)
- Complex open-world 3D tasks (e.g., ObtainDiamond); highly sample-inefficient.
- Native speed: ~20 FPS (steps/second) in real-time mode; headless ~6–70 FPS depending on setup.
- Alternatives (e.g., CraftGround simulator): significantly faster (hundreds–thousands FPS).
- Training competitive agents (e.g., VPT models): billions of steps; original MineRL is slow, making high throughput critical.
- Minimum for practical training: **Hundreds of steps/second** (via optimized simulators) to make billion-step training feasible in weeks/months.

#### Other Complex 3D/Procedural Environments
- MuJoCo/DM Control (continuous control, e.g., Humanoid): millions of steps/second in batched/JAX-accelerated versions (e.g., Brax/MJX: 1–10 million SPS on GPU).
- For truly complex 3D games (e.g., VizDoom, DMLab proxies for FPS): often 100–1,000 FPS per instance, requiring heavy parallelism.

**General Minimum for Complex 3D Games**: No strict universal threshold, but for competitive (human-level or better) agents requiring 10^9–10^10 steps:
- **Total throughput: 100,000–1,000,000+ steps/second** (via GPU acceleration or massive CPU parallelism) to achieve training in days–weeks.
- Below ~10,000 total SPS, training becomes impractical (months+ wall-clock).

### Relationship Between Environment Speed and Wall-Clock Training Time

Wall-clock time ≈ (Total required environment steps) / (Effective throughput in steps/second)

- **Direct inverse relationship**: Doubling throughput halves training time, assuming fixed sample complexity.
- Sample complexity (steps to human-level) is algorithm-dependent (e.g., PPO vs. Rainbow) and often 10^8–10^10 for complex tasks.
- Faster environments enable:
  - More algorithm iterations/experimentation.
  - Scaling to harder tasks (e.g., sparse rewards, long horizons).
- Trade-offs: Very fast environments (e.g., Procgen) prioritize benchmarking sample efficiency; slower ones (e.g., photorealistic 3D) better simulate real-world physics but increase time.
- Historical trend: Progress in RL often combines better sample efficiency (fewer steps needed) with higher throughput (faster per step), yielding exponential wall-clock reductions.

In summary, for complex 3D games, competitive agents demand environments supporting **at least hundreds of thousands of steps/second total** to keep wall-clock time reasonable (days–weeks on modern hardware). Slower speeds shift focus to ultra-sample-efficient algorithms or pre-training.

### Key Lightweight Alternatives to Full Minecraft for AI Agent Training

Running the full Minecraft Java Edition (via frameworks like **MineRL**, **MineDojo**, or **Project Malmo**) provides high fidelity but is resource-intensive and slow for large-scale training due to the Java-based engine. Below are prominent alternatives, including the requested ones and recent developments in simplified voxel environments, neural simulators, and world models. These aim to enable faster training while approximating Minecraft's open-ended voxel world.

| Environment/Framework | Fidelity to Real Minecraft | Speed (approx. steps/second) | Transfer to Real Minecraft | Notes |
|------------------------|----------------------------|-----------------------------|----------------------------|-------|
| **MineRL** (2019+, based on Malmo/Minecraft) | High (full Minecraft with RGB observations, keyboard/mouse actions) | Low (~20-100 steps/sec; RL training often takes days/weeks) | Yes (direct; agents trained here run in vanilla Minecraft) | Benchmark with human demonstration dataset; used for imitation learning (e.g., OpenAI's VPT fine-tuned here to craft diamonds). Slow due to Java overhead. |
| **MineDojo** (2022, builds on MineRL) | High (full Minecraft, supports Overworld/Nether/End, thousands of open-ended tasks) | Low (similar to MineRL; simulator noted as slow for RL) | Yes (direct; Voyager agent uses skill library in new worlds) | Open-ended with internet-scale knowledge (YouTube/Wiki); Voyager (LLM-powered agent) achieves strong generalization in new instances. |
| **Project Malmo** (2016+, Microsoft) | High (full Minecraft mod) | Low (Java-based, similar constraints) | Yes (direct) | Early platform; foundation for MineRL/MineDojo. Less active now. |
| **Craftium** (2024-2025, based on Luanti/Minetest engine) | Medium-High (Minecraft-like voxels, procedural generation, destructible worlds; extensible via Lua mods) | Very High (>2000 steps/sec reported) | Partial/Limited (similar mechanics but different engine; behaviors may need adaptation) | Open-source, C++ engine (no Java); Gymnasium/PettingZoo APIs for single/multi-agent RL. Ideal for fast prototyping and scaling. |
| **Crafter** (2022+, 2D simplified crafting env) | Low (2D top-down, basic crafting/survival) | High (thousands of steps/sec possible) | Low/No (too simplified; no 3D/voxels) | Fast open-world survival benchmark; often recommended as stepping stone before Minecraft. |
| **MineWorld** (2025, neural interactive world model) | Medium (generative video model simulating Minecraft dynamics) | High (real-time interaction via diagonal decoding; parallel inference) | Potential via simulation (not direct; for planning/imagination in agents) | Open-source; trained to predict next frames/actions. Enables fast "dreaming" for RL without running game engine. |
| **Dreamer (DeepMind, 2025)** | High in simulation (world model learns full mechanics like crafting/tools from pixels) | High (offline RL in imagined rollouts; scales to complex tasks quickly) | Promising (learns diamond mining from scratch; world model captures Minecraft physics) | Scalable world model agent; trains inside high-fidelity internal simulation without external engine. No human demos needed. |
| **Oasis (Decart, 2024)** | Medium-High (generative AI simulating playable Minecraft from video data) | High (real-time playable demo) | Limited (pure simulation; no direct code transfer) | Next-frame prediction from millions of hours of footage; no underlying game code. |

#### Additional Insights
- **Fidelity vs. Speed Trade-off**: Full Minecraft setups (MineRL/MineDojo) offer exact mechanics (e.g., precise crafting recipes, physics) but are bottlenecked by rendering and Java. Simplified alternatives like Craftium sacrifice minor details (e.g., exact block textures) for orders-of-magnitude faster training.
- **Transferability**: Direct transfer is easiest with full Minecraft-based envs. For proxies/simulators, transfer often involves using them for pre-training or planning (e.g., Dreamer's world model for imagination-based RL). OpenAI's **VPT** (2022) demonstrated strong transfer by pre-training on videos then fine-tuning in MineRL to achieve human-level tasks like diamond pickaxe crafting.
- **Recent Trends (2024-2025)**: Shift toward neural world models (MineWorld, Dreamer) for scalable, engine-free simulation. These allow agents to "imagine" millions of steps internally, accelerating learning while approximating real Minecraft. Multi-agent setups (e.g., TeamCraft benchmark) are emerging for collaboration research.
- **Recommendations**: For maximum speed in voxel-like training, start with Craftium. For high-fidelity open-ended research with transfer, use MineDojo/Voyager. World models like Dreamer represent the cutting edge for avoiding slow engines altogether.

The **state-of-the-art** in learned world models for training game-playing agents without accessing the actual game engine centers on generative models (often diffusion-based) that simulate environment dynamics from pixel observations. These models enable agents to train via "dreamed" (imagined) rollouts in the model's autoregressive predictions, bypassing real-time interaction with the game engine for sample efficiency and safety.

### Key Recent Approaches
Recent works like **DIAMOND** (2024, NeurIPS Spotlight), **GameNGen** (2024), and **Genie** series (2024–2025) from Google DeepMind represent the frontier. These build on earlier ideas (e.g., World Models from 2018, Dreamer series) but achieve higher fidelity and interactivity using diffusion or transformer-based generation.

- **DIAMOND** (Diffusion As a Model Of eNvironment Dreams)  
  Uses a diffusion model to generate next frames conditioned on past frames and actions. It trains RL agents (e.g., actor-critic) entirely within imagined trajectories. Achieves state-of-the-art on Atari 100k benchmark (mean human-normalized score 1.46, surpassing prior world-model agents like STORM/IRIS). Also creates a playable neural simulator for Counter-Strike: Global Offensive (trained on static gameplay data).

- **GameNGen** (2024, Google Research)  
  A diffusion-based engine that fully simulates Doom (including dynamics like health, items, enemies) without the original engine. Trained in two phases: RL agent collects data by playing Doom, then diffusion model learns to predict frames from actions/past frames. Runs real-time interactive simulation at ~20 FPS on a single TPU.

- **Genie** series (DeepMind)  
  Foundation world models trained unsupervised on internet videos.  
  - Genie (2024): Generates 2D platformer-like environments from videos, supports frame-by-frame interaction via latent actions.  
  - Genie 2 (late 2024): Creates diverse 3D worlds from a single image prompt, interactive for up to ~1 minute, with emergent physics and object interactions.  
  - Genie 3 (2025): Text-to-interactive 3D worlds at 720p/24 FPS, consistent for minutes, with promptable events and real-time navigation.

Other notable models include Sora (OpenAI, 2024–2025), which simulates processes like Minecraft gameplay zero-shot but focuses more on video generation than agent training. Dreamer 4 (recent) scales latent world models for offline long-horizon tasks like Minecraft diamond mining.

### Resolution and Temporal Fidelity
These models vary in visual quality and rollout length:

| Model       | Resolution                  | Frame Rate / Temporal Fidelity                  | Rollout Duration / Consistency                  |
|-------------|-----------------------------|-------------------------------------------------|-------------------------------------------------|
| DIAMOND    | Native Atari (~160x210) or CS:GO screenshots | Autoregressive (1–few diffusion steps per frame)| Long trajectories (stable for Atari/CS:GO play) |
| GameNGen   | Native Doom (320x200)      | ~20 FPS real-time interactive                  | Long (full levels, but short-term memory limits)|
| Genie 2    | Variable (high-quality 3D)| Autoregressive generation                      | Up to 1 minute consistent                      |
| Genie 3    | 720p HD                    | 24 FPS real-time                               | Several minutes consistent                     |

Diffusion models like DIAMOND/GameNGen excel at high visual detail (preserving subtle cues critical for RL), while transformer-based (Genie) scale to higher resolutions and diversity.

### Transfer of Agents Trained in Dreamed Rollouts to Real Environments
Yes, transfer is possible and a key advantage, though fidelity gaps can limit it:

- In DIAMOND, agents train purely in the diffusion model but evaluate (and often deploy) directly in the real Atari/CS:GO environments, achieving superhuman scores on Atari 100k.
- GameNGen simulations are indistinguishable in short clips from real Doom; human players interact seamlessly.
- Classic examples (e.g., original World Models, Dreamer) show policies trained in imagination transfer well to real envs.
- Limitations: Model inaccuracies (e.g., drifting physics over long horizons) cause compounding errors, reducing perfect transfer. Techniques like noise augmentation (GameNGen) or multi-step sampling improve robustness.
- Overall, high-fidelity models (2024–2025) enable strong zero-shot or near-zero-shot transfer, especially for visually detailed games, outperforming prior latent-based world models.

These advances point toward engine-less game simulation and scalable agent training, with diffusion proving particularly effective for preserving visual details essential to RL decision-making.

### Continuous vs. Discrete Action Spaces

Minecraft's native controls are **hybrid**: discrete keyboard inputs (e.g., move forward, jump, attack) combined with continuous mouse movements for camera control (pitch/yaw deltas). This makes pure continuous spaces challenging for RL due to high-dimensional exploration and precision requirements (e.g., exact aiming).

- **Discrete actions** are predominant in successful Minecraft RL setups. Most benchmarks, including **MineRL** (the primary competition for sample-efficient RL in Minecraft) and **DreamerV3** (the first agent to obtain diamonds from scratch without human data), use fully discretized or categorical action spaces. Discretization simplifies exploration, enables algorithms like PPO or world-model-based methods, and improves stability. For camera control, common approaches include binning mouse deltas (e.g., small/medium/large turns) or using "tank-like" controls.

- **Continuous actions** are harder to learn, often preventing convergence in complex environments like Minecraft (e.g., precise mouse aiming or simultaneous movements). Studies on action space shaping show continuous controls are sensitive to implementation and generally underperform discrete variants unless heavily tuned. Hybrid (discrete + continuous parameters) can work but adds complexity without clear benefits in Minecraft.

**Conclusion**: Discrete (or carefully discretized hybrid) spaces are far more effective for sample efficiency in Minecraft RL.

### Hierarchical vs. Flat Action Spaces

Minecraft tasks are long-horizon with sparse rewards (e.g., obtaining a diamond requires chaining dozens of sub-tasks like tree punching, crafting tools, mining).

- **Flat policies** (single-level RL over primitive actions) struggle with credit assignment and exploration over thousands of steps. Pure flat RL rarely solves advanced tasks like ObtainDiamond without massive samples or priors.

- **Hierarchical RL (HRL)** decomposes the problem: a high-level policy selects sub-goals or **options** (temporally extended actions), while low-level policies execute them. This reduces the effective horizon, improves exploration, and enables skill reuse. Key examples:
  - **Options framework**: High-level selects reusable skills (e.g., "navigate to tree", "mine block"); low-level executes primitives.
  - Successful agents: JueWu-MC (hierarchical with imitation-guided options), H-DRLN (deep skill arrays for lifelong learning), ForgER (hierarchical with forgetful replay).

HRL consistently outperforms flat approaches in Minecraft, enabling transfer across tasks (e.g., navigation skills reused for mining).

**Conclusion**: Hierarchical is strongly preferred for complex, open-world tasks in Minecraft.

### Primitive Actions vs. Learned Skills/Options

- **Primitive actions**: Low-level (e.g., move forward, turn camera, use item). Sufficient for basic tasks but sample-inefficient for long-horizon goals due to poor exploration and no temporal abstraction.

- **Learned skills/options**: Temporally extended macro-actions discovered or refined via RL (e.g., "chop tree until wood obtained", "follow compass"). These act as building blocks:
  - Enable composition for complex behaviors.
  - Improve sample efficiency by operating at higher abstraction levels.
  - Facilitate transfer: Skills learned on one task (e.g., navigation) accelerate others (e.g., resource gathering).
  - Examples: Deep Skill Networks (DSNs) in lifelong Minecraft agents; skill priors in SPiRL-like methods; option-critic architectures.

Learned skills outperform primitives in transfer and efficiency, especially when combined with hierarchies (e.g., skill distillation to compress multiple options into one network).

**Conclusion**: Learned skills/options are superior for sample efficiency and zero-shot/few-shot transfer in Minecraft.

### Most Effective Action Abstractions for Sample Efficiency and Transfer

Proven abstractions balance expressiveness with learnability:

- **Craft-item macros**: Abstract crafting (e.g., one action to "craft wooden pickaxe" instead of placing items in grid). Used in MineRL v0.4.4 and DreamerV3—dramatically boosts efficiency by skipping repetitive micro-steps. Essential for tech-tree progression (e.g., obtaining diamond).

- **Navigation primitives**: Higher-level movement (e.g., "go to waypoint", "follow compass", pathfinding sub-policies). Reusable across biomes/tasks; critical for exploration in procedurally generated worlds.

- **Mining/breaking acceleration**: Auto-hold attack until block breaks (common modification in benchmarks). Prevents learning to hold buttons for hundreds of steps.

- **Other effective ones**:
  - Block placement/use macros.
  - Inventory management skills.
  - Combat/evasion options.

These abstractions (often hand-designed or semi-automated) enable agents like DreamerV3 to solve ObtainDiamond in ~50 episodes and support transfer (e.g., skills from navigation tasks aid mining). Pure primitives fail; full API-level macros (e.g., Voyager) excel but reduce low-level challenge.

**Overall Recommendation**: Use a **hierarchical discrete action space** with **learned options** over primitives, incorporating **crafting macros** and **navigation skills**. This maximizes sample efficiency (e.g., diamonds from scratch) and transfer across Minecraft's open-ended tasks.

**For Minecraft agents, particularly in benchmarks like MineRL, the most effective observation resolutions and modality combinations depend on the agent type (e.g., pure RL vs. LLM-driven) and task complexity (e.g., short-horizon atomic tasks like tree chopping vs. open-ended long-horizon exploration).**

### Key Modalities and Comparisons
| Modality | Description | Pros | Cons | Best Use Cases & Performance Notes |
|----------|-------------|------|------|----------------------------|
| **Raw Pixels** | First-person RGB POV (includes hotbar, health, GUI). Standard resolutions: 64×64×3 (MineRL default, downscaled/grayscale for efficiency), 128×128×3 (VPT), up to 640×360×3 (Optimus-2, full Minecraft render). | Captures full visual richness (textures, distances, animations). Essential for low-level control (e.g., aiming attacks). | High-dimensional; prone to aliasing (similar-looking blocks). Compute-heavy at high res. | Atomic/short tasks: VPT at 128×128 crafts tools (e.g., iron pickaxe at 80% reliability post-RL fine-tune). Long-horizon: Optimus-2 at 640×360 hits 99% success on wood/stone tasks, 13% on diamonds. |
| **Segmentation Masks** | Object/instance masks for key entities (e.g., trees, ores). Often added to pixels (e.g., via foundation models like Cutie). | Reduces aliasing; focuses on relevant objects. Improves RL sample efficiency. | Requires annotation/preprocessing; not standard in benchmarks. | Object-centric RL: Boosts performance in sparse-reward tasks by extracting features (e.g., K×2048 vectors at 64×64 pixels). Rarely standalone; best as pixel augment. |
| **Inventory State** | Dict of item counts (e.g., {'log': 5, 'stone': 3}). Always low-dim vector. | Critical for planning (e.g., know if axe needed). Cheap to process. | Doesn't capture environment. | Pairs with pixels: MineRL agents using pixels + inventory outperform pixels alone (e.g., better climbing via dirt count). Voyager uses it as core for tech-tree mastery (3.3× more items discovered). |
| **Text Descriptions** | Parsed text: nearby blocks/entities, biome, position, health/hunger (e.g., Voyager frontend; LLM-generated captions). | Semantic abstraction; enables LLM reasoning. Zero-shot generalization. | Loses fine-grained visuals (e.g., can't aim precisely). | Open-ended/LLM agents: Voyager (text-only) unlocks 15.3× faster milestones vs. baselines; no pixels needed. MineDojo adds voxels/text for multi-task. |

**Top Combinations:**
1. **Pixels (64-128) + Inventory + Compass**: Best for classic RL in MineRL (e.g., Navigate/Treechop). Compass (angle vector) aids goal direction; boosts success 2-5× in sparse tasks.
2. **High-Res Pixels (640×360) + Text Goals**: SOTA for multimodal (Optimus-2/STEVE): 27% gain on atomic, 10% on long-horizon vs. priors.
3. **Text + Inventory (No Pixels)**: Best for LLM/open-world (Voyager): Excels in exploration/tech tree; 2.3× longer distances.
4. **Pixels + Masks + Inventory**: Emerging for efficiency (e.g., object-centric); aids RL in aliased visuals.

### Minimum Visual Resolution
No explicit "failure threshold" in literature, but agents reliably learn at **64×64** (MineRL standard; distinguishes blocks/textures). Below ~32×32, severe aliasing occurs: Agents fail to differentiate block types (e.g., log vs. stone), leading to exploration collapse and 0% success on visual tasks like Treechop. VPT succeeds at 128×128 for precise control (e.g., mining); 64×64 suffices for coarser tasks but limits fine motor skills.

**How to Determine:** Train DQN/PPO ablating res (e.g., via code_execution if needed). Failure mode: Reward plateaus near random due to indistinguishable states.

### Compute-Performance Tradeoff
| Resolution | Compute (Rel. FLOPs for CNN/Transformer) | Performance Gain | Notes |
|------------|------------------------------------------|------------------|-------|
| **32×32** | Low (~1×) | Poor (basic navigation only) | Fast prototyping; fails detailed tasks. |
| **64×64** | Medium (~4×) | Good (MineRL baseline) | Efficient RL; 80% of high-res perf at 25% compute. |
| **128×128** | High (~16×) | Strong (VPT tools/crafting) | Better aliasing resistance; RL fine-tune needed. |
| **640×360** | Very High (~100×+) | SOTA (Optimus) | Transformers/MLLMs handle; inference 2.3× slower for larger models. |

- **Tradeoff Rule:** 64×64 optimal for RL (balances speed/accuracy); scale to 128+ with imitation pretraining (VPT). Text/hybrid cuts compute 10-50× for open tasks.
- **Mitigations:** Grayscale (2× faster), patches, or masks reduce effective dim without full loss.

### Design Principles for High-Throughput RL Training Environments

Recent discussions (primarily from 2024–2025 papers and frameworks) emphasize shifting RL pipelines toward **maximum simulation throughput** to support sample-hungry modern algorithms like PPO, SAC, and their variants. The core goal is to minimize wall-clock training time by generating millions of environment steps per second, enabling rapid iteration and scaling to complex tasks (e.g., robotics, reasoning agents).

Key principles include:

1. **Vectorized (Batch) Environments**:
   - Run thousands of environment instances in parallel (vectorized) to amortize computation.
   - Synchronous vectorization (e.g., classic Gym vector envs) is simple but limited by slowest env; asynchronous variants allow better utilization.
   - Hierarchical parallelism: parallel envs → parallel agents per env → parallel training steps (e.g., EvoRL framework, 2025).

2. **GPU-Accelerated Simulators**:
   - Move physics/simulation to GPU for massive parallelism (thousands–tens of thousands of envs on a single GPU).
   - Examples:
     - **Isaac Gym** (NVIDIA): GPU-based PhysX engine; achieves 2–3 orders of magnitude speedup over CPU pipelines, enabling thousands of parallel robotic envs.
     - **Brax** (Google, JAX-based): Differentiable physics on GPU/TPU; excels in batched continuous control but lower per-env speed for small batches.
     - These allow end-to-end GPU pipelines (simulation + policy inference + gradients), reducing CPU-GPU transfers.
   - Trade-off: Domain-specific (often rigid-body physics); less general than CPU-based.

3. **Async Actor-Learner Architectures**:
   - Decouple rollout collection (actors) from learning (learner) to overlap computation and hide latencies.
   - Actors generate trajectories with potentially stale policies; learner updates centrally.
   - Corrections for off-policy data: V-trace (IMPALA), staleness-aware objectives, or bounded asynchronous ratios.
   - Recent extensions (2025): Fine-grained parallelism + rollout-train decoupling (ROLL Flash); modular async modules for GUI/agentic RL (DART); full async for language reasoning (AReaL).
   - Benefits: Higher GPU utilization, scalability to clusters; handles stochastic/long-horizon envs (e.g., LLMs, agents).

4. **Hybrid/Decoupled Systems**:
   - Full end-to-end GPU (e.g., EvoRL for evolutionary RL).
   - CPU parallelism for general envs (e.g., EnvPool: C++-based, NUMA-optimized).
   - Avoid bottlenecks: Minimize data transfers, use redundant rollouts for reliability.

These principles are driven by needs in robotics (Isaac Gym/Brax), games/Atari/MuJoCo (EnvPool), and emerging agentic/LLM reasoning (async systems like ROLL Flash, AReaL).

### Steps-per-Second Benchmarks and Targets for Custom Environments

Benchmarks vary by hardware, task, and parallelism level. High-end setups (e.g., DGX-A100 with 256 CPU cores or high-end GPU):

| Simulator/Engine       | Environment Type | Peak Throughput (steps/frames per second) | Notes |
|------------------------|------------------|-------------------------------------------|-------|
| **EnvPool** (CPU-based) | Atari           | ~1 million                               | General; 3–15x faster than Gym subprocess on modest/high-end hardware. |
| **EnvPool** (CPU-based) | MuJoCo          | ~3 million                               | Highly parallel C++; compatible with many libraries. |
| **Isaac Gym** (GPU)    | Robotics/MuJoCo-like | Thousands of envs → millions overall    | 100–1000x speedup vs. CPU; e.g., trains complex tasks in minutes on single GPU. |
| **Brax** (GPU/JAX)     | Continuous physics | High batched (millions aggregate)        | Strong for large batches; single-env slower than CPU. |
| **IMPALA/SEED-style**  | General         | ~250,000                                 | Older distributed; modern async variants higher. |

For viability with **modern RL algorithms** (e.g., PPO, SAC in 2024–2025 setups):

- **Minimum target**: 10,000–100,000 steps/second overall (across parallel envs) on typical hardware (e.g., consumer GPU + multicore CPU). This supports efficient training without env becoming the bottleneck.
- **Recommended for state-of-the-art**: 500,000–1 million+ steps/second.
  - PPO often uses 1,024–4,096 parallel envs; needs fast simulation to keep policy network busy.
  - Off-policy (SAC) tolerates slower envs but benefits from high throughput for exploration.
  - Custom envs should aim for **at least 3–10x speedup over naive Gym** (via vectorization or GPU) to enable training complex policies in hours rather than days/weeks.
- If your custom env runs <1,000 steps/second aggregated, it risks being too slow for iterative research with deep networks.

To achieve this: Start with vectorized Gym wrappers; port to EnvPool for CPU speed; use Brax/Isaac Gym if physics-heavy. For async scaling, integrate with frameworks like RLlib or custom actor-learner setups.

**Handling the Sparse Reward Problem in Minecraft Long-Horizon Tasks**

The "ObtainDiamond" task in Minecraft (from the MineRL benchmark) exemplifies sparse rewards: agents start empty-handed in a procedurally generated world and must execute thousands of steps—punching trees for logs, crafting tools (wooden → stone → iron pickaxe), building a furnace, smelting ore, strip-mining deep caves—to reach diamonds, which spawn rarely at Y<16. Success requires exploration, tool progression, hazard avoidance (lava, falls, mobs), and credit assignment over 15-30 minute episodes. Pure random exploration fails; sample budgets are tight (e.g., 8M steps in competitions).

**MineRL's Built-in Reward Shaping:** The benchmark provides **potential-based milestone rewards** (additive, non-decreasing): +1 (log), +2 (planks), +4 (sticks), +8 (crafting table), +16 (wooden pickaxe), +32 (cobblestone), +64 (stone pickaxe), +128 (iron ore), +256 (furnace), +512 (iron ingot), +1024 (iron pickaxe), +2048 (diamond). This dense shaping guides progress without policy distortion, but early stages saturate quickly, leaving late-game sparsity.

**Most Effective Strategies:**

1. **Hierarchical Reinforcement Learning (HRL):** Decomposes into 5-12 subtasks (e.g., ChopTree, CraftPickaxe, MineIron, SearchDiamond). Train low-level policies per subtask via imitation learning (IL) from human demos + RL fine-tuning; high-level meta-policy selects/sequences via IL or RL. Handles sparsity by shortening horizons per subtask and bootstrapping with demos. 
   - **JueWu-MC (MineRL 2021 Winner):** Auto-extracts subtasks from demos; uses A2RL (action-aware reps) + DSIL (self-imitation) for gathering, EBC (ensemble BC) for crafting. Score: 76.97/100 (2nd track, unlimited methods); 100% conditional success on early subtasks.
   - **SEIHAI (2020 Winner):** 5 subtasks; LarMI/SQIL for mining, BC for crafting. Score: 39.55.

2. **World Model-Based RL (e.g., DreamerV3):** Learns a dynamics model from pixels/actions, simulates trajectories ("imagination") for planning/actor-critic training. Excels on sparse rewards via latent exploration and farsighted value estimation—no extra shaping/curriculum needed. 
   - **Performance:** First diamond ~30M steps (17 in-game days); all 40 seeds succeed within 100M steps; masters full tech tree (e.g., iron pickaxe near-100%). Outperforms IMPALA/Rainbow (stop at iron) and VPT (2.5% diamond rate w/ massive data/compute).

3. **LLM-Augmented Methods:**
   - **Auto MC-Reward:** LLMs iteratively design **dense rewards** (Reward Designer codes Python funcs; Critic verifies; Analyzer refines via failures). E.g., + for ore proximity/tree approach, - for lava. Boosts PPO/A2C success 2-5x on diamond subtasks/exploration.
   - **Plan4MC:** Learns primitive skills (e.g., "Finding" for object search) with intrinsic rewards; LLM builds skill graph for planning sequences (>10 skills). Solves 40 tech-tree tasks (incl. diamond-adjacent); most sample-efficient demo-free RL.

**Curriculum Learning Strategies:**
- **Reverse/Top-Down HRL:** Train from goal backward (e.g., "mine diamond assuming iron pickaxe" → full task). Simplifies credit assignment; pairs w/ IL.
- **Automatic/Exploration-Driven:** Voyager's curriculum proposes novel goals via LLM (maximizes item/distance); not RL but composes code-skills. DreamerV3 implicitly curricula via model rollouts.

**Useful Intermediate Objectives & Intrinsic Signals:**
| Category | Examples | Benefits |
|----------|----------|----------|
| **Milestones** | Item acquisition (log → diamond) | Guides tech tree; prevents local optima. |
| **Dense Shaping (LLM/Manual)** | Proximity to ore/trees, -lava/death, +crafting progress | 2-10x faster convergence; handles hazards. |
| **Intrinsic (Curiosity/Count)** | Prediction error in world models; visit-novelty for "Finding" skills | Fuels exploration in open worlds; bootstraps primitives. |
| **Subtask Goals** | HRL options (e.g., "gather cobble") | Shortens horizons; composable. |

**Comparison of Top Methods:**
| Method | Core Technique | Diamond Success | Human Data? | Steps/Compute |
|--------|----------------|---------------|-------------|---------------|
| DreamerV3 | World Model RL | Yes (~30M steps first; 100% in 100M) | No | 100M env steps |
| JueWu-MC | HRL + IL | 77% score (equiv. high) | Yes | <8M |
| Voyager | LLM Skills/Curric. | Yes (tech tree) | LLM only | Hours |
| Plan4MC | Intrinsic Skills + LLM Plan | Tech tree (40 tasks) | No | Efficient |
| Auto MC-Reward + PPO | LLM Dense Rewards | Subtasks: 2-5x boost | No | Faster conv. |

**Key Insights:** HRL+IL wins competitions (data helps); DreamerV3 generalizes best demo-free. LLMs automate shaping/curricula, scaling to open-world. Future: Combine world models + LLM planning for ultimate efficiency.

In Minecraft reinforcement learning (RL), agent policies are typically trained directly in the full Minecraft game environment using platforms like **MineRL** or **MineDojo**, which run the actual Java Edition game engine. There are no "lightweight simulators" distinct from the full game; training occurs in the real Minecraft simulation itself, ensuring perfect fidelity by design. This avoids traditional sim-to-real transfer issues seen in robotics, where policies trained in simplified simulators must adapt to physical hardware.

Key examples include:
- **Video PreTraining (VPT)**: Agents are pretrained on unlabeled human gameplay videos, then fine-tuned via behavioral cloning or RL directly in Minecraft, achieving tasks like crafting diamond tools.
- **Dreamer 4**: A world model agent trained offline on video data, then performing RL in imagination with high-fidelity simulation of Minecraft dynamics, enabling real-time performance and diamond acquisition from offline data alone.

Since training and deployment use the same environment, no explicit transfer from a lower-fidelity simulator is required.

### Studies on Minimal World Model Accuracy
Research focuses on learned world models rather than simplified rule-based simulators:
- **Dreamer 4** demonstrates that high-fidelity world models (using block-causal transformers) enable effective RL entirely in imagination, achieving state-of-the-art Minecraft performance (e.g., diamond retrieval) with minimal labeled data and real-time inference.
- No direct studies quantify a "minimal" accuracy threshold, but works like VPT and MineDojo emphasize that accurate replication of Minecraft's visuals, physics, and interactions (via the full engine) is crucial for complex, long-horizon tasks. Lower-fidelity approximations (e.g., grid-world proxies) fail to capture open-ended behaviors.

### Distributional Shifts and Agent Performance
Distributional shifts in Minecraft RL arise from:
- **Procedural world generation** — Each episode starts in a new, randomly generated world (different biomes, resource locations, terrains).
- **Task variations** — Shifts in goals, inventories, or subtasks.
- **Adversarial or out-of-distribution elements** — Rare events, novelties, or interventions.

These shifts significantly degrade performance if not addressed:
- Agents overfit to common training distributions (e.g., frequent biomes or resource spawns), leading to poor generalization in rare scenarios.
- Studies show robustness techniques mitigate this:
  - Domain randomization (varying dynamics or visuals during training) improves transfer, though less common in Minecraft due to the fixed engine.
  - Hierarchical RL, meta-RL, or data diversification (e.g., large video datasets in VPT) help agents adapt.
  - In multi-agent setups or benchmarks like TeamCraft, shifts from agent switching or environmental changes cause drops in coordination or success rates.
- Goal misgeneralization examples (e.g., VPT agents confusing villagers for trees under specific visual conditions) highlight how subtle shifts lead to catastrophic failures.

Overall, Minecraft's inherent variability makes distributional robustness a core challenge, addressed primarily through diverse training data, imitation from humans, and model-based methods rather than fidelity reductions.