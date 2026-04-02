# CLAUDE.md — Project Guide

## Project Overview

**Reinforcement Learning: DQN vs PPO on LunarLander-v3**

A portfolio project demonstrating reinforcement learning competency by implementing, training, and comparing two foundational RL algorithms — **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** — on the Gymnasium LunarLander-v3 environment.

This project showcases:
- From-scratch implementation of both DQN and PPO (no RL libraries like Stable-Baselines)
- Neural network design with PyTorch
- Hyperparameter tuning and training analysis
- Rigorous evaluation methodology (separate train/eval, 100-episode greedy evaluation)
- Rich visualization and storytelling (training progression, algorithm comparison)

## Repository Structure

```
.
├── CLAUDE.md              ← You are here
├── README.md              ← Public-facing documentation with embedded visuals
├── .gitignore
├── notebooks/
│   ├── lunar_lander_dqn_v1.py      # v1: DQN only (Databricks notebook format)
│   ├── lunar_lander_dqn_v1.ipynb   # v1: Jupyter version with cell outputs
│   └── lunar_lander_dqn_v2.py      # v2: DQN + PPO comparison (final version)
├── models/
│   ├── best_agent.pt               # DQN — best 100-episode average during training
│   ├── solved_agent.pt             # DQN — checkpoint when env was first solved
│   └── ppo_best_agent.pt           # PPO — best 100-episode average
├── assets/                          # All training visualizations (URL-friendly names)
│   ├── trained-dqn-agent.gif       # Demo: trained agent landing
│   ├── training-progression.gif    # Demo: random → trained evolution
│   ├── dqn-vs-ppo-rewards.png      # Comparison: smoothed reward curves
│   ├── dqn-vs-ppo-eval-distribution.png
│   ├── dqn-vs-ppo-episode-length.png
│   ├── training-rewards.png        # DQN training reward curve
│   ├── epsilon-decay.png           # Exploration schedule
│   ├── episode-length.png          # Steps per episode over training
│   ├── training-loss.png           # MSE loss curve
│   ├── rolling-success-rate.png    # % of last 100 eps with reward >= 200
│   ├── reward-distribution.png     # Histogram of all training rewards
│   ├── random-baseline.png         # Random agent reward distribution
│   └── random-vs-trained.png       # Overlay: random vs trained distributions
└── metrics.json                    # Full training config, results, per-episode data
```

## Notebook Format

The `.py` files use **Databricks notebook source format**:
- `# COMMAND ----------` separates cells
- `# MAGIC %md` prefix indicates markdown cells
- `# MAGIC %pip` prefix indicates pip install cells
- These can be imported directly into any Databricks workspace
- For local execution: strip `# MAGIC` prefixes and `# COMMAND ----------` delimiters

The `.ipynb` file is a standard Jupyter notebook with rendered outputs.

## Key Technical Decisions

### Why DQN for this environment?
- State space is only 8 continuous floats (no CNN needed)
- Action space is 4 discrete actions
- DQN is the natural fit: simple, sample-efficient, fast convergence on low-dimensional discrete problems

### Why also implement PPO?
- Demonstrates versatility across algorithm families (value-based vs policy-based)
- Side-by-side comparison on the same environment is a strong portfolio differentiator
- PPO is the industry standard for more complex problems — showing both covers the RL fundamentals

### Architecture choices
- **DQN**: 2-layer MLP (128×128), experience replay (100K buffer), target network (hard update every 10 episodes), epsilon-greedy with 0.995 decay, gradient clipping at 1.0
- **PPO**: Shared 2-layer MLP (128×128) with separate actor/critic heads, GAE (λ=0.95), clipped surrogate (ε=0.2), entropy bonus (0.01), 4 update epochs per rollout

### Training results
- **DQN**: Solved at episode 708, best avg reward 260.2, eval success rate 80%
- Both algorithms trained for 1000 episodes each
- All metrics saved in `metrics.json` for reproducibility

## Model Checkpoint Format

All `.pt` files are PyTorch `torch.save()` dictionaries:

**DQN checkpoints** contain:
```python
{
    "q_network": state_dict,       # Main Q-network weights
    "target_network": state_dict,  # Target network weights
    "optimizer": state_dict,       # Adam optimizer state
    "epsilon": float,              # Current exploration rate
    "episode_count": int,          # Training episodes completed
    "training_step": int,          # Total gradient steps
}
```

**PPO checkpoints** contain:
```python
{
    "network": state_dict,         # Actor-Critic network weights
    "optimizer": state_dict,       # Adam optimizer state
    "episode_count": int,          # Training episodes completed
}
```

## Dependencies

```
torch
gymnasium[box2d]
plotly
imageio[ffmpeg]
pandas
numpy
```

## Conventions

- Seed: 42 (for reproducibility)
- Device: auto-detected (CUDA if available, else CPU)
- Visualization: Plotly Express with `plotly_white` template
- Logging: every 50 episodes during training
- Evaluation: 100 episodes, fully greedy (no exploration)
- Environment solved: average reward >= 200 over 100 consecutive episodes

## What This Project Demonstrates

| Skill | Evidence |
|-------|----------|
| **RL fundamentals** | Correct implementation of Bellman equation, TD targets, policy gradients |
| **PyTorch proficiency** | Custom nn.Module classes, gradient clipping, checkpoint save/load |
| **Algorithm design** | Experience replay, target networks, GAE, clipped surrogate objective |
| **Experiment methodology** | Random baseline, train/eval split, 100-episode evaluation, metrics logging |
| **Data visualization** | 13 Plotly charts covering every aspect of the training process |
| **Software engineering** | Clean class design, modular code, reproducible seeds, checkpoint management |
| **Technical communication** | Detailed markdown documentation in notebooks explaining every decision |
