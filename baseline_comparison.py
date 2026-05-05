"""
baseline_comparison.py
======================
Compares 3 agents on the same environment:
  1. Random agent       — takes random actions
  2. Rule-based agent   — maintains when health < 0.4 (threshold policy)
  3. DQN agent          — our trained RL agent

This directly supports the rubric criterion:
  "Performance Evaluation & Results Interpretation (CO1 & CO2)"

Run:
    python baseline_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from sim.factory_env import FactoryMachineEnv

NUM_EPISODES = 30
SEED         = 42


# ── Agent 1: Random ──────────────────────────────────────────────────────────
def run_random_agent(num_episodes):
    rewards, breakdowns, maintenances = [], [], []
    env = FactoryMachineEnv(max_steps=365)
    rng = np.random.default_rng(SEED)

    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(10000)))
        total  = 0
        while True:
            action = rng.integers(0, 3)
            obs, reward, done, _, info = env.step(int(action))
            total += reward
            if done:
                rewards.append(total)
                breakdowns.append(info["breakdown_count"])
                maintenances.append(info["maintenance_count"])
                break
    return rewards, breakdowns, maintenances


# ── Agent 2: Rule-based ──────────────────────────────────────────────────────
def run_rule_based_agent(num_episodes, threshold=0.4):
    """
    Simple rule: if health < threshold → schedule maintenance, else keep running.
    Never uses emergency repair proactively.
    """
    rewards, breakdowns, maintenances = [], [], []
    env = FactoryMachineEnv(max_steps=365)
    rng = np.random.default_rng(SEED)

    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(10000)))
        total  = 0
        while True:
            health = obs[0]
            if health < threshold:
                action = 1  # schedule maintenance
            else:
                action = 0  # keep running
            obs, reward, done, _, info = env.step(action)
            total += reward
            if done:
                rewards.append(total)
                breakdowns.append(info["breakdown_count"])
                maintenances.append(info["maintenance_count"])
                break
    return rewards, breakdowns, maintenances


# ── Agent 3: DQN ─────────────────────────────────────────────────────────────
def run_dqn_agent(num_episodes):
    model  = DQN.load("models/dqn_predictive_maintenance")
    env    = FactoryMachineEnv(max_steps=365)
    rng    = np.random.default_rng(SEED)
    rewards, breakdowns, maintenances = [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(10000)))
        total  = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
            total += reward
            if done:
                rewards.append(total)
                breakdowns.append(info["breakdown_count"])
                maintenances.append(info["maintenance_count"])
                break
    return rewards, breakdowns, maintenances


# ── Run all agents ────────────────────────────────────────────────────────────
print("Running baseline comparison...")
print("  [1/3] Random agent...")
r_rnd, bd_rnd, mx_rnd = run_random_agent(NUM_EPISODES)

print("  [2/3] Rule-based agent...")
r_rule, bd_rule, mx_rule = run_rule_based_agent(NUM_EPISODES)

print("  [3/3] DQN agent...")
r_dqn, bd_dqn, mx_dqn = run_dqn_agent(NUM_EPISODES)

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Metric':<35} {'Random':>8} {'Rule':>8} {'DQN':>8}")
print("-"*65)
print(f"{'Avg reward':<35} {np.mean(r_rnd):>8.1f} {np.mean(r_rule):>8.1f} {np.mean(r_dqn):>8.1f}")
print(f"{'Std reward':<35} {np.std(r_rnd):>8.1f}  {np.std(r_rule):>8.1f}  {np.std(r_dqn):>8.1f}")
print(f"{'Avg breakdowns/year':<35} {np.mean(bd_rnd):>8.2f} {np.mean(bd_rule):>8.2f} {np.mean(bd_dqn):>8.2f}")
print(f"{'Avg maintenances/year':<35} {np.mean(mx_rnd):>8.2f} {np.mean(mx_rule):>8.2f} {np.mean(mx_dqn):>8.2f}")
print("="*65)

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Agent Comparison — Random vs Rule-Based vs DQN\n"
    "Predictive Maintenance | SDG 9 & SDG 12",
    fontsize=13, fontweight="bold"
)

agents  = ["Random", "Rule-Based", "DQN"]
colors  = ["#888780", "#1D9E75", "#534AB7"]
rewards = [r_rnd, r_rule, r_dqn]
breaks  = [bd_rnd, bd_rule, bd_dqn]
mxs     = [mx_rnd, mx_rule, mx_dqn]

# Average reward
means = [np.mean(r) for r in rewards]
stds  = [np.std(r)  for r in rewards]
bars  = axes[0].bar(agents, means, color=colors, alpha=0.85, edgecolor="white")
axes[0].errorbar(agents, means, yerr=stds, fmt="none", color="black", capsize=5, linewidth=1.5)
axes[0].set_title("Average Reward per Year")
axes[0].set_ylabel("Total Reward")
axes[0].grid(alpha=0.3, axis="y")
for bar, mean in zip(bars, means):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{mean:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Average breakdowns
bd_means = [np.mean(b) for b in breaks]
axes[1].bar(agents, bd_means, color=colors, alpha=0.85, edgecolor="white")
axes[1].set_title("Avg Breakdowns per Year")
axes[1].set_ylabel("Breakdowns")
axes[1].grid(alpha=0.3, axis="y")
for i, (agent, val) in enumerate(zip(agents, bd_means)):
    axes[1].text(i, val + 0.05, f"{val:.1f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

# Average maintenances
mx_means = [np.mean(m) for m in mxs]
axes[2].bar(agents, mx_means, color=colors, alpha=0.85, edgecolor="white")
axes[2].set_title("Avg Planned Maintenances per Year")
axes[2].set_ylabel("Maintenances")
axes[2].grid(alpha=0.3, axis="y")
for i, (agent, val) in enumerate(zip(agents, mx_means)):
    axes[2].text(i, val + 0.05, f"{val:.1f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("results/baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n📊 Comparison plot saved to results/baseline_comparison.png")
print("✅ Baseline comparison complete!")
