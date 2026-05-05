"""
evaluate.py
===========
Loads the trained DQN model and evaluates it over multiple episodes.
Generates performance plots and prints a summary report.

Run:
    python evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import DQN

from sim.factory_env import FactoryMachineEnv

# ── Load model and environment ───────────────────────────────────────────────
print("\n" + "="*60)
print("  Evaluating Trained DQN Agent")
print("="*60)

model = DQN.load("models/dqn_predictive_maintenance")
env   = FactoryMachineEnv(max_steps=365)

# ── Run evaluation episodes ──────────────────────────────────────────────────
NUM_EPISODES = 20
ACTION_NAMES = {0: "Run", 1: "Maintenance", 2: "Emergency"}
ACTION_COLORS= {0: "#5B8BD0", 1: "#1D9E75", 2: "#E8593C"}

all_rewards         = []
all_breakdowns      = []
all_maintenances    = []
all_health_traces   = []
all_action_traces   = []

for ep in range(NUM_EPISODES):
    obs, _        = env.reset()
    total_reward  = 0
    health_trace  = []
    action_trace  = []

    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))

        total_reward += reward
        health_trace.append(info["health"])
        action_trace.append(int(action))

        if done:
            all_rewards.append(total_reward)
            all_breakdowns.append(info["breakdown_count"])
            all_maintenances.append(info["maintenance_count"])
            all_health_traces.append(health_trace)
            all_action_traces.append(action_trace)
            break

# ── Print summary ────────────────────────────────────────────────────────────
print(f"\n{'Metric':<30} {'Value':>12}")
print("-" * 44)
print(f"{'Episodes evaluated':<30} {NUM_EPISODES:>12}")
print(f"{'Avg total reward':<30} {np.mean(all_rewards):>12.2f}")
print(f"{'Std total reward':<30} {np.std(all_rewards):>12.2f}")
print(f"{'Avg breakdowns per year':<30} {np.mean(all_breakdowns):>12.2f}")
print(f"{'Avg maintenances per year':<30} {np.mean(all_maintenances):>12.2f}")
print(f"{'Min reward (worst episode)':<30} {np.min(all_rewards):>12.2f}")
print(f"{'Max reward (best episode)':<30} {np.max(all_rewards):>12.2f}")

# Action distribution across all episodes
all_actions_flat = [a for trace in all_action_traces for a in trace]
total_actions    = len(all_actions_flat)
run_pct  = all_actions_flat.count(0) / total_actions * 100
mx_pct   = all_actions_flat.count(1) / total_actions * 100
em_pct   = all_actions_flat.count(2) / total_actions * 100

print(f"\n{'Action distribution':}")
print(f"  Keep running     : {run_pct:.1f}%")
print(f"  Schedule service : {mx_pct:.1f}%")
print(f"  Emergency repair : {em_pct:.1f}%")

# ── Generate plots ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Evaluation Results — DQN Predictive Maintenance Agent\n"
    "Smart Manufacturing Production Optimization | SDG 9 & SDG 12",
    fontsize=14, fontweight="bold"
)

# ── Plot 1: Health trace of best episode ─────────────────────────────────────
best_ep       = int(np.argmax(all_rewards))
health_trace  = all_health_traces[best_ep]
action_trace  = all_action_traces[best_ep]
days          = list(range(len(health_trace)))

axes[0, 0].plot(days, health_trace, color="#5B8BD0", linewidth=1.5, label="Machine health")
axes[0, 0].fill_between(days, health_trace, alpha=0.15, color="#5B8BD0")
axes[0, 0].axhline(y=0.3, color="#E8593C", linestyle="--", linewidth=1, alpha=0.7, label="Critical threshold (0.3)")
axes[0, 0].axhline(y=0.7, color="#1D9E75", linestyle="--", linewidth=1, alpha=0.7, label="Smart service zone (0.7)")

# Mark maintenance actions
for day, action in enumerate(action_trace):
    if action == 1:
        axes[0, 0].axvline(day, color="#1D9E75", alpha=0.5, linewidth=0.8)
    elif action == 2:
        axes[0, 0].axvline(day, color="#E8593C", alpha=0.7, linewidth=1)

axes[0, 0].set_xlabel("Day")
axes[0, 0].set_ylabel("Machine Health")
axes[0, 0].set_title(f"Machine Health Trace — Best Episode (Reward: {all_rewards[best_ep]:.1f})")
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_ylim(-0.05, 1.05)
axes[0, 0].grid(alpha=0.3)

# ── Plot 2: Reward distribution ───────────────────────────────────────────────
axes[0, 1].hist(all_rewards, bins=12, color="#534AB7", alpha=0.7, edgecolor="white", linewidth=0.5)
axes[0, 1].axvline(np.mean(all_rewards), color="#E8593C", linestyle="--", linewidth=2, label=f"Mean: {np.mean(all_rewards):.1f}")
axes[0, 1].set_xlabel("Total Reward per Episode")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Reward Distribution Across Evaluation Episodes")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# ── Plot 3: Breakdowns vs Maintenances per episode ───────────────────────────
x = np.arange(NUM_EPISODES)
width = 0.35
axes[1, 0].bar(x - width/2, all_breakdowns,  width, label="Breakdowns",  color="#E8593C", alpha=0.8)
axes[1, 0].bar(x + width/2, all_maintenances,width, label="Maintenances",color="#1D9E75", alpha=0.8)
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Breakdowns vs Planned Maintenances per Episode")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis="y")

# ── Plot 4: Action distribution pie chart ────────────────────────────────────
sizes  = [run_pct, mx_pct, em_pct]
labels = [f"Keep running\n({run_pct:.1f}%)",
          f"Schedule service\n({mx_pct:.1f}%)",
          f"Emergency repair\n({em_pct:.1f}%)"]
colors = ["#5B8BD0", "#1D9E75", "#E8593C"]
axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, pctdistance=0.75,
               wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[1, 1].set_title("Agent Action Distribution (All Episodes)")

plt.tight_layout()
plt.savefig("results/evaluation_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 Evaluation plots saved to results/evaluation_results.png")
print("\n✅ Evaluation complete!")
