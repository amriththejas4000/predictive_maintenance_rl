"""
train.py
========
Trains a DQN agent on the FactoryMachineEnv using Stable Baselines3.
Reads hyperparameters from a YAML config file for reproducibility.

Run:
    python train.py --config configs/dqn_v1.yaml
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.factory_env import FactoryMachineEnv

# ── Argument Parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train DQN agent for Predictive Maintenance")
parser.add_argument(
    "--config",
    type=str,
    default="configs/dqn_v1.yaml",
    help="Path to YAML config file (default: configs/dqn_v1.yaml)"
)
args = parser.parse_args()

# ── Load Config ──────────────────────────────────────────────────────────────
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

print(f"\n📄 Loaded config: {args.config}")
print(f"   Run ID     : {cfg['experiment']['run_id']}")
print(f"   Description: {cfg['experiment']['description']}\n")

# ── Output folders ───────────────────────────────────────────────────────────
os.makedirs(cfg["logging"]["model_save_path"], exist_ok=True)
os.makedirs(cfg["logging"]["log_path"],        exist_ok=True)
os.makedirs("results",                          exist_ok=True)
os.makedirs("experiments",                      exist_ok=True)


# ── Custom callback to track reward per episode ──────────────────────────────
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self):
        reward = self.locals["rewards"][0]
        self.current_rewards.append(reward)
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        return True


# ── Create and wrap environments ─────────────────────────────────────────────
def make_env():
    env = FactoryMachineEnv(max_steps=cfg["environment"]["max_steps"])
    env = Monitor(env)
    return env

train_env = DummyVecEnv([make_env])
eval_env  = DummyVecEnv([make_env])

# ── Callbacks ─────────────────────────────────────────────────────────────────
reward_logger = RewardLogger()

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = cfg["logging"]["model_save_path"],
    log_path             = cfg["logging"]["log_path"],
    eval_freq            = cfg["training"]["eval_freq"],
    n_eval_episodes      = cfg["training"]["n_eval_episodes"],
    deterministic        = True,
    verbose              = 0
)

# ── Define DQN model from config ──────────────────────────────────────────────
model = DQN(
    policy                = cfg["model"]["policy"],
    env                   = train_env,
    learning_rate         = cfg["model"]["learning_rate"],
    buffer_size           = cfg["model"]["buffer_size"],
    learning_starts       = cfg["model"]["learning_starts"],
    batch_size            = cfg["model"]["batch_size"],
    gamma                 = cfg["model"]["gamma"],
    exploration_fraction  = cfg["exploration"]["exploration_fraction"],
    exploration_final_eps = cfg["exploration"]["exploration_final_eps"],
    train_freq            = cfg["model"]["train_freq"],
    target_update_interval= cfg["model"]["target_update_interval"],
    policy_kwargs         = dict(net_arch=cfg["model"]["net_arch"]),
    verbose               = 1,
    tensorboard_log       = cfg["logging"]["log_path"]
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Training DQN agent on FactoryMachineEnv")
print("  SDGs: 9 (Industry & Innovation), 12 (Responsible Consumption)")
print("=" * 60 + "\n")

model.learn(
    total_timesteps = cfg["training"]["total_timesteps"],
    callback        = [reward_logger, eval_callback],
    progress_bar    = True
)

# ── Save final model ──────────────────────────────────────────────────────────
model.save("models/dqn_predictive_maintenance")
model.save("models/policy_v1")
print("\n✅ Model saved to models/dqn_predictive_maintenance.zip")
print("✅ Policy saved to models/policy_v1.zip")

# ── Plot training reward curve ────────────────────────────────────────────────
rewards = reward_logger.episode_rewards

if rewards:
    window   = 20
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        "DQN Training — Predictive Maintenance RL\n"
        "Smart Manufacturing Production Optimization | SDG 9 & 12",
        fontsize=14, fontweight="bold"
    )

    axes[0].plot(rewards, alpha=0.35, color="#5B8BD0", linewidth=0.8, label="Episode reward")
    axes[0].plot(
        range(window-1, len(rewards)),
        smoothed, color="#E8593C", linewidth=2.0, label=f"Smoothed (window={window})"
    )
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Episode Reward Over Training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(window-1, len(rewards)), smoothed, color="#1D9E75", linewidth=2.0)
    axes[1].fill_between(range(window-1, len(rewards)), smoothed, alpha=0.2, color="#1D9E75")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Smoothed Reward")
    axes[1].set_title("Smoothed Learning Curve (Agent Improvement Over Time)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Training curve saved to results/training_curve.png")

# ── Log to experiments CSV ────────────────────────────────────────────────────
import csv, os

csv_path   = cfg["logging"]["results_path"]
file_exists = os.path.isfile(csv_path)

avg_reward = round(float(np.mean(rewards)), 2) if rewards else 0.0

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["run_id", "episodes", "avg_reward", "avg_breakdowns",
                         "maintenance_count", "learning_rate", "epsilon", "notes"])
    writer.writerow([
        cfg["experiment"]["run_id"],
        cfg["training"]["total_timesteps"],
        avg_reward,
        "—",
        "—",
        cfg["model"]["learning_rate"],
        cfg["exploration"]["exploration_final_eps"],
        cfg["experiment"]["description"]
    ])

print(f"📋 Run logged to {csv_path}")
print("\n✅ Training complete! Run evaluate.py to see agent performance.")